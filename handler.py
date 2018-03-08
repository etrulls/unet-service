import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from os.path import isfile
import os
from IPython import embed
from time import time
import tifffile
import h5py
import pickle
from shutil import copy2
from skimage.segmentation import mark_boundaries
from networks.unet import UNet, stamp, print


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--pickle", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--use-cpu", dest="use_cpu", action="store_true")
    parser.add_argument("--input", type=str, required=True)
    # parser.add_argument("--output_seg", type=str, required=True)
    # parser.add_argument("--output_vis", type=str, required=True)
    parser.set_defaults(use_cpu=False)
    params = parser.parse_args()

    # Config
    with open(params.pickle, 'rb') as f:
        config = pickle.load(f)

    # Number of threads
    torch.set_num_threads(params.threads)
    if not params.use_cpu:
        print(stamp("Running on GPU {}".format(os.environ["CUDA_VISIBLE_DEVICES"])))
    else:
        print(stamp("Running on CPU"))

    config["batch_size"] = params.batch_size
    config["use_cpu"] = params.use_cpu

    # Compute input shape
    c = UNet.get_optimal_shape(
        output_shape_lower_bound=config["output_size"],
        steps=config["num_unet_steps"],
        num_convs=config["num_unet_convs"])
    input_size = [int(ci) for ci in c["input"]]
    m = np.asarray(input_size) - np.asarray(config["output_size"])
    if len(np.unique(m)) == 1:
        config["margin"] = m[0]
    else:
        raise RuntimeError("Should never be here?")

    if config["model"] == "UNet":
        print(stamp("Instantiating UNet"))
        model = UNet(
            steps=config["num_unet_steps"],
            num_input_channels=np.int64(1),
            first_layer_channels=config["num_unet_filters"],
            num_classes=np.int64(2),
            num_convs=config["num_unet_convs"],
            output_size=config["output_size"],
            pooling=config["pooling"],
            activation=config["activation"],
            use_dropout=config["use_dropout"],
            use_batchnorm=config["use_batchnorm"],
            init_type=config["init_type"],
            final_unit=config["final_unit"],
            use_cpu=params.use_cpu,
        )

        # Need to overwrite this
        if model.is_3d:
            config["input_size"] = model.input_size
        else:
            config["input_size"] = [model.input_size[1], model.input_size[2]]
        config["margin"] = model.margin
        print(stamp("UNet -> Input size: {}. Output size: {}".format(
            config["input_size"], config["output_size"])))
    else:
        raise RuntimeError("Unknown model")

    if not config["use_cpu"]:
        model.cuda()

    if model.is_3d:
        config["input_size"] = model.input_size
    else:
        config["input_size"] = [model.input_size[1], model.input_size[2]]

    # Load some data
    ext = params.input.split('.')[-1]
    if ext in ['tif', 'tiff']:
        stack = tifffile.imread(params.input)
    elif ext == 'h5':
        with h5py.File(params.input, 'r') as f:
            stack = f['data'].value.copy()
    else:
        raise RuntimeError('Bad input format')

    if len(stack.shape) == 3:
        stack = np.expand_dims(stack, axis=1)

    # TODO try mean/std used in training instead
    data, mean, std = [], [], []
    for s in stack:
        data.append(s)
        mean.append(s.mean())
        std.append(s.std())

    # pad the image
    pad = config['margin']
    if model.is_3d:
        raise RuntimeError("TODO idc right now")
    else:
        pad_dims = []
        for p in pad:
            p = int(p / 2)
            pad_dims.append([p] * 2)

    data_padded = []
    for s in data:
        data_padded.append(np.pad(s, pad_dims, 'reflect'))

    # Load model
    print(stamp("Loading model: '{}'".format(params.weights)))
    model.load_state_dict(torch.load(params.weights))

    model.eval()

    # Test a full stack with mirroring, accounting for boundaries etc.
    # For 2D data:
    # "images" should be a list of slices size 1xMxN, mirror-padded (config['margin']/2) on each side (x-y)
    # "mean"/"std" should be a list the same size with a single value (can be constant)
    prediction = model.inference(
        {
            "images": data_padded,
            "mean": mean,
            "std": std,
        },
        config['batch_size'],
        config['use_lcn'],
    )

    # Treshold
    for i in range(len(prediction)):
        if model.is_3d:
            prediction[i] = prediction[i][:, 1, :, :, :] - \
                prediction[i][:, 0, :, :, :]
        else:
            prediction[i] = np.expand_dims(
                prediction[i][0, 1, :, :] - prediction[i][0, 0, :, :],
                axis=0)

    # Save
    root = "{}/workspace/{}/runs/model-{}/data-{}".format(
        os.path.dirname(os.path.realpath(__file__)),
        params.username,
        params.model_name,
        params.dataset_name,
    )
    if not os.path.isdir(root):
        os.makedirs(root)

    print(stamp('Saving results...'))
    out = root + '/output.h5'
    with h5py.File(out, 'w') as f:
        f.create_dataset(name='data', data=np.array(prediction)[:, 0, :, :], chunks=True)
    print(stamp('Done!'))
