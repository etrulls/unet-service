from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from time import time
from datetime import datetime
from IPython import embed
from tqdm import tqdm
# from utils import get_Wb
import tifffile


def print(s, end='\n', file=sys.stdout):
    '''
    Quick hack to print and flush.
    '''
    file.write(s + end)
    file.flush()


def get_Wb(model):
    _w, _b = [], []
    for name, param in model.named_parameters():
        if '.weight' in name:
            _w.append(param.view(-1).data.numpy())
        elif '.bias' in name:
            _b.append(param.view(-1).data.numpy())
    _w = np.concatenate(_w, axis=0)
    _b = np.concatenate(_b, axis=0)
    return _w, _b


def stamp(string):
    return datetime.fromtimestamp(time()).strftime('%H:%M:%S ') + string


class UNetBlock(nn.Module):
    def __init__(
            self,
            is_3d,
            num_convs,
            features,
            downsample,
            upsample,
            pooling,
            activation,
            use_dropout,
            use_batchnorm):
        super().__init__()

        rect = getattr(nn, activation)

        layers = []
        if downsample:
            if pooling == 'MaxPool':
                if is_3d:
                    layers += [nn.MaxPool3d(2, stride=2, ceil_mode=True)]
                else:
                    layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
            else:
                raise RuntimeError('Invalid pooling type')

        # if len(features) < 2 or len(features) > 3:
        #     raise RuntimeError(
        #         'Number of convolutions per layer can be 1 or 2')

        for c in range(1, num_convs + 1):
            if is_3d:
                layers += [nn.Conv3d(features[c - 1], features[c], 3)]
            else:
                layers += [nn.Conv2d(features[c - 1], features[c], 3)]
            layers += [rect()]
            if use_batchnorm:
                if is_3d:
                    layers += [nn.BatchNorm3d(features[c])]
                else:
                    layers += [nn.BatchNorm2d(features[c])]

        if use_dropout:
            layers += [nn.Dropout()]

        if upsample:
            if is_3d:
                layers += [nn.ConvTranspose3d(
                    features[-1], features[-1] // 2, 2, stride=2)]
            else:
                layers += [nn.ConvTranspose2d(
                    features[-1], features[-1] // 2, 2, stride=2)]
            # layers += [rect()]
            if use_batchnorm:
                if is_3d:
                    layers += [nn.BatchNorm3d(features[-1] // 2)]
                else:
                    layers += [nn.BatchNorm2d(features[-1] // 2)]

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class UNet(nn.Module):

    @staticmethod
    def margin(steps, num_convs):
        '''
        Compute number of pixels lost with convolutions/pooling.
        '''

        s = UNet.get_optimal_shape((0,), steps=steps, num_convs=num_convs)
        return (s['input'][0] - s['output'][0])

    @staticmethod
    def get_output_shape(input_shape, steps, num_convs):
        '''
        Get output size given an input size.
        '''

        shave = 2 * num_convs
        first = lambda x: x - shave
        down = lambda x: (x - 1) // 2 + 1 - shave
        up = lambda x: (x * 2) - shave

        s = np.asarray(input_shape)

        s = first(s)
        for i in range(steps):
            s = down(s)
        for i in range(steps):
            s = up(s)

        return tuple(s)

    @staticmethod
    def get_optimal_shape(output_shape_lower_bound, steps, num_convs):
        '''
        Get optimal input and output size given a lower bound for the output.
        '''

        shave = 2 * num_convs
        up = lambda x: (x * 2) - shave
        rev_first = lambda x: x + shave
        rev_down = lambda x: (x + shave) * 2
        rev_up = lambda x: (x + shave - 1) // 2 + 1

        s = np.asarray(output_shape_lower_bound)

        for i in range(steps):
            s = rev_up(s)

        # Compute correct out shape from minimum shape
        o = np.copy(s)
        for i in range(steps):
            o = up(o)

        # Best input shape
        for i in range(steps):
            s = rev_down(s)

        s = rev_first(s)

        return {'input': tuple(s), 'output': tuple(o)}

    @staticmethod
    def get_shape_combinations(upper_bound, steps, num_convs):
        valid = []
        for s in range(upper_bound):
            r = UNet.get_optimal_shape([s], steps=steps, num_convs=num_convs)
            if r not in valid:
                valid.append(r)

        print(stamp('Valid combinations: {}'.format(len(valid))))
        for v in valid:
            print(stamp('Input: {} -> Output: {}'.format(v['input'][0], v['output'][0])))

    @staticmethod
    def pad_widths(output_shape, steps, num_convs):
        s = UNet.get_optimal_shape(
            output_shape,
            steps=steps,
            num_convs=num_convs)
        s_in = s['input']
        s_out = s['output']

        input_padding = [((sh_o - sh_i) // 2, (sh_o - sh_i - 1) // 2 + 1)
                         for sh_i, sh_o in zip(output_shape, s_in)]
        output_padding = [((sh_o - sh_i) // 2, (sh_o - sh_i - 1) // 2 + 1)
                          for sh_i, sh_o in zip(output_shape, s_out)]

        return {'input': input_padding, 'output': output_padding}

    def __init__(self,
                 steps,
                 num_input_channels,
                 first_layer_channels,
                 num_classes,
                 num_convs,
                 output_size,
                 pooling='maxpool',
                 activation='relu',
                 use_dropout=False,
                 use_batchnorm=False,
                 init_type='xavier_normal',
                 final_unit=None,
                 use_cpu=True):
        super().__init__()

        # torch does not like combinations of ints and np.ints
        steps = int(steps)
        num_input_channels = int(num_input_channels)
        first_layer_channels = int(first_layer_channels)
        num_classes = int(num_classes)
        for i in range(len(output_size)):
            output_size[i] = int(output_size[i])

        # save some to attributes
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.is_3d = True if len(output_size) == 3 else False
        self.init_type = init_type
        self.use_cpu = use_cpu

        # Sanity check
        if any(s == 0 for s in output_size) == 1:
            raise RuntimeError('Input patch values should be > 1')

        # Check output shape
        pad = UNet.pad_widths(
            output_shape=output_size,
            steps=steps,
            num_convs=num_convs)
        if sum([sum(p) for p in pad['output']]) > 0:
            raise RuntimeError(
                'Invalid output size: requires padding {}'.format(
                    pad['output']))

        # Get input shape
        c = UNet.get_optimal_shape(
            output_shape_lower_bound=output_size,
            steps=steps,
            num_convs=num_convs)
        self.input_size = [int(ci) for ci in c['input']]
        if not self.is_3d:
            self.input_size = [1] + self.input_size
        self.output_size = [int(ci) for ci in c['output']]
        if not self.is_3d:
            self.output_size = [1] + self.output_size
        self.margin = np.asarray(self.input_size) - np.asarray(self.output_size)
        # if not self.is_3d:
        #     self.margin = [0] + self.margin
        for margin in self.margin:
            if margin % 2:
                raise RuntimeError('Network margin is odd?')

        # Calls such as cpu(), cuda(), parameters() etc pass through only
        # for nn.Module objects, but not e.g. lists of the same.
        # This is a bit hacky but works.
        self.steps = steps

        # First
        num_down = 1
        setattr(self,
                'down{}'.format(num_down),
                UNetBlock(
                    self.is_3d,
                    features=[num_input_channels] + [first_layer_channels] * num_convs,
                    num_convs=num_convs,
                    downsample=False,
                    upsample=False,
                    pooling=pooling,
                    activation=activation,
                    use_dropout=False,
                    use_batchnorm=use_batchnorm,
                ))

        # Contracting path
        s = first_layer_channels
        for i in range(1, steps):
            # Use dropout on the last down step only
            num_down += 1
            setattr(self,
                    'down{}'.format(num_down),
                    UNetBlock(
                        self.is_3d,
                        features=[s] + [2 * s] * num_convs,
                        num_convs=num_convs,
                        downsample=True,
                        upsample=False,
                        pooling=pooling,
                        activation=activation,
                        use_dropout=False,
                        use_batchnorm=use_batchnorm,
                    ))
            # use_dropout=(steps == i) and use_dropout,
            s *= 2

        self.center = UNetBlock(
            self.is_3d,
            features=[s] + [2 * s] * num_convs,
            num_convs=num_convs,
            downsample=True,
            upsample=True,
            pooling=pooling,
            activation=activation,
            use_dropout=use_dropout,
            use_batchnorm=use_batchnorm,
        )

        # Up
        num_up = 0
        for i in range(steps - 1):
            num_up += 1
            setattr(self,
                    'up{}'.format(num_up),
                    UNetBlock(
                        self.is_3d,
                        features=[2 * s] + [s] * num_convs,
                        num_convs=num_convs,
                        downsample=False,
                        upsample=True,
                        pooling=pooling,
                        activation=activation,
                        use_dropout=False,
                        use_batchnorm=use_batchnorm,
                    ))
            s = s // 2

        # Last
        num_up += 1
        setattr(self,
                'up{}'.format(num_up),
                UNetBlock(
                    self.is_3d,
                    features=[2 * s] + [s] * num_convs,
                    num_convs=num_convs,
                    downsample=False,
                    upsample=False,
                    pooling=pooling,
                    activation=activation,
                    use_dropout=False,
                    use_batchnorm=use_batchnorm,
                ))

        layers = []
        if self.is_3d:
            layers += [nn.Conv3d(s, num_classes, 1)]
        else:
            layers += [nn.Conv2d(s, num_classes, 1)]

        if final_unit == "relu":
            layers += [nn.ReLU()]
        if final_unit == "relu+tanh":
            layers += [nn.ReLU()]
            layers += [nn.Tanh()]
        elif final_unit != "none":
            raise RuntimeError("Unsupported final unit type")

        self.final = nn.Sequential(*layers)

        # Initialize weights
        units = []
        for i in range(1, self.steps + 1):
            units.append(getattr(self, 'down{}'.format(i)).seq)
            units.append(getattr(self, 'up{}'.format(i)).seq)
        units.append(self.center.seq)
        units.append(self.final)

        # Get random values
        # _w, _b = get_Wb(self)

        for unit in units:
            unit.apply(self.weights_init)

        # Get initialized values
        # Keep in mind that we're only initializing some layers (e.g. not batchnorm)
        # _w, _b = get_Wb(self)

    def weights_init(self, m):
        for valid in [
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose1d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d]:
            if isinstance(m, valid):
                getattr(torch_init, self.init_type)(m.weight.data)
                # set bias to zero
                m.bias.data.fill_(0)

    def forward(self, x):
        y_down = []
        for i in range(1, self.steps + 1):
            x = getattr(self, 'down{}'.format(i))(x)
            y_down.append(x)

        x = self.center(x)

        # Get number of pixels to crop from downwards branch
        # Then concatenate inputs
        for i in range(1, self.steps + 1):
            crop = y_down[-i].size(-1) - x.size(-1)
            crop = [int(crop // 2), int(crop // 2 + crop % 2)]

            if self.is_3d:
                x = getattr(self, 'up{}'.format(i))(
                    torch.cat([x,
                               y_down[-i][
                                   :,
                                   :,
                                   crop[0]:-crop[1],
                                   crop[0]:-crop[1],
                                   crop[0]:-crop[1]
                               ]
                               ], 1))
            else:
                crop = y_down[-i].size(-1) - x.size(-1)
                crop = [int(crop // 2), int(crop // 2 + crop % 2)]

                x = getattr(self, 'up{}'.format(i))(
                    torch.cat([x,
                               y_down[-i][
                                   :,
                                   :,
                                   crop[0]:-crop[1],
                                   crop[0]:-crop[1]
                               ]
                               ], 1))

        return self.final(x) / 10

    def chunkify(self, image_size):
        acc = [0, 0, 0]
        n = [0, 0, 0]
        for dim in range(3):
            n[dim] += 1
            acc[dim] = acc[dim] + self.input_size[dim]
            for i in range(1, image_size[dim]):
                n[dim] += 1
                acc[dim] = acc[dim] + self.input_size[dim] - self.margin[dim]
                if acc[dim] >= image_size[dim]:
                    break
        return n

    def get_chunk(self, indices, chunks, image_size):
        inner, outer = [], []
        for i in range(3):
            buf = int(self.margin[i] / 2)
            if indices[i] == 0:
                outer.append([
                    0,
                    (indices[i] + 1) * self.input_size[i]
                ])
                inner.append([
                    0,
                    (indices[i] + 1) * self.output_size[i]
                ])
            elif indices[i] == chunks[i] - 1:
                outer.append([
                    image_size[i] - self.input_size[i],
                    image_size[i]
                ])
                inner.append([
                    image_size[i] - 2 * buf - self.output_size[i],
                    image_size[i] - 2 * buf
                ])
            else:
                outer.append([
                    indices[i] * self.output_size[i],
                    indices[i] * self.output_size[i] + self.input_size[i],
                ])
                inner.append([
                    indices[i] * self.output_size[i],
                    (indices[i] + 1) * self.output_size[i]
                ])

            if outer[i][0] < 0 or outer[i][1] > image_size[i]:
                raise RuntimeError("Out of bounds")

        return outer, inner

    def inference(self, data, batch_size, use_lcn):
        '''
        Compute prediction. Data should be a list of images or stacks.
        '''
        self.eval()

        for key in ["images", "mean", "std"]:
            if not isinstance(data[key], list):
                data[key] = list(data[key])

        res_list = []
        # Feedback per image basis when doing 2D
        # if not self.is_3d:
        #     bar = tqdm(total=len(data['images']))
        for it, (image, mean, std) in enumerate(zip(data['images'], data['mean'], data['std'])):
            if self.is_3d:
                imsize = image.shape
            else:
                imsize = [1, image.shape[1], image.shape[2]]
            n = self.chunkify(imsize[len(imsize) - 3:])
            num_n = n[0] * n[1] * n[2]

            # Feedback per stack basis when doing 3D
            # if self.is_3d:
            #     bar = tqdm(total=num_n)

            if self.is_3d:
                res = np.zeros(
                    shape=[
                        1,
                        self.num_classes,
                        imsize[1] - self.margin[0],
                        imsize[2] - self.margin[1],
                        imsize[3] - self.margin[2]],
                    dtype=np.float)
                batch = torch.Tensor(
                    batch_size,
                    self.num_input_channels,
                    self.input_size[0],
                    self.input_size[1],
                    self.input_size[2])
            else:
                res = np.zeros(
                    shape=[
                        imsize[0] - self.margin[0],
                        self.num_classes,
                        imsize[1] - self.margin[1],
                        imsize[2] - self.margin[2]],
                    dtype=np.float)
                batch = torch.Tensor(
                    batch_size,
                    self.num_input_channels,
                    self.input_size[1],
                    self.input_size[2])

            curr_outer, curr_inner = [], []
            last_batch = False
            for z in range(n[0]):
                for y in range(n[1]):
                    for x in range(n[2]):
                        if self.is_3d:
                            try:
                                outer, inner = self.get_chunk(
                                    [z, y, x], n, imsize[len(imsize) - 3:])
                                batch[len(curr_outer)] = torch.from_numpy(
                                    image[
                                        :,
                                        outer[0][0]:outer[0][1],
                                        outer[1][0]:outer[1][1],
                                        outer[2][0]:outer[2][1]
                                    ].astype(np.float)
                                )
                            except:
                                print(stamp("Can't do this (3D)"))
                                embed()
                        else:
                            try:
                                outer, inner = self.get_chunk(
                                    [z, y, x],
                                    n,
                                    [1, image.shape[1], image.shape[2]])
                                batch[len(curr_outer)] = torch.from_numpy(
                                    image[
                                        :,
                                        outer[1][0]:outer[1][1],
                                        outer[2][0]:outer[2][1]
                                    ].astype(np.float)
                                )
                            except:
                                print(stamp("Can't do this (2D)"))
                                embed()

                        # LCN
                        if use_lcn:
                            batch[len(curr_outer)] = (
                                batch[len(curr_outer)]
                                - batch[len(curr_outer)].mean()) \
                                / batch[len(curr_outer)].std()
                        else:
                            batch[len(curr_outer)] = (
                                batch[len(curr_outer)] - mean) / std

                        curr_outer.append(outer)
                        curr_inner.append(inner)

                        if (z + 1) * (y + 1) * (x + 1) == num_n:
                            last_batch = True

                        if len(curr_outer) == batch_size or last_batch:
                            if self.use_cpu:
                                r = self.forward(Variable(
                                    batch[:len(curr_outer)]
                                )).data.numpy()
                            else:
                                r = self.forward(Variable(
                                    batch[:len(curr_outer)]
                                ).cuda()).data.cpu().numpy()

                            # r = batch[:len(curr_outer)].numpy() * data['std'] + data['mean']
                            for i, c in enumerate(curr_inner):
                                if self.is_3d:
                                    res[:,
                                        :,
                                        c[0][0]:c[0][1],
                                        c[1][0]:c[1][1],
                                        c[2][0]:c[2][1],
                                        ] = r[i]
                                else:
                                    res[c[0][0]:c[0][1],
                                        :,
                                        c[1][0]:c[1][1],
                                        c[2][0]:c[2][1],
                                        ] = r[i]

                            # Inner loop (xyz)
                            # if self.is_3d:
                                # bar.update(len(curr_outer))

                            curr_outer, curr_inner = [], []

            # End of loop over (list of) images/stacks
            if not self.is_3d:
                # bar.update(1)
                print(stamp('Testing: {}/{}'.format(it + 1, len(data['images']))))

            res_list.append(res)
        # bar.close()

        self.train()

        return res_list
