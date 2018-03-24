# Remote U-Net service for Ilastik

This repository contains the a module wrapping pre-trained
[U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) models
written in Pytorch, allowing them to be used as an external service for
Ilastik.  This component can be called from Ilastik with the remote server
plug-in, currently available [here](https://github.com/etrulls/ilastik), which
interfaces with the [remote server](https://github.com/etrulls/cvlab-server),
which in turns calls this service. This work has been developed by the
[Computer Vision lab at EPFL](https://cvlab.epfl.ch) within the context of the
Human Brain Project.

This repository is designed to be placed or symlinked inside the remote server
folder. A (possibly overtuned) list of requirements is given in `reqs.txt`, for
reference.

Pre-trained models for HBP datasets are too large to be uploaded to
Github: they are available
[here](http://icwww.epfl.ch/~trulls/shared/models.tar.gz); just unzip the
tarball inside this folder.

![Teaser](https://raw.githubusercontent.com/etrulls/unet-service/master/img/teaser_unet.jpg "Teaser")
