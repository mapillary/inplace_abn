# In-Place Activated BatchNorm

[**In-Place Activated BatchNorm for Memory-Optimized Training of DNNs**](http://arxiv.org/abs/1234.56789)

In-Place Activated BatchNorm (InPlace-ABN) is a novel approach to reduce the memory required for training deep networks.
It allows for up to 50% memory savings in modern architectures such as ResNet, ResNeXt and Wider ResNet by redefining
BN + non linear activation as a single in-place operation, while smartly dropping or recomputing intermediate buffers as
needed.

This repository contains a [PyTorch](http://pytorch.org/) implementation of the InPlace-ABN layer, as well as some
training scripts to reproduce the ImageNet classification results reported in our paper.

- [Overview](#overview)
- [Installation](#installation)
- [Training on ImageNet](#training-on-imagenet)

## Overview

<p align="center"><img width="70%" src="inplace_abn.png" /></p>

TODO: method summary

## Installation

Our code has only been tested under Linux with CUDA 8.0 / 9.0 and CUDNN 7.0.

### Requirements

To install PyTorch, please refer to https://github.com/pytorch/pytorch#installation.

**NOTE: due to unresolved conflicts with PyTorch master, our code _requires_ PyTorch v0.2**.

For all other dependencies, just run:
```bash
pip install -r requirements.txt
```

### Compiling

Some parts of InPlace-ABN have a native CUDA implementation, which must be compiled with the following commands:
```bash
cd modules
sh build.sh
python build.py
``` 
The `build.sh` script assumes that the `nvcc` compiler is available in the current system search path.
The CUDA kernels are compiled for `sm_50`, `sm_52` and `sm_61` by default.
To change this (_e.g._ if you are using a Kepler GPU), please edit the `CUDA_GENCODE` variable in `build.sh`.

## Training on ImageNet

Results from our arXiv paper (top-5 / top-10):

| Network                           | Batch | 224            | 224, 10-crops  | 320           |
|-----------------------------------|-------|----------------|----------------|---------------|
| [ResNeXt101, Std-BN][1]           | 256   | 77.04 / 93.50  | 78.72 / 94.47  | 77.92 / 94.28 |
| [ResNeXt101, InPlace-ABN][2]      | 512   | 78.08 / 93.79  | 79.52 / 94.66  | 79.38 / 94.67 |
| [ResNeXt152, InPlace-ABN][3]      | 256   | 78.28 / 94.04  | 79.73 / 94.82  | 79.56 / 94.67 |
| [WideResNet38, InPlace-ABN][4]    | 256   | 79.72 / 94.78  | 81.03 / 95.43  | 80.69 / 95.27 |
| [ResNeXt101, InPlace-ABN sync][5] | 256   | 77.70 / 93.78  | 79.18 / 94.60  | 78.98 / 94.56 |

[1]: experiments/resnext101_stdbn_lr_256.json
[2]: experiments/resnext101_ipabn_lr_512.json
[3]: experiments/resnext152_ipabn_lr_256.json
[4]: experiments/wider_resnet38_ipabn_lr_256.json
[5]: experiments/resnext101_ipabn-sync_lr_256.json

### Data preparation

Our scripts uses [torchvision.datasets.ImageFolder](http://pytorch.org/docs/master/torchvision/datasets.html#torchvision.datasets.ImageFolder)
for loading ImageNet data, which expects folders organized as follows:
```
root/train/[class_id1]/xxx.{jpg,png,jpeg}
root/train/[class_id1]/xxy.{jpg,png,jpeg}
root/train/[class_id2]/xxz.{jpg,png,jpeg}
...

root/val/[class_id1]/asdas.{jpg,png,jpeg}
root/val/[class_id1]/123456.{jpg,png,jpeg}
root/val/[class_id2]/__32_.{jpg,png,jpeg}
...
```
Images can have any name, as long as the extension is that of a recognized image format.
Class ids are also free-form, but they are expected to match between train and validation data.
Note that the training data in the standard ImageNet distribution is already given in the required format, while
validation images need to be split into class sub-folders as described above.  

### Training

The main training script is `train_imagenet.py`: this supports training on ImageNet, or any other dataset formatted
as described above, while keeping a log of relevant metrics in Tensorboard format and periodically saving snapshots.
Most training parameters can be specified as a `json`-formatted configuration file (look [here](imagenet/config.py)
for a complete list of configurable parameters).
All parameters not explicitly specified in the configuration file are set to their defaults, also available in
[imagenet/config.py](imagenet/config.py).

Our arXiv results can be reproduced by running `train_imagenet.py` with the configuration files in `./experiments`.
As an example, the command to train `ResNeXt101` with InPlace-ABN, Leaky ReLU and `batch_size = 512` is:
```bash
python train_imagenet.py --log_dir /path/to/tensorboard/logs experiments/resnext101_ipabn_lr_512.json /path/to/imagenet/root
```

### Validation

Validation is run by `train_imagenet.py` at the end of every training epoch.
To validate a trained model, you can use the `test_imagenet.py` script, which allows for 10-crops validation and
transferring weights across compatible networks (_e.g._ from `ResNeXt101` with ReLU to `ResNeXt101` with Leaky
ReLU).
This script accepts the same configuration files as `train_imagenet.py`, but note that the `scale_val` and `crop_val`
parameters are ignored in favour of the `--scale` and `--crop` command-line arguments.

As an example, to validate the `ResNeXt101` trained above using 10-crops of size `224` from images scaled to `256`
pixels, you can run:
```bash
python test_imagenet.py --crop 224 --scale 256 --ten_crops experiments/resnext101_ipabn_lr_512.json /path/to/checkpoint /path/to/imagenet/root
```