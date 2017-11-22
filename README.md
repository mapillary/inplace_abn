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

![InPlace-ABN forward and backward computation](inplace_abn.png | width=500)

TODO: method summary

## Installation

Note our code has only been tested under Linux with CUDA 8.0 / 9.0 and CUDNN 7.0. 

### Requirements

To install PyTorch, please refer to https://github.com/pytorch/pytorch#installation.

**NOTE: due to currently unresolved bugs in PyTorch master, our code _requires_ PyTorch v0.2**.

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
This assumes that the `nvcc` compiler is available in the current `PATH`.

## Training on ImageNet

TODO: ImageNet training scripts instructions