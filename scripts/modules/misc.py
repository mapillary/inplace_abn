# Copyright (c) Facebook, Inc. and its affiliates.

import torch.nn as nn


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class SingleGPU(nn.Module):
    def __init__(self, module):
        super(SingleGPU, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x.cuda(non_blocking=True))
