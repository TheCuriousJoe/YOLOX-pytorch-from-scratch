# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn


class SiLU(nn.Module):
    # export-friendly version of nn.SiLU(), but PyTorch 1.8.0+ starts to support exporting nn.SiLU to ONNX
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(inplace=inplace)
    else:
        raise AttributeError("Unsupported activation type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """
    Basic convolution layer in YOLOX: Conv2d -> Batchnorm -> silu/leaky relu block
    """
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """
    Depth-wise convolution layer for insufficient computing resources
    """
    def __init__(self, in_channels, out_channels, ksize, stride, act="silu"):
        super().__init__()

        """
        深度可分离卷积中的depth-wise conv，实现方式为torch.nn.Conv2d(..., groups=in_channels, ...):
        - group=1，跟普通卷积一样，特征图的通道数不分离
        - group=2，把特征图分成两部分，如640 * 640 * 4的特征图，分为640 * 640 * 2，640 * 640 * 2两张特征图，
          此时有两个3 * 3 * 2卷积核，各自对两张特征图进行卷积后拼接
        - group=in_channels，按照输入特征图的通道数对特征图进行切割，如640 * 640 * 4的特征图，分成640 * 640 * 1 * 4四张特征图，
          此时有四个3 * 3 * 1卷积核，各自对4张特征图进行卷积后拼接
        """
        self.space_conv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)

        """
           深度可分离卷积中的point-wise conv，实现特征图的升维
        """
        self.depth_conv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        return self.depth_conv(self.space_conv(x))

