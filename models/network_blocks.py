# -*- encoding: utf-8 -*-
import numpy as np
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

    # todo: do not know the effect
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


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_top_right, patch_bot_left, patch_bot_right), dim=1)
        return self.conv(x)


# todo: do not know the effect
class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels, mid_channels, ksize=1, stride=1, act="lrelu")
        self.layer2 = BaseConv(mid_channels, in_channels, ksize=3, stride=1, act="lrelu")

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class Bottleneck(nn.Module):
    """Standard bottleneck resblock"""
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu"):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), act="silu"):
        super(SPPBottleneck, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks//2) for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, ksize=1, stride=1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv1(x)
        return x


# todo: draw csplayer struct
class CSPLayer(nn.Module):
    """
    C3 in yolov5, CSP Bottleneck with 3 convolutions

    Args:
        in_channels (int): input channels.
        out_channels (int): output channels.
        n (int): number of Bottlenecks. Default value: 1.
    """
    def __init__(self, in_channels, out_channels, n=1, short_cut=True, expansion=0.5, depthwise=False, act="silu"):
        super(CSPLayer, self).__init__()
        hidden_channels = int(in_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, short_cut, 1.0, depthwise, act=act) for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

