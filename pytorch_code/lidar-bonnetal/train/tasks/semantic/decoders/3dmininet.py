# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different strides in the

import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch


# Depthwise Separable convolution
class moduleERS(nn.Module):
    def __init__(self, in_channels, out_channles, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, dropprob=0.,track_running_stats=True):
        super(moduleERS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channles, 1, 1, 0, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3, track_running_stats=track_running_stats)
        self.bn = nn.BatchNorm2d(out_channles, eps=1e-3, track_running_stats=track_running_stats)

        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, inputs):

        # depthwise convolution
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        # point-wise convolution
        x = self.pointwise(x)
        x = self.bn(x)


        if self.dropout.p != 0:
            x = self.dropout(x)


        if x.shape[1] == inputs.shape[1]:
            return self.relu2(x)+ inputs
        else:
            return self.relu2(x)


class BasicBlock_half(nn.Module):
    def __init__(self, inplanes, out_planes, dropprob=0):
        super(BasicBlock_half, self).__init__()
        self.conv1 = moduleERS(inplanes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, dropprob=dropprob)

    def forward(self, x):
        out = self.conv1(x)

        return out


# ******************************************************************************

class Decoder(nn.Module):
    """
       Class for DarknetSeg. Subclasses PyTorch's own "nn" module
    """

    def __init__(self, params, stub_skips, OS=16, feature_depth=1024):
        super(Decoder, self).__init__()
        self.backbone_OS = OS
        self.backbone_feature_depth = feature_depth
        print(params)
        self.block_1 = params["block_1"]
        self.block_2 = params["block_2"]
        self.features_bottleneck = int(params["features_bottleneck"])


        self.strides = [2, 2, 2, 2]
        self.strides_2 = [2, 2, 1, 1]

        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Decoder original OS: ", int(current_os))
        # redo strides according to needed stride
        for i, stride in enumerate(self.strides):
            if int(current_os) != self.backbone_OS:
                if stride == 2:
                    current_os /= 2
                    self.strides[i] = 1
                if int(current_os) == self.backbone_OS:
                    break
        print("Decoder new OS: ", int(current_os))
        print("Decoder strides: ", self.strides)

        # decoder

        self.dec4 = self._make_dec_layer(BasicBlock_half, [self.backbone_feature_depth, self.features_bottleneck],  n_blocks=self.block_1)
        self.dec3 = self._make_dec_layer(BasicBlock_half, [self.features_bottleneck, int(self.features_bottleneck/2)],  n_blocks=self.block_2)

        # layer list to execute with skips
        self.layers = [self.dec4, self.dec3]


        # last channels
        self.last_channels = int(self.features_bottleneck/2)

    def _make_dec_layer(self, block, planes, n_blocks=1, dropprob=0):
        layers = []


        #  blocks
        layers.append(("residual", block(planes[0], planes[1], dropprob=dropprob)))
        for i in range(n_blocks - 1):
            layers.append(("residual" + str(i), block(planes[1], planes[1], dropprob=dropprob)))




        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os, detach_skip = True):
        if detach_skip:
            x = x + skips[os].detach()  # add skip (detach is for non-gradient)
        else:
            x = x + skips[os]


        feats = layer(x)  # up

        x = feats
        return x, skips, int(os/2)

    def forward(self, x, skips):
        os = self.backbone_OS
        os /= 4

        x = nn.functional.interpolate(x, size=(x.shape[2]*2, x.shape[3] * 2), mode='bilinear', align_corners=True)

        x, skips, os = self.run_layer(x, self.dec4, skips, os, detach_skip = False) # detach false porque es aun encoder

        x = nn.functional.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode='bilinear', align_corners=True)


        x, skips, os = self.run_layer(x, self.dec3, skips, os, detach_skip = False) # detach false porque es la otra branch
        x = nn.functional.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode='bilinear', align_corners=True)


        return x

    def get_last_depth(self):
        return self.last_channels
