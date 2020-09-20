# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different strides in the

import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

import torch

from torch.nn import functional as f





class moduleProyection(nn.Module):
    def __init__(self, channels_in, channels_out, channels=[16, 32, 64, 128], conv_feature=128, neighbours=16,
                 track_running_stats=True):
        super(moduleProyection, self).__init__()


        # Local Feature Extractor

        # 1st FC
        self.conv_fc1 = nn.Conv2d(channels_in, channels[0], 1, 1, 0, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0], eps=1e-3, track_running_stats=track_running_stats)
        self.relu1 = nn.ReLU()

        # 2nd FC
        self.conv_fc2 = nn.Conv2d(channels[0], channels[1], 1, 1, 0, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1], eps=1e-3, track_running_stats=track_running_stats)
        self.relu2 = nn.ReLU()

        # 3rd FC
        self.conv_fc3 = nn.Conv2d(channels[1], channels[2], 1, 1, 0, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[2], eps=1e-3, track_running_stats=track_running_stats)
        self.relu3 = nn.ReLU()

        # 4th FC
        self.conv_fc4 = nn.Conv2d(channels[2], channels[3], 1, 1, 0, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(channels[3], eps=1e-3, track_running_stats=track_running_stats)
        self.relu4 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=(neighbours, 1), stride=(1, 1), padding=0)



        # Context Feature Extractor
        self.pool_context = nn.MaxPool2d(kernel_size=(9, 1), stride=(1, 1), padding=0)

        # Context with dilation 1
        self.conv_fc_context_1 = nn.Conv2d(channels[1], channels[2], 1, 1, 0, 1, 1, bias=False)
        self.bn_context_1 = nn.BatchNorm2d(channels[2], eps=1e-3, track_running_stats=track_running_stats)
        self.relu_context_1 = nn.ReLU()

        # Context with dilation 2
        self.conv_fc_context_2 = nn.Conv2d(channels[1], channels[1], 1, 1, 0, 1, 1, bias=False)
        self.bn_context_2 = nn.BatchNorm2d(channels[1], eps=1e-3, track_running_stats=track_running_stats)
        self.relu_context_2 = nn.ReLU()

        # Context with dilation 3
        self.conv_fc_context_3 = nn.Conv2d(channels[1], channels[1], 1, 1, 0, 1, 1, bias=False)
        self.bn_context_3 = nn.BatchNorm2d(channels[1], eps=1e-3, track_running_stats=track_running_stats)
        self.relu_context_3 = nn.ReLU()

        # Spatial Feature Extractor
        self.conv = nn.Conv2d(channels_in, conv_feature, (neighbours, 1), 1, 0, 1, 1, bias=False)
        self.bn_conv = nn.BatchNorm2d(conv_feature, eps=1e-3, track_running_stats=track_running_stats)
        self.relu_conv = nn.ReLU()


        # Feature Fusion
        self.conv_atten = nn.Conv2d(channels[-1] + conv_feature + channels[3],
                                    channels[-1] + conv_feature + channels[3], kernel_size=1, bias=True)
        self.sigmoid_atten = nn.Sigmoid()

        # Bottleneck
        self.conv_out = nn.Conv2d(channels[-1] + conv_feature + channels[3], channels_out, 1, 1, 0, 1, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels_out, eps=1e-3, track_running_stats=track_running_stats)
        self.relu_out = nn.ReLU()

    def forward(self, inputs, size=[16, 512]):

        # Local Feature Extractor

        # 1st FC
        x = self.conv_fc1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        # 2nd FC
        x = self.conv_fc2(x)
        x = self.bn2(x)
        x2 = self.relu2(x)

        # 3rd FC
        x = self.conv_fc3(x2)
        x = self.bn3(x)
        x = self.relu3(x)

        # 4th FC
        x = self.conv_fc4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool(x)


        # Context Feature Extractor
        x_context = self.pool(x2)

        # reshape to image-like shape
        x_context = x_context.squeeze(2)
        n, c, _ = x_context.size()
        h = size[0]
        w = size[1]
        x_context = x_context.view(n, c, h, w)

        # fast point grouping for dilation 1, 2 and 3
        windows_size = 3
        list_dil_1 = []
        list_dil_2 = []
        list_dil_3 = []
        for b in range(int(x_context.shape[0])):
            dil_1 = f.unfold(x_context[b, ...].unsqueeze(1), kernel_size=windows_size, stride=1, padding=1, dilation=1)
            dil_2 = f.unfold(x_context[b, ...].unsqueeze(1), kernel_size=windows_size, stride=1, padding=2, dilation=2)
            dil_3 = f.unfold(x_context[b, ...].unsqueeze(1), kernel_size=windows_size, stride=1, padding=3, dilation=3)
            list_dil_1.append(dil_1.unsqueeze(0))
            list_dil_2.append(dil_2.unsqueeze(0))
            list_dil_3.append(dil_3.unsqueeze(0))

        x_context_1 = torch.cat(list_dil_1, dim=0)
        x_context_2 = torch.cat(list_dil_2, dim=0)
        x_context_3 = torch.cat(list_dil_3, dim=0)


        # Context with dilation 1
        x_context_1 = self.conv_fc_context_1(x_context_1)
        x_context_1 = self.bn_context_1(x_context_1)
        x_context_1 = self.relu_context_1(x_context_1)
        x_context_1 = self.pool_context(x_context_1)

        # Context with dilation 2
        x_context_2 = self.conv_fc_context_2(x_context_2)
        x_context_2 = self.bn_context_2(x_context_2)
        x_context_2 = self.relu_context_2(x_context_2)
        x_context_2 = self.pool_context(x_context_2)

        # Context with dilation 3
        x_context_3 = self.conv_fc_context_3(x_context_3)
        x_context_3 = self.bn_context_3(x_context_3)
        x_context_3 = self.relu_context_3(x_context_3)
        x_context_3 = self.pool_context(x_context_3)

        x_context_1 = torch.cat((x_context_1, x_context_2, x_context_3), dim=1)

        # Reshape to image-like
        x_context_1 = x_context_1.squeeze(2)
        n, c, _ = x_context_1.size()
        x_context_1 = x_context_1.view(n, c, h, w)


        # Spatial Feature Extractor
        x_conv = self.conv(inputs)
        x_conv = self.bn_conv(x_conv)
        x_conv = self.relu_conv(x_conv)
        x = torch.cat((x, x_conv), dim=1)

        # reshape to image-like shape
        x = x.squeeze(2)
        n, c, _ = x.size()
        h = size[0]
        w = size[1]
        x = x.view(n, c, h, w)

        x = torch.cat((x, x_context_1), dim=1)


        # Feature Fusion
        atten = F.adaptive_avg_pool2d(x, 1)
        atten = self.conv_atten(atten)
        atten = self.sigmoid_atten(atten)
        x = torch.mul(x, atten)

        # Bottleneck
        x = self.conv_out(x)
        x = self.bn_out(x)
        x = self.relu_out(x)

        return x

# Depthwise Separable convolution
class moduleERS(nn.Module):
    def __init__(self, in_channels, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, dropprob=0., mul = 1, track_running_stats=True):
        super(moduleERS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_planes*mul, 1, 1, 0, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3, track_running_stats=track_running_stats)
        self.bn = nn.BatchNorm2d(out_planes*mul, eps=1e-3, track_running_stats=track_running_stats)
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

# Multi-dilation Depthwise Separable convolution
class moduleERS_muldil(nn.Module):
    def __init__(self, in_channels, out_planes, kernel_size=3, stride=1, padding=1, dilation=[1, 8], bias=False, dropprob=0., mul=1, track_running_stats=True):
        super(moduleERS_muldil, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, dilation, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_planes*mul, 1, 1, 0, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3, track_running_stats=track_running_stats)
        self.bn2 = nn.BatchNorm2d(in_channels, eps=1e-3, track_running_stats=track_running_stats)
        self.bn = nn.BatchNorm2d(out_planes*mul, eps=1e-3, track_running_stats=track_running_stats)

        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.relu3 = nn.LeakyReLU(0.1)

        self.dropout = nn.Dropout2d(dropprob)


    def forward(self, inputs):

        # depthwise convolution
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        # dilated depthwise convolution
        x2 = self.conv2(inputs)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x += x2

        # point-wise convolution
        x = self.pointwise(x)
        x = self.bn(x)

        if self.dropout.p != 0:
            x = self.dropout(x)

        if x.shape[1] == inputs.shape[1]:
            return self.relu3(x)+ inputs
        else:
            return self.relu3(x)

class BasicBlock_mul(nn.Module):
    def __init__(self, inplanes, out_planes, dilation=1, dropprob=0., mul=1):
        super(BasicBlock_mul, self).__init__()
        self.conv1 = moduleERS_muldil(inplanes, inplanes, kernel_size=3, stride=1, padding=1, dilation=dilation, bias=False,
                                      dropprob=dropprob)
        self.conv2 = moduleERS_muldil(inplanes, out_planes, kernel_size=3, stride=1, padding=1,
                                      dilation=[dilation[0]*2, dilation[1] * 2], bias=False, dropprob=dropprob, mul=mul)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, out_planes, dilation=1, dropprob=0., mul=1):
        super(BasicBlock, self).__init__()
        self.conv1 = moduleERS(inplanes, inplanes, kernel_size=3, stride=1, padding=1, dilation=dilation, bias=False,
                               dropprob=dropprob)
        self.conv2 = moduleERS(inplanes, out_planes, kernel_size=3, stride=1, padding=1, dilation=dilation, bias=False,
                               dropprob=dropprob, mul=mul)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


# ******************************************************************************



class Backbone(nn.Module):
    """
       Class for DarknetSeg. Subclasses PyTorch's own "nn" module
    """

    def __init__(self, params):
        super(Backbone, self).__init__()
        self.use_range = params["input_depth"]["range"]
        self.use_xyz = params["input_depth"]["xyz"]
        self.use_remission = params["input_depth"]["remission"]
        self.OS = params["OS"]
        self.block_1 = params["block_1"]
        self.block_2 = params["block_2"]
        self.features_bottleneck = int(params["features_bottleneck"])

        # input depth calc
        self.input_depth = 0
        self.input_idxs = []

        # 2TIMES,  FOR ABSOLUTE AND RELATIVE DATA
        if self.use_range:
            self.input_depth += 2
            self.input_idxs.append(0)
            self.input_idxs.append(5)
        if self.use_xyz:
            self.input_depth += 6
            self.input_idxs.extend([1, 2, 3])
            self.input_idxs.extend([6, 7, 8])
        if self.use_remission:
            self.input_depth += 2
            self.input_idxs.append(4)
            self.input_idxs.append(9)

        self.input_depth += 1
        self.input_idxs.append(10)

        self.input_depth_absolute = int(self.input_depth / 2)

        print("Depth of backbone input = ", self.input_depth)

        # stride play
        self.strides = [2, 2, 2, 2]
        self.strides_2 = [2, 2, 2, 2]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Original OS: ", current_os)

        # make the new stride
        if self.OS > current_os:
            print("Can't do OS, ", self.OS,
                  " because it is bigger than original ", current_os)
        else:
            # redo strides according to needed stride
            for i, stride in enumerate(reversed(self.strides), 0):
                if int(current_os) != self.OS:
                    if stride == 2:
                        current_os /= 2
                        self.strides[-1 - i] = 1
                    if int(current_os) == self.OS:
                        break
            print("New OS: ", int(current_os))
            print("Strides: ", self.strides)


        self.proj = moduleProyection(self.input_depth, self.features_bottleneck, [int(self.features_bottleneck/8),
                    int(self.features_bottleneck/4), int(self.features_bottleneck/2), self.features_bottleneck],
                    conv_feature = self.features_bottleneck, neighbours=16)

        # encoder  block, planes, blocks, stride, bn_d=0.1, dilation=1, dropprob=0.)
        self.enc1 = self._make_enc_layer(BasicBlock, [self.input_depth_absolute, self.features_bottleneck, self.features_bottleneck, self.features_bottleneck], 0,
                                         stride=(self.strides[0], self.strides_2[0]),  dilation=1)

        self.enc2 = self._make_enc_layer(BasicBlock, [self.features_bottleneck, self.features_bottleneck, self.features_bottleneck, self.features_bottleneck], self.block_1, downsampling=False,
                                         stride=(self.strides[1], self.strides_2[1]),  dilation=1, dropprob=0.25)
        self.enc3 = self._make_enc_layer(BasicBlock_mul, [self.features_bottleneck, self.features_bottleneck, self.features_bottleneck, self.features_bottleneck], self.block_2,
                                         stride=(self.strides[2], self.strides_2[2]),dilation=2, dropprob=0.25)

        # last channels
        self.last_channels = self.features_bottleneck


    # make layer useful function
    def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1, dilation=1, dropprob=0., multiplier=1, downsampling=True):
        #planes: inplanes, downsample planes, working planes, output planes
        layers = []



        #  downsample
        if downsampling:
            if blocks == 0:
                layers.append(("conv0", nn.Conv2d(planes[0], planes[0],
                                                 kernel_size=3,
                                                 stride=[stride[1], stride[0]], dilation=1,
                                                 padding=1, groups=planes[0], bias=False)))
                layers.append(("bn0", nn.BatchNorm2d(planes[0], momentum=bn_d)))
                layers.append(("relu0", nn.LeakyReLU(0.1)))

                layers.append(("conv01", nn.Conv2d(planes[0], planes[1]*multiplier,
                                                 kernel_size=1,
                                                 stride=1, dilation=1,
                                                 padding=0, bias=False)))
                layers.append(("bn01", nn.BatchNorm2d(planes[1]*multiplier, momentum=bn_d)))
                layers.append(("relu01", nn.LeakyReLU(0.1)))
            else:
                layers.append(("conv00", nn.Conv2d(planes[0], planes[0],
                                                 kernel_size=3,
                                                 stride=[stride[1], stride[0]], dilation=1,
                                                 padding=1,groups= planes[0], bias=False)))
                layers.append(("bn00", nn.BatchNorm2d(planes[0], momentum=bn_d)))
                layers.append(("relu00", nn.LeakyReLU(0.1)))

                layers.append(("conv01", nn.Conv2d(planes[0], planes[1],
                                                 kernel_size=1,
                                                 stride=1, dilation=1,
                                                 padding=0, bias=False)))
                layers.append(("bn01", nn.BatchNorm2d(planes[1], momentum=bn_d)))
                layers.append(("relu01", nn.LeakyReLU(0.1)))


        max_dil = 8
        i_reset = 0

        if downsampling:
            inplanes = planes[1]
        else:
            inplanes = planes[0]

        out_planes = planes[2]

        if dilation > 1:
            dil = [int(dilation/2), int(dilation/2)]
        else:
            dil = dilation

        for i in range(0, blocks):
            if i == blocks - 1:
                mul = multiplier
                out_planes = planes[3]
            else:
                mul = 1

            layers.append(("residual_{}".format(i),
                           block(inplanes, out_planes,  dilation=dil, dropprob=dropprob, mul=mul)))

            inplanes = planes[2]

            if dilation > 1:
                dil = [dil[0], dil[1] * 4]
                if dil[1] > max_dil:
                    i_reset += 1
                    dil = [1, 1]

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os):
        y = layer(x)

        return y, skips, os

    def forward(self, x_in):
        image = x_in[0]
        points = x_in[1]

        projection = self.proj(points)
        skips = {}
        os = 1

        x, skips, os = self.run_layer(image[:,:self.input_depth_absolute], self.enc1, skips, os)
        skips[2] = x

        x, skips, os = self.run_layer(projection, self.enc2, skips, os)
        skips[4] = x


        x, skips, os = self.run_layer(x, self.enc3, skips, os)


        return x, skips

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth
