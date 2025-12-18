# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
from modules.AFPN import AFPN, AFPN222

from pytorch_wavelets import DTCWTForward, DTCWTInverse
from transformers.models.swin.modeling_swin import SwinModel
from model.convnext1 import convnext_small


##############################
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FFT(nn.Module):
    def __init__(self,inchannel,outchannel):
        super().__init__()
        self.DWT = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')

        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.conv1 = BasicConv2d(inchannel, inchannel)
        self.conv2 = BasicConv2d(inchannel, inchannel)
        self.conv3 = BasicConv2d(outchannel, outchannel)

    def forward(self, x, y):
        y = self.conv2(y)
        Xl, Xh = self.DWT(x)
        Yl, Yh = self.DWT(y)

        x_y = self.conv1(Xl) + self.conv1(Yl)

        x_m = self.IWT((x_y, Xh))
        x_y=torch.cat([x_m,y],dim=1)
        out = self.conv3(x_y)
        return out


class Fusion_net_simple_point_cat4(nn.Module):
    def __init__(self, rgb=None, depth=None, clip=None, ingredient=None, points=None):
        super(Fusion_net_simple_point_cat4, self).__init__()

        self.depth = depth
        self.rgb = rgb
        self.clip = clip
        self.ingredient = ingredient

        self.conv1x1 = nn.Conv2d(2048, 1024, kernel_size=1)

        self.con1_1 = nn.Conv2d(128, 96, kernel_size=1, stride=1, padding=0)
        self.con2_1 = nn.Conv2d(256, 192, kernel_size=1, stride=1, padding=0)
        self.con3_1 = nn.Conv2d(512, 384, kernel_size=1, stride=1, padding=0)
        self.con4_1 = nn.Conv2d(1024, 768, kernel_size=1, stride=1, padding=0)

        self.con1 = nn.Conv2d(256, 192, kernel_size=1, stride=1, padding=0)
        self.con2 = nn.Conv2d(512, 384, kernel_size=1, stride=1, padding=0)
        self.con3 = nn.Conv2d(1024, 768, kernel_size=1, stride=1, padding=0)
        self.con4 = nn.Conv2d(2048, 1536, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.f3 = nn.Linear(1536, 768)
        self.f311 = nn.Linear(1536, 768)
        self.f322 = nn.Linear(1536, 768)
        self.f333 = nn.Linear(1536, 768)
        # self.f3 = nn.Linear(8, 768)

        self.f_1 = nn.Linear(768, 1)
        self.f_2 = nn.Linear(768, 1)
        self.f_3 = nn.Linear(768, 1)
        self.f_4 = nn.Linear(768, 1)
        self.f_5 = nn.Linear(768, 1)

        self.dropout1 = nn.Dropout(p=0.5)

        self.fft1 = FFT(96, 192)
        self.fft2 = FFT(192, 384)

        self.conv_y1 = nn.Conv1d(in_channels=257, out_channels=24 * 24, kernel_size=1)
        self.conv_y2 = nn.Conv1d(in_channels=257, out_channels=24 * 12, kernel_size=1)

        self.conv_y3 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=1)

        # self.conv_y4 = nn.Conv2d(768, 768, kernel_size=1 )
        self.conv_y4 = nn.Conv2d(768*2, 768*2, kernel_size=1)
        self.conv_y5 = nn.Conv2d(768, 1536, kernel_size=3,stride=2, padding=1)
        self.param1 = nn.Parameter(torch.rand(1))
    def _upsample_add(self, x, y):
        # 将输入x上采样两倍
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    ########################################################
    def forward(self, rgb, depth, clip, dino_feature, points):
        # dino [b, 257, 768]

        """
            dino 前面的系数  能是否改变
        """
        dino3 = self.conv_y1(dino_feature)
        dino3 = dino3.reshape(dino3.shape[0], 768, 24, 24)

        dino4 = self.conv_y2(dino_feature)
        dino4 = dino4.reshape(dino4.shape[0], 768 * 2, 12, 12)

        points1 = self.conv_y4(points)
        # points2 = self.conv_y5(points)

        rgb[0] = self.con1_1(rgb[0])
        cat1 = self.fft1(rgb[0], depth[0])

        rgb[1] = self.con2_1(rgb[1])
        cat2 = self.fft2(rgb[1], depth[1])

        rgb[2] = self.con3_1(rgb[2])
        cat3 = torch.cat((rgb[2], depth[2]), dim=1)
        cat3 = self.con3(clip[2]) + cat3 + dino3    # 768

        rgb[3] = self.con4_1(rgb[3])  # 1536
        cat4 = torch.cat((rgb[3], depth[3]), dim=1)
        cat4 = self.con4(clip[3]) + cat4 + 0.1 * points1   #
        list = tuple((cat1, cat2, cat3, cat4))

        return list

class Fusion_net_cat4(nn.Module):
    def __init__(self, rgb=None, depth=None, clip=None, ingredient=None, points=None):
        super(Fusion_net_cat4, self).__init__()

        self.depth = depth
        self.rgb = rgb
        self.clip = clip
        self.ingredient = ingredient

        self.conv1x1 = nn.Conv2d(2048, 1024, kernel_size=1)

        self.con1_1 = nn.Conv2d(128, 96, kernel_size=1, stride=1, padding=0)
        self.con2_1 = nn.Conv2d(256, 192, kernel_size=1, stride=1, padding=0)
        self.con3_1 = nn.Conv2d(512, 384, kernel_size=1, stride=1, padding=0)
        self.con4_1 = nn.Conv2d(1024, 768, kernel_size=1, stride=1, padding=0)

        self.con1 = nn.Conv2d(256, 192, kernel_size=1, stride=1, padding=0)
        self.con2 = nn.Conv2d(512, 384, kernel_size=1, stride=1, padding=0)
        self.con3 = nn.Conv2d(1024, 768, kernel_size=1, stride=1, padding=0)
        self.con4 = nn.Conv2d(2048, 1536, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.f3 = nn.Linear(1536, 768)
        self.f311 = nn.Linear(1536, 768)
        self.f322 = nn.Linear(1536, 768)
        self.f333 = nn.Linear(1536, 768)
        # self.f3 = nn.Linear(8, 768)

        self.f_1 = nn.Linear(768, 1)
        self.f_2 = nn.Linear(768, 1)
        self.f_3 = nn.Linear(768, 1)
        self.f_4 = nn.Linear(768, 1)
        self.f_5 = nn.Linear(768, 1)

        self.dropout1 = nn.Dropout(p=0.5)

        self.fft1 = FFT(96, 192)
        self.fft2 = FFT(192, 384)

        self.conv_y1 = nn.Conv1d(in_channels=257, out_channels=24 * 24, kernel_size=1)
        self.conv_y2 = nn.Conv1d(in_channels=257, out_channels=24 * 12, kernel_size=1)

        self.conv_y3 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=1)

        # self.conv_y4 = nn.Conv2d(768, 768, kernel_size=1 )
        self.conv_y4 = nn.Conv2d(768*2, 768*2, kernel_size=1)
        self.conv_y5 = nn.Conv2d(768, 1536, kernel_size=3,stride=2, padding=1)
        self.param1 = nn.Parameter(torch.rand(1))
    def _upsample_add(self, x, y):
        # 将输入x上采样两倍
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    ########################################################
    def forward(self, rgb, depth, clip, dino_feature):
        # dino [b, 257, 768]

        """
            dino 前面的系数  能是否改变
        """
        dino3 = self.conv_y1(dino_feature)
        dino3 = dino3.reshape(dino3.shape[0], 768, 24, 24)

        dino4 = self.conv_y2(dino_feature)
        dino4 = dino4.reshape(dino4.shape[0], 768 * 2, 12, 12)


        rgb[0] = self.con1_1(rgb[0])
        cat1 = self.fft1(rgb[0], depth[0])

        rgb[1] = self.con2_1(rgb[1])
        cat2 = self.fft2(rgb[1], depth[1])

        rgb[2] = self.con3_1(rgb[2])
        cat3 = torch.cat((rgb[2], depth[2]), dim=1)
        cat3 = self.con3(clip[2]) + cat3 + dino3    # 768

        rgb[3] = self.con4_1(rgb[3])  # 1536
        cat4 = torch.cat((rgb[3], depth[3]), dim=1)
        cat4 = self.con4(clip[3]) + cat4
        list = tuple((cat1, cat2, cat3, cat4))

        return list
