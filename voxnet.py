#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: voxnet.py
Created: 2020-01-21 21:32:40
Author : Yangmaonan
Email : 59786677@qq.com
Description: VoxNet 网络结构
'''

import torch
import torch.nn as nn
import dynamic_conv
from collections import OrderedDict
from dynamic_conv import Dynamic_conv3d


class VoxNet(nn.Module):
    def __init__(self, n_classes=10, input_shape=(32, 32, 32)):
        super(VoxNet, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.feat = torch.nn.Sequential(OrderedDict([
            # ('conv3d_1', torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2)),
            ('conv3d_1', Dynamic_conv3d(in_planes=1, out_planes=8, kernel_size=5, stride=2, K=4, temperature=43)),
            # ('bn1', torch.nn.BatchNorm3d(16)),
            ('bn2', torch.nn.BatchNorm3d(8)),
            ('relu1', torch.nn.ReLU()),
            # ('drop1', torch.nn.Dropout(p=0.1)),
            ('pool1', torch.nn.MaxPool3d(2)),

            # ('conv3d_11', Dynamic_conv3d(in_planes=8, out_planes=16, kernel_size=3, stride=1, K=4, temperature=43)),
            # ('bn12', torch.nn.BatchNorm3d(16)),
            # ('relu11', torch.nn.ReLU()),
            # ('drop11', torch.nn.Dropout(p=0.1)),
            # ('pool11', torch.nn.MaxPool3d(2)),

            # ('conv3d_2', Dynamic_conv3d(in_planes=8, out_planes=16, kernel_size=3, stride=1, K=4)),
            # ('relu1', torch.nn.ReLU()),

            # ('conv3d_3', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('conv3d_2', Dynamic_conv3d(in_planes=8, out_planes=16, kernel_size=3,K=4,temperature=43)),
            ('bn3', torch.nn.BatchNorm3d(16)),
            ('relu2', torch.nn.ReLU()),
            # ('pool2', torch.nn.MaxPool3d(2)),
            # ('drop2', torch.nn.Dropout(p=0.65))
        ]))
        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        self.mlp = torch.nn.Sequential(OrderedDict([
            # ('fc1', torch.nn.Linear(dim_feat, 128)),
            # ('relu1', torch.nn.ReLU()),
            # ('drop3', torch.nn.Dropout(p=0.4)),
            ('fc2', torch.nn.Linear(dim_feat, self.n_classes))
        ]))

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    voxnet = VoxNet()
    data = torch.rand([256, 1, 32, 32, 32])
    voxnet(data)
