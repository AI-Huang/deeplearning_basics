#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-17-20 02:55
# @Author  : Kelly Hwong (you@example.org)
# @RefLink : https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# @RefLink : https://github.com/pytorch/examples/blob/master/mnist/main.py

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class LeNet5(nn.Module):
    """LeNet5 implemented with PyTorch
    LeNet5 的结构，是 3x conv2d 和 2x FC
    """

    def __init__(self, output_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # padding 28*28 to 32*32
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # conv1 pool
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))  # conv2 pool
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class TwoConvNet(nn.Module):
    """简单二层卷积网络，用了 Dropout
    # @RefLink : https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self):
        super(TwoConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MLP(nn.Module):
    """MLP with BatchNorm, ReLU and Dropout
    """

    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(784, 548)
        self.bc1 = nn.BatchNorm1d(548)

        self.fc2 = nn.Linear(548, 252)
        self.bc2 = nn.BatchNorm1d(252)

        self.fc3 = nn.Linear(252, 10)

    def forward(self, x):
        x = x.view((-1, 784))
        h = self.fc1(x)
        h = self.bc1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        h = self.fc2(h)
        h = self.bc2(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)

        h = self.fc3(h)
        out = F.log_softmax(h)
        return out
