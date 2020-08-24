#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-21-20 20:54
# @Author  : Kelly Hwong (you@example.org)
# @Link    : https://pytorch.org/docs/master/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss

import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    # x is of size N x C = 3 x 4
    N, C = 1, 4
    x = torch.randn(N, C)
    print(x)
    # target = torch.FloatTensor([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
    target = torch.FloatTensor([[0, 0, 0, 1]])
    m = nn.Sigmoid()
    loss = nn.BCELoss()
    print(loss(m(x), target))

    # x 和 CE loss
    # tensor([[0.9813,  1.1011,  1.2295, -0.1908]])
    # tensor(1.2417)


if __name__ == "__main__":
    main()
