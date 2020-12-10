#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-18 17:15:04
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt


class LinearRegression(nn.Module):
    """线性回归"""

    def __init__(self, input_size, out_size):
        """初始化"""
        super(LinearRegression, self).__init__()
        self.x2o = nn.Linear(input_size, out_size)

    def forward(self, x):
        """前向传递"""
        return self.x2o(x)


def main():
    x = np.random.randn(1000, 1) * 4
    w, bias = np.array([0.5, ]), -1.68

    y_true = np.dot(x, w) + bias  # 真实数据
    y = y_true + np.random.randn(x.shape[0])  # 加噪声的数据

    batch_size = 10
    epochs = 50

    model = LinearRegression(1, 1)  # 回归模型
    criterion = nn.MSELoss()  # 损失函数
    # 调用cuda
    model.cuda()
    criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    losses = []

    for i in range(epochs):
        loss = 0
        optimizer.zero_grad()  # 清空上一步的梯度
        idx = np.random.randint(x.shape[0], size=batch_size)
        batch_cpu = Variable(torch.from_numpy(x[idx])).float()
        batch = batch_cpu.cuda()  # 很重要

        target_cpu = Variable(torch.from_numpy(y[idx])).float()
        target = target_cpu.cuda()  # 很重要
        output = model.forward(batch)
        loss += criterion(output, target)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Loss at epoch[%s]: %.3f' % (i, loss.data))
        losses.append(loss.data)

    plt.plot(losses, '-or', label="loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
