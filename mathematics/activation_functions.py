#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-06-20 15:47
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist  # 画布工具


def relu(x):
    pass


def activation(activation):
    pass


def sigmoid(x):
    """sigmoid function
    x: input, np array
    """
    return 1/(1+np.exp(-x))


def softmax():
    scores = [1, 2, 3]
    scores = np.exp(scores)
    scores = scores / np.sum(scores)
    print(scores)


def test_sigmoid():
    # -10, 10, 0.1
    x = np.arange(-10, 10, 0.1)
    y = np.empty(x.shape)
    # TODO map
    for i in range(len(x)):
        y[i] = sigmoid(x[i])

    # 创建画布
    fig = plt.figure(figsize=(8, 8))
    # 使用axisartist.Subplot方法创建一个绘图区对象ax
    ax = axisartist.Subplot(fig, 111)
    # 将绘图区对象添加到画布中
    fig.add_axes(ax)

    # 通过set_visible方法设置绘图区所有坐标轴隐藏
    ax.axis[:].set_visible(False)
    # ax.new_floating_axis代表添加新的坐标轴
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    # 给x坐标轴加上箭头
    ax.axis["x"].set_axisline_style("->", size=1.0)
    # 添加y坐标轴，且加上箭头
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["y"].set_axisline_style("-|>", size=1.0)
    # 设置x、y轴上刻度显示方向
    ax.axis["x"].set_axis_direction("top")
    ax.axis["y"].set_axis_direction("right")

    # 生成x步长为0.1的列表数据
    x = np.arange(-15, 15, 0.1)
    # 生成sigmiod形式的y数据
    y = 1/(1+np.exp(-x))
    # 设置x、y坐标轴的范围
    plt.xlim(-12, 12)
    plt.ylim(-1, 1)
    # 绘制图形
    plt.plot(x, y, c='b')
    plt.show()


def main():
    # test_sigmoid()
    softmax()


if __name__ == "__main__":
    main()
