#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-29-20 22:39
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np
import matplotlib.pyplot as plt

# vstack, Stack arrays in sequence vertically (row wise).


def vstack_test1():
    # for example, a kalman filter
    # x0 = np.asarray([120, 160])  # coordinate of x0
    x0 = np.asarray([0, 0])  # coordinate of x0
    xs = np.vstack((np.arange(20)*3, np.arange(20)*2)).T + x0  # 这里有个转置
    print(xs.shape)
    print(xs)
    plt.scatter(xs[:, 0], xs[:, 1])
    plt.show()


def vstack_test2():
    a = np.asarray([0, 1])
    b = np.asarray([2, 3])
    c = np.vstack((a, b))
    d = c + [2, 3]  # 矩阵加上行向量
    print(c)
    print(d)


def hstack_test():
    a = np.asarray([0, 1])
    b = np.asarray([2, 3])
    c = np.sstack((a, b))
    d = c + [2, 3]  # 矩阵加上行向量
    print(c)
    print(d)


def main():
    x = np.arange(1, 5+1) * 4  # mulptiplied by scaler
    print(x)


if __name__ == "__main__":
    main()
