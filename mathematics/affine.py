#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-23-20 10:21
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np
import matplotlib.pyplot as plt


def rotate(p, theta):
    """to_center
    p: tuple, 2D coordinates
    theta: radian angle
    """
    A = np.asarray([[np.math.cos(theta), -np.math.sin(theta)],
                    [np.math.sin(theta), np.math.cos(theta)]])  # kernel, 2D rotate
    p_out = A.dot(p)
    return p_out


def trans(p, m, n):
    """to_center
    p: tuple, 2D coordinates
    m, n: translation factors
    """
    A = np.asarray([[m, 0],
                    [0, n]])  # kernel, 2D translation
    p_out = A.dot(p)
    return p_out


def main1():
    x = np.asarray([1, 1])  # point, row vector
    A = np.asarray([[1, 2],
                    [1, 0]])  # kernel, D*D matrix
    b = np.asarray([0, 0])  # bias, for translation transformation
    x_out = A.dot(x) + b
    print(x_out)


def rotate_test():
    p = [1, 0]

    plt.scatter(p[0], p[1], label="p")
    plt.scatter(p_out[0], p_out[1], label="p_out")
    plt.legend()
    plt.grid()
    plt.show()


def trans_then_rotate():
    p = [1, 1]
    plt.scatter(0, 0, label="origin")
    plt.scatter(p[0], p[1], label="point")
    p_out = trans(p, 2, 3)
    plt.scatter(p_out[0], p_out[1], label="p_out1 trans")
    p_out = rotate(p_out, 90/180*np.math.pi)
    plt.scatter(p_out[0], p_out[1], label="p_out2 rotate")
    plt.legend()
    plt.grid()
    plt.show()


def rotate_then_trans():
    p = [1, 1]
    plt.scatter(0, 0, label="origin")
    plt.scatter(p[0], p[1], label="point")
    p_out = rotate(p, 90/180*np.math.pi)
    plt.scatter(p_out[0], p_out[1], label="p_out1 rotate")
    p_out = trans(p_out, 2, 3)
    plt.scatter(p_out[0], p_out[1], label="p_out2 trans")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    # trans_then_rotate()
    rotate_then_trans()


if __name__ == "__main__":
    main()
