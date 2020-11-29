#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-23-20 19:33
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


def line_point_dist():
    p = [1, 1]
    # y=-x or x+y=0 or [1,1]*[x,y]=0
    xx = np.linspace(-10, 10, 100)
    yy = -xx
    A = np.asarray([1, 1])
    b = 0
    d = (A.dot(p)+b)/np.linalg.norm(A)
    print(d)
    return
    # d =
    plt.plot(xx, yy)
    plt.scatter(p[0], p[1])
    plt.scatter(0, 0, c="r")
    plt.grid()
    plt.show()


def line_normal_vector():
    theta, p = 135/180*np.math.pi, np.sqrt(2)/2
    a = np.asarray([np.math.cos(theta), np.math.sin(theta)])
    b = -p
    # 与仿射变换有联系
    # a*x+b=0 or a1 x + a2 y + b=0 -> y=-a1/a2 x- b/a2
    xx = np.linspace(-5, 5, 100)
    yy = -a[0]/a[1]*xx-b/a[1]  # for a1 is not 0!
    # point = [0, 5]
    # out = a.dot(point)+b
    # print(out)  # >0 正方向
    # return
    endpoint = np.asarray([p, 0])
    endpoint = rotate(endpoint, theta=theta)
    # vector O->endpoint就是法线，法线指向的方向是正方向
    plt.plot(xx, yy)
    plt.scatter(0, 0, c="r")
    plt.scatter(endpoint[0], endpoint[1], label="endpoint", s=4)
    plt.arrow(0, 0, endpoint[0], endpoint[1], head_width=0.1, head_length=0.1)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend()
    plt.grid()
    plt.show()


def main():
    line_normal_vector()


if __name__ == "__main__":
    main()
