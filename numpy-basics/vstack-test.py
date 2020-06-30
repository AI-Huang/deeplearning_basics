#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-29-20 22:39
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np
import matplotlib.pyplot as plt

# vstack, Stack arrays in sequence vertically (row wise).


def main():
    x0 = np.asarray([120, 160])  # coordinate of x0
    xs = np.vstack((np.arange(20)*3, np.arange(20)*2)).T + x0
    # print(xs.shape)
    # print(xs)
    plt.scatter(xs[:, 0], xs[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
