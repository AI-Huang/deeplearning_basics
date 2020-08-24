#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-05-20 12:34
# @Author  : Your Name (you@example.org)
# @Link    : http://cs231n.github.io/python-numpy-tutorial/

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Compute the x and y coordinates for points on a sine curve
    x = np.arange(0, 3 * np.pi, 0.1)
    y = np.sin(x)

    # Plot the points using matplotlib
    plt.plot(x, y)
    plt.show()  # You must call plt.show() to make graphics appear.


if __name__ == "__main__":
    main()
