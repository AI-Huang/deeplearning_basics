#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-02-20 10:40
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    xx = np.array([_ * 0.01 for _ in range(1, 1000)])
    yy = np.log(xx)
    plt.figure()
    plt.plot(xx, yy)
    plt.show()


if __name__ == "__main__":
    main()
