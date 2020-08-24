#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-18-20 21:00
# @Author  : Your Name (you@example.org)
# @Link    : https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html

import os
import numpy as np


def main():
    a = np.linspace(1, 10, 10)
    a = np.ones(20)
    v = np.ones(10)
    print(a)
    print(v)
    conv = np.convolve(a, v)
    print(conv)


if __name__ == "__main__":
    main()
