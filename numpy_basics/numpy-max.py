#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : May-22-20 13:44
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np


def main():
    a = np.random.randint(0, 10, size=(4, 5))
    print(a)
    out = np.max(a)  # default, all number in matrix a
    print(out)
    out = np.max(a, axis=0)  # axis 0, max value of each column
    print(out)
    print(type(out))  # <class 'numpy.ndarray'>

    out = np.max(a, axis=1)  # axis 1, max value of each row
    print(out)


if __name__ == "__main__":
    main()
