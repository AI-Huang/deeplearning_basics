#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jul-09-20 14:46
# @Author  : Kelly Hwong (you@example.org)
# @Link    : https://numpy.org/doc/stable/reference/generated/numpy.where.html

import numpy as np


def main():
    a = np.arange(10)
    result = np.where(a < 5)
    print(result)


if __name__ == "__main__":
    main()
