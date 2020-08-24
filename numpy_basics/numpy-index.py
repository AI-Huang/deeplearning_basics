#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jul-28-20 21:35
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np


def main():
    a = np.asarray([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12]])
    print(a[1:3])  # 从最外围开始索引，如果a是矩阵就是第begin到第end-1行

    a = np.arange(10)
    print(a[0:9])


if __name__ == "__main__":
    main()
