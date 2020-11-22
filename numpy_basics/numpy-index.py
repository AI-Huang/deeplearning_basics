#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jul-28-20 21:35
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np


def np_index_test1():
    a = np.asarray([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12]])
    print(a[1:3])  # 从最外围开始索引，如果a是矩阵就是第begin到第end-1行

    a = np.arange(10)
    print(a[0:9])


def np_index_test2():
    """slice index
    """
    a = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # left closed, right opened interval, so this result in [4 5 6 7], without 8
    b = a[3:7]
    print(b)


def np_index_test3():
    """index over the length of an array
    """
    a = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # will result in [11, 12] because 20 is larger than the last index number 11 (12th), so b will cut out the left elements
    b = a[10:20]
    print(b)


def main():
    np_index_test2()


if __name__ == "__main__":
    main()
