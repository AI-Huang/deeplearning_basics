#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-18-19 14:44
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np


def test1():
    l1 = np.asarray([1, 2, 3, 4, 5])
    l2 = np.asarray([6, 7, 8, 9, 10])
    l3 = l1 + l2 + l1 + l2
    l4 = l3/3.33
    l5 = l1 * l2  # ??????
    l5 = np.dot(l1, l2)
    l5 = np.outer(l1, l2)
    print(l3)
    print(l4)
    print(l5)


def test2():
    a = np.asarray([[1, 2],
                    [3, 4]])
    a += 1
    print(a)


def random_test():
    ttt = np.random.random_sample([3, 4])
    print(ttt)
    print(ttt[:, 1])  # index from 0


def append_test():
    ttt = np.asarray([])
    ttt = np.append(ttt, [[1, 2, 3]], axis=0)
    ttt = np.append(ttt, [4, 5, 6], axis=0)
    print(ttt)


def matrix_index():
    m = np.random.random([4, 5])
    print(m)
    r = m[[0, 1, 2],  [0, 1, 0]]
    print(r)


def matrix_dot():
    a = np.asarray([[1, 2], [3, 4]])
    b = np.asarray([[1, 2, 3], [4, 5, 6]])
    print(a.dot(b))


def matrix_multiply():
    a = np.asarray([[1, 2], [3, 4]])
    b = np.asarray([[2, 3], [4, 5]])
    print(a * b)
    print(a.dot(b))


def tile_test():
    a = np.array([0, 1, 2])  # 1 x 3
    n = 7
    print(a)
    print(a.T)
    print(np.tile(a, (n, 1)).T)
    b = np.tile(a, (3, 4))  # 3 x 12
    c = np.tile(a, (3, 4, 2))  # 3 x 12
    print(c)


def matrix_star():
    a = np.asarray([[1, 2, 3], [4, 5, 6]])
    print(a)
    p = [1, 2, 3]
    print(a * p)

    a_sum = np.sum(a, axis=1)
    print(a_sum)
    print(np.divide(a.T, a_sum).T)


def matrix_div():
    def f_sigmoid(x): return 1.0 / (1.0 + np.exp(-x))  # sigmoid
    a = np.asarray([[1, 2, 3], [4, 5, 6]])
    print(f_sigmoid(a))
    print(1/a)


def onehot():
    """"""
    num_classes = 3
    y = np.asarray([1, 2, 0, 1, 1, 1])
    y_onehot = np.eye(num_classes)
    print(y_onehot)
    y_onehot = y_onehot[y]  # ???????
    print(y_onehot)


def weighted_matrix():
    """ multiply columns by weight
    """
    a = np.asarray([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    b = np.asarray([1, 2, 3])
    # equivalent
    b = np.asarray([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    print(a*b)  # b is weights, a is mat
    print(b*a)  # same as above

    # ??????????????????????
    # b = np.asarray([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    a = np.asarray([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    b = np.asarray([1, 2, 3])
    print((a.T*b))  # ??????? a ??????????????????????? b ??????
    print((a.T*b).T)  # ??????????

    print(a[0])


def main():
    # matrix_index()
    # matrix_dot()
    # matrix_multiply()
    # tile_test()
    # matrix_star()
    # matrix_div()
    # onehot()
    # weighted_matrix()

    a = np.asarray([1, 2, 3, 4])
    print(a[:1])


if __name__ == "__main__":
    main()
