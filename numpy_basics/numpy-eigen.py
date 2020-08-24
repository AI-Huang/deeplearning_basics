#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-13-20 17:12
# @Author  : Your Name (you@example.org)
# @RefLink : https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html#numpy.linalg.eig
# @RefLink : https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html

import numpy as np


def solve_test():
    """solve equation a * x = b
    """
    a = np.array([[3, 1], [1, 2]])
    b = np.array([9, 8])
    x = np.linalg.solve(a, b)
    print(f"x:{x}")


def eigen_test():
    """
    v: The normalized (unit “length”) eigenvectors, such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    """
    # A = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    A = np.asarray([[1, 2], [3, 4]])
    print(f"A:{A}")

    w, v = np.linalg.eig(A)

    print(f"w:{w}")
    print(f"v:{v}")
    A_restore = np.dot(np.dot(v, np.diag(w)), np.linalg.inv(v))
    print(f"A_restore:{A_restore}")

    error = np.sum(A_restore-A)
    print(f"error:{error}")


def main():
    eigen_test()


if __name__ == "__main__":
    main()
