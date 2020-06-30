#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-05-20 12:32
# @Author  : Your Name (you@example.org)
# @Link    : http://cs231n.github.io/python-numpy-tutorial/

import os
import numpy as np


def choice_test():
    c = np.random.choice(5, 3)  # has repeated elements
    print(c)
    num = 10
    c = np.arange(num)
    print(c)

    c = np.random.permutation(c)
    print(c)


def random_test():
    np.random.seed(42)
    out = np.random.random(5)
    print(out)


def main():
    random_test()


if __name__ == "__main__":
    main()
