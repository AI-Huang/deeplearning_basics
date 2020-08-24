#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-05-20 12:32
# @Author  : Your Name (you@example.org)
# @RefLink : http://cs231n.github.io/python-numpy-tutorial/
# @RefLink : https://numpy.org/doc/stable/reference/random/index.html

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
    # out = np.random.random(5)
    out = np.random.normal(5)
    print(out)


def randint_test():
    M, N = 4, 3
    rnd = np.random.randint(low=0, high=2, size=(M, N))
    print(rnd)


def main():
    rng = np.random.default_rng()
    vals = rng.standard_normal(10)
    more_vals = rng.standard_normal(10)
    print(vals)


if __name__ == "__main__":
    main()
