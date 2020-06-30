#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : May-26-20 21:55
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np


def main():
    a = np.asarray([1, 2, 3, 4, 5])
    b = np.asarray([1, 2, 3, 4, 5])
    # np.save("a.npy", a)
    a = np.load("a.npy")
    print(a)

    np.savez("ab.npz", a=a)
    f = np.load("ab.npz")
    print(f)  # <numpy.lib.npyio.NpzFile object at 0x000002335C981108>
    print(f.files)
    # a, b = f["a"], f["b"]
    a = f["a"]
    print(a)


if __name__ == "__main__":
    main()
