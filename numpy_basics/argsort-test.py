#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-04-20 18:30
# @Author  : Your Name (you@example.org)
# @Link    : https://thispointer.com/find-the-index-of-a-value-in-numpy-array/

import os
import numpy as np


def main():
    l = [3, 5, 42, 3, 4, 21]
    ret = np.argsort(l)
    print(type(ret))
    # print(ret.find(0))
    result = np.where(ret == 1)
    print(type(result))
    print(result[0][0])


if __name__ == "__main__":
    main()
