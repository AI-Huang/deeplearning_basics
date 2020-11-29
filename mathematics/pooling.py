#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-21-20 12:41
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np


def max_pool(x):
    # max pool with 2x2 filters and stride 2
    # x N*N
    # stride S
    # output F*F F=N/S
    input_size = x.shape[0]
    stride = 2
    num_filters = int(input_size/stride)

    output = np.zeros([num_filters, num_filters])
    for i in range(num_filters):
        for j in range(num_filters):
            print("Now filtering...")
            sub_x = x[i*stride:(i+1)*stride, j*stride:(j+1)*stride]
            print(sub_x)
            downsample = np.max(sub_x)
            print(f"Downsample: {downsample}")
            output[i, j] = downsample
    print(f"Output:\n{output}")

    return output


def main():
    x = np.asarray([[1, 1, 2, 4],
                    [5, 6, 7, 8],
                    [3, 2, 1, 0],
                    [1, 2, 3, 4]])
    output = max_pool(x)


if __name__ == "__main__":
    main()
