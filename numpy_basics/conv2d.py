#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-31-20 16:05
# @Author  : Kan HUANG (you@example.org)
# @Link    : http://example.org

import os
import numpy as np
import matplotlib.pyplot as plt


def conv2d(image, kernel, stride=2):
    """the image has three channels
    """
    from matplotlib import pyplot as plt

    N = image.shape[0]
    F = kernel.shape[0]
    channels = image.shape[-1]  # 3

    O = int(np.ceil((N - F) / stride + 1))
    output = np.zeros([O, O, channels])

    for channel in range(channels):
        _image = image[:, :, channel]  # (20, 20)

        N = _image.shape[0]
        F = kernel.shape[0]
        if (N-F) % stride != 0:
            dN = stride - (N-F) % stride  # 补零
            for _ in range(dN):
                _image = np.concatenate([_image, np.zeros([N, 1])], axis=1)
                # print(_image.shape) # 20, 21
            N += dN
            for _ in range(dN):
                _image = np.concatenate([_image, np.zeros([1, N])], axis=0)
                # print(_image.shape) # 21, 21

        O = int((N - F) / stride + 1)
        channel_output = np.zeros([O, O])
        for i in range(O):
            for j in range(O):
                _sub_image = _image[stride*i:stride*i+F, stride*j:stride*j+F]
                _o = 0
                for f1 in range(F):
                    for f2 in range(F):
                        _o += _sub_image[f1, f2] * kernel[f1, f2]
                channel_output[i][j] += _o

        output[:, :, channel] = channel_output

    return output


def main():
    kernel = np.random.random([3, 3])
    image = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 1, 1]], dtype=np.uint8)
    # image = base_img[:, :, np.newaxis]
    plt.imshow(image, cmap="gray")
    plt.show()

    image = np.expand_dims(image, axis=2)
    image = np.concatenate((image, image, image), axis=2)

    image_conv1 = conv2d(image, kernel, stride=2)


if __name__ == "__main__":
    main()
