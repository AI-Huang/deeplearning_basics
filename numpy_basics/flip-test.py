#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : May-16-20 03:02
# @Author  : Kelly Hwong (you@example.org)
# @RefLink : https://github.com/aleju/imgaug/blob/fc228dc4a13b9ad161f6b3ace6573d7ea503c651/imgaug/augmenters/flip.py#L990

import os
import numpy as np
import matplotlib.pyplot as plt
import PIL


def test1():
    kernel = np.random.random([3, 3])
    image = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 1, 1]], dtype=np.uint8)
    # image = base_img[:, :, np.newaxis]
    # plt.imshow(image, cmap="gray")
    # plt.show()

    # revert the first dim i.e. row for y, this will flip the image vertically
    image_flipped = image[::-1, ...]

    # revert the last dim i.e. column for x, this will flip the image horizontally
    image_flipped = image[:, ::-1, ...]

    plt.imshow(image_flipped, cmap="gray")
    plt.show()


def main():
    image = PIL.Image.open("./assets/bobo_cat.JPG")
    image = np.array(image)
    image_flipped = image[:, ::-1, ...]

    plt.imshow(image_flipped, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
