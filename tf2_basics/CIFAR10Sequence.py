#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-02-20 23:44
# @Author  : Kelly Hwong (you@example.org)
# @Link    : https://www.cs.toronto.edu/~kriz/cifar.html
# @Link    : https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/utils/data_utils.py#L437

import os
import math
import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize


class CIFAR10Sequence(tf.keras.utils.Sequence):
    # Here, `x_set` is list of path to the images
    # and `y_set` are the associated classes.

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        return np.array([resize(imread(file_name), (200, 200)) for file_name in batch_x]), np.array(batch_y)


def main():
    pass


if __name__ == "__main__":
    main()
