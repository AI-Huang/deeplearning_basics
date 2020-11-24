#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-11-20 20:01
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import os
import numpy as np
from tensorflow import keras


def main():
    # data

    # load fashion_mnist
    fashion_mnist = keras.datasets.fashion_mnist

    # load mnist
    mnist = keras.datasets.mnist
    (train_images, train_labels), \
        (test_images, test_labels) = mnist.load_data()
    num_classes = np.max(train_labels) + 1  # 10

    # fit options
    epochs = 200
    batch_size = 32

    train_images = np.expand_dims(train_images, -1)
    input_shape = train_images.shape[1:]
    train_labels = keras.utils.to_categorical(train_labels)

    print(train_labels.shape)
    print(train_labels[0])

    np.random.seed(42)
    np.random.shuffle(train_labels)
    print(train_labels.shape)
    print(train_labels[0])


if __name__ == "__main__":
    main()
