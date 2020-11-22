#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-03-20 19:19
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras  # keras-tf
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    # data
    fashion_mnist = keras.datasets.fashion_mnist
    mnist = keras.datasets.mnist
    (train_images, train_labels), \
        (test_images, test_labels) = mnist.load_data()
    num_classes = np.max(train_labels) + 1  # 10

    padding = 2

    train_images_padded = np.zeros(
        (len(train_images),
         train_images.shape[1]+2*padding,
         train_images.shape[2]+2*padding)
    )
    for i in tqdm(range(len(train_images))):
        img = train_images[i]
        h_ones = np.zeros((padding, img.shape[1]))
        img = np.vstack([h_ones, img, h_ones])

        v_ones = np.zeros((img.shape[0], padding))
        img = np.hstack([v_ones, img, v_ones])
        train_images_padded[i] = img

    import matplotlib.pyplot as plt
    plt.imshow(train_images_padded[0], cmap="gray")
    plt.show()
    return

    np.save("./mnist_train_images_padded.npy", train_images_padded)


if __name__ == "__main__":
    main()
