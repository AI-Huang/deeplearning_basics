#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-07-20 11:21
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras  # keras-tf
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    # data
    # fashion_mnist = keras.datasets.fashion_mnist
    mnist = keras.datasets.mnist
    (train_images, train_labels), \
        (test_images, test_labels) = mnist.load_data()
    num_classes = np.max(train_labels) + 1  # 10

    padding = 2
    paddings = tf.constant([[padding, padding], [padding, padding]])

    train_images_padded = tf.zeros(
        (len(train_images),
         train_images.shape[1]+2*padding,
         train_images.shape[2]+2*padding)
    )  # tf.float32
    train_images_padded = tf.Variable(train_images_padded)
    for i in tqdm(range(len(train_images))):
        if i == 10:
            break
        img1 = tf.pad(train_images[i], paddings, "CONSTANT")
        img2 = tf.cast(img1, tf.float32)
        train_images_padded[i].assign(tf.add(train_images_padded[i], img2))

    import matplotlib.pyplot as plt
    plt.imshow(train_images_padded[0], cmap="gray")
    plt.show()
    return

    np.save("./mnist_train_images_padded.npy", train_images_padded)


if __name__ == "__main__":
    main()
