#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : May-09-20 19:58
# @Update  : Sep-11-20 20:01
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://keras.io/examples/mnist_cnn/


import os
import numpy as np
import tensorflow as tf


def load_data1():
    # load fashion_mnist dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # load mnist
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), \
        (test_images, test_labels) = mnist.load_data()
    num_classes = np.max(train_labels) + 1  # 10

    # fit options
    epochs = 200
    batch_size = 32

    train_images = np.expand_dims(train_images, -1)
    input_shape = train_images.shape[1:]
    train_labels = tf.keras.utils.to_categorical(train_labels)

    print(train_labels.shape)
    print(train_labels[0])

    np.random.seed(42)
    np.random.shuffle(train_labels)
    print(train_labels.shape)
    print(train_labels[0])


def load_data2():
    from tensorflow.keras.datasets.mnist import load_data
    (x_train, y_train), (x_test, y_test) = load_data()
    print(y_train.shape)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    num_classes = 10
    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)


def main():
    load_data1()


if __name__ == "__main__":
    main()
