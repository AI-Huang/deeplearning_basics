#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-08-20 21:49
# @Author  : Kelly Hwong (you@example.org)
# @RefLink : https://www.kaggle.com/curiousprogrammer/lenet-5-cnn-with-keras-99-48

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D,  Flatten


def LeNet5(input_shape=(28, 28, 1), num_classes=10):
    """LeNet-5 network built with Keras
    """
    model = Sequential()
    # 6 filters
    model.add(Conv2D(filters=6, kernel_size=(5, 5), padding="same",
                     activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(filters=6, kernel_size=(5, 5),
                     padding="valid", activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.build()
    model.summary()

    return model


def LeNet5_test():
    num_classes = 10
    input_shape = (28, 28, 1)
    model = LeNet5(input_shape=input_shape, num_classes=num_classes)


def main():
    LeNet5_test()


if __name__ == "__main__":
    main()
