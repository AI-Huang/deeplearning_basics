#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-24-20 01:50
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks?hl=zh-cn


import os
import datetime
import tensorflow as tf


def main():
    # 下载 FashionMNIST 数据集并对其进行缩放
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    def create_model():
        # 创建一个非常简单的模型
        return tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def train_model():
        # 使用 Keras 和 TensorBoard 回调训练模型
        model = create_model()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        logdir = os.path.join(
            "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            logdir, histogram_freq=1)

        model.fit(x=x_train,
                  y=y_train,
                  epochs=5,
                  validation_data=(x_test, y_test),
                  callbacks=[tensorboard_callback])

    train_model()


if __name__ == "__main__":
    main()
