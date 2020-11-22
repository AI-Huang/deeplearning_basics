#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-20-20 16:27
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D,  Flatten
from lenet import LeNet5
from model_config import get_confusion_matrix_metrics


def main():
    # Experiment configs
    model_type = "LeNet5-tf"

    # Paths
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", model_type, current_time)

    # Prepare data
    dataset_name = "mnist"

    if dataset_name == "cifar10":
        dataset = tf.keras.datasets.cifar10
        input_shape = (32, 32, 3)
    elif dataset_name == "mnist":
        dataset = tf.keras.datasets.mnist
        input_shape = (28, 28, 1)
    else:
        dataset = tf.keras.datasets.mnist
    num_classes = 10

    (train_images, train_labels), \
        (test_images, test_labels) = dataset.load_data()
    train_labels = tf.keras.utils.to_categorical(train_labels)  # to one-hot

    if dataset_name == "mnist":
        train_images = np.expand_dims(train_images, -1)
        input_shape = train_images.shape[1:]

    model = LeNet5(input_shape=input_shape, num_classes=num_classes)

    target_class_id = 1
    metrics = get_confusion_matrix_metrics(class_id=target_class_id)

    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.optimizers import Adam

    model.compile(loss=CategoricalCrossentropy(),
                  optimizer=Adam(),
                  metrics=metrics)

    from tensorflow.keras.callbacks import TensorBoard
    tensorboard_callback = TensorBoard(log_dir=log_dir, update_freq="batch")

    callbacks = [tensorboard_callback]

    model.fit(
        train_images,
        train_labels,
        batch_size=32,
        epochs=10,
        callbacks=callbacks,
        verbose=1)


if __name__ == "__main__":
    main()
