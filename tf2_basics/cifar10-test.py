#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-03-20 00:05
# @Author  : Kelly Hwong (you@example.org)
# @Link    : https://keras.io/zh/examples/cifar10_cnn/

from __future__ import print_function
import os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# 数据，切分为训练和测试集。
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
