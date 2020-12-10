#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-18-20 17:12
# @Author  : Your Name (you@example.org)
# @Link    : https://stackoverflow.com/questions/44544766/how-do-i-check-if-keras-is-using-gpu-version-of-tensorflow/44547144#44547144


import tensorflow as tf
from tensorflow.python.client import device_lib
import keras
from keras import backend as K
# keras.backend.tensorflow_backend


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    global _LOCAL_DEVICES
    if _LOCAL_DEVICES is None:
        if _is_tf_1():
            devices = get_session().list_devices()
            _LOCAL_DEVICES = [x.name for x in devices]
        else:
            devices = tf.config.list_logical_devices()
            _LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in _LOCAL_DEVICES if 'device:gpu' in x.lower()]


print(tf.__version__)
print(keras.__version__)

""" 列出计算设备 """
print(device_lib.list_local_devices())  # list of DeviceAttributes
print(tf.test.is_gpu_available())  # True/False

""" 检查Keras是否得到GPU """
print(K.tensorflow_backend._get_available_gpus())
print(K.tensorflow_backend._get_current_tf_device())
print(K.tensorflow_backend._is_tf_1())
