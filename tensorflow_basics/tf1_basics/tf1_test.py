#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-18-20 17:12
# @Author  : kan.huang.hkust@gmail.com

"""TensorFlow 1.x tests
## Requirements
- Tensorflow (tested with v1.15.0)
- Keras (tested with v2.3.0)
- horovod (tested with v0.22.1)

"""


import tensorflow as tf
import keras
import horovod.tensorflow as hvd


def _get_available_gpus():
    # https://stackoverflow.com/questions/44544766/how-do-i-check-if-keras-is-using-gpu-version-of-tensorflow/44547144#44547144
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


def requirements_test():
    print(tf.__version__)
    print(keras.__version__)
    # horovod 的 version 用 pip install 查看


def tf1_gpu_test():
    """ 列出计算设备 """
    from tensorflow.python.client import device_lib
    from keras import backend as K

    print(device_lib.list_local_devices())  # list of DeviceAttributes
    print(tf.test.is_gpu_available())  # True/False

    """ 检查Keras是否得到GPU """
    print(K.tensorflow_backend._get_available_gpus())
    print(K.tensorflow_backend._get_current_tf_device())
    print(K.tensorflow_backend._is_tf_1())


def tensorflow_session():
    """
    Create tensorflow session with horovod
    """
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    return sess


def run_sess_hvd(args):
    # Create tensorflow session
    sess = tensorflow_session()  # sess which's config is initialized with hvd
    sess.graph.finalize()


def run_sess_test():
    """传统的 session 与 feed_dict
    """
    a = tf.add(2, 5)
    b = tf.multiply(a, 3)

    sess = tf.Session()  # 默认的 config
    # sess.run(x_sampled, {Y: _y, m.eps_std: _eps_std})
    sess.run(b)

    # 或者多嵌套一层
    # with tf.Session() as sess:
    # sess.run(b)


def main():
    # requirements_test()
    # tf1_gpu_test()
    run_sess_test()


if __name__ == "__main__":
    main()
