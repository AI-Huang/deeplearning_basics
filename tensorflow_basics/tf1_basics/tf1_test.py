#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-18-20 17:12
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

"""TensorFlow 1.x tests
# Requirements
- Tensorflow (tested with v1.15.0)
- Keras (tested with v2.3.0)
- horovod (tested with v0.22.1)

"""


from tensorflow.contrib.framework.python.ops import add_arg_scope
import tensorflow as tf
import keras
import horovod.tensorflow as hvd
import numpy as np


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


def tf1_check_device_test():
    # @RefLink : https://stackoverflow.com/questions/44544766/how-do-i-check-if-keras-is-using-gpu-version-of-tensorflow/44547144#44547144

    # Creates a graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))


def tf1_list_gpu_test():
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


def abstract_model_xy(sess, hps, feeds, train_iterator, test_iterator, data_init, lr, f_loss):

    # == Create class with static fields and methods
    class m(object):
        pass
    m.sess = sess
    m.feeds = feeds
    m.lr = lr


def encoder(z, objective):
    eps = []
    for i in range(hps.n_levels):
        z, objective = revnet2d(str(i), z, objective, hps)
        if i < hps.n_levels-1:
            z, objective, _eps = split2d(
                "pool"+str(i), z, objective=objective)
            eps.append(_eps)
    return z, objective, eps


@add_arg_scope
def my_affine_tmp(name, x, w):
    with tf.variable_scope(name):
        return x * w


def default_initializer(std=0.05):
    return tf.random_normal_initializer(0., std)


@add_arg_scope
def my_affine(name, input_, units):
    # input_: a placeholder
    with tf.variable_scope(name):
        input_dims = input_.shape[1]
        w = tf.get_variable("W", [input_dims, units], tf.float32,
                            initializer=default_initializer())
        return tf.matmul(input_, w)


def model(sess, input_):
    with tf.name_scope('input'):
        Y = tf.placeholder(tf.int32, [None], name='label')


def run_layer_test():
    # Build graph first
    input_ = tf.placeholder(tf.float32, [None, 3], name='input_')
    output = my_affine("my_affine_1", input_, units=2)

    # Execute with a session
    with tf.Session() as sess:
        # Initialize variables
        # tf.global_variables_initializer().run() # Same effect with below
        sess.run(tf.global_variables_initializer())

        x = tf.random.uniform((32, 3))
        output = sess.run(output, feed_dict={input_: x.eval()})

        print(type(output))
        print(output.shape)

    # Check variables
    # with tf.variable_scope(name)


def main():
    # requirements_test()
    # tf1_gpu_test()
    # run_sess_test()
    run_layer_test()


if __name__ == "__main__":
    main()
