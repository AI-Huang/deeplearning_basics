#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-18-20 19:12
# @Author  : Your Name (you@example.org)
# @Link    : https://stackoverflow.com/questions/44544766/how-do-i-check-if-keras-is-using-gpu-version-of-tensorflow/44547144#44547144

import tensorflow as tf  # tf 1
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
