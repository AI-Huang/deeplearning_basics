#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-20-20 16:16
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @RefLink : https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric?version=nightly

import tensorflow as tf


class ClassTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives


class BinaryTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)  # Equivalent to that threshold=0.5
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives


def main():
    import numpy as np
    binary_true_positives = BinaryTruePositives()
    y_true = np.asarray([1, 1, 0, 0])
    y_pred = np.asarray([0.981, 1, 0, 0.6])  # tp, tp, tn, fn

    binary_true_positives.update_state(y_true, y_pred)
    result = binary_true_positives.result().numpy()
    print(f"binary_true_positives: {result}")


if __name__ == "__main__":
    main()
