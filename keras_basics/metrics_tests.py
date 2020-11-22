#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-08-20 12:52
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @RefLink : https://www.tensorflow.org/api_docs/python/tf/keras/metrics/TruePositives

import os
import numpy as np
import tensorflow as tf
import confusion_matrix_v2_1_0 as confusion_matrix


class MyTests(object):
    def __init__(self):
        y_true = np.asarray([1, 1, 0, 0])
        y_true = np.vstack((y_true, 1-y_true))  # multi-dim pred array
        self.y_true = y_true.T

        y_pred = np.asarray([0.981, 1, 0, 0.6])  # tp, tp, tn, fn
        y_pred = np.vstack((y_pred, 1-y_pred))  # multi-dim pred array
        self.y_pred = y_pred.T

        self.thresholds = [0.5, 0.5]

    def TruePositives_test(self):
        m = tf.keras.metrics.TruePositives(thresholds=self.thresholds)
        m.update_state(self.y_true, self.y_pred)
        result = m.result().numpy()
        print(f"TruePositives: {result}")

    def TrueNegatives_test(self):
        m = tf.keras.metrics.TrueNegatives(thresholds=self.thresholds)
        m.update_state(self.y_true, self.y_pred)
        result = m.result().numpy()
        print(f"TrueNegatives: {result}")

    def FalsePositives_test(self):
        m = tf.keras.metrics.FalsePositives(thresholds=self.thresholds)
        m.update_state(self.y_true, self.y_pred)
        result = m.result().numpy()
        print(f"FalsePositives: {result}")

    def FalseNegatives_test(self):
        m = tf.keras.metrics.FalseNegatives(thresholds=self.thresholds)
        m.update_state(self.y_true, self.y_pred)
        result = m.result().numpy()
        print(f"FalseNegatives: {result}")

    def BinaryAccuracy_test(self):
        m = tf.keras.metrics.BinaryAccuracy()
        m.update_state(self.y_true, self.y_pred)
        result = m.result().numpy()
        print(result)


class OfficialTests(object):
    def __init__(self, thresholds=None, class_id=None, multi_label=False, two_dimentional=False):
        # fp, fn, tp, tp, fp
        self.y_true = [0, 1, 1, 1, 0]  # n, p, p, p, n
        self.y_pred = [1, 0, 1, 1, 1]  # p, n, p, p, p

        if two_dimentional:
            y_true = np.asarray([0, 1, 1, 1, 0])
            y_true = np.vstack((y_true, 1-y_true))  # multi-dim pred array
            self.y_true = y_true.T

            # p, p, n, p
            y_pred = np.asarray([1, 0, 1, 1, 1])
            # tp, tp, tn, fn # actually
            y_pred = np.vstack((y_pred, 1-y_pred))  # multi-dim pred array
            self.y_pred = y_pred.T

        self.thresholds = thresholds
        self.class_id = class_id
        self.multi_label = multi_label

    def TruePositives_test(self):
        m = tf.keras.metrics.TruePositives()
        m.update_state(self.y_true, self.y_pred)
        result = m.result().numpy()
        print(f"TruePositives_test: {result}")

    def TrueNegatives_test(self):
        m = tf.keras.metrics.TrueNegatives()
        m.update_state(self.y_true, self.y_pred)
        result = m.result().numpy()
        print(f"TrueNegatives_test: {result}")

    def FalseNegatives_test(self):
        m = tf.keras.metrics.FalseNegatives()
        m.update_state(self.y_true, self.y_pred)
        result = m.result().numpy()
        print(f"FalseNegatives_test: {result}")

    def FalsePositives_test(self):
        m = tf.keras.metrics.FalsePositives()
        m.update_state(self.y_true, self.y_pred)
        result = m.result().numpy()
        print(f"FalsePositives_test: {result}")


class ModifiedOfficialTests(object):
    """
    test adding class_id
    """

    def __init__(self, thresholds=None, class_id=None, multi_label=False, two_dimentional=False):
        # fp, fn, tp, tp, fp
        self.y_true = [0, 1, 1, 1, 0]  # n, p, p, p, n
        self.y_pred = [1, 0, 1, 1, 1]  # p, n, p, p, p

        if two_dimentional:
            y_true = np.asarray([0, 1, 1, 1, 0])
            y_true = np.vstack((y_true, 1-y_true))  # multi-dim pred array
            self.y_true = y_true.T

            # p, p, n, p
            y_pred = np.asarray([1, 0, 1, 1, 1])
            # tp, tp, tn, fn # actually
            y_pred = np.vstack((y_pred, 1-y_pred))  # multi-dim pred array
            self.y_pred = y_pred.T

        self.thresholds = thresholds
        self.class_id = class_id
        self.multi_label = multi_label

    def TruePositives_test(self):
        m = confusion_matrix.TruePositives(thresholds=self.thresholds,
                                           class_id=self.class_id, multi_label=self.multi_label)
        m.update_state(self.y_true, self.y_pred)
        result = m.result().numpy()
        print(f"TruePositives_test: {result}")

    def TrueNegatives_test(self):
        m = confusion_matrix.TrueNegatives(thresholds=self.thresholds,
                                           class_id=self.class_id, multi_label=self.multi_label)
        m.update_state(self.y_true, self.y_pred)
        result = m.result().numpy()
        print(f"TrueNegatives_test: {result}")

    def FalseNegatives_test(self):
        m = confusion_matrix.FalseNegatives(thresholds=self.thresholds,
                                            class_id=self.class_id, multi_label=self.multi_label)
        m.update_state(self.y_true, self.y_pred)
        result = m.result().numpy()
        print(f"FalseNegatives_test: {result}")

    def FalsePositives_test(self):
        m = confusion_matrix.FalsePositives(thresholds=self.thresholds,
                                            class_id=self.class_id, multi_label=self.multi_label)
        m.update_state(self.y_true, self.y_pred)
        result = m.result().numpy()
        print(f"FalsePositives_test: {result}")


def main():
    # t = OfficialTests()
    # t = MyTests()
    # t = ModifiedOfficialTests(multi_label=False)
    t = ModifiedOfficialTests(
        class_id=None, multi_label=True, two_dimentional=True)
    # t = OfficialTests(class_id=0, two_dimentional=True) # pass
    print(t.y_true)
    print(t.y_pred)
    t.TruePositives_test()
    t.TrueNegatives_test()
    t.FalsePositives_test()
    t.FalseNegatives_test()


if __name__ == "__main__":
    main()
