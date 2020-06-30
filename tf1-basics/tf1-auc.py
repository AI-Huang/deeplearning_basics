#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-24-20 08:01
# @Author  : Your Name (you@example.org)
# @Link    : https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC

import os
import numpy as np
import tensorflow as tf


def main():
    tf.enable_eager_execution()
    # print(tf.executing_eagerly())  # True
    m = tf.keras.metrics.AUC(num_thresholds=3)
    m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])

    # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
    # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
    # recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
    # auc = ((((1+0.5)/2)*(1-0))+ (((0.5+0)/2)*(0-0))) = 0.75

    print('Final result: ', m.result().numpy())  # Final result: 0.75


if __name__ == "__main__":
    main()
