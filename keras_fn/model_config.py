#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-21-20 16:08
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org
"""model_config
general model configurations.
"""

import tensorflow as tf


def get_confusion_matrix_metrics(class_id):
    import confusion_matrix_v2_1_0 as confusion_matrix
    metrics = [confusion_matrix.FalsePositives(class_id=class_id),
               confusion_matrix.FalseNegatives(class_id=class_id),
               confusion_matrix.TruePositives(class_id=class_id),
               confusion_matrix.TrueNegatives(class_id=class_id),
               confusion_matrix.AUC(class_id=class_id),
               tf.keras.metrics.Recall(class_id=class_id),
               tf.keras.metrics.Precision(class_id=class_id)]

    return metrics


def get_confusion_matrix_metrics_test():
    metrics = get_confusion_matrix_metrics(class_id=1)
    return metrics


def main():
    get_confusion_matrix_metrics_test()


if __name__ == "__main__":
    main()
