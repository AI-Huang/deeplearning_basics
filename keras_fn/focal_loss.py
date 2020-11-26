#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-24-20 20:17
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : https://github.com/tensorflow/models/blob/master/official/vision/keras_cv/losses/focal_loss.py

import tensorflow as tf
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import backend_config
epsilon = backend_config.epsilon

"""Focal losses implementation
Classes:
 - BinaryFocalLoss: binary focal loss function.
 - FocallossSigmoid: Keras official focal loss function, will pass the y_pred into sigmoid function before calculation.

Reference:
https://github.com/tensorflow/models/blob/master/official/vision/keras_cv/losses/focal_loss.py
"""


def clip_by_epsilon(y_pred):
    """clip_by_epsilon
    clip boundary values on y_pred by epsilon.
    This is for the case that log(0) is calculated and an NaN yeilds.
    """
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    epsilon_ = constant_op.constant(
        epsilon(), dtype=y_pred.dtype.base_dtype)
    y_pred_ones_mask = tf.equal(y_pred, 1.)
    y_pred_zeros_mask = tf.equal(y_pred, 0.)
    y_pred = tf.where(y_pred_ones_mask, clip_ops.clip_by_value(
        y_pred, epsilon_, 1. - epsilon_), y_pred)
    y_pred = tf.where(y_pred_zeros_mask, clip_ops.clip_by_value(
        y_pred, epsilon_, 1. - epsilon_), y_pred)

    return y_pred


class BinaryFocalLoss(tf.keras.losses.Loss):
    """Implements a Focal loss for classification problems.

    Reference:
      [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
    """

    def __init__(self,
                 alpha,
                 gamma,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name=None):
        """Initializes `BinaryFocalLoss`.

        Arguments:
          alpha: The `alpha` weight factor for binary class imbalance.
          gamma: The `gamma` focusing parameter to re-weight loss.
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
              https://www.tensorflow.org/tutorials/distribute/custom_training) for
                more details.
          name: Optional name for the op. Defaults to 'retinanet_class_loss'.
        """
        self._alpha = alpha
        self._gamma = gamma
        super(BinaryFocalLoss, self).__init__(
            reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        """Invokes the `BinaryFocalLoss`.

        Arguments:
          y_true: A tensor of size [batch, num_anchors, num_classes]
          y_pred: A tensor of size [batch, num_anchors, num_classes]

        Returns:
          Summed loss float `Tensor`.
        """
        with tf.name_scope('focal_loss'):
            y_true = tf.cast(y_true, dtype=tf.float32)
            y_pred = tf.cast(y_pred, dtype=tf.float32)

            y_pred = clip_by_epsilon(y_pred)

            positive_label_mask = tf.equal(y_true, 1.0)

            binary_cross_entropy = - y_true * math_ops.log(y_pred)
            binary_cross_entropy += - (1-y_true) * math_ops.log(1-y_pred)

            probs_gt = tf.where(positive_label_mask, y_pred, 1.0 - y_pred)
            # With small gamma, the implementation could produce NaN during back prop.
            modulator = tf.pow(1.0 - probs_gt, self._gamma)
            loss = modulator * binary_cross_entropy
            weighted_loss = tf.where(positive_label_mask, self._alpha * loss,
                                     (1.0 - self._alpha) * loss)

        return weighted_loss


class FocalLossSigmoid(tf.keras.losses.Loss):
    """Implements a Focal loss for classification problems.

    Reference:
      [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
    """

    def __init__(self,
                 alpha,
                 gamma,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name=None):
        """Initializes `FocalLossSigmoid`.

        Arguments:
          alpha: The `alpha` weight factor for binary class imbalance.
          gamma: The `gamma` focusing parameter to re-weight loss.
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
              https://www.tensorflow.org/tutorials/distribute/custom_training) for
                more details.
          name: Optional name for the op. Defaults to 'retinanet_class_loss'.
        """
        self._alpha = alpha
        self._gamma = gamma
        super(FocalLossSigmoid, self).__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        """Invokes the `FocalLossSigmoid`.

        Arguments:
          y_true: A tensor of size [batch, num_anchors, num_classes]
          y_pred: A tensor of size [batch, num_anchors, num_classes]

        Returns:
          Summed loss float `Tensor`.
        """
        with tf.name_scope('focal_loss'):
            y_true = tf.cast(y_true, dtype=tf.float32)
            y_pred = tf.cast(y_pred, dtype=tf.float32)
            positive_label_mask = tf.equal(y_true, 1.0)
            cross_entropy = (
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
            probs = tf.sigmoid(y_pred)
            probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
            # With small gamma, the implementation could produce NaN during back prop.
            modulator = tf.pow(1.0 - probs_gt, self._gamma)
            loss = modulator * cross_entropy
            weighted_loss = tf.where(positive_label_mask, self._alpha * loss,
                                     (1.0 - self._alpha) * loss)

        return weighted_loss

    def get_config(self):
        config = {
            'alpha': self._alpha,
            'gamma': self._gamma,
        }
        base_config = super(FocalLossSigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
