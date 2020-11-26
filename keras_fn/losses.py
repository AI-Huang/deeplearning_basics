#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-24-20 00:12
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @RefLink : https://blog.csdn.net/JNingWei/article/details/80038594

import tensorflow as tf


def focal_loss(logits, targets, alpha, gamma, normalizer):
    with tf.name_scope('focal_loss'):
        positive_label_mask = tf.math.equal(targets, 1.0)
        cross_entropy = (tf.nn.sigmoid_cross_entropy_with_logits(
            labels=targets, logits=logits))
        neg_logits = -1.0 * logits
        modulator = tf.math.exp(gamma * targets * neg_logits -
                                gamma * tf.math.log1p(tf.math.exp(neg_logits)))
        loss = modulator * cross_entropy
        weighted_loss = tf.where(
            positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
        weighted_loss /= normalizer
    return weighted_loss


def main():
    logits = tf.constant([0.4, 0.6])
    targets = tf.constant([0, 1.])
    weighted_loss = focal_loss(logits, targets, alpha=1, gamma=2, normalizer=1)
    print(weighted_loss)


if __name__ == "__main__":
    main()
