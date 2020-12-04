#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-21-20 21:04
# @Author  : Kelly Hwong (you@example.org)
# @RefLink : https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy

import numpy as np


def np_clip_by_epsilon(x):
    """np_clip_by_epsilon
    Input:
        x: numpy array.
    """
    epsilon = np.finfo(x.dtype).eps

    x_ones_mask = np.equal(x, 1.)
    x_zeros_mask = np.equal(x, 0.)

    x = np.where(x_ones_mask, 1. - epsilon, x)
    x = np.where(x_zeros_mask, 0. + epsilon, x)

    return x


def np_softmax(x):
    """softmax y_pred function
    """
    _ = np.exp(x)
    return _/np.sum(_)


def np_entropy(p, ):
    """entropy
    p: positve probability
    """
    return -(p*np.log(p)+(1-p)*np.log(1-p))


def np_cross_entropy(y_true, y_pred, reduction="sum"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np_clip_by_epsilon(y_pred)

    loss = -y_true*np.log(y_pred)

    if reduction == "sum":
        loss = np.sum(loss)
    elif reduction == "mean":
        loss = np.mean(loss)

    return loss


def np_binary_cross_entropy(y_true, y_pred, reduction="mean"):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np_clip_by_epsilon(y_pred)

    loss = -y_true * np.log(y_pred)
    loss += -(1-y_true) * np.log(1-y_pred)

    # Compute mean on the num_classes axis
    # This is equivalent with:
    # loss = np.sum(loss, axis=-1) / num_classes
    # where
    # num_classes = loss.shape[-1]
    loss = np.mean(loss, axis=-1)

    # Batch samples reduction
    if reduction == "sum":
        loss = np.sum(loss, axis=0)
    elif reduction == "mean":
        loss = np.mean(loss, axis=0)

    return loss


def np_binary_focal_loss():
    """
    docstring
    """
    pass


def test():
    x = np.asarray([-0.9258, -0.3433,  0.1002,  0.3339, -0.7066])
    y_pred = np_softmax(x)
    # y_true = np.asarray([1, 0, 4])  # y_pred 有0元素怎么办
    y_true = np.asarray([0, 1, 0, 0, 0])  # y_pred 有0元素怎么办
    ce = np_cross_entropy(y_pred, y_true)
    bce = np_binary_cross_entropy(y_pred, y_true)
    print(y_pred)
    print(ce)
    print(bce)


def main():
    test()


if __name__ == "__main__":
    main()
