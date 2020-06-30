#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-21-20 21:04
# @Author  : Kelly Hwong (you@example.org)
# @RefLink : https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy

import os
import numpy as np
import torch
import torch.nn as nn


def softmax(x):
    """softmax logits function
    """
    _ = np.exp(x)
    return _/np.sum(_)


def entropy(p):
    """entropy
    p: positve probability
    """
    return -(p*np.log(p)+(1-p)*np.log(1-p))


def cross_entropy(logits, y_true):
    loss = np.sum(-logits*np.log(y_true))
    return loss


def binary_cross_entropy(logits, y_true):
    return np.sum(-logits*np.log(y_true)) + np.sum(-(1-logits)*np.log(1-y_true))


def main():
    x = np.asarray([-0.9258, -0.3433,  0.1002,  0.3339, -0.7066])
    logits = softmax(x)
    # y_true = np.asarray([1, 0, 4])  # y_pred 有0元素怎么办
    y_true = np.asarray([0, 1, 0, 0, 0])  # y_pred 有0元素怎么办
    ce = cross_entropy(logits, y_true)
    bce = binary_cross_entropy(logits, y_true)
    print(logits)
    print(ce)
    print(bce)


if __name__ == "__main__":
    main()
