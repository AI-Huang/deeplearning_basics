#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-02-20 00:45
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import os
import numpy as np
from sklearn.model_selection import KFold


def main():
    # X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    num_samples = 20
    X = np.random.randint(0, 10, size=(num_samples, 2))
    n_splits = 10
    print("X:", X)
    # y = np.array([1, 2, 3, 4])
    y = np.linspace(1, num_samples, num=num_samples)
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(X)  # 2
    print(kf)  # KFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    # TRAIN: [2 3] TEST: [0 1]
    # TRAIN: [0 1] TEST: [2 3]


if __name__ == "__main__":
    main()
