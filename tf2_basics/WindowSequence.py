#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-03-20 00:27
# @Author  : Kelly Hwong (you@example.org)
# @Link    : https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/utils/data_utils.py#L437

import os
import numpy as np
import tensorflow as tf


class WindowSequence(tf.keras.utils.Sequence):
    # Here, `x_set` is list of path to the images
    # and `y_set` are the associated classes.
    def __init__(
        self,
        sequence,
        y,
        window_size,
        stride=1,
        batch_size=32,
        shuffle=True
    ):
        """Initialize the windowed sub-sequences dataset generated from a sequence data.

        Arguments:
            sequence: sequence data, a numpy array.
            y: corresponding y to the sub-sequences data.
            batch_size: batch size, default 32.
            shuffle:
        """
        self.sequence = sequence
        N = sequence.shape[0]
        self.y = y[:N-window_size+1]
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        # index of windows
        self.index = np.arange((sequence.shape[0]-window_size)//stride)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, position):
        """Gets batch at position `position`.

        Arguments:
            position: position of the batch in the Sequence.

        Returns:
            A batch
        """
        sequence = self.sequence
        y = self.y
        window_size = self.window_size
        stride = self.stride

        batch_size = self.batch_size
        batch_index = self.index[position *
                                 batch_size:(position + 1) * batch_size]

        # end_idx <= sequence.shape[0]
        # stride*i+window_size<=sequence.shape[0]
        # 比如， stride 为 1，i<=sequence.shape[0]-window_size，i从0开始计数

        batch_x = np.empty(window_size)
        batch_y = np.empty(1)
        for i in batch_index:
            _x = sequence[i:i + window_size]
            _y = y[i]
            batch_x = np.vstack((batch_x, _x))
            batch_y = np.vstack((batch_y, _y))
        batch_x, batch_y = batch_x[1:], batch_y[1:]

        return batch_x, batch_y

    def __len__(self):
        """Number of batch in the Sequence.

        Returns:
            The number of batches in the Sequence.
        """
        return self.y.shape[0] // self.batch_size + 1

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        if self.shuffle == True:
            self.index = np.arange(self.y.shape[0])
            np.random.shuffle(self.index)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item


def main_test():
    N = 200000
    window_size = 150000
    stride = 24
    sequence = np.random.rand(N)
    y = np.random.rand(N-window_size+1)
    window_sequence = WindowSequence(
        sequence, y, window_size=window_size, stride=stride, batch_size=32, shuffle=True)

    batch_x, batch_y = window_sequence.__getitem__(0)


if __name__ == "__main__":
    main_test()
