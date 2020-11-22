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
        self.y = y
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        # index of the beginning element of the windows
        self.index = np.arange(0, N-window_size, step=stride)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, batch_index):
        """Gets batch at batch_index `batch_index`.

        Arguments:
            batch_index: batch_index of the batch in the Sequence.

        Returns:
            A batch
        """
        sequence = self.sequence
        y = self.y
        window_size = self.window_size

        batch_size = self.batch_size
        sample_index = self.index[batch_index *
                                  batch_size:(batch_index + 1) * batch_size]

        # end_idx <= sequence.shape[0]
        # stride*i+window_size<=sequence.shape[0]
        # 比如， stride 为 1，i<=sequence.shape[0]-window_size，i从0开始计数

        batch_x = np.empty((batch_size, window_size))
        batch_y = np.empty(batch_size)
        for _, i in enumerate(sample_index):
            batch_x[_, ] = sequence[i:i + window_size]
            # label element on the right edge of the window
            batch_y[_] = y[i+window_size]
        # batch_x must have dimensions (N, D1, D2)
        batch_x = np.expand_dims(batch_x, -1)
        return batch_x, batch_y

    def __len__(self):
        """Number of batch in the Sequence.

        Returns:
            The number of batches in the Sequence.
        """
        return self.index.shape[0] // self.batch_size + 1

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        if self.shuffle == True:
            N = self.sequence.shape[0]
            window_size = self.window_size
            stride = self.stride
            self.index = np.arange(0, N-window_size, step=stride)
            np.random.shuffle(self.index)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def normalize_y(self):
        """Normalize y on the whole dataset
        """
        self.y_mean = self.y.mean()
        self.y_std = self.y.std()
        self.y = (self.y - self.y_mean) / self.y_std

    def denormalize_y(self):
        raise NotImplementedError


class WindowSequences(WindowSequence):
    pass


def main_test():
    N = 200000

    window_size = 7200
    stride = 24

    sequence = np.random.rand(N)
    y = np.random.rand(N)
    window_sequence = WindowSequence(
        sequence, y, window_size=window_size, stride=stride, batch_size=32, shuffle=True)

    batch_x, batch_y = window_sequence.__getitem__(0)

    print(f"num_samples: {len(window_sequence.index)}")
    print(f"num_batches: {len(window_sequence)}")
    print(f"total_size: {window_size*len(window_sequence.index)}")


if __name__ == "__main__":
    main_test()
