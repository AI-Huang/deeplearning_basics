#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-13-20 13:00
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv1D, LSTM, Bidirectional


def lstm_model(inputs):
    x = inputs
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    model = Model(inputs=inputs, outputs=x)
    return model


def main():
    pass


if __name__ == "__main__":
    main()
