#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Oct-31-20 00:30
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import numpy as np
import tensorflow as tf


def main():
    # Manually build a graph.
    graph = tf.Graph()

    with graph.as_default():
        a = tf.constant(5.0)
        b = tf.constant(6.0)
        c = a * b

    # Launch the graph in a session.
    # sess = tf.compat.v1.Session()
    sess = tf.Session(graph=graph)

    # Evaluate the tensor `c`.
    print(sess.run(c))


if __name__ == "__main__":
    main()
