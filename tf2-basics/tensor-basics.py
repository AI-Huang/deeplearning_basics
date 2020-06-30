#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-25-20 04:44
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import tensorflow as tf


def main():
    c = tf.constant([[1.0, 2.0, 3.0, 4.0]])
    c = tf.range(10)
    print(1-c)
    print(c[:2])


if __name__ == "__main__":
    main()
