#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-05-20 10:11
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np


def main():
    # 设置类别的数量
    num_classes = 10
    # 需要转换的整数
    arr = [1, 3, 4, 5, 9]
    # 将整数转为一个10位的one hot编码
    print(np.eye(10))
    print(np.eye(10)[0])
    print(np.eye(10)[arr])


if __name__ == "__main__":
    main()
