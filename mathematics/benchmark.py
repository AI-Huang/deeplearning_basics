#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-05-20 08:25
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import time
import math
from sqrt_newton import sqrt_newton

"""
测试写的函数的运行速度的benchmark
"""
TEST_NUM = 1000 * 1000  # 运行100次


def test_my_sqrt():
    test_func = sqrt_newton
    test_param = 9

    start = time.process_time()
    for _ in range(TEST_NUM):
        ret = test_func(test_param)
        # print(ret)
    elapsed = (time.process_time() - start)
    print("运行次数：", TEST_NUM)
    print("总时间：", elapsed, "s")
    print("平均时间：", elapsed/TEST_NUM, "s")


def main():
    test_param = 9

    start = time.process_time()
    for _ in range(TEST_NUM):
        ret = math.sqrt(test_param)
        # print(ret)
    elapsed = (time.process_time() - start)
    print("运行次数：", TEST_NUM)
    print("总时间：", elapsed, "s")
    print("平均时间：", elapsed/TEST_NUM, "s")


if __name__ == "__main__":
    main()
