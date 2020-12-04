#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-17-20 14:09
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os


def countOnes(n: int):
    count = 0
    while count < n:
        n &= n-1  # 清除最低位的1
        count += 1
    return count


def hammingDistance(x: int, y: int) -> int:
    n = x ^ y
    count = 0
    while count < n:
        n &= n-1  # 清除最低位的1
        count += 1
    return count


def main():
    M = 14  # 1 的个数
    a, b = 0b00011001010011010010101111001, 0
    print(hammingDistance(a, b))


if __name__ == "__main__":
    main()
