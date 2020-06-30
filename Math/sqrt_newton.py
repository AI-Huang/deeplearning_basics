#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-01 20:13:43
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os
import math


def sqrt_newton(y, err=1e-10):  # err=1e-7
    if y < 0:
        return None
    xn = y  # x_next
    while abs(y - xn*xn) > err:
        xn = (y/xn + xn) / 2.0
        # print(xn)
    return xn


def main():
    """
    print(sqrt_newton(2))
    print(math.sqrt(2))
    """

    print("%.64f" % math.sqrt(3))
    print("%.64f" % sqrt_newton(3))


if __name__ == '__main__':
    main()
