#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-23 13:52:57
# @Author  : Your Name (you@example.org)
# @Link    : http://www.runoob.com/numpy/numpy-array-from-existing-data.html
# @Version : $Id$

import os
import numpy as np

"""
numpy.asarray(a, dtype = None, order = None)
a   任意形式的输入参数，可以是，列表, 列表的元组, 元组, 元组的元组, 元组的列表，多维数组

"""

"""
numpy.zeros(shape, dtype = float, order = 'C')
"""

_ = np.zeros([2, 3, 4], dtype=float, order='C')
print(_)
l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
t = (4, 2)
l_np = np.asarray(l)
