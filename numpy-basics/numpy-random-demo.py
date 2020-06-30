#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-21 22:48:08
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os
import numpy as np

# zeros = np.zeros(1*2).reshape(1,2)
# print(zeros)

N, M = 4, 3

ones = np.ones(shape=(M, N))
rnd = np.random.randint(low=0, high=2, size=(M, N))
print(rnd)
