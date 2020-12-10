#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-18-20 15:23
# @Author  : Kelly Hwong (you@example.org)
# @RefLink    : https://pytorch.org/docs/master/tensor_view.html

import os
import torch
import numpy as np


def main():
    a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    print(a)
    print(a.view(1, 6))
    print(a.view(-1, 2))  # -1 means by default
    print(a)  # won't change original shape
    a.resize(2, 3)
    print(a.size())  # torch.Size([1, 2, 3])
    print(a.size()[0])
    a_np = a.numpy()
    print("a_np: ", a_np)
    print(a_np.size)
    print(a_np.shape)  # (1, 2, 3)


if __name__ == "__main__":
    main()
