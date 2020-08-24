#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-04-20 21:33
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import numpy as np


def main():
    np1 = np.ones(1)
    print(np1)
    int1from_np = int(np1)
    print(int1from_np)
    print(int1from_np == np1)


if __name__ == "__main__":
    main()
