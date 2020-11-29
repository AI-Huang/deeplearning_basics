#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-19-19 00:27
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os


def main():
    # result = (-80538738812075974)³ + 80435758145817515³ + 12602123297335631³
    result = pow(-80538738812075974, 3) + \
        pow(80435758145817515, 3) + pow(12602123297335631, 3)
    print(result)


if __name__ == "__main__":
    main()
