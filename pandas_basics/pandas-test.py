#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : May-26-20 21:37
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import pandas as pd


def main():
    path = os.path.dirname(__file__)
    path = os.path.join(path, "test.csv")
    df = pd.read_csv(path, encoding="utf-8")
    print(df)
    v = df["name"].values
    print(v)
    print(type(v))


if __name__ == "__main__":
    main()
