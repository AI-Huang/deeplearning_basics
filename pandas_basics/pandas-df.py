#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : May-22-20 13:36
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import pandas as pd


def main():
    # Creating the dataframe
    df = pd.DataFrame({"A": [12, 4, 5, 44, 1],
                       "B": [5, 2, 54, 3, 2],
                       "C": [20, 16, 7, 3, 8],
                       "D": [14, 3, 17, 2, 6]})
    print(df)
    print("max min...")
    out = df.max(axis=0)
    print(out)
    out = df.min(axis=0)
    print(out)
    print(type(out))  # <class 'pandas.core.series.Series'>

    # out = df.max()
    out = df.max(axis=0)  # default axis is 0, max of each column
    print(out)

    out = df.max(axis=1)  # axis 1, max of each row
    print(out)


if __name__ == "__main__":
    main()
