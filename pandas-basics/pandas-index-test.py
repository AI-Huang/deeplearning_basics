#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-26-20 16:01
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import pandas as pd
import os


def main():
    df =
    # print(df)pd.read_csv("./data/friends.csv", encoding="utf-8")
    # print(df["user_id"])
    # print(df.keys())

    # print(df["time_zone"])
    tmp = df["nick_name"]  # Series
    user_id = 491746682
    out = df[df["user_id"].isin(
        [user_id])]["time_zone"].tolist()[0]  # .to_string()
    print(type(out))


if __name__ == "__main__":
    main()
