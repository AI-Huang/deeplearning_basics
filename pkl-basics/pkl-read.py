#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-02-19 21:31
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import pickle
import pprint


def main():
    file = open("dataset-cornell-length10-filter1-vocabSize40000.pkl", "rb")
    data = pickle.load(file)
    pprint.pprint(data)
    file.close()


if __name__ == "__main__":
    main()
