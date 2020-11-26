#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-26-20 14:26
# @Update  : Nov-25-20 01:04
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import os
import numpy as np
import pandas as pd


def plotxxxxx():
    # df['label'].value_counts().plot.bar()
    # plt.show()

    # print("Sample image...")
    # sample = random.choice(filenames)
    # image = load_img("./data/train/"+sample)
    # plt.imshow(image)
    # plt.show()
    # train_df['label'].value_counts().plot.bar()

    # """ Example Generation """
    # example_df = train_df.sample(n=1).reset_index(drop=True)
    # example_generator = train_datagen.flow_from_dataframe(
    #     example_df,
    #     TRAIN_DATA_DIR,
    #     x_col='filename',
    #     y_col='label',
    #     target_size=IMAGE_SIZE,
    #     class_mode='categorical'
    # )
    """ Example Generation Ploting """
    # plt.figure(figsize=(12, 12))
    # for i in range(0, 15):
    #     plt.subplot(5, 3, i+1)
    #     for X_batch, Y_batch in example_generator:
    #         image = X_batch[0]
    #         plt.imshow(image)
    #         break
    # plt.tight_layout()
    # plt.show()

    # Heatmap


if __name__ == "__main__":
    main()
