#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-17-20 14:32
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import sys
from torchvision import transforms
import torchvision
from PIL import Image


def main():
    # For example: python image_transformation.py kaggle_data/train/bad_1 13033
    data_transforms = transforms.Compose([
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomAffine(degrees=2, translate=(
            0.15, 0.1), scale=(0.75, 1.05))
    ])

    my_dir = sys.argv[1]
    bad_ones = os.listdir(my_dir)
    N = int(sys.argv[2])
    n = 0
    while n < N:
        for filename in bad_ones:
            new_img = data_transforms(
                Image.open(os.path.join(my_dir, filename)))
            new_filename = 'generate_'+str(n)+'.jpg'
            new_img.save(os.path.join(my_dir, new_filename))
            n = n+1
            if n == N:
                break


if __name__ == "__main__":
    main()
