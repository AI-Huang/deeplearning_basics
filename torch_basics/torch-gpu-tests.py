#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-15-20 13:41
# @Author  : Kelly Hwong (you@example.org)
# @Link    : https://pytorch.org/docs/stable/cuda.html

import os
import torch


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # "0", 在 Windows 和 notebook 上无效
    print(
        f"""os.environ["CUDA_VISIBLE_DEVICES"]: {os.environ["CUDA_VISIBLE_DEVICES"]}""")

    use_gpu = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {use_gpu}")

    if use_gpu:
        ret = torch.cuda.get_device_name()
        print(f"torch.cuda.get_device_name(): {ret}")

        ret = torch.cuda.device_count()
        print(f"torch.cuda.device_count(): {ret}")

        ret = torch.cuda.get_device_capability()
        print(f"torch.cuda.get_device_capability(): {ret}")


if __name__ == "__main__":
    main()
