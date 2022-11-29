#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jul-16-21 16:38
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import os
import torch
import perfplot


def copy_tensor_test():
    # https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
    perfplot.show(
        setup=lambda n: torch.randn(n),
        kernels=[
            lambda a: a.new_tensor(a),
            lambda a: a.clone().detach(),
            lambda a: torch.empty_like(a).copy_(a),
            lambda a: torch.tensor(a),
            lambda a: a.detach().clone(),
        ],
        labels=["new_tensor()", "clone().detach()", "empty_like().copy()",
                "tensor()", "detach().clone()"],
        n_range=[2 ** k for k in range(15)],
        xlabel="len(a)",
        logx=False,
        logy=False,
        # title='Timing comparison for copying a pytorch tensor',
    )


def main():
    copy_tensor_test()


if __name__ == "__main__":
    main()
