#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-31-20 16:05
# @Update  : Jan-11-21 22:42
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://zhuanlan.zhihu.com/p/63974249

"""conv2d
2D convolution, where every channel has its own kernel and its weights.
二维卷积，每个通道分配一个核和一个权重的版本。
"""

import numpy as np
import matplotlib.pyplot as plt


def im2col(images, ksize, stride=1):
    """im2col algorithm

    # Arguments:
        images: a image batch with 4-dims ([N, H, W, C_in]).
        kernels: number of C_out kernels of C_out filters.
    # Return:
        col: col representation of batch of images
    """
    N, H, W, C = images.shape
    out_h = (H - ksize) // stride + 1
    out_w = (W - ksize) // stride + 1

    outsize = out_w * out_h
    col = np.empty((N * out_h * out_w, ksize * ksize * C))

    for y in range(out_h):
        y_min = y * stride
        y_max = y_min + ksize
        y_start = y * out_w
        for x in range(out_w):
            x_min = x * stride
            x_max = x_min + ksize
            # 核心代码, extract image patch
            # y_min:y_max, x_min:x_max, means image area
            # for every sample, for every channel
            # to N rows, ksize * ksize * C cols
            # for every outsize steps
            col[y_start+x::outsize, :] = \
                images[:, y_min:y_max, x_min:x_max, :].reshape(N, -1)
    return col


def conv2d_im2col(images, kernels, stride=1, padding="same"):
    """im2col-style conv2d on batch of images

    Notation: without kernel sharing, kernels must have same channels with the input images.

    Arguments:
        images: a batch of images with 4-dims ([N, H, W, C]).
        kernels: kernels with 4-dims ([N_filters, ksize, ksize, C]).
    """
    N, H, W, C = images.shape
    N_filters, ksize, ksize, C = kernels.shape

    assert padding in ["same", "valid"]
    if padding == "same":
        p = ksize // 2
        images = np.pad(images, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')
        # belows are WRONG
        out_h = (H - ksize) // stride + 1
        out_w = (W - ksize) // stride + 1
    elif padding == "valid":
        out_h = (H - ksize) // stride + 1
        out_w = (W - ksize) // stride + 1

    # get X patches with im2col
    col = im2col(images, ksize, stride)

    # calculate convolution
    z = np.dot(col, kernels.reshape((kernels.shape[0], -1)).T)
    z = z.reshape((N, z.shape[0] // N, -1))
    z = z.reshape((N, out_h, out_w, N_filters))

    return z


def im2col_test():
    N, H, W, C = 16, 32, 32, 3
    images = np.random.randint(0, 256, size=(N, H, W, C))
    col = im2col(images, ksize=3, stride=1)
    print(f"col.shape: {col.shape}")


def conv2d_im2col_test():
    N, H, W, C = 16, 32, 32, 3
    N_filters, ksize, ksize, C = 6, 3, 3, 3
    images = np.random.randint(0, 256, size=(N, H, W, C))
    kernels = np.random.random(size=(N_filters, ksize, ksize, C))
    z = conv2d_im2col(images, kernels, padding="valid")
    print(f"z.shape: {z.shape}")


def filt(sub_image, kernel):
    """filt, filt a sub_image with kernel, sub_image and kernel **must** be the same size.
    Inputs:
        sub_image:
        kernel: np.array like, the kernel of the filter
    Return:
        pixel: a pixel value
    """
    assert kernel.shape[0] == kernel.shape[1]
    assert sub_image.shape == kernel.shape

    F = kernel.shape[0]

    pixel = 0
    for f1 in range(F):
        for f2 in range(F):
            pixel += sub_image[f1, f2] * kernel[f1, f2]

    return pixel


def conv2d_on_one_image(image, kernel, stride=2):
    """the image has 3 channels for RGB, 4 if it's RGBA
    Arguments:
        image: one 3-dim ([H, W, C]) image.
    """

    N = image.shape[0]
    F = kernel.shape[0]
    channels = image.shape[-1]  # 3

    O = int(np.ceil((N - F) / stride + 1))
    output = np.zeros([O, O, channels])

    for channel in range(channels):
        _image = image[:, :, channel]  # (20, 20)

        N = _image.shape[0]
        F = kernel.shape[0]
        if (N-F) % stride != 0:
            dN = stride - (N-F) % stride  # 补零
            for _ in range(dN):
                _image = np.concatenate([_image, np.zeros([N, 1])], axis=1)
                # print(_image.shape) # 20, 21
            N += dN
            for _ in range(dN):
                _image = np.concatenate([_image, np.zeros([1, N])], axis=0)
                # print(_image.shape) # 21, 21

        O = int((N - F) / stride + 1)
        channel_output = np.zeros([O, O])
        for i in range(O):
            for j in range(O):
                _sub_image = _image[stride*i:stride*i+F, stride*j:stride*j+F]
                _o = 0
                for f1 in range(F):
                    for f2 in range(F):
                        _o += _sub_image[f1, f2] * kernel[f1, f2]
                channel_output[i][j] += _o

        output[:, :, channel] = channel_output

    return output


def conv2d(images, kernels, stride=2):
    """conv2d on batch of images

    Notation: without kernel sharing, C_out must be same with C_in

    Arguments:
        images: a batch of images with 4-dims ([N, H, W, C_in]).
        kernels: number of C_out kernels of C_out filters.
    """
    C_in = images.shape[-1]  # channel_last
    C_out = kernels.shape[-1]  # channel_last
    assert C_in == C_out

    N = images.shape[0]
    F = kernels.shape[0]
    O = int(np.ceil((N - F) / stride + 1))
    output = np.zeros([O, O, C_in])

    for channel in range(C_in):
        _image = images[:, :, channel]  # (20, 20)
        _kernel = kernels[:, :, channel]

        # padding
        if (N-F) % stride != 0:
            dN = stride - (N-F) % stride
            for _ in range(dN):
                _image = np.concatenate([_image, np.zeros([N, 1])], axis=1)
                # print(_image.shape) # 20, 21
            N += dN
            for _ in range(dN):
                _image = np.concatenate([_image, np.zeros([1, N])], axis=0)
                # print(_image.shape) # 21, 21

        O = int((N - F) / stride + 1)
        c_output = np.zeros([O, O])
        for i in range(O):
            for j in range(O):
                _sub_image = _image[stride*i:stride*i+F, stride*j:stride*j+F]
                _o = 0
                for f1 in range(F):
                    for f2 in range(F):
                        _o += _sub_image[f1, f2] * _kernel[f1, f2]
                c_output[i][j] += _o

        output[:, :, channel] = c_output

    return output


def conv2d_test():
    kernel = np.random.random([3, 3])
    image = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 1, 1]], dtype=np.uint8)
    # image = base_img[:, :, np.newaxis]
    # plt.matshow(image, cmap="gray")
    # plt.colorbar()
    # plt.show()

    image = np.expand_dims(image, axis=2)
    images = np.concatenate((image, image, image), axis=2)
    kernel = np.expand_dims(kernel, axis=2)
    kernels = np.concatenate((kernel, kernel, kernel), axis=2)

    plt.imshow(np.asarray(images, dtype=np.float32))
    # plt.show()
    image_conv = conv2d(images, kernels, stride=2)


def test_once(func, *args, **kwargs):
    import time
    start = time.perf_counter()
    func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    # print(f"func_name, time used: {elapsed}s")
    return elapsed


def performance_test():
    N, H, W, C = 16, 32, 32, 3
    N_filters, ksize, ksize, C = 6, 3, 3, 3
    images = np.random.randint(0, 256, size=(N, H, W, C))
    kernels = np.random.random(size=(N_filters, ksize, ksize, C))

    import time

    start = time.perf_counter()
    conv2d_im2col(images, kernels, padding="valid")
    elapsed = time.perf_counter() - start
    print(f"conv2d_im2col, time used: {elapsed}s")

    start = time.perf_counter()
    for i in range(N):
        for j in range(N_filters):
            conv2d(images[i], kernels[j], stride=1)
    elapsed = time.perf_counter() - start
    print(f"conv2d, time used: {elapsed}s")


def main():
    performance_test()
    # im2col_test()
    # conv2d_im2col_test()


if __name__ == "__main__":
    main()
