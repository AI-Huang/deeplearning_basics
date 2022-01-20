#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-20-22 16:05
# @Author  : Kelley Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://karpathy.github.io/2015/05/21/rnn-effectiveness/

import numpy as np
import torch
import torch.nn as nn


class RNN(object):
    """

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    """

    def __init__(self, word_to_idx, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        self.word_to_idx = word_to_idx
        self.wordvec_dim = wordvec_dim
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type
        self.dtype = dtype

        self.h = None
        self.params = {}
        self.reset_parameters()

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def reset_parameters(self):
        wordvec_dim = self.wordvec_dim
        hidden_dim = self.hidden_dim
        cell_type = self.cell_type

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params["W_xh"] = np.random.randn(
            wordvec_dim, dim_mul * hidden_dim)
        self.params["W_xh"] /= np.sqrt(wordvec_dim)
        self.params["b_xh"] = np.zeros(dim_mul * hidden_dim)

        self.params["W_hh"] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["W_hh"] /= np.sqrt(hidden_dim)
        self.params["b_hh"] = np.zeros(dim_mul * hidden_dim)

    def set_parameter(self, name, value):
        """
        Input:
            name: e.g.: "W_xh"
        """
        if name not in self.params:
            raise ValueError("There is no parameter name: {name}.")
        param = self.params[name]
        if value.shape != param.shape:
            raise ValueError(
                "Value's shape  {value.shape} does NOT equal to parameter's shape {param.shape}.")
        param = value

    def step(self, x, hidden_state=None):
        """
        x: should only have one timestep
        """
        W_xh = self.params["W_xh"]
        b_xh = self.params["b_xh"]
        W_hh = self.params["W_hh"]
        b_hh = self.params["b_hh"]

        # W_hy = self.params["W_hy"]
        # b_hy = self.params["b_hy"]

        # update the hidden state
        if hidden_state is not None:
            self.h = hidden_state

        self.h = np.tanh(x.dot(W_xh) + b_xh +
                         self.h.dot(W_hh) + b_hh)

        # compute the output vector
        # y = np.dot(W_hy, self.h) + b_hy
        y = None

        # and return the hidden state after this time step

        return y, self.h

    def forward(self, x, hidden_state=None, transpose=False):
        """
        Inputs:
            - x: Input data for the entire timeseries, of shape (T, N, D).
        """
        if hidden_state is not None:
            self.h = hidden_state

        if transpose:
            x = x.transpose(1, 0, 2)  # (N, T, D) -> (T, N, D)
        T, N, D = x.shape

        output = np.empty((T, N, self.hidden_dim))

        seq_len = T
        for i in range(seq_len):
            y, hn = self.step(x[i], self.h)
            output[i] = hn

        return output, hn


def main():
    # input_size = 10
    # hidden_size = 20

    N, D, W, H = 16, 20, 30, 40
    word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
    V = len(word_to_idx)
    T = 13
    wordvec_dim = W  # 30
    input_size = wordvec_dim
    hidden_size = H  # 40

    # N: number of samples
    rnn = RNN(word_to_idx, wordvec_dim=wordvec_dim, hidden_dim=hidden_size)

    # Store init weights
    params = {}
    params["weight_ih_l0"] = rnn.params["W_xh"]
    params["weight_hh_l0"] = rnn.params["W_hh"]
    params["bias_ih_l0"] = rnn.params["b_xh"]
    params["bias_hh_l0"] = rnn.params["b_hh"]

    # Forward
    _input = np.random.randn(5, N, input_size)  # (L, N, H_in)
    h0 = np.random.randn(1, N, hidden_size)  # (S, N, H_out)
    output, hn = rnn.forward(_input, h0[0])  # (N, H_out)

    print(f"hn.shape: {hn.shape}")
    print(f"hn.sum: {hn.sum()}")

    print(f"output.shape: {output.shape}")  # (L, N, H_out)
    print(f"output.sum: {output.sum()}")

    # Compare with PyTorch nn.RNN
    rnn_torch = nn.RNN(input_size=input_size, hidden_size=hidden_size)
    state_dict = rnn_torch.state_dict()
    state_dict.keys()

    # Cast parameters to correct dtype
    for k, v in params.items():
        params[k] = torch.from_numpy(v.astype(np.float32))

    # Load parameters to state_dict
    for k, v in params.items():
        state_dict[k] = params[k].T
    rnn_torch.load_state_dict(state_dict)

    _input = torch.from_numpy(_input.astype(np.float32))  # (L, N, H_in)
    h0 = torch.from_numpy(h0.astype(np.float32))  # (S, N, H_out)
    output, hn = rnn_torch(_input, h0)
    print(f"hn.shape: {hn.shape}")
    print(f"hn.sum: {hn.sum()}")

    print(f"output.shape: {output.shape}")  # (L, N, H_out)
    print(f"output.sum: {output.sum()}")


if __name__ == "__main__":
    main()
