# encoding: utf8

"""
From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label
Classification. André F. T. Martins, Ramón Fernandez Astudillo
In: Proc. of ICML 2016, https://arxiv.org/abs/1602.02068
"""

from __future__ import division

import numpy as np
import torch
from torch import nn
from .base import base_forward, base_backward
from torch.autograd import Function


def project_simplex(v, z=1):
    v_sorted, _ = torch.sort(v, dim=0, descending=True)
    cssv = torch.cumsum(v_sorted, dim=0) - z
    ind = torch.arange(1, 1 + len(v)).to(dtype=v.dtype, device=v.device)
    cond = v_sorted - cssv / ind > 0
    rho = ind.masked_select(cond)[-1]
    tau = cssv.masked_select(cond)[-1] / rho
    w = torch.clamp(v - tau, min=0)

    return w


def sparsemax_grad(dout, w_star):
    supp = w_star > 0
    masked = dout.masked_select(supp)
    nnz = supp.to(dtype=dout.dtype).sum()
    masked -= masked.sum() / nnz
    out = dout.new(dout.size()).zero_()
    out[supp] = masked

    return out


class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, x, lengths=None):
        return base_forward(ctx, x, project_simplex, lengths=lengths)

    @staticmethod
    def backward(ctx, dout):
        return base_backward(ctx, dout, sparsemax_grad)


sparsemax_function = SparsemaxFunction.apply


class Sparsemax(nn.Module):
    def forward(self, x, lengths=None):

        return sparsemax_function(x, lengths)
