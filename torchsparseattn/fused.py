"""Fusedmax attention

Clusters neighboring attention weights into groups with equal weight.

A Regularized Framework for Sparse and Structured Neural Attention
Vlad Niculae, Mathieu Blondel
https://arxiv.org/abs/1705.07704
"""

from __future__ import division

import torch
from torch import nn
from torch import autograd as ta
import warnings
import torchsparse 
from .base import base_forward, base_backward
from .sparsemax import sparsemax_function
from torchsparse._fused import prox_tv1d


def _inplace_fused_prox_jv_slow(y_hat, dout):
    """not efficient in python for long seqs, but template for a cython impl"""

    n_features = len(dout)

    for i in range(n_features + 1):
        if i in (0, n_features) or y_hat[i] != y_hat[i - 1]:
            if i > 0:
                dout[last_ix:i] = acc / n

            if i < n_features:
                last_ix = i
                acc = dout[i]
                n = 1
        else:
            acc += dout[i]
            n += 1

    return dout


try:
    from torchsparse._fused_jv import _inplace_fused_prox_jv
except ImportError:
    warnings.warn(
        "Could not import cython implementation of fused backward "
        "pass. Slow implementation used instead."
    )
    _inplace_fused_prox_jv = _inplace_fused_prox_jv_slow


def fused_prox_jv_slow(y_hat, dout):
    dout = dout.clone()
    _inplace_fused_prox_jv_slow(y_hat, dout)

    return dout


def fused_prox_jv_fast(y_hat, dout):
    dout = dout.clone()
    _inplace_fused_prox_jv(y_hat.detach().cpu().numpy(), dout.cpu().numpy())

    return dout


def project(x, alpha):
    x_np = x.detach().cpu().numpy().copy()
    prox_tv1d(x_np, alpha)
    
    y_hat = torch.from_numpy(x_np)

    return y_hat


def project_jv(dout, y_hat):
    dout = dout.clone()
    _inplace_fused_prox_jv(y_hat.detach().cpu().numpy(), dout.cpu().numpy())

    return dout


class FusedProxFunction(ta.Function):
    @staticmethod
    def forward(ctx, x, alpha, lengths=None):
        return base_forward(ctx, x, lambda x: project(x, alpha), lengths=lengths)

    @staticmethod
    def backward(ctx, dout):
        grad, _ = base_backward(ctx, dout, project_jv)

        return grad, None, None


fusedprox_function = FusedProxFunction.apply


class Fusedmax(nn.Module):
    def __init__(self, alpha=1):
        self.alpha = alpha
        super(Fusedmax, self).__init__()

    def forward(self, x, lengths=None):
        return sparsemax_function(fusedprox_function(x, self.alpha, lengths), lengths)
    
class Fusedmax3D(nn.Module):
    def __init__(self, alpha=1):
        super(Fusedmax3D, self).__init__()
        self.alpha = alpha

    def forward(self, x, lengths=None):
        # x is expected to be a 3D tensor: [batch_size, seq_length, features]
        batch_size, seq_length, features = x.shape

        # Process each item in the batch
        output = torch.zeros_like(x)
        for i in range(batch_size):
            # Apply the Fusedmax to each sequence in the batch individually
            output[i] = sparsemax_function(fusedprox_function(x[i], self.alpha, lengths), lengths)

        return output

if __name__ == "__main__":
    from timeit import timeit

    torch.manual_seed(1)

    for dim in (5, 10, 50, 100, 500, 1000):

        x = torch.randn(dim)
        x_var = ta.Variable(x, requires_grad=True)
        y_hat = fusedprox_function(x_var,0.2).data
        dout = torch.arange(0, dim)
        print("dimension={}".format(dim))
        print(
            "slow",
            timeit("fused_prox_jv_slow(y_hat, dout)",
                   globals=globals(), number=10000),
        )
        print(
            "fast",
            timeit("fused_prox_jv_fast(y_hat, dout)",
                   globals=globals(), number=10000),
        )
