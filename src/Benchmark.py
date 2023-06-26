""" Wavelet memory sequence modeling layer """

import math
import torch
import torch.nn as nn
import pywt

import numpy as np
class MultiresLayer(nn.Module):
    def __init__(self, d_model, kernel_size=None, depth=None, wavelet_init=None, tree_select="fading", 
                 seq_len=None, dropout=0., memory_size=None, indep_res_init=False):
        super().__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.tree_select = tree_select
        if depth is not None:
            self.depth = depth
        elif seq_len is not None:
            self.depth = self.max_depth(seq_len)
        else:
            raise ValueError("Either depth or seq_len must be provided.")
        print("depth:", self.depth)

        if tree_select == "fading":
            self.m = self.depth + 1
        elif memory_size is not None:
            self.m = memory_size
        else:
            raise ValueError("memory_size must be provided when tree_select != 'fading'")

        with torch.no_grad():
            if wavelet_init is not None:
                self.wavelet = pywt.Wavelet(wavelet_init)
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
                self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [d_model, 1, 1]))
                self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [d_model, 1, 1]))
            elif kernel_size is not None:
                self.h0 = nn.Parameter(
                    torch.empty(d_model, 1, kernel_size).uniform_(-1., 1.) * 
                    math.sqrt(2.0 / (kernel_size * 2))
                )
                self.h1 = nn.Parameter(
                    torch.empty(d_model, 1, kernel_size).uniform_(-1., 1.) * 
                    math.sqrt(2.0 / (kernel_size * 2))
                )
            else:
                raise ValueError("kernel_size must be specified for non-wavelet initialization.")

            w_init = torch.empty(
                d_model, self.m + 1).uniform_(-1., 1.) * math.sqrt(2.0 / (2*self.m + 2))
            if indep_res_init:
                w_init[:, -1] = torch.empty(d_model).uniform_(-1., 1.)
            self.w = nn.Parameter(w_init)

        self.activation = nn.GELU()
        dropout_fn = nn.Dropout1d
        self.dropout = dropout_fn(dropout) if dropout > 0. else nn.Identity()

    def max_depth(self, L):
        depth = math.ceil(math.log2((L - 1) / (self.kernel_size - 1) + 1))
        return depth

    def forward(self, x):
        if self.tree_select == "fading":
            y = forward_fading(x, self.h0, self.h1, self.w, self.depth, self.kernel_size)
        elif self.tree_select == "uniform":
            y = forward_uniform(x, self.h0, self.h1, self.w, self.depth, self.kernel_size, self.m)
        else:
            raise NotImplementedError()
        y = self.dropout(self.activation(y))
        return y


def forward_fading(x, h0, h1, w, depth, kernel_size):
    res_lo = x
    y = 0.
    dilation = 1
    for i in range(depth, 0, -1):
        padding = dilation * (kernel_size - 1)
        res_lo_pad = torch.nn.functional.pad(res_lo, (padding, 0), "constant", 0)
        res_hi = torch.nn.functional.conv1d(res_lo_pad, h1, dilation=dilation, groups=x.shape[1])
        res_lo = torch.nn.functional.conv1d(res_lo_pad, h0, dilation=dilation, groups=x.shape[1])
        y += w[:, i:i + 1] * res_hi
        dilation *= 2

    y += w[:, :1] * res_lo
    y += x * w[:, -1:]
    return y


def forward_uniform(x, h0, h1, w, depth, kernel_size, memory_size):
    # x: [bs, d_model, L]
    coeff_lst = []
    dilation_lst = [1]
    dilation = 1
    res_lo = x
    for _ in range(depth):
        padding = dilation * (kernel_size - 1)
        res_lo_pad = torch.nn.functional.pad(res_lo, (padding, 0), "constant", 0)
        res_hi = torch.nn.functional.conv1d(res_lo_pad, h1, dilation=dilation, groups=x.shape[1])
        res_lo = torch.nn.functional.conv1d(res_lo_pad, h0, dilation=dilation, groups=x.shape[1])
        coeff_lst.append(res_hi)
        dilation *= 2
        dilation_lst.append(dilation)
    coeff_lst.append(res_lo)
    coeff_lst = coeff_lst[::-1]
    dilation_lst = dilation_lst[::-1]

    # y: [bs, d_model, L]
    y = uniform_tree_select(coeff_lst, dilation_lst, w, kernel_size, memory_size)
    y = y + x * w[:, -1:]
    return y


def uniform_tree_select(coeff_lst, dilation_lst, w, kernel_size, memory_size):
    latent_dim = 1
    y_lst = [coeff_lst[0] * w[:, 0, None]]
    layer_dim = 1
    dilation_lst[0] = 1
    for l, coeff_l in enumerate(coeff_lst[1:]):
        if latent_dim + layer_dim > memory_size:
            layer_dim = memory_size - latent_dim
        # layer_w: [d, layer_dim]
        layer_w = w[:, latent_dim:latent_dim + layer_dim]
        # coeff_l_pad: [bs, d, L + left_pad]
        left_pad = (layer_dim - 1) * dilation_lst[l]
        coeff_l_pad = torch.nn.functional.pad(coeff_l, (left_pad, 0), "constant", 0)
        # y: [bs, d, L]
        y = torch.nn.functional.conv1d(
            coeff_l_pad,
            torch.flip(layer_w[:, None, :], (-1,)),
            dilation=dilation_lst[l],
            groups=coeff_l.shape[1],
        )
        y_lst.append(y)
        latent_dim += layer_dim
        if latent_dim >= memory_size:
            break
        layer_dim = 2 * (layer_dim - 1) + kernel_size
    return sum(y_lst)

class MultiresNet(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        batchnorm=False,
        encoder="linear",
        n_tokens=None, 
        layer_type="multires",
        max_length=None,
        hinit=None,
        depth=None,
        tree_select="fading",
        d_mem=None,
        kernel_size=2,
        indep_res_init=False,
    ):
        super().__init__()

        self.batchnorm = batchnorm
        self.max_length = max_length
        self.depth = depth
        if encoder == "linear":
            self.encoder = nn.Conv1d(d_input, d_model, 1)
        elif encoder == "embedding":
            self.encoder = nn.Embedding(n_tokens, d_model)
        self.activation = nn.GELU()

        # Stack sequence modeling layers as residual blocks
        self.seq_layers = nn.ModuleList()
        self.mixing_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        if batchnorm:
            norm_func = nn.BatchNorm1d
        else:
            norm_func = nn.LayerNorm

        for _ in range(n_layers):
            if layer_type == "multires":
                layer = MultiresLayer(
                    d_model, 
                    kernel_size=kernel_size, 
                    depth=depth,
                    wavelet_init=hinit,
                    tree_select=tree_select,
                    seq_len=max_length,
                    dropout=dropout, 
                    memory_size=d_mem,
                    indep_res_init=indep_res_init,
                )
            else:
                raise NotImplementedError()
            self.seq_layers.append(layer)

            activation_scaling = 2
            mixing_layer = nn.Sequential(
                nn.Conv1d(d_model, activation_scaling * d_model, 1),
                nn.GLU(dim=-2),
                nn.Dropout1d(dropout),
            )

            self.mixing_layers.append(mixing_layer)
            self.norms.append(norm_func(d_model))

        # Linear layer maps to logits
        self.output_mapping = nn.Linear(d_model, d_output)

    def forward(self, x, **kwargs):
        """Input shape: [bs, d_input, seq_len]. """
        # conv: [bs, d_input, seq_len] -> [bs, d_model, seq_len]
        # embedding: [bs, seq_len] -> [bs, seq_len, d_model]
        x = self.encoder(x)
        if isinstance(self.encoder, nn.Embedding):
            x = x.transpose(-1, -2)

        for layer, mixing_layer, norm in zip(
                self.seq_layers, self.mixing_layers, self.norms):
            x_orig = x
            x = layer(x)
            x = mixing_layer(x)
            x = x + x_orig

            x = apply_norm(x, norm, self.batchnorm)

        # mean_pooling: [bs, d_model, seq_len] -> [bs, d_model]
        lengths = kwargs.get("lengths", None)
        if lengths is not None:
            lengths = lengths.to(x.device)
            # only pooling over the steps corresponding to actual inputs
            x = masked_meanpool(x, lengths)
        else:
            x = x.mean(dim=-1)

        # out: [bs, d_output]
        out = self.output_mapping(x)
        return out

def apply_norm(x, norm, batch_norm=False):
    if batch_norm:
        return norm(x)
    else:
        return norm(x.transpose(-1, -2)).transpose(-1, -2)


import torch.utils.benchmark as benchmark


print("Whole model")
N, C, W = 64, 3, 1024    
input = torch.randn(N, C, W).cuda()

t0 = benchmark.Timer(
    stmt='layer(input)',
    setup='layer = MultiresNet(d_model=256,d_input=3,kernel_size=2, depth=10, dropout=0.1).cuda()',
    globals={'input': input, 'MultiresNet': MultiresNet})
m = t0.timeit(100)
print(f"Mean:   {m.mean * 1e6:6.2f} us")
print(f"Median: {m.median * 1e6:6.2f} us")
print(m)

print("Single layer")
N, C, W = 64, 256, 1024
input = torch.randn(N, C, W).cuda()

t0 = benchmark.Timer(
    stmt='layer(input)',
    setup='layer = MultiresLayer(d_model=256,kernel_size=2, depth=10).cuda()',
    globals={'input': input, 'MultiresLayer': MultiresLayer})
m = t0.timeit(100)
print(f"Mean:   {m.mean * 1e6:6.2f} us")
print(f"Median: {m.median * 1e6:6.2f} us")
print(m)

print("10 layers")

t0 = benchmark.Timer(
    stmt='layer(input)',
    setup='''layer = nn.Sequential(
        MultiresLayer(d_model=256,kernel_size=2, depth=10),
        MultiresLayer(d_model=256,kernel_size=2, depth=10),
        MultiresLayer(d_model=256,kernel_size=2, depth=10),
        MultiresLayer(d_model=256,kernel_size=2, depth=10),
        MultiresLayer(d_model=256,kernel_size=2, depth=10),
        MultiresLayer(d_model=256,kernel_size=2, depth=10),
        MultiresLayer(d_model=256,kernel_size=2, depth=10),
        MultiresLayer(d_model=256,kernel_size=2, depth=10),
        MultiresLayer(d_model=256,kernel_size=2, depth=10),
        MultiresLayer(d_model=256,kernel_size=2, depth=10)).cuda()''',
    globals={'input': input, 'MultiresLayer': MultiresLayer, 'nn': nn})
m = t0.timeit(100)
print(f"Mean:   {m.mean * 1e6:6.2f} us")
print(f"Median: {m.median * 1e6:6.2f} us")
print(m)