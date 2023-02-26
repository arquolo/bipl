from __future__ import annotations

__all__ = ['fc_densenet', 'max_vit', 'vit']

from collections.abc import Iterable
from itertools import accumulate
from typing import Literal

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

from .modules import (ConvCtx, DenseBlock, DenseDelta, Encoder, Ensemble,
                      LazyBias2d, MaxVitBlock, VitBlock)


def fc_densenet(num_classes: int,
                init: int = 48,
                depths: Iterable[int] = (4, 4),
                step: int = 12,
                bottleneck: bool = False):
    # TODO: define Fold/Unfold or Stash/Braid module
    # TODO: use flat structure instead of recursive nesting used currently
    """
    Implementation of
    [The One Hundred Layers Tiramisu](https://arxiv.org/pdf/1611.09326.pdf)

    See [FC-DenseNet](https://github.com/SimJeg/FC-DenseNet) as reference
    Theano/Lasagne-based implementation,
    and (https://github.com/bfortuner/pytorch_tiramisu) as pytorch fork.
    """
    *depths, = depths
    *dims, _ = accumulate([init] + [step * depth for depth in depths])

    core: list[nn.Module] = []

    ctx = ConvCtx(parity=0)
    depth = depths.pop()
    while depths:
        dim = dims.pop()
        branch = [
            ctx.norm_act_conv(ctx.conv(dim, 1)),
            ctx.max_pool(),
            *core,
            DenseDelta(depth, step, bottleneck, ctx),
            # ? norm + act
            ctx.conv_unpool(depth * step, bias=False),
        ]
        depth = depths.pop()
        core = [
            DenseBlock(depth, step, bottleneck, ctx),
            Ensemble(nn.Identity(), branch, mode='cat'),
        ]

    core = [
        ctx.conv(init, 3, bias=False),
        *core,
        DenseBlock(depth, step, bottleneck, ctx),
        # ? norm + act
        ctx.conv(num_classes, 1),
    ]
    return nn.Sequential(*core)


# ------------------------------ token actions -------------------------------


class CatToken(nn.Module):  # [B, N, D] -> [B, 1 + N, D]
    def __init__(self,
                 dim: int,
                 count: int = 1,
                 device: torch.device | None = None):
        super().__init__()
        self.token = nn.Parameter(torch.empty(count, dim, device=device))
        nn.init.normal_(self.token)

    def extra_repr(self) -> str:
        num_tokens, features = self.token.shape
        return f'{num_tokens=}, {features=}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tokens = self.token.broadcast_to(b, -1, -1)
        return torch.cat((tokens, x), dim=1)


class PopToken(nn.Module):  # [B, N, D] -> [B, D]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0, :]


# -------------------------------- models ------------------------------------


def vit(num_classes: int,
        patch_size: int,
        depth: int,
        dim: int,
        dim_head: int = 64,
        mlp_ratio: float = 4.,
        pool: Literal['cls', 'mean'] = 'cls',
        dropout: float = 0.,
        dropout_emb: float = 0.,
        qkv_bias: bool = True,
        reattn: bool = False) -> Encoder:
    assert pool in {'cls', 'mean'}

    stem = nn.Sequential(
        nn.LazyConv2d(dim, patch_size, patch_size),
        LazyBias2d(),
        Rearrange('b c h w -> b (h w) c'),
    )

    # ViT (Vision Transformer)
    #   = patchify (convAxA/sA) + cat-token + pop-token
    # DeepViT
    #   = ViT + reattention
    # CVT (Compact Vision Transformer)
    #   = patchify + identity + mean
    # CCT (Compact Convolutional Transformer)
    #   = CVT + patchify replaced with one of
    #   - 1-2 x [conv3x3 - relu - maxpool(2,2)], i.e. eff.stride = 2-4
    #   - 1-2 x [conv7x7/s2 - relu - maxpool(2,2)], i.e. eff.stride = 4-16
    transformer = nn.Sequential(
        # Operates in (b n d) domain
        nn.Identity() if pool == 'mean' else CatToken(dim),
        nn.Dropout(dropout_emb),
        *(VitBlock(dim, dim_head, mlp_ratio, dropout, qkv_bias, reattn)
          for _ in range(depth)),
        Reduce('b n d -> b d', 'mean') if pool == 'mean' else PopToken(),
    )
    head = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes),
    )
    return Encoder(stem, transformer, head)


def max_vit(num_classes: int,
            depths: tuple[int, ...],
            dim: int,
            dim_head: int = 32,
            dim_stem: int | None = None,
            window_size: int = 7,
            bn_ratio: float = 4,
            se_ratio: float = 0.25,
            mlp_ratio: float = 4,
            qkv_bias: bool = False,
            dropout: float = 0.1) -> Encoder:
    dim_stem = dim_stem or dim
    # ctx = ConvCtx(activation=nn.GELU, inplace=False, overlap=0)  # 1
    # ctx = ConvCtx(activation=nn.GELU, inplace=False, parity=0, overlap=0)  # 2
    # ctx = ConvCtx(activation=nn.GELU, inplace=False)  # 3
    ctx = ConvCtx(activation=nn.GELU, inplace=False, parity=0)  # 4
    # ctx = ConvCtx(activation=nn.SiLU, parity=0)

    stem = nn.Sequential(
        ctx.conv(dim_stem, stride=2, bias=False),
        ctx.conv(dim_stem, 3),
    )

    dims = *(dim << i for i, _ in enumerate(depths)),
    dims = (dim_stem, *dims)

    layers: list[nn.Module] = []
    for dim, depth in zip(dims[1:], depths):
        layers += [
            MaxVitBlock(dim, dim_head, window_size, stride, bn_ratio, se_ratio,
                        mlp_ratio, dropout, qkv_bias, ctx)
            for stride in [2] + [1] * (depth - 1)
        ]

    head = nn.Sequential(
        Reduce('b d h w -> b d', 'mean'),
        nn.LayerNorm(dims[-1]),
        nn.Linear(dims[-1], num_classes),
    )
    return Encoder(stem, nn.Sequential(*layers), head)
