from __future__ import annotations

__all__ = [
    'BottleneckResidualBlock',
    'DenseBlock',
    'DenseDelta',
    'ResNeXtBlock',
    'ResidualBlock',
    'SplitAttention',
    'SqueezeExcitation',
    'mbconv',
    'mobilenet_v2_block',
    'mobilenet_v3_block',
    'resnest_block',
]

import torch
from einops.layers.torch import Rearrange, Reduce
from torch import nn
from torchvision.ops.stochastic_depth import StochasticDepth

from .aggregates import Cat, Ensemble, Gate, Residual
from .context import ConvCtx
from .util import ActivationFn, NameMixin, round8

# --------------------------------- densenet ---------------------------------


class DenseBlock(nn.ModuleList):
    expansion = 4
    efficient = True
    __constants__ = ['step', 'depth', 'bottleneck']

    def __init__(self,
                 depth: int = 4,
                 step: int = 16,
                 bottleneck: bool = True,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx()
        dim_inner = round8(step * self.expansion)

        layers = []
        for _ in range(depth):
            layer: list[nn.Module] = []
            if bottleneck:
                layer += ctx.norm_act_conv(ctx.conv(dim_inner, 1))
            layer += ctx.norm_act_conv(ctx.conv(step, 3))
            layers.append(
                Cat(*layer) if self.efficient else nn
                .Sequential(Cat(), *layer))

        super().__init__(layers)

        self.step = step
        self.depth = depth
        self.bottleneck = bottleneck

    def __repr__(self) -> str:
        dim_in = next(m for m in self.modules()
                      if isinstance(m, nn.modules.conv._ConvNd)).in_channels

        dim_out = self.step * self.depth
        if not isinstance(self, DenseDelta):
            dim_out = (dim_in + dim_out) * (dim_in != 0)

        line = f'{dim_in}, {dim_out}, step={self.step}, depth={self.depth}'
        if self.bottleneck:
            line += ', bottleneck=True'
        return f'{type(self).__name__}({line})'

    def base_forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        xs = [x]
        for m in self:
            xs.append(m(xs))
        return xs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.base_forward(x)
        return torch.cat(xs, 1)


class DenseDelta(DenseBlock):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.base_forward(x)
        return torch.cat(xs[1:], 1)  # Omit original x


# ---------------------------------- se-net ----------------------------------


class SqueezeExcitation(NameMixin, Gate):
    def __init__(self,
                 dim: int,
                 ratio: float = 0.25,
                 activation: ActivationFn = nn.SiLU,
                 scale_activation: ActivationFn = nn.Sigmoid):
        dim_inner = round8(dim * ratio)
        super().__init__(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, dim_inner, bias=False),
            activation(),
            nn.Linear(dim_inner, dim, bias=False),
            scale_activation(),
            Rearrange('b c -> b c 1 1'),
        )
        self.name = f'{dim}, {dim_inner=}'


# ---------------------------------- resnet ----------------------------------


class ResidualBlock(nn.Sequential):
    """BasicBlock from ResNet-18/34"""
    def __init__(self,
                 dim: int,
                 dropout: float = 0.,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx()
        # TODO: Support stride
        super().__init__(
            Residual(
                ctx.conv(dim, 3, bias=False),
                ctx.norm(),
                ctx.activation_(),
                ctx.conv(dim, 3, bias=False),
                ctx.norm(),
                StochasticDepth(dropout, 'row'),
            ),
            ctx.activation_(),
        )


class BottleneckResidualBlock(nn.Sequential):
    """BottleneckBlock from ResNet-50/101/152"""

    # https://arxiv.org/abs/1512.03385
    def __init__(self,
                 dim: int,
                 bn_ratio: float = 0.25,
                 se_ratio: float = 0.25,
                 groups: int | None = 1,
                 dropout: float = 0.,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx()
        dim_inner = round8(dim * bn_ratio)
        # TODO: Support stride
        super().__init__(
            Residual(
                ctx.conv(dim_inner, 1, bias=False),
                ctx.norm(),
                ctx.activation_(),
                ctx.conv(dim_inner, 3, groups=groups, bias=False),
                ctx.norm(),
                ctx.activation_(),
                ctx.conv(dim, 1, bias=False),
                ctx.norm(),
                *([SqueezeExcitation(dim, se_ratio, ctx.activation)]
                  if se_ratio else []),
                StochasticDepth(dropout, 'row'),
            ),
            ctx.activation_(),
        )


class ResNeXtBlock(BottleneckResidualBlock):
    def __init__(self,
                 dim: int,
                 se_ratio: float = 0.25,
                 dropout: float = 0.,
                 ctx: ConvCtx | None = None):
        ctx = ctx or ConvCtx()
        # TODO: Support stride
        super().__init__(dim, 0.5, se_ratio, 32, dropout, ctx)


def _mbconv_base(dim: int,
                 dim_inner: int,
                 dim_out: int,
                 stride: int = 1,
                 se_ratio: float = 0.25,
                 ctx: ConvCtx | None = None) -> list[nn.Module]:
    ctx = ctx or ConvCtx()  # or nn.HardSwish

    children: list[nn.Module] = []
    if dim != dim_inner:
        children += [
            ctx.conv(dim_inner, 1, bias=False),
            ctx.norm(),
            ctx.activation_(),
        ]
    children += [
        ctx.conv(dim_inner, 3, groups=dim_inner, bias=False) if stride == 1
        else ctx.conv(dim_inner, stride=stride, groups=dim_inner, bias=False)
    ]
    children += [
        ctx.norm(),
        ctx.activation_(),
    ]
    if se_ratio:
        children += [
            SqueezeExcitation(dim_inner, se_ratio, ctx.activation)
            # SqueezeExcitation(dim_inner, se_ratio, ctx.activation,
            #                   nn.Hardsigmoid)
        ]
    children += [
        ctx.conv(dim_out, 1, bias=False),
        ctx.norm(),
    ]
    return children


def mobilenet_v3_block(dim: int,
                       dim_inner: int,
                       dim_out: int | None = None,
                       stride: int = 1,
                       se_ratio: float = 0.,
                       dropout: float = 0.,
                       ctx: ConvCtx | None = None):
    dim_out = dim_out or dim
    children = _mbconv_base(dim, dim_inner, dim_out, stride, se_ratio, ctx)

    if stride != 1 or dim != dim_out:
        return nn.Sequential(*children)
    if dropout:
        children += [StochasticDepth(dropout, 'row')]
    return Residual(*children)


def mobilenet_v2_block(dim: int,
                       dim_out: int | None = None,
                       stride: int = 1,
                       bn_ratio: float = 6.,
                       dropout: float = 0.,
                       ctx: ConvCtx | None = None):
    ctx = ctx or ConvCtx(activation=nn.ReLU6)
    dim_inner = round8(dim * bn_ratio)
    return mobilenet_v3_block(dim, dim_inner, dim_out, stride, 0., dropout,
                              ctx)


# ------------------------------- efficientnet -------------------------------


def mbconv(dim: int,
           stride: int = 1,
           bn_ratio: float = 3.,
           se_ratio: float = 0.25,
           dropout: float = 0.,
           ctx: ConvCtx | None = None) -> nn.Module:
    """
    According to the original article or MaxViT, this one should be:
    ```
    => x + seq(
        norm,  # ! this is redundant in lack of consecutive gelu
        conv1x1, norm, gelu, conv-dw, norm, gelu,
        se,
        conv1x1,
    )
    ```
    for stride == 1, and:
    ```
    => seq(pool, conv1x1) + seq(
        norm,  # ! this is redundant in lack of consecutive gelu
        conv1x1, norm, gelu, conv-dw-pool, norm, gelu,
        se,
        conv1x1,
    )
    ```
    otherwise.
    But this implementation uses post-norm instead, making MBConv the same as
    mobilenet_v3_block.
    """
    dim_inner = round8(dim * bn_ratio)
    return mobilenet_v3_block(dim, dim_inner, dim, stride, se_ratio, dropout,
                              ctx)


# --------------------------------- resnest ----------------------------------


class SplitAttention(NameMixin, nn.Module):
    """
    Split-Attention (aka Splat) block from ResNeSt.
    If radix == 1, equals to SqueezeExitation block from SENet.
    """
    __constants__ = NameMixin.__constants__ + ['radix']

    def __init__(self,
                 dim: int,
                 groups: int = 1,
                 radix: int = 2,
                 ctx: ConvCtx | None = None):
        assert dim % (groups * radix) == 0
        ctx = ctx or ConvCtx()
        dim_inner = dim * radix // 4

        super().__init__()
        self.radix = radix
        self.to_radix = Rearrange('b (r gc) h w -> b r gc h w', r=radix)
        self.attn = nn.Sequential(
            # Mean by radix and spatial dims
            Reduce('b r gc h w -> b gc 1 1', 'mean'),

            # Core
            ctx.conv(dim_inner, 1, groups=groups, bias=False),
            ctx.norm(),
            ctx.activation_(),
            ctx.conv(dim * radix, 1, groups=groups, bias=False),

            # Normalize
            Rearrange('b (g r c) 1 1 -> b r (g c)', g=groups, r=radix),
            nn.Sigmoid() if radix == 1 else nn.Softmax(1),
        )
        self.name = f'{dim} -> {dim_inner} -> {dim}x{radix}, groups={groups}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rchw = self.to_radix(x)
        rc = self.attn(x * self.radix if self.radix > 1 else x)
        chw = torch.einsum('b r c h w, b r c -> b c h w', rchw, rc)
        return chw.contiguous()


def resnest_block(dim: int,
                  stride: int = 1,
                  radix: int = 1,
                  groups: int = 1,
                  rate: float = 0.25,
                  dropout: float = 0,
                  ctx: ConvCtx | None = None) -> nn.Module:
    ctx = ctx or ConvCtx()
    # dim_inner = round8(dim * rate) * groups  # TODO
    if stride != 1:
        dim_inner = (dim // 4) * groups
        pool = [ctx.avg_pool(stride)]
    else:
        dim_inner = round8(dim * rate)
        pool = []

    core: list[nn.Module] = [
        ctx.conv(dim_inner, 1, bias=False),
        ctx.norm(),
        ctx.activation_(),
        ctx.conv(dim_inner * radix, 3, groups=groups * radix, bias=False),
        ctx.norm(),
        ctx.activation_(),
        SplitAttention(dim_inner, groups, radix),
        *pool,
        ctx.conv(dim, 1, bias=False),
        ctx.norm(),
        StochasticDepth(dropout, 'row'),
    ]

    residual: nn.Module
    if stride != 1:
        downsample = [
            ctx.avg_pool(stride),
            ctx.conv(dim, 1, bias=False),
            ctx.norm(),
        ]
        residual = Ensemble(core, downsample, mode='sum')
    else:
        residual = Residual(*core)

    return nn.Sequential(residual, ctx.activation_())
