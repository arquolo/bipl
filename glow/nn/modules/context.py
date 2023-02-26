from __future__ import annotations

__all__ = ['ConvCtx']

import warnings
from dataclasses import dataclass, replace
from typing import Literal

from torch import nn

from .lazy import LazyBlurPool2d
from .simple import Decimate2d
from .util import ActivationFn, LazyConvFn, LazyNormFn


@dataclass(frozen=True)
class ConvCtx:
    """
    Parity:
    - 0: kernel = 2 * overlap + stride
    - 1: kernel = 2 * overlap + 1
    """
    conv_fn: LazyConvFn = nn.LazyConv2d
    # TODO: accept kernel_pool
    # TODO: compute padding from kernel_pool
    parity: Literal[0, 1] = 1
    overlap: int = 1
    pad: bool = True

    norm: LazyNormFn = nn.LazyBatchNorm2d

    activation: ActivationFn = nn.ReLU
    inplace: bool = True

    def _get_k(self, stride: int, overlap: int | None = None) -> int:
        if overlap is None:
            overlap = self.overlap
        return (stride if self.parity == 0 else 1) + 2 * overlap

    def _get_p(self, kernel: int, stride: int, dilation: int) -> int:
        assert stride == 1 or dilation == 1, \
            'one of stride/dilation should be always 1'
        if stride == 1 and kernel % 2 == 0 and dilation % 2 != 0:
            raise ValueError('Even kernel with odd dilation is not supported')

        if self.parity == 0:
            total_padding = kernel - stride
            assert total_padding >= 0, \
                'kernel should be same or greater than stride'
            assert total_padding >= 0
        else:
            total_padding = kernel - 1

        assert total_padding % 2 == 0, \
            'padding is not symmetric, offset kernel by 1'
        if not self.pad:
            return 0
        return (total_padding // 2) * dilation

    def _invert(self) -> ConvCtx:
        return replace(self, parity=1 - self.parity)

    def conv(self,
             dim: int,
             kernel: int | None = None,
             stride: int = 1,
             overlap: int | None = None,
             dilation: int = 1,
             groups: int | None = 1,
             bias: bool = True) -> nn.modules.conv._ConvNd:
        """
        Basic convolutions, pass kernel explicitly:
        ```
        >>> ctx.conv(64, 1)  # conv1x1, 64f
        >>> ctx.conv(64, 3)  # conv3x3/pad=1, 64f
        >>> ctx.conv(64, 3, groups=64)  # dw-conv3x3/pad=1
        ```
        Dilations, will fail with stride != 1:
        ```
        >>> ctx.conv(64, 3, dilation=2)  # conv3x3/d2/pad=2
        ```

        Strided convolutions.
        Better use implicit kernel if parity switch is desired.
        Even mode (parity = 0):
        ```
        >>> ctx.conv(64, stride=2)     # conv4x4/s2/pad=1
        >>> ctx.conv(64, 4, stride=2)  # same, explicit
        >>> ctx.conv(64, 3, stride=2)  # fail
        ```
        Odd mode (parity = 1), can introduce aliasing:
        ```
        >>> ctx.conv(64, stride=2)     # conv3x3/s2/pad=1
        >>> ctx.conv(64, 3, stride=2)  # same, explicit
        >>> ctx.conv(64, 4, stride=2)  # fail
        ```
        """
        if kernel is None:
            kernel = self._get_k(stride, overlap)

        elif stride == 1 and kernel % 2 == 0:
            warnings.warn(f'Parity is altered because of {kernel=}')

        elif stride != 1 and kernel != self.parity:
            raise ValueError(f'Used kernel does not match used parity '
                             f'({self.parity=}). Network is likely to break')

        padding = self._get_p(kernel, stride, dilation)
        groups = groups or dim
        return self.conv_fn(dim, kernel, stride, padding, dilation, groups,
                            bias)

    def avg_pool(self,
                 stride: int = 2,
                 overlap: int | None = None) -> Decimate2d | nn.AvgPool2d:
        assert stride > 1

        kernel = self._get_k(stride, overlap)
        if kernel == 1:
            return Decimate2d(stride)

        padding = self._get_p(kernel, stride, 1)
        return nn.AvgPool2d(kernel, stride, padding)

    def max_pool(self,
                 stride: int = 2,
                 overlap: int | None = None) -> Decimate2d | nn.MaxPool2d:
        assert stride > 1

        kernel = self._get_k(stride, overlap)
        if kernel == 1:
            return Decimate2d(stride)

        padding = self._get_p(kernel, stride, 1)
        return nn.MaxPool2d(kernel, stride, padding)

    def conv_unpool(self,
                    dim: int,
                    stride: int = 2,
                    overlap: int | None = None,
                    groups: int | None = 1,
                    bias: bool = True) -> nn.ConvTranspose2d:
        groups = groups or dim
        kernel = self._get_k(stride, overlap)
        padding = self._get_p(kernel, stride, 1)
        return nn.LazyConvTranspose2d(dim, kernel, stride, padding, 0, groups,
                                      bias)

    def activation_(self) -> nn.Module:
        if self.inplace:
            return self.activation(inplace=True)
        return self.activation()

    def norm_act(self, act: bool = True) -> list[nn.Module]:
        norm = self.norm()
        return [norm, self.activation_()] if act else [norm]

    def conv_norm(self, mod: nn.modules.conv._ConvNd) -> nn.Sequential:
        assert self.norm
        mod.bias = None
        return nn.Sequential(mod, self.norm())

    def conv_norm_act(self, mod: nn.modules.conv._ConvNd) -> nn.Sequential:
        norms: list[nn.Module] = []
        if self.norm:
            mod.bias = None  # Normalization overrides this with its own bias
            norms += [self.norm()]
        return nn.Sequential(mod, *norms, self.activation_())

    def norm_act_conv(self, mod: nn.modules.conv._ConvNd) -> nn.Sequential:
        assert self.norm
        mod.bias = None
        return nn.Sequential(self.norm(), self.activation_(), mod)

    def blur_pool(self,
                  stride: int = 2,
                  overlap: int | None = None) -> nn.Module:
        """Replacement (and generalization) for AvgPool2d(2, 2)"""
        kernel = self._get_k(stride, overlap)
        padding = self._get_p(kernel, stride, 1)
        return LazyBlurPool2d(kernel, stride, padding)

    def max_blur_pool(self, stride: int = 2, overlap: int = 1) -> nn.Module:
        """
        Replacement for MaxPool2d(2, 2)
        Only for even inputs. When applied to odd inputs, loses 1 sample.
        """
        assert self.parity == 0
        return nn.Sequential(
            # switches parity
            nn.MaxPool2d(2, 1),
            # correct pool
            self._invert().blur_pool(stride, overlap),
        )

    def conv_blur_pool(self,
                       dim: int,
                       stride: int = 2,
                       overlap: int = 1) -> nn.Module:
        """Replacement for [Conv2d(..., 3, 2, 1), norm, act]"""
        return nn.Sequential(
            self.conv(dim, 3),
            self.norm(),
            self.activation_(),
            self.blur_pool(stride, overlap),
        )
