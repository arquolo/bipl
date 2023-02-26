from __future__ import annotations

__all__ = ['ActivationFn', 'LazyConvFn', 'LazyNormFn', 'NameMixin', 'round8']

from typing import Protocol, TypeVar

from torch import nn

_T = TypeVar('_T')


def pair(t: _T | tuple[_T, ...]) -> tuple[_T, ...]:
    return t if isinstance(t, tuple) else (t, t)


class NameMixin:
    __constants__ = ['name']
    name: str

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return self.name


class LazyConvFn(Protocol):
    def __call__(self,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True) -> nn.modules.conv._ConvNd:
        ...


class ActivationFn(Protocol):
    def __call__(self, inplace: bool = ...) -> nn.Module:
        ...


class LazyNormFn(Protocol):
    def __call__(self) -> nn.Module:
        ...


def round8(v: float, divisor: int = 8) -> int:
    """Ensure that number rounded to nearest 8, and error is less than 10%

    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    n = v / divisor
    return int(max(n + 0.5, n * 0.9 + 1)) * divisor
