__all__ = [
    'Bias2d', 'BlurPool2d', 'Conv2dWs', 'Decimate2d', 'Noise', 'Upscale2d'
]

from string import ascii_lowercase
from typing import Optional

import torch
import torch.nn.functional as TF
from torch import nn

from .. import functional as F


class Noise(nn.Module):
    __constants__ = ['std']

    def __init__(self, std: float):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        return torch.empty_like(x).normal_(std=self.std).add_(x)

    def extra_repr(self) -> str:
        return f'std={self.std}'


class Bias2d(nn.Module):
    def __init__(self,
                 dim: int,
                 *size: int,
                 device: Optional[torch.device] = None):
        super().__init__()
        assert len(size) == 2
        self.bias = nn.Parameter(torch.empty(1, dim, *size, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.bias)

    def extra_repr(self) -> str:
        _, dim, *space = self.bias.shape
        return f'features={dim}, size={tuple(space)}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        size = [x.shape[2], x.shape[3]]

        if torch.jit.is_tracing() or bias.shape[2:] != size:
            # Stretch to input size
            bias = TF.interpolate(
                bias, size, mode='bicubic', align_corners=False)

        return x + bias


class Decimate2d(nn.Module):
    __constants__ = ['stride']

    def __init__(self, stride: int = 2):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, ::self.stride, ::self.stride]

    def extra_repr(self) -> str:
        return f'stride={self.stride}'


class Upscale2d(nn.Module):
    """Upsamples input tensor in `scale` times.
    Use as inverse for `nn.Conv2d(kernel=3, stride=2)`.

    There're 2 different methods:

    - Pixels are thought as squares. Aligns the outer edges of the outermost
      pixels.
      Used in `torch.nn.Upsample(align_corners=True)`.

    - Pixels are thought as points. Aligns centers of the outermost pixels.
      Avoids the need to extrapolate sample values that are outside of any of
      the existing samples.
      In this mode doubling number of pixels doesn't exactly double size of the
      objects in the image.

    This module implements the second way (match centers).
    New image size will be computed as follows:
        `destination size = (source size - 1) * scale + 1`

    For comparison see [here](http://entropymine.com/imageworsener/matching).
    """
    __constants__ = ['stride']

    def __init__(self, stride: int = 2):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.upscale2d(x, self.stride)

    def extra_repr(self):
        return f'stride={self.stride}'


class Conv2dWs(nn.Conv2d):
    """
    [Weight standartization](https://arxiv.org/pdf/1903.10520.pdf).
    Better use with GroupNorm(32, features).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d_ws(x, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)


# --------------------------------- blurpool ---------------------------------


def _pascal_triangle(n: int) -> list[int]:
    values = [1]
    for _ in range(n - 1):
        values = [a + b for a, b in zip(values + [0], [0] + values)]
    return values[:n]


class BlurPool2d(nn.Conv2d):
    def __init__(self,
                 dim: int,
                 kernel: int = 4,
                 stride: int = 2,
                 padding: int = 1,
                 padding_mode: str = 'reflect'):
        super().__init__(dim, dim, kernel, stride, padding, 1, dim, False,
                         padding_mode)
        del self.weight
        self.register_buffer(
            'weight', torch.empty(dim, 1, kernel, kernel), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not self.in_channels:
            return

        weights = [
            torch.as_tensor(_pascal_triangle(k)).float()
            for k in self.kernel_size
        ]
        letters = ascii_lowercase[:len(weights)]
        weight = torch.einsum(','.join(letters) + ' -> ' + letters, *weights)
        weight /= weight.sum()

        self.weight.copy_(weight, non_blocking=True)
