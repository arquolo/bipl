__all__ = ['Show']

import weakref

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# TODO: rewrite like `def traced(nn.Module) -> nn.Module`
# TODO: use pyqt/matplotlib to create window


class Show(nn.Module):
    """Shows contents of tensors during forward pass"""
    nsigmas = 2
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, colored: bool = False):
        super().__init__()
        self.name = f'{type(self).__name__}_0x{id(self):x}'
        self.colored = colored

        weight = torch.tensor(128 / self.nsigmas)
        bias = torch.tensor(128.)
        self.register_buffer('weight', weight, persistent=False)
        self.register_buffer('bias', bias, persistent=False)

        self.close = weakref.finalize(self, cv2.destroyWindow, self.name)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        c = inputs.shape[1]

        with torch.no_grad():
            weight = self.weight.expand(c)
            bias = self.bias.expand(c)
            ten = F.instance_norm(inputs, None, None, weight, bias)
            image: np.ndarray = ten.clamp_(0, 255).byte().cpu().numpy()

        if self.colored:
            groups = c // 3
            image = image[:, :groups * 3, :, :]
            image = rearrange(image, 'b (g c) h w -> (b h) (g w) c', c=3)
        else:
            image = rearrange(image, 'b c h w -> (b h) (c w)')

        cv2.imshow(self.name, image)
        cv2.waitKey(1)

        return inputs
