__all__ = [
    'LazyBias2d', 'LazyBlurPool2d', 'LazyConv2dWs', 'LazyGroupNorm',
    'LazyLayerNorm'
]

import warnings

import torch
from torch import nn
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from .simple import Bias2d, BlurPool2d, Conv2dWs

warnings.filterwarnings('ignore', module='torch.nn.modules.lazy')


def _materialize_cls(m: nn.modules.lazy._LazyProtocol):
    # Fixes incomplete implementation of LazyModuleMixin._lazy_load_hook

    # Does the same as LazyModuleMixin._infer_parameters does,
    # except no input is needed

    # By default, if all module's parameters are loaded during load_state_dict,
    # _lazy_load_hook doen't mutate class.
    # Because of that even completely initialized modules are left as lazy,
    # and require calling forward() to trigger class mutation.

    # This function cancels this requirement.

    # FIXME: When merged to an upstream, do nothing
    assert isinstance(m, nn.modules.lazy.LazyModuleMixin)

    m._initialize_hook.remove()
    m._load_hook.remove()
    delattr(m, '_initialize_hook')
    delattr(m, '_load_hook')
    if m.cls_to_become is not None:
        m.__class__ = m.cls_to_become


class _LazyModuleMixinV2(nn.modules.lazy.LazyModuleMixin):
    def _lazy_load_hook(self: nn.modules.lazy._LazyProtocol, *args, **kwargs):
        super()._lazy_load_hook(*args, **kwargs)  # type: ignore

        if not self.has_uninitialized_params():  # type: ignore
            _materialize_cls(self)


class _LazyBase(_LazyModuleMixinV2):
    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params():  # type: ignore
            super().reset_parameters()  # type: ignore

    def initialize_parameters(self, x: torch.Tensor) -> None:
        if self.has_uninitialized_params():  # type: ignore
            self.materialize(x.shape)
            self.reset_parameters()

    def materialize(self, shape: torch.Size) -> None:
        raise NotImplementedError


class LazyLayerNorm(_LazyBase, nn.LayerNorm):
    cls_to_become = nn.LayerNorm  # type: ignore

    weight: UninitializedParameter  # type: ignore
    bias: UninitializedParameter  # type: ignore

    def __init__(self,
                 rank: int = 1,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__([0] * rank, eps, False)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = UninitializedParameter()
            self.bias = UninitializedParameter()

    def materialize(self, shape: torch.Size) -> None:
        rank = len(self.normalized_shape)
        self.normalized_shape = *shape[-rank:],
        if self.elementwise_affine:
            self.weight.materialize(self.normalized_shape)
            self.bias.materialize(self.normalized_shape)


class LazyGroupNorm(_LazyBase, nn.GroupNorm):
    cls_to_become = nn.GroupNorm  # type: ignore

    weight: UninitializedParameter  # type: ignore
    bias: UninitializedParameter  # type: ignore

    def __init__(self,
                 num_groups: int,
                 eps: float = 1e-5,
                 affine: bool = True):
        super().__init__(num_groups, 0, eps, False)
        self.affine = affine
        if self.affine:
            self.weight = UninitializedParameter()
            self.bias = UninitializedParameter()

    def materialize(self, shape: torch.Size) -> None:
        self.num_channels = shape[1]
        if self.affine:
            self.weight.materialize((self.num_channels, ))
            self.bias.materialize((self.num_channels, ))


class LazyConv2dWs(nn.LazyConv2d):
    cls_to_become = Conv2dWs  # type: ignore


class LazyBlurPool2d(_LazyBase, BlurPool2d):
    cls_to_become = BlurPool2d  # type: ignore

    weight: UninitializedBuffer

    def __init__(self,
                 kernel: int = 4,
                 stride: int = 2,
                 padding: int = 1,
                 padding_mode: str = 'reflect'):
        super().__init__(0, kernel, stride, padding, padding_mode)
        self.weight = UninitializedBuffer()

    def materialize(self, shape: torch.Size) -> None:
        self.in_channels = self.out_channels = shape[-3]
        self.weight.materialize((self.out_channels, 1, *self.kernel_size))


class LazyBias2d(_LazyBase, Bias2d):
    cls_to_become = Bias2d  # type: ignore

    bias: UninitializedParameter  # type: ignore

    def __init__(self):
        super().__init__(0, 0, 0)
        self.bias = UninitializedParameter()

    def materialize(self, shape: torch.Size) -> None:
        self.bias.materialize((1, *shape[1:]))
