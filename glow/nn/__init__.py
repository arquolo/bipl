from importlib import import_module
from typing import TYPE_CHECKING

from ._loader import get_loader
from ._trainer import Trainer
from .amp import get_grads
from .modules import (Ensemble, Noise, Residual, SplitAttention,
                      SqueezeExcitation, Upscale2d)
from .optimizers import SGDW, AdamW, RAdam
from .util import (detach_, device, dump_to_onnx, eval_, frozen, inference,
                   param_count, profile)

__all__ = [
    'AdamW', 'Ensemble', 'Noise', 'RAdam', 'Residual', 'SGDW',
    'SplitAttention', 'SqueezeExcitation', 'Trainer', 'Upscale2d', 'detach_',
    'device', 'dump_to_onnx', 'eval_', 'frozen', 'get_gpu_capability',
    'get_gpu_memory_info', 'get_grads', 'get_loader', 'inference',
    'param_count', 'plot_model', 'profile'
]

_exports = {
    '.plot': ['plot_model'],
    '.driver': ['get_gpu_capability', 'get_gpu_memory_info'],
}
_submodule_by_name = {
    name: modname for modname, names in _exports.items() for name in names
}

if TYPE_CHECKING:
    from .driver import get_gpu_capability, get_gpu_memory_info
    from .plot import plot_model

else:

    def __getattr__(name: str):
        if modname := _submodule_by_name.get(name):
            mod = import_module(modname, __package__)
            globals()[name] = obj = getattr(mod, name)
            return obj
        raise AttributeError(f'No attribute {name}')

    def __dir__():
        return __all__
