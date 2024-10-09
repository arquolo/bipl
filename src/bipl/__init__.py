from importlib import import_module
from typing import TYPE_CHECKING

from ._env import env
from .io import Dzi
from .ops import Mosaic

_exports = {
    '.io': ['Slide'],
}
_submodule_by_name = {
    name: modname for modname, names in _exports.items() for name in names
}

if TYPE_CHECKING:
    from .io import Slide
else:

    def __getattr__(name: str):
        if modname := _submodule_by_name.get(name):
            mod = import_module(modname, __package__)
            globals()[name] = obj = getattr(mod, name)
            return obj
        raise AttributeError(f'No attribute {name}')

    def __dir__():
        return __all__


__all__ = ['Dzi', 'Mosaic', 'Slide', 'env']
