from importlib import import_module
from typing import TYPE_CHECKING

from ._dzi import Dzi

_exports = {
    '._slide': ['Slide'],
}
_submodule_by_name = {
    name: modname for modname, names in _exports.items() for name in names
}

if TYPE_CHECKING:
    from ._slide import Slide
else:

    def __getattr__(name: str):
        if modname := _submodule_by_name.get(name):
            mod = import_module(modname, __package__)
            globals()[name] = obj = getattr(mod, name)
            return obj
        raise AttributeError(f'No attribute {name}')

    def __dir__():
        return __all__


__all__ = ['Dzi', 'Slide']
