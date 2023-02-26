from __future__ import annotations

__all__ = ['register_post_import_hook', 'when_imported']

import sys
from collections.abc import Callable
from importlib import abc, util
from threading import RLock
from typing import Any, TypeVar

_Hook = Callable[[Any], object]
_HookVar = TypeVar('_HookVar', bound=_Hook)

_INITIALIZED = False
_LOCK = RLock()
_HOOKS: dict[str, list[_Hook]] = {}


class _ImportHookChainedLoader:
    def __init__(self, loader):
        self.loader = loader

    def load_module(self, fullname):
        module = self.loader.load_module(fullname)

        with _LOCK:
            name = getattr(module, '__name__', None)
            if hooks := _HOOKS.get(name):  # type: ignore[arg-type]
                while hooks:
                    hooks.pop()(module)

        return module


class _ImportHookFinder(abc.MetaPathFinder, set):
    def find_module(self, fullname, path=None):
        with _LOCK:
            if fullname not in _HOOKS or fullname in self:
                return None

            self.add(fullname)
            try:
                if (spec := util.find_spec(fullname)) and spec.loader:
                    return _ImportHookChainedLoader(spec.loader)
                return None
            finally:
                self.remove(fullname)


def register_post_import_hook(hook: _Hook, name: str) -> None:
    """Register a new post import hook for the target module name.

    This will result in a proxy callback being registered which will defer
    loading of the specified module containing the callback function until
    required.

    Simplified version of wrapt.register_post_import_hook.
    """
    with _LOCK:
        global _INITIALIZED
        if not _INITIALIZED:
            _INITIALIZED = True
            sys.meta_path.insert(0, _ImportHookFinder())

        if (module := sys.modules.get(name)) is not None:
            hook(module)
        else:
            _HOOKS.setdefault(name, []).append(hook)


def when_imported(name: str) -> Callable[[_HookVar], _HookVar]:
    """
    Decorator for marking that a function should be called as a post
    import hook when the target module is imported.

    Simplified version of wrapt.when_imported.
    """
    def wrapper(hook: _HookVar) -> _HookVar:
        register_post_import_hook(hook, name)
        return hook

    return wrapper
