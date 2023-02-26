__all__ = ['env']

from collections import ChainMap
from collections.abc import Iterator
from contextlib import contextmanager

from ..core import repr_as_obj


class _Env(ChainMap):
    """Environment with scopes.
    The outermost scope takes highest priority.

    As example:

        >>> print(env)  # Empty in global scope
        _Env()
        >>> with env(a=1):  # Overrides `a` from inner scope
        ...     print(env)
        ...     with env(a=2, b=3):  # Set defaults for `a` and `b`
        ...         print(env)
        ...     print(env)
        _Env(a=1)
        _Env(a=1, b=3)
        _Env(a=1)
        >>> print(env)  # Reset to initial state
        _Env()

    """
    @contextmanager
    def __call__(self, **items) -> Iterator[None]:
        self.maps.append(items)
        try:
            yield
        finally:
            self.maps.pop()

    def __repr__(self) -> str:
        return f'{type(self).__name__}({repr_as_obj({**self})})'


env = _Env()
