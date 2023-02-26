__all__ = ['make_key']

from collections.abc import Hashable

_KWD_MARK = object()


class _HashedSeq(list):  # List is mutable, that's why not NamedTuple
    __slots__ = 'hashvalue',

    def __init__(self, tup: tuple):
        self[:] = tup
        self.hashvalue = hash(tup)  # Memorize hash

    def __hash__(self):
        return self.hashvalue


def make_key(*args, **kwargs) -> Hashable:
    """Copied from functools._make_key, as private function"""
    if len(args) == 1 and isinstance(args[0], (int, str)):
        return args[0]
    if kwargs:
        args = sum(kwargs.items(), (*args, _KWD_MARK))
    return _HashedSeq(args)
