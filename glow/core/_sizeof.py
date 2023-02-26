from __future__ import annotations

__all__ = ['sizeof']

import ctypes
import functools
import sys
from collections import abc
from ctypes import pythonapi
from enum import Enum
from inspect import isgetsetdescriptor, ismemberdescriptor
from numbers import Number
from types import FunctionType, ModuleType

import numpy as np

from ._import_hook import when_imported
from ._repr import si_bin

_ZERO_DEPTH_BASES = (str, bytes, bytearray, range, Number, Enum)
_SINGLETONS = (type, bool, FunctionType, ModuleType)

pythonapi._PyObject_GetDictPtr.argtypes = (ctypes.py_object, )
pythonapi._PyObject_GetDictPtr.restype = ctypes.POINTER(ctypes.py_object)


def unique_only(fn):
    """Protection from self-referencing"""
    def wrapper(obj, seen: set[int]) -> int:
        if (id_ := id(obj)) not in seen:
            seen.add(id_)
            return fn(obj, seen)
        return 0

    return functools.update_wrapper(wrapper, fn)


def true_vars(obj) -> dict[str, object] | None:
    """Get instance's vars(), even it's hidden.

    Inspired by [magic_get_dict](https://stackoverflow.com/a/45315745/9868257)

    For more details see:
    - https://utcc.utoronto.ca/~cks/space/blog/python/HowSlotsWorkI
    - https://utcc.utoronto.ca/~cks/space/blog/python/DictoffsetNotes
    """
    tp = type(obj)

    if sys.implementation.name == 'cpython':
        if (ptr := pythonapi._PyObject_GetDictPtr(obj)) and ptr.contents:
            return ptr.contents.value
        return None

        # if offset := tp.__dictoffset__:
        #     if offset < 0:
        #         offset += tp.__sizeof__(obj)  # type: ignore
        #     addr = id(obj) + offset
        #     ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.py_object))
        #     return ptr.contents.value
        # return None

    if hasattr(obj, '__dict__'):
        for cls in tp.__mro__:
            if (attr := vars(cls).get('__dict__')) is not None:
                return (vars(obj) if (isgetsetdescriptor(attr)
                                      or ismemberdescriptor(attr)) else None)
    return None


@functools.singledispatch
@unique_only
def _sizeof(obj, seen: set[int]) -> int:
    if obj is None:
        return 0

    tp = type(obj)  # Real type, may differ from obj.__class__
    if issubclass(tp, _SINGLETONS):
        return 0

    size = sys.getsizeof(obj)
    if issubclass(tp, _ZERO_DEPTH_BASES):
        return size

    size += _sizeof(true_vars(obj), seen)

    if hasattr(obj, '__wrapped__'):  # Unwrap
        size += _sizeof(obj.__wrapped__, seen)
        return size

    if issubclass(tp, dict):
        size += sum(_sizeof(k, seen) for k in obj.keys())
        size += sum(_sizeof(v, seen) for v in obj.values())
    elif issubclass(tp, abc.Collection):
        size += sum(_sizeof(item, seen) for item in obj)

    if hasattr(obj, '__slots__'):
        size += sum(
            _sizeof(getattr(obj, slot, None), seen)
            for cls in tp.__mro__ if (slots := getattr(cls, '__slots__', ()))
            for slot in ([slots] if isinstance(slots, str) else slots))

    return size


@unique_only
def _sizeof_numpy(obj, seen: set[int]) -> int:
    return _sizeof(obj.base, seen) + sys.getsizeof(obj)


@unique_only
def _sizeof_torch(obj, _: set[int]) -> int:
    size = sys.getsizeof(obj)
    if not obj.is_cuda:
        size += obj.numel() * obj.element_size()
    return size  # TODO: fix if useless when grads are attached


_sizeof.register(np.ndarray, _sizeof_numpy)
when_imported('torch')(
    lambda torch: _sizeof.register(torch.Tensor, _sizeof_torch))


def sizeof(obj) -> int:
    """Computes size of object, no matter how complex it is.

    Inspired by
    [PySize](https://github.com/bosswissam/pysize/blob/master/pysize.py)
    """
    return si_bin(_sizeof(obj, set()))
