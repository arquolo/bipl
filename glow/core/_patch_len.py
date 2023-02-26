"""Make builtin iterators compatible with `len`"""
__all__ = ['apply']

import builtins
import functools
import math
import operator
from collections.abc import Iterable, Iterator, Sized
from dataclasses import dataclass
from itertools import (accumulate, count, cycle, islice, product, repeat,
                       starmap, tee, zip_longest)
from typing import Generic, Protocol, TypeVar, runtime_checkable

# -------------------------- iterable-size proxies ---------------------------

_T_co = TypeVar('_T_co', covariant=True)
_INF = float('inf')


@runtime_checkable
class SizedIterable(Sized, Iterable[_T_co], Protocol[_T_co]):
    ...


@dataclass(repr=False, frozen=True)
class _SizedIterable(Generic[_T_co]):
    _it: Iterable[_T_co]
    _size: int

    def __iter__(self) -> Iterator[_T_co]:
        return iter(self._it)

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        line = object.__repr__(self._it)
        line = line.removeprefix('<').removesuffix('>')
        return f'<{line} with {self._size} items>'


# --------------------------------- builtins ---------------------------------

len_hint = functools.singledispatch(builtins.len)

_tee: type = tee(())[0].__class__
_iterables: list[Iterable] = [
    '', b'',
    range(0), (), [], {}, {}.keys(), {}.values(), {}.items(),
    reversed(()),
    reversed([]),
    set(),
    frozenset()
]
_transparent_types: tuple[type, ...] = tuple(
    it.__iter__().__class__ for it in _iterables)
for _tp in _transparent_types:
    len_hint.register(_tp, operator.length_hint)


def _are_definitely_independent(iters):
    return (len({id(it) for it in iters}) == len(iters)
            and all(isinstance(it, _transparent_types) for it in iters))


@len_hint.register(zip)
def _len_zip(x):  # type: ignore[misc]
    _, iters = x.__reduce__()
    if not iters:
        return 0
    if len(iters) == 1:
        return len(iters[0])

    # Do not compute zip size when it's constructed from multiple iterables.
    # as there's currently no reliable way to check whether underlying
    # iterables are independent or not
    if _are_definitely_independent(iters):
        return min(map(len, iters))

    raise TypeError


@len_hint.register(map)
def _len_map(x):  # type: ignore[misc]
    _, (__fn, *iters) = x.__reduce__()
    if len(iters) == 1:
        return len(iters[0])

    # Same as for zip above
    if _are_definitely_independent(iters):
        return min(map(len, iters))

    raise TypeError


# --------------------------- itertools.infinite ---------------------------


@len_hint.register(count)
def _len_count(_):  # type: ignore[misc]
    return _INF


@len_hint.register(cycle)
def _len_cycle(x):  # type: ignore[misc]
    _, [iterable], (buf, pos) = x.__reduce__()
    if buf or len(iterable):
        return _INF
    return 0


@len_hint.register(repeat)
def _len_repeat(x):  # type: ignore[misc]
    _, (obj, *left) = x.__reduce__()
    return left[0] if left else _INF


# ---------------------------- itertools.finite ----------------------------


@len_hint.register(accumulate)
def _len_accumulate(x):  # type: ignore[misc]
    _, (seq, _fn), _total = x.__reduce__()
    return len(seq)


# @len_hint.register(chain)


@len_hint.register(islice)
def _len_islice(x):  # type: ignore[misc]
    _, (it, start, *stop_step), done = x.__reduce__()
    if not stop_step:
        return 0
    stop, step = stop_step
    total = len(it) + done
    stop = total if stop is None else min(total, stop)
    if math.isinf(stop):
        # range can't handle inf
        return _INF
    return len(range(start, stop, step))


@len_hint.register(starmap)
def _len_starmap(x):  # type: ignore[misc]
    _, (_fn, seq) = x.__reduce__()
    return len(seq)


@len_hint.register(_tee)
def _len_tee(x):
    _, [empty_tuple], (dataobject, pos) = x.__reduce__()
    _, (src, buf, none) = dataobject.__reduce__()
    return len(src) + len(buf) - pos


@len_hint.register(zip_longest)
def _len_zip_longest(x):  # type: ignore[misc]
    _, iters, _pad = x.__reduce__()
    if _are_definitely_independent(iters):
        return max(map(len, iters))
    raise TypeError


# -------------------------- itertools.combinatoric -------------------------


@len_hint.register(product)
def _len_product(x):  # type: ignore[misc]
    _, seqs, *pos = x.__reduce__()

    # Greedy caches all input iterables, no need to check interference
    lens = *map(len, seqs),
    total = functools.reduce(operator.mul, lens, 1)
    if not pos:
        return total

    strides = accumulate((1, ) + lens[:0:-1], operator.mul)
    offset = sum(map(operator.mul, strides, reversed(pos[0])))
    return total - offset - 1


# @len_hint.register(permutations)
# @len_hint.register(combinations)
# @len_hint.register(combinations_with_replacement)


def apply():
    builtins.len = len_hint
