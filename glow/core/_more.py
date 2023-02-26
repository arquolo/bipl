from __future__ import annotations

__all__ = ['as_iter', 'chunked', 'eat', 'ichunked', 'roundrobin', 'windowed']

import threading
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from functools import partial
from itertools import chain, cycle, islice, repeat
from typing import Protocol, TypeVar, overload

_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)


class SupportsSlice(Sized, Protocol[_T_co]):
    def __getitem__(self, __s: slice) -> _T_co:
        ...


# ----------------------------------------------------------------------------


def as_iter(obj: Iterable[_T] | _T, limit: int | None = None) -> Iterator[_T]:
    """Make iterator with at most `limit` items"""
    if isinstance(obj, Iterable):
        return islice(obj, limit)
    return repeat(obj) if limit is None else repeat(obj, limit)


# ----------------------------------------------------------------------------


def _dispatch(fallback_fn, fn, it, *args):
    if (not isinstance(it, Sized) or not hasattr(it, '__getitem__')
            or isinstance(it, Mapping)):
        return fallback_fn(it, *args)

    r = fn(it, *args)
    if isinstance(it, Sequence):
        return r

    try:
        # Ensure that slice is supported by prefetching 1st item
        first_or_none = *islice(r, 1),
        return chain(first_or_none, r)

    except TypeError:
        return fallback_fn(it, *args)


# ----------------------------------------------------------------------------


def window_hint(it, size):
    return len(it) + 1 - size


def chunk_hint(it, size):
    return len(range(0, len(it), size))


def _sliced_windowed(s: SupportsSlice[_T], size: int) -> Iterator[_T]:
    indices = range(len(s) + 1)
    slices = map(slice, indices[:-size], indices[size:])
    return map(s.__getitem__, slices)


def _windowed(it: Iterable[_T], size: int) -> Iterator[tuple[_T, ...]]:
    if size == 1:  # Trivial case
        return zip(it)

    it = iter(it)
    w = deque(islice(it, size), maxlen=size)

    if len(w) != size:
        return iter(())
    return map(tuple, chain([w], map(w.__iadd__, zip(it))))


def _sliced(s: SupportsSlice[_T], size: int) -> Iterator[_T]:
    indices = range(len(s) + size)
    slices = map(slice, indices[::size], indices[size::size])
    return map(s.__getitem__, slices)


def _chunked(it: Iterable[_T], size: int) -> Iterator[tuple[_T, ...]]:
    if size == 1:  # Trivial case
        return zip(it)

    fetch_chunk = partial(islice, iter(it), size)
    chunks = iter(fetch_chunk, None)
    return iter(map(tuple, chunks).__next__, ())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------


@overload
def windowed(it: SupportsSlice[_T], size: int) -> Iterator[_T]:
    ...


@overload
def windowed(it: Iterable[_T], size: int) -> Iterator[tuple[_T, ...]]:
    ...


def windowed(it, size):
    """Retrieve overlapped windows from iterable.
    Tries to use slicing if possible.

    >>> [*windowed(range(6), 3)]
    [range(0, 3), range(1, 4), range(2, 5), range(3, 6)]

    >>> [*windowed(iter(range(6)), 3)]
    [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)]
    """
    return _dispatch(_windowed, _sliced_windowed, it, size)


@overload
def chunked(__it: SupportsSlice[_T], size: int) -> Iterator[_T]:
    ...


@overload
def chunked(__it: Iterable[_T], size: int) -> Iterator[tuple[_T, ...]]:
    ...


def chunked(it, size):
    """
    Splits iterable to chunks of at most size items each.
    Uses slicing if possible.
    Each next() on result will advance passed iterable to size items.

    >>> [*chunked(range(10), 3)]
    [range(0, 3), range(3, 6), range(6, 9), range(9, 10)]

    >>> [*chunked(iter(range(10)), 3)]
    [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]
    """
    return _dispatch(_chunked, _sliced, it, size)


# ----------------------------------------------------------------------------


def _deiter(q: deque[_T]) -> Iterator[_T]:
    # Same as iter_except(q.popleft, IndexError) from docs of itertools
    try:
        while True:
            yield q.popleft()
    except IndexError:
        return


def ichunked(it: Iterable[_T], size: int) -> Iterator[Iterator[_T]]:
    """Split iterable to chunks of at most size items each.

    Does't consume items from passed iterable to return complete chunk
    unlike chunked, as yields iterators, not sequences.

    >>> s = ichunked(range(10), 3)
    >>> len(s)
    4
    >>> [[*chunk] for chunk in s]
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    if size == 1:  # Trivial case
        yield from map(iter, zip(it))  # type: ignore[arg-type]
        return

    it = iter(it)
    while head := deque(islice(it, 1)):
        # Remaining chunk items
        body = islice(it, size - 1)

        # Cache for not-yet-consumed
        tail = deque[_T]()

        # Include early fetched item into chunk
        yield chain(_deiter(head), body, _deiter(tail))

        # Advance and fill internal cache, expand tail with items from body
        tail += body


# ----------------------------------------------------------------------------


def eat(iterable: Iterable, daemon: bool = False) -> None:
    """Consume iterable, daemonize if needed (move to background thread)"""
    if daemon:
        threading.Thread(target=deque, args=(iterable, 0), daemon=True).start()
    else:
        deque(iterable, 0)


def roundrobin(*iterables: Iterable[_T]) -> Iterator[_T]:
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    iters = cycle(iter(it) for it in iterables)
    for pending in range(len(iterables) - 1, -1, -1):
        yield from map(next, iters)
        iters = cycle(islice(iters, pending))
