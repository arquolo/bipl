__all__ = ['NDIndex', 'NumpyLike', 'Shape', 'Span', 'Tile', 'Vec']

from collections.abc import Sequence
from typing import NamedTuple, Protocol

import numpy as np

NDIndex = tuple[int, ...]  # Index vector
Shape: tuple[int, ...]  # N-dim shape
Span = tuple[int, int]  # start/stop for `slice()`
Vec = tuple[int, ...]  # N-dim radius-vector to some point


class NumpyLike(Protocol):
    @property
    def shape(self) -> Sequence[int]:
        ...

    @property
    def dtype(self) -> np.dtype:
        ...

    def __getitem__(self, key: slice | tuple[slice, ...]) -> np.ndarray:
        ...


class Tile(NamedTuple):
    idx: NDIndex
    vec: Vec
    data: np.ndarray
