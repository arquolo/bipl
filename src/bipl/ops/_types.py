__all__ = ['NumpyLike', 'Tile', 'Vec']

from collections.abc import Sequence
from typing import NamedTuple, Protocol

import numpy as np

Vec = tuple[int, ...]


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
    idx: Vec
    vec: Vec
    data: np.ndarray
