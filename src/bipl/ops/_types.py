__all__ = ['NumpyLike', 'Tile', 'Vec']

from typing import NamedTuple, Protocol

import numpy as np

Vec = tuple[int, int]


class NumpyLike(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...

    def __getitem__(self, key: slice | tuple[slice, ...]) -> np.ndarray:
        ...


class Tile(NamedTuple):
    idx: Vec
    vec: Vec
    data: np.ndarray
