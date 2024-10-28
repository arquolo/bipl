__all__ = [
    'HasParts',
    'NDIndex',
    'NumpyLike',
    'Patch',
    'Shape',
    'Span',
    'Tile',
    'Vec',
]

from collections.abc import Iterator, Sequence
from typing import NamedTuple, Protocol

import numpy as np
from glow import map_n

type NDIndex = tuple[int, ...]  # Index vector
type Shape = tuple[int, ...]  # N-dim shape
type Span = tuple[int, int]  # start/stop for `slice()`
type Vec = tuple[int, ...]  # N-dim radius-vector to some point


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


class Patch(NamedTuple):
    loc: tuple[Span, ...]
    data: np.ndarray


class HasParts:
    def part(self, *loc: Span) -> np.ndarray:
        """Reads crop of LOD. Overridable"""
        raise NotImplementedError

    def parts(self,
              locs: Sequence[tuple[Span, ...]],
              max_workers: int = 0) -> Iterator[Patch]:
        return map_n(
            lambda loc: Patch(loc, self.part(*loc)),
            locs,
            max_workers=max_workers,
        )
