__all__ = [
    'HasPartsAbc',
    'NDIndex',
    'NumpyLike',
    'Patch',
    'Shape',
    'Span',
    'Tile',
    'Vec',
]

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import NamedTuple, Protocol, TypeAlias

import numpy as np

NDIndex: TypeAlias = tuple[int, ...]  # Index vector
Shape: TypeAlias = tuple[int, ...]  # N-dim shape
Span: TypeAlias = tuple[int, int]  # start/stop for `slice()`
Vec: TypeAlias = tuple[int, ...]  # N-dim radius-vector to some point


class NumpyLike(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...

    @property
    def dtype(self) -> np.dtype: ...

    def __getitem__(self, key: slice | tuple[slice, ...], /) -> np.ndarray: ...


class Tile(NamedTuple):
    idx: NDIndex
    vec: Vec
    data: np.ndarray

    def shape(self) -> Shape:
        return self.data.shape


class Patch(NamedTuple):
    loc: tuple[Span, ...]
    data: np.ndarray


class HasPartsAbc(ABC):
    @abstractmethod
    def parts(
        self, locs: Sequence[tuple[Span, ...]], max_workers: int = 0
    ) -> Iterator[Patch]: ...
