"""
Driver based on cv2
- thread-safe
- compatible with single image formats: bmp/png/jpg/jpeg/webp
"""

__all__ = ['Raw']

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np

from bipl._types import Patch, Span
from bipl.ops._util import keep3d, padslice

from ._slide_bases import Driver, Image, ImageLevel
from ._util import Icc


@dataclass(frozen=True, kw_only=True)
class _Level(ImageLevel):
    data: np.ndarray
    index: int

    def numpy(self) -> np.ndarray:
        return self._postprocess(self.data)

    def part(self, *loc: Span) -> np.ndarray:
        im = padslice(self.data, *loc)
        return self._postprocess(im)

    def parts(
        self, locs: Sequence[tuple[Span, ...]], max_workers: int = 0
    ) -> Iterator[Patch]:
        data = self.numpy()
        return (Patch(loc, padslice(data, *loc)) for loc in locs)

    @property
    def icc(self) -> Icc | None:
        return None


class Raw(Driver):
    def __init__(self, path: str) -> None:
        self.path = path

    def get_mpp(self) -> float | None:
        return None

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.path})'

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Image:
        im = cv2.imread(self.path, cv2.IMREAD_ANYCOLOR)
        if im is None:
            raise ValueError(f'Cannot read {self.path!r}')

        im = keep3d(im)
        if im.ndim == im.shape[-1] == 3:  # BGR
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        return _Level(im.shape, data=im, index=index)

    def keys(self) -> list[str]:
        return []

    def get(self, key: str) -> Image:
        raise NotImplementedError

    @property
    def bbox(self) -> tuple[slice, ...]:
        return slice(None), slice(None)

    @cached_property
    def icc(self) -> Icc | None:
        return None
