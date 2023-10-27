__all__ = ['Mosaic']

import dataclasses
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from itertools import chain
from math import ceil
from typing import TypeVar, cast

import cv2
import numpy as np
from glow import chunked, map_n

from ._tile import BlendCropper, Cropper, Decimator, Reconstructor, Zipper
from ._types import NumpyLike, Tile, Vec
from ._util import get_trapz, normalize_loc

# TODO: allow result of .map/.map_batched to have different tile and step
# NOTE: all classes here are stateless & their iterators are reentrant.

# ---------------------------- utility functions -----------------------------


def _apply(fn: Callable[[np.ndarray], np.ndarray], obj: Tile) -> Tile:
    r = fn(obj.data)
    assert r.shape[:2] == obj.data.shape[:2], \
        'Tile shape alteration (besides channel count) is forbidden'
    return obj._replace(data=r)


def _apply_batched(fn: Callable[[list[np.ndarray]], Iterable[np.ndarray]],
                   ts: tuple[Tile, ...]) -> list[Tile]:
    *rs, = fn([t.data for t in ts])
    assert len(rs) == len(ts)
    assert all(r.shape[:2] == t.data.shape[:2]
               for r, t in zip(rs, ts)), \
        'Tile shape alteration (besides channel count) is forbidden'
    return [t._replace(data=r) for t, r in zip(ts, rs)]


def _reweight(weight: np.ndarray, tile: np.ndarray) -> np.ndarray:
    assert tile.dtype.kind == 'f'
    return np.einsum('hwc,h,w -> hwc', tile, weight, weight, optimize=True)


def _padslice(a: NumpyLike, *loc: slice) -> np.ndarray:
    loc = normalize_loc(loc, a.shape)

    pos_loc = *(slice(max(s.start, 0), s.stop) for s in loc),
    a = a[pos_loc]

    pad = [(pos.start - raw.start, pos.stop - pos.start - size)
           for raw, pos, size in zip(loc, pos_loc, a.shape)]
    return np.pad(a, pad) if np.any(pad) else a


# ------------------------------ mosaic setting ------------------------------


@dataclass(frozen=True, slots=True)
class Mosaic:
    """
    Helper to split image to tiles and process them.

    Parameters:
    - step - Step between consecutive tiles
    - overlap - Count of pixels that will be shared among overlapping tiles

    So tile size = overlap + non-overlapped area + overlap = step + overlap,
    and non-overlapped area = step - overlap.
    """
    step: int
    overlap: int

    def __post_init__(self):
        if self.overlap < 0:
            raise ValueError('overlap should be non-negative, '
                             f'got: {self.overlap}')
        # That may be optional
        if self.overlap % 2 != 0:
            raise ValueError(f'overlap should be even, got: {self.overlap}')
        if self.overlap > self.step:
            raise ValueError('overlap should be lower than step, got: '
                             f'overlap={self.overlap} and step={self.step}')

    def get_kernel(self) -> np.ndarray:
        return get_trapz(self.step, self.overlap)

    def iterate(self,
                image: NumpyLike,
                max_workers: int = 1) -> '_TiledArrayView':
        """Read tiles from input image"""
        shape = image.shape[:2]
        ishape = *(len(range(0, s + self.overlap, self.step)) for s in shape),
        cells = np.ones(ishape, dtype=np.bool_)

        return _TiledArrayView(self, shape, cells, image, max_workers)


# --------------------------------- actions ----------------------------------

_Self = TypeVar('_Self', bound='_BaseView')


@dataclass(frozen=True)
class _BaseView:
    m: Mosaic
    shape: tuple[int, ...]
    cells: np.ndarray

    @property
    def ishape(self) -> tuple[int, ...]:
        return self.cells.shape

    def __len__(self) -> int:
        return int(self.cells.sum())

    def report(self) -> dict[str, str]:
        """Cells and area usage"""
        used = int(self.cells.sum())
        total = self.cells.size
        coverage = (used / total) * (1 + self.m.overlap / self.m.step) ** 2
        return {'cells': f'{used}/{total}', 'coverage': f'{coverage:.0%}'}

    def __iter__(self) -> Iterator[Tile]:
        raise NotImplementedError

    def map(self: _Self,
            fn: Callable[[np.ndarray], np.ndarray],
            /,
            max_workers: int = 0) -> _Self:
        """
        Applies function to each tile.
        Note: change of tile shape besides channel count is forbidden.
        Each tile is HWC-ordered ndarray.
        Supports threading.
        """
        tile_fn = partial(_apply, fn)
        tiles = map_n(tile_fn, self, max_workers=max_workers)
        return cast(
            _Self,  # don't narrow type
            _IterView(self.m, self.shape, self.cells, tiles),
        )

    def pool(self: _Self, stride: int = 1) -> _Self:
        """Resizes each tile to desired stride"""
        if stride == 1:
            return self
        if self.m.step % stride != 0 or self.m.overlap % stride != 0:
            raise ValueError('stride should be a divisor of '
                             f'{self.m.step}, {self.m.overlap}. Got: {stride}')

        m = Mosaic(self.m.step // stride, self.m.overlap // stride)
        shape = *(len(range(0, s, stride)) for s in self.shape),
        return cast(  # don't narrow type
            _Self,
            _DecimatedView(m, shape, self.cells, stride, self),
        )

    def crop(self) -> '_BaseView':
        it = map(Cropper(self.shape), self)
        return _IterView(self.m, self.shape, self.cells, it)

    def zip_with(self, view: np.ndarray, v_scale: float) -> '_ZipView':
        """Extracts tiles from `view` simultaneously with tiles from self"""
        if v_scale > 1:
            raise ValueError('v_scale should be less than 1, '
                             f'got: {v_scale}')
        return _ZipView(self, view, v_scale)

    def with_cm(self: _Self, ctx: AbstractContextManager) -> _Self:
        return cast(
            _Self,
            _ScopedView(self.m, self.shape, self.cells, self, ctx),
        )


@dataclass(frozen=True)
class _ZipView:
    source: _BaseView
    view: NumpyLike
    v_scale: float

    def __len__(self) -> int:
        return len(self.source)

    def __iter__(self) -> Iterator[tuple[Vec, Vec, np.ndarray, np.ndarray]]:
        assert self.v_scale >= 1
        return map(Zipper(self.view, self.v_scale), self.source)


@dataclass(frozen=True)
class _View(_BaseView):
    def map_batched(self,
                    fn: Callable[[list[np.ndarray]], Iterable[np.ndarray]],
                    /,
                    batch_size: int = 1,
                    max_workers: int = 0) -> '_View':
        """
        Applies function to batches of tiles.
        Note: change of tile shape besides channel count is forbidden.
        Each tile is HWC-ordered ndarray.
        Supports threading.
        """
        tile_fn = partial(_apply_batched, fn)

        chunks = chunked(self, batch_size)
        batches = map_n(tile_fn, chunks, max_workers=max_workers)
        tiles = chain.from_iterable(batches)

        return _IterView(self.m, self.shape, self.cells, tiles)

    def transform(self, fn: Callable[[Iterable[Tile]],
                                     Iterable[Tile]]) -> '_View':
        """
        TODO: fill docs
        """
        tiles = fn(self)
        return _IterView(self.m, self.shape, self.cells, tiles)

    def reweight(self) -> '_View':
        """
        Applies weight to tile edges to prepare them for summation
        if overlap exists.
        Note: no need to call this method if you have already applied it.
        """
        if not self.m.overlap:  # No need
            return self

        weight = self.m.get_kernel()
        tile_fn = partial(_reweight, weight)
        return self.map(tile_fn)

    def merge(self) -> _BaseView:
        """
        Removes overlapping regions from all the tiles if any.
        Tiles should be reweighted if overlap exists.
        """
        if self.m.overlap:
            return _BlendCropsView(self.m, self.shape, self.cells, self)
        return self


@dataclass(frozen=True)
class _IterView(_View):
    source: Iterable[Tile]

    def __iter__(self) -> Iterator[Tile]:
        return iter(self.source)


@dataclass(frozen=True)
class _ScopedView(_BaseView):
    source: Iterable[Tile]
    _ctx: AbstractContextManager

    def __iter__(self) -> Iterator[Tile]:
        """Iterator over tiles. Reentrant if nested context is reentrant"""
        with self._ctx:
            yield from self.source


@dataclass(frozen=True)
class _DecimatedView(_View):
    """
    Decimates tiles.
    Doesn't change size uniformity.
    Yields decimated views of original tiles.
    """
    stride: int
    source: Iterable[Tile]

    def __iter__(self) -> Iterator[Tile]:
        return map(Decimator(self.stride), self.source)


@dataclass(frozen=True)
class _TiledArrayView(_View):
    """
    Extracts tiles from array.
    Yields same-sized tiles with overlaps.
    """
    data: NumpyLike
    max_workers: int

    def select(self,
               mask: np.ndarray,
               scale: float | None = None) -> '_TiledArrayView':
        """
        Subset non-masked (i.e. non-zeros in mask) tiles for iteration.
        Mask size is "scale"-multiple of source image.
        """
        if mask.ndim == 3:  # Strip extra dim
            mask = mask.squeeze(2)
        if mask.ndim != 2:
            raise ValueError(f'Mask should be 2D, got shape: {mask.shape}')
        if scale is None:
            scale = max(ms / s for ms, s in zip(mask.shape, self.data.shape))

        mask = np.where(mask, np.uint8(255), np.uint8(0))

        tile = (self.m.step + self.m.overlap) * scale
        if (h_tile := ceil(tile / 2)):
            kernel = np.ones((3, 3), dtype='u1')
            mask = cv2.dilate(mask, kernel, iterations=h_tile)

        ih, iw = self.ishape
        cells = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_AREA)

        return dataclasses.replace(self, cells=cells)

    def _slice_tile(self, iy: int, ix: int) -> Tile:
        """Slice source image to get overlapping tiles"""
        (y0, y1), (x0, x1) = ((
            self.m.step * i - self.m.overlap,
            self.m.step * (i + 1),
        ) for i in (iy, ix))

        data = _padslice(self.data, slice(y0, y1), slice(x0, x1))
        return Tile(idx=(iy, ix), vec=(y0, x0), data=data)

    def _get_tile(self, iy: int, ix: int) -> Tile:
        """Read non-overlapping tile of source image"""
        # First tile is [-overlap: step], as `step + overlap = tile`
        (y0, y1), (x0, x1) = ((
            self.m.step * i - self.m.overlap,
            self.m.step * (i + 1),
        ) for i in (iy, ix))

        # We have tile above to pick upper edge from
        if iy and self.cells[iy - 1, ix]:
            y0 += self.m.overlap
        # We have previous tile to pick left edge from
        if ix and self.cells[iy, ix - 1]:
            x0 += self.m.overlap

        return Tile(
            idx=(iy, ix),
            vec=(y0, x0),
            data=self.data[y0:y1, x0:x1],
        )

    def __iter__(self) -> Iterator[Tile]:
        """
        Yield complete tiles built from source image.
        Each tile will have size `(step + overlap)`
        """
        ys, xs = np.where(self.cells)
        if isinstance(self.data, np.ndarray):
            return map(self._slice_tile, ys, xs)

        parts = map_n(self._get_tile, ys, xs, max_workers=self.max_workers)
        if not self.m.overlap:
            return iter(parts)

        rcr = Reconstructor(self.m.overlap, self.m.step, self.cells)
        return map(rcr, parts)

    def get_tile(self, idx: int) -> Tile:
        iy, ix = np.argwhere(self.cells)[idx]
        return self._slice_tile(iy, ix)


@dataclass(frozen=True)
class _BlendCropsView(_BaseView):
    """
    Applies weighted average over overlapping regions.
    Yields tiles without overlaps, so their size can differ.
    """
    source: Iterable[Tile]

    def __iter__(self) -> Iterator[Tile]:
        assert self.m.overlap
        bcr = BlendCropper(self.m.step, self.m.overlap, self.cells)
        return map(bcr, self.source)
