__all__ = ['Mosaic']

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from itertools import chain, starmap
from math import ceil
from typing import Literal, TypeVar, cast

import cv2
import numpy as np
import numpy.typing as npt
from glow import chunked, map_n, starmap_n

from ._tile import BlendCropper, Cropper, Decimator, Reconstructor, Zipper
from ._types import NumpyLike, Tile, Vec
from ._util import get_trapz, padslice

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

    @property
    def tile(self) -> int:
        return self.step + self.overlap

    def get_kernel(self) -> np.ndarray:
        return get_trapz(self.step, self.overlap)

    def iterate(self,
                image: NumpyLike,
                max_workers: int = 1) -> '_TiledArrayView':
        """Read tiles from input image"""
        # Source shape
        shape = image.shape[:2]

        # Index
        ih, iw = ishape = *((s + self.tile - 1) // self.step for s in shape),
        cells = np.ones(ishape, dtype=np.bool_)

        # Align slide & cells centers
        view = *(round(i * self.step + self.overlap) for i in ishape),
        origin = *((s - v) // 2 for s, v in zip(shape, view)),

        # All of (ih iw YX)
        # First tile is [origin: origin + tile]
        iyx = np.mgrid[:ih, :iw].transpose(1, 2, 0)
        yx0 = self.step * iyx + origin
        yx1 = self.tile + yx0
        locs = np.stack([yx0, yx1], -1)  # (ih iw YX lo-hi)

        # TODO: read row by row - process rectangular tiles
        # TODO: merge consecutive tiles from single row
        return _TiledArrayView(self, shape, cells, locs, image, max_workers)


# --------------------------------- actions ----------------------------------

_Self = TypeVar('_Self', bound='_BaseView')


@dataclass(frozen=True)
class _BaseView:
    m: Mosaic
    shape: Sequence[int]
    cells: np.ndarray

    @property
    def ishape(self) -> Sequence[int]:
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
        shape = *((s + stride - 1) // stride for s in self.shape),
        return cast(  # don't narrow type
            _Self,
            _DecimatedView(m, shape, self.cells, stride, self),
        )

    def crop(self) -> '_BaseView':
        it = map(Cropper(self.shape), self)
        return _IterView(self.m, self.shape, self.cells, it)

    def zip_with(
        self,
        view: NumpyLike,
        v_scale: float,
        interpolation: Literal[0, 1, 2, 3, 4] = 0,
    ) -> '_ZipView':
        """
        For each tile read & resample matching region from `view`.
        Similar to RoIAlign used in some detectors.

        Interpolation is Nearest (0) by default, but Linear (1), Bicubic (2),
        Area (3) & Lanczos-4 (4) are also supported (OpenCV codes).
        """
        if v_scale <= 0:
            raise ValueError(f'v_scale should be positive, got: {v_scale}')
        return _ZipView(self, view, v_scale, interpolation)

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
    interpolation: int

    def __len__(self) -> int:
        return len(self.source)

    def __iter__(self) -> Iterator[tuple[Vec, Vec, np.ndarray, np.ndarray]]:
        return map(
            Zipper(self.view, self.v_scale, self.interpolation),
            self.source,
        )


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
    locs: npt.NDArray[np.int64]  # (ih iw YX lo-hi)
    data: NumpyLike
    max_workers: int

    def select(self,
               mask: np.ndarray,
               scale: float | None = None) -> '_TiledArrayView':
        """
        Subset non-masked (i.e. non-zeros in mask) tiles for iteration.
        Mask size is "scale"-multiple of source image.

        NOTE: `select(mask1).select(mask2)` is equal to `select(mask1 & mask2)`
        """
        if mask.ndim == 3:  # Strip extra dim
            mask = mask.squeeze(2)
        if mask.ndim != 2:
            raise ValueError(f'Mask should be 2D, got shape: {mask.shape}')
        if scale is None:
            scale = max(ms / s for ms, s in zip(mask.shape, self.data.shape))

        mask = np.where(mask, np.uint8(255), np.uint8(0))

        if half_tile := ceil(self.m.tile * scale / 2):
            kernel = np.ones((3, 3), dtype='u1')
            mask = cv2.dilate(mask, kernel, iterations=half_tile)

        # Align mask & cells centers
        ih, iw = ishape = self.cells.shape
        view = *(round(i * self.m.step * scale) for i in ishape),
        loc = *(slice((s0 - s1) // 2, (s0 - s1) // 2 + s1)
                for s0, s1 in zip(mask.shape, view)),
        mask = padslice(mask, *loc)

        cells = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_AREA)

        return dataclasses.replace(self, cells=self.cells & cells.astype(bool))

    def _get_tile(self, idx: Vec, *loc: slice) -> Tile:
        """Do `data[loc]`. Result could be cropped."""
        vec = *(s.start for s in loc),
        return Tile(idx=idx, vec=vec, data=self.data[loc])

    def _get_tile_padded(self, idx: Vec, *loc: slice) -> Tile:
        """Do `data[loc]` padding data in necessary."""
        vec = *(s.start for s in loc),
        return Tile(idx=idx, vec=vec, data=padslice(self.data, *loc))

    def _ilocs(self,
               locs: npt.NDArray[np.int64]) -> list[tuple[Vec, slice, slice]]:
        """Precompute tile coordinates: index & Y/X-slices"""
        idx = np.argwhere(self.cells).tolist()
        boxes = locs[self.cells].tolist()
        return [(tuple(i), slice(*ys), slice(*xs))
                for i, (ys, xs) in zip(idx, boxes)]

    def _drop_overlaps(self,
                       grid: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        grid = grid.copy()  # (ih iw YX lo-hi)

        # Cut top edge (y0) from tiles having neighbor from above (iy)
        grid[1:, :, 0, 0][self.cells[:-1, :]] += self.m.overlap

        # Cut left edge (x0) from tiles having neighbor on the left (ix)
        grid[:, 1:, 1, 0][self.cells[:, :-1]] += self.m.overlap
        return grid

    def __iter__(self) -> Iterator[Tile]:
        """
        Yield complete tiles built from source image.
        Each tile will have size `(step + overlap)`
        """
        if isinstance(self.data, np.ndarray):
            ilocs = self._ilocs(self.locs)
            return starmap(self._get_tile_padded, ilocs)

        # De-overlap if we don't know whether `data` has cheap slices
        # Though we hope that data don't use index wrapping i.e. `index % size`
        locs = self._drop_overlaps(self.locs)
        ilocs = self._ilocs(locs)
        parts = starmap_n(self._get_tile, ilocs, max_workers=self.max_workers)
        if not self.m.overlap:
            return iter(parts)

        rcr = Reconstructor(self.m.overlap, self.cells)
        return map(rcr, parts)

    def get_tile(self, idx: int) -> Tile:
        """Get full tile by its flattened index"""
        i = *np.argwhere(self.cells)[idx].tolist(),
        ys, xs = self.locs[i].tolist()
        return self._get_tile_padded(i, slice(*ys), slice(*xs))


@dataclass(frozen=True)
class _BlendCropsView(_BaseView):
    """
    Applies weighted average over overlapping regions.
    Yields tiles without overlaps, so their size can differ.

    NOTE: can output 0-sized tiles.
    """
    source: Iterable[Tile]

    def __iter__(self) -> Iterator[Tile]:
        assert self.m.overlap
        bcr = BlendCropper(self.m.step, self.m.overlap, self.cells)
        return map(bcr, self.source)
