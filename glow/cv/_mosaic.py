from __future__ import annotations

__all__ = ['Mosaic', 'Tile', 'get_fusion']

import dataclasses
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import NamedTuple, Protocol, TypeVar, cast

import cv2
import numpy as np

from .. import chunked, map_n
from ._util import get_trapz

Vec = tuple[int, int]

# TODO: allow result of .map/.map_batched to have different tile and step

# ------------------------------- basic types --------------------------------


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


def _crop(tiles: Iterable[Tile], shape: tuple[int, ...]) -> Iterator[Tile]:
    h, w = shape
    for iyx, (y, x), tile in tiles:
        yield Tile(iyx, (y, x), tile[:h - y, :w - x])


def get_fusion(tiles: Iterable[Tile],
               shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    r: np.ndarray | None = None

    if shape is None:  # Collect all the tiles to compute destination size
        tiles = [*tiles]
        # N arrays of (yx + hw)
        yx_hw = np.array([[[t.vec, t.data.shape[:2]] for t in tiles]]).sum(1)
        # bottom left most edge
        shape = *yx_hw.max(0).tolist(),
    else:
        assert len(shape) == 2

    for _, (y, x), tile in tiles:
        if not tile.size:
            continue

        h, w, c = tile.shape
        if r is None:  # First iteration, initilize
            r = np.zeros((*shape, c), tile.dtype)

        assert c == r.shape[2]
        v = r[y:, x:][:h, :w]  # View to destination
        v[:] = tile[:v.shape[0], :v.shape[1]]  # Crop if needed

    return r


# ------------------------------ mosaic setting ------------------------------


@dataclass
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
        assert 0 <= self.overlap <= self.step
        assert self.overlap % 2 == 0  # That may be optional

    def get_kernel(self) -> np.ndarray:
        return get_trapz(self.step, self.overlap)

    def iterate(self,
                image: NumpyLike,
                max_workers: int = 1) -> _TiledArrayView:
        """Read tiles from input image"""
        shape = image.shape[:2]
        ishape = *(len(range(0, s + self.overlap, self.step)) for s in shape),
        cells = np.ones(ishape, dtype=np.bool_)

        return _TiledArrayView(self, shape, cells, image, max_workers)


# --------------------------------- actions ----------------------------------

_Self = TypeVar('_Self', bound='_BaseView')


@dataclass
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
            _IterView(self.m, self.shape, self.cells, tiles))

    def pool(self: _Self, stride: int = 1) -> _Self:
        """Resizes each tile to desired stride"""
        if stride == 1:
            return self
        shape = *(len(range(0, s, stride)) for s in self.shape),
        return cast(  # don't narrow type
            _Self,
            _DecimatedView(
                Mosaic(self.m.step // stride, self.m.overlap // stride), shape,
                self.cells, stride, self))

    def crop(self) -> _BaseView:
        return _IterView(self.m, self.shape, self.cells, _crop(
            self, self.shape))

    def zip_with(
            self, view: np.ndarray,
            v_scale: int) -> Iterator[tuple[Vec, Vec, np.ndarray, np.ndarray]]:
        """Extracts tiles from `view` simultaneously with tiles from self"""
        assert v_scale >= 1
        for tile in self:
            tw, th = tile.data.shape[:2]
            v = view[tile.vec[0] // v_scale:,
                     tile.vec[1] // v_scale:][:tw // v_scale, :th // v_scale]
            yield *tile, v


@dataclass
class _View(_BaseView):
    def map_batched(self,
                    fn: Callable[[list[np.ndarray]], Iterable[np.ndarray]],
                    /,
                    batch_size: int = 1,
                    max_workers: int = 0) -> _View:
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
                                     Iterable[Tile]]) -> _View:
        """
        TODO: fill docs
        """
        tiles = fn(self)
        return _IterView(self.m, self.shape, self.cells, tiles)

    def reweight(self) -> _View:
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
            return _UniqueTileView(self.m, self.shape, self.cells, self)
        return self


@dataclass
class _IterView(_View):
    source: Iterable[Tile]

    def __iter__(self) -> Iterator[Tile]:
        return iter(self.source)


@dataclass
class _DecimatedView(_View):
    """
    Decimates tiles.
    Doesn't change size uniformity.
    Yields decimated views of original tiles.
    """
    stride: int
    source: Iterable[Tile]

    def __iter__(self) -> Iterator[Tile]:
        for t in self.source:
            yield Tile(
                t.idx,
                tuple(c // self.stride for c in t.vec),  # type: ignore
                t.data[::self.stride, ::self.stride],
            )


@dataclass
class _TiledArrayView(_View):
    """
    Extracts tiles from array.
    Yields same-sized tiles with overlaps.
    """
    data: NumpyLike
    max_workers: int

    def select(self, mask: np.ndarray, scale: int) -> _TiledArrayView:
        """Drop tiles where `mask` is 0"""
        assert mask.ndim == 2
        mask = mask.astype('u1')

        ih, iw = self.ishape
        step = self.m.step // scale
        pad = self.m.overlap // (scale * 2)

        mh, mw = (ih * step), (iw * step)
        if mask.shape[:2] != (mh, mw):
            mask_pad = [(0, s1 - s0) for s0, s1 in zip(mask.shape, (mh, mw))]
            mask = np.pad(mask, mask_pad)[:mh, :mw]

        if self.m.overlap:
            kernel = np.ones((3, 3), dtype='u1')
            mask = cv2.dilate(mask, kernel, iterations=pad)

        if pad:
            mask = np.pad(mask[:-pad, :-pad], [[pad, 0], [pad, 0]])

        cells = mask.reshape(ih, step, iw, step).any((1, 3))
        return dataclasses.replace(self, cells=cells)

    def _get_tile(self, iy: int, ix: int) -> Tile:
        """Read non-overlapping tile of source image"""
        (y0, y1), (x0, x1) = ((
            self.m.step * i - self.m.overlap,
            self.m.step * (i + 1),
        ) for i in (iy, ix))
        if iy and self.cells[iy - 1, ix]:
            y0 += self.m.overlap
        if ix and self.cells[iy, ix - 1]:
            x0 += self.m.overlap
        return Tile((iy, ix), (y0, x0), self.data[y0:y1, x0:x1])

    def _rejoin_tiles(self, image_parts: Iterable[Tile]) -> Iterator[Tile]:
        """Joins non-overlapping parts to tiles"""
        assert self.m.overlap
        overlap = self.m.overlap
        cells = np.pad(self.cells, [(0, 1), (0, 1)])
        step = self.m.step
        row = defaultdict[int, np.ndarray]()

        for (iy, ix), _, part in image_parts:
            # Lazy init, first part is always whole
            if row.default_factory is None:
                row.default_factory = partial(np.zeros, part.shape, part.dtype)

            if (tile := row.pop(ix, None)) is not None:
                tile[-part.shape[0]:, -part.shape[1]:] = part
            else:
                tile = part

            yield Tile((iy, ix), (iy * step - overlap, ix * step - overlap),
                       tile)

            if cells[iy, ix + 1]:
                row[ix + 1][:, :overlap] = tile[:, -overlap:]
            if cells[iy + 1, ix]:
                row[ix][:overlap, :] = tile[-overlap:, :]

    def __iter__(self) -> Iterator[Tile]:
        """
        Yield complete tiles built from source image.
        Each tile will have size `(step + overlap)`
        """
        ys, xs = np.where(self.cells)
        parts = map_n(self._get_tile, ys, xs, max_workers=self.max_workers)
        return self._rejoin_tiles(parts) if self.m.overlap else iter(parts)


@dataclass
class _UniqueTileView(_BaseView):
    """
    Applies weighted average over overlapping regions.
    Yields tiles without overlaps, so their size can differ.
    """
    source: Iterable[Tile]

    _cells: np.ndarray = field(init=False, repr=False)
    _row: dict[int, np.ndarray] = field(init=False, repr=False)
    _carry: list[np.ndarray] = field(init=False, repr=False)

    def __post_init__(self):
        self._cells = np.pad(self.cells, [(0, 1), (0, 1)])
        self._row = {}
        self._carry = []

    def _update(self, obj: Tile) -> Tile:
        """
        Blends edges of overlapping tiles and returns non-overlapping parts
        """
        (iy, ix), (y, x), tile = obj
        overlap = self.m.overlap
        step = self.m.step

        if iy and self._cells[iy - 1, ix]:  # TOP exists
            top = self._row.pop(ix)
            tile[:overlap, step - top.shape[1]:step] += top
        else:
            tile = tile[overlap:]  # cut TOP
            y += overlap

        if ix and self._cells[iy, ix - 1]:  # LEFT exists
            left = self._carry.pop()
            if self._cells[iy + 1, [ix - 1, ix]].all():
                tile[-left.shape[0]:, :overlap] += left
            else:  # cut BOTTOM-LEFT
                tile[-left.shape[0] - overlap:-overlap, :overlap] += left
        else:
            tile = tile[:, overlap:]  # cut LEFT
            x += overlap

        tile, right = np.split(tile, [-overlap], axis=1)
        if self._cells[iy, ix + 1]:  # RIGHT exists
            if not (iy and self._cells[iy - 1, [ix, ix + 1]].all()):
                right = right[-step:]  # cut TOP-RIGHT
            if not self._cells[iy + 1, [ix, ix + 1]].all():
                right = right[:-overlap]  # cut BOTTOM-RIGHT
            self._carry.append(right)

        tile, bottom = np.split(tile, [-overlap])
        if self._cells[iy + 1, ix]:  # BOTTOM exists
            if not (ix and self._cells[[iy, iy + 1], ix - 1].all()):
                # cut BOTTOM-LEFT
                bottom = bottom[:, -(step - overlap):]
            self._row[ix] = bottom

        return Tile((iy, ix), (y, x), tile)

    def __iter__(self) -> Iterator[Tile]:
        assert self.m.overlap
        return map(self._update, self.source)
