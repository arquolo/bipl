__all__ = ['BlendCropper', 'Cropper', 'Decimator', 'Reconstructor', 'Zipper']

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from math import ceil, floor

import cv2
import numpy as np

from ._types import NumpyLike, Tile, Vec
from ._util import crop_to, padslice


@dataclass(frozen=True, slots=True)
class Cropper:
    """Crops tile to be exactly within [0 .. data.shape]"""
    shape: Sequence[int]

    def __call__(self, tile: Tile) -> Tile:
        idx, vec, data = tile
        vec, data = crop_to(vec, data, self.shape)
        return Tile(idx=idx, vec=vec, data=data)


@dataclass(frozen=True, slots=True)
class Decimator:
    stride: int

    def __call__(self, tile: Tile) -> Tile:
        stride = self.stride
        vec = *(c // stride for c in tile.vec),
        data = tile.data[::stride, ::stride]
        return Tile(idx=tile.idx, vec=vec, data=data)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Zipper:
    v: NumpyLike
    v_scale: float
    interpolation: int

    def __call__(self, tile: Tile) -> tuple[Vec, Vec, np.ndarray, np.ndarray]:
        v_scale = self.v_scale
        th, tw = tshape = tile.data.shape[:2]
        if not tile.data.size:
            _, _, *extra_dims = self.v.shape
            empty = np.empty((th, tw, *extra_dims), self.v.dtype)
            return tile.idx, tile.vec, tile.data, empty

        lo = *(v_scale * o for o in tile.vec),
        hi = *(v_scale * (o + s) for o, s in zip(tile.vec, tshape)),

        # Select approximate slice
        loc = *(slice(floor(lo_), ceil(hi_)) for lo_, hi_ in zip(lo, hi)),
        r = padslice(self.v, *loc)

        # Align pixel grids of incoming tile and existing view and resize
        dy, dx = (lo_ - floor(lo_) + (v_scale - 1) / 2 for lo_ in lo)
        r = cv2.warpAffine(
            r,
            np.array([[v_scale, 0, dx], [0, v_scale, dy]], 'f4'),
            (tw, th),
            flags=self.interpolation | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return tile.idx, tile.vec, tile.data, r


class Reconstructor:
    """Joins non-overlapping parts to tiles"""
    def __init__(self, overlap: int, cells: np.ndarray):
        assert overlap
        self.overlap = overlap
        self.cells = np.pad(cells, ((0, 1), (0, 1)))
        self.row = defaultdict[int, np.ndarray]()

    def __call__(self, t: Tile) -> Tile:
        (iy, ix), (y, x), part = t

        # Lazy init, first part is always whole
        if self.row.default_factory is None:
            self.row.default_factory = partial(np.zeros, part.shape,
                                               part.dtype)

        if (tile := self.row.pop(ix, None)) is not None:
            # Incoming tile is always bottom-right aligned to the full one
            tile[-part.shape[0]:, -part.shape[1]:] = part
            y += part.shape[0] - tile.shape[0]
            x += part.shape[1] - tile.shape[1]
        else:
            tile = part

        # Cache right & bottom edges for adjacent tiles
        if self.cells[iy, ix + 1]:
            self.row[ix + 1][:, :self.overlap] = tile[:, -self.overlap:]
        if self.cells[iy + 1, ix]:
            self.row[ix][:self.overlap, :] = tile[-self.overlap:, :]

        return Tile(idx=(iy, ix), vec=(y, x), data=tile)


class BlendCropper:
    """
    Applies blends edges of incoming tiles.
    Returns crops excluding overlaps. Crop size depends on its surrounding.
    """
    def __init__(self, step: int, overlap: int, cells: np.ndarray):
        self.step = step
        self.overlap = overlap
        self.cells = np.pad(cells, [(0, 1), (0, 1)])
        self.row: dict[int, np.ndarray] = {}
        self.carry: np.ndarray | None = None

    def __call__(self, obj: Tile) -> Tile:
        """
        Blends edges of overlapping tiles and returns non-overlapping parts
        """
        (iy, ix), (y, x), tile = obj
        overlap = self.overlap
        step = self.step

        if iy and self.cells[iy - 1, ix]:  # TOP exists
            top = self.row.pop(ix)
            tile[:overlap, step - top.shape[1]:step] += top
        else:
            tile = tile[overlap:]  # cut TOP
            y += overlap

        if ix and self.cells[iy, ix - 1]:  # LEFT exists
            left = self.carry
            assert left is not None
            if self.cells[iy + 1, [ix - 1, ix]].all():
                tile[-left.shape[0]:, :overlap] += left
            else:  # cut BOTTOM-LEFT
                tile[-left.shape[0] - overlap:-overlap, :overlap] += left
        else:
            tile = tile[:, overlap:]  # cut LEFT
            x += overlap

        tile, right = np.split(tile, [-overlap], axis=1)
        if self.cells[iy, ix + 1]:  # RIGHT exists
            if not (iy and self.cells[iy - 1, [ix, ix + 1]].all()):
                right = right[-step:]  # cut TOP-RIGHT
            if not self.cells[iy + 1, [ix, ix + 1]].all():
                right = right[:-overlap]  # cut BOTTOM-RIGHT
            self.carry = right

        tile, bottom = np.split(tile, [-overlap])
        if self.cells[iy + 1, ix]:  # BOTTOM exists
            if not (ix and self.cells[[iy, iy + 1], ix - 1].all()):
                # cut BOTTOM-LEFT
                bottom = bottom[:, -(step - overlap):]
            self.row[ix] = bottom

        return Tile(idx=(iy, ix), vec=(y, x), data=tile)
