__all__ = ['BlendCropper', 'Cropper', 'Decimator', 'Reconstructor', 'Zipper']

from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import numpy as np

from ._types import NumpyLike, Tile, Vec


@dataclass(frozen=True, slots=True)
class Cropper:
    shape: tuple[int, ...]

    def __call__(self, tile: Tile) -> Tile:
        h, w = self.shape
        iyx, (y, x), data = tile
        return Tile(idx=iyx, vec=(y, x), data=data[:h - y, :w - x])


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

    def __call__(self, tile: Tile) -> tuple[Vec, Vec, np.ndarray, np.ndarray]:
        v_scale = self.v_scale
        loc = *(slice(round(o * v_scale), round((o + s) * v_scale))
                for o, s in zip(tile.vec, tile.data.shape[:2])),
        return tile.idx, tile.vec, tile.data, self.v[loc]


class Reconstructor:
    """Joins non-overlapping parts to tiles"""
    def __init__(self, overlap: int, step: int, cells: np.ndarray):
        assert overlap
        self.overlap = overlap
        self.step = step
        self.cells = np.pad(cells, [(0, 1), (0, 1)])
        self.row = defaultdict[int, np.ndarray]()

    def __call__(self, t: Tile) -> Tile:
        (iy, ix), _, part = t

        # Lazy init, first part is always whole
        if self.row.default_factory is None:
            self.row.default_factory = partial(np.zeros, part.shape,
                                               part.dtype)

        if (tile := self.row.pop(ix, None)) is not None:
            tile[-part.shape[0]:, -part.shape[1]:] = part
        else:
            tile = part

        # Cache right & bottom edges for adjacent tiles
        if self.cells[iy, ix + 1]:
            self.row[ix + 1][:, :self.overlap] = tile[:, -self.overlap:]
        if self.cells[iy + 1, ix]:
            self.row[ix][:self.overlap, :] = tile[-self.overlap:, :]

        return Tile(
            idx=(iy, ix),
            vec=(iy * self.step - self.overlap, ix * self.step - self.overlap),
            data=tile,
        )


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
