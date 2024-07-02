__all__ = ['BlendCropper', 'Decimator', 'Reconstructor', 'Stripper', 'Zipper']

from collections import defaultdict, deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import numpy.typing as npt

from ._types import NDIndex, NumpyLike, Tile, Vec
from ._util import crop_to, rescale_crop

_Op = Callable[[Tile], tuple[Tile, ...]]


@dataclass(frozen=True, slots=True)
class Stripper:
    """Strips regions outside of `data.shape`. Produces non-square patches"""
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

    def __call__(
        self,
        tile: Tile,
    ) -> tuple[NDIndex, Vec, np.ndarray, np.ndarray]:
        tsize = tile.data.shape[:2]
        if not tile.data.size:
            _, _, *extra_dims = self.v.shape
            empty = np.empty((*tsize, *extra_dims), self.v.dtype)
            return tile.idx, tile.vec, tile.data, empty

        loc = (slice(c0, c0 + size) for c0, size in zip(tile.vec, tsize))
        r = rescale_crop(
            self.v, *loc, scale=self.v_scale, interpolation=self.interpolation)

        return tile.idx, tile.vec, tile.data, r


class Reconstructor:
    """Rebuilds full tiles from parts"""
    def __init__(self, overlap: int, cells: npt.NDArray[np.bool_]):
        assert overlap
        self.overlap = overlap
        self.cells = np.pad(cells, (0, 1))
        self.row = defaultdict[int, np.ndarray]()

    def __call__(self, t: Tile) -> Tile:
        (iy, ix), (y, x), part = t
        ov = self.overlap

        # Lazy init, first part is always whole
        if self.row.default_factory is None:
            self.row.default_factory = partial(np.empty, part.shape,
                                               part.dtype)

        if (tile := self.row.pop(ix, None)) is not None:
            # Incoming tile is always bottom-right aligned to the full one
            dy, dx = (tile.shape[a] - part.shape[a] for a in (0, 1))
            tile[dy:, dx:] = part
            y -= dy
            x -= dx
        else:
            tile = part

        # Use current right edge as new left edge
        if self.cells[iy, ix + 1]:
            self.row[ix + 1][:, :ov] = tile[:, -ov:]

        # Use current bottom edge as new top edge
        if self.cells[iy + 1, ix]:
            if ix and self.cells[iy + 1, ix - 1]:  # Don't write angle twice
                self.row[ix][:ov, ov:] = tile[-ov:, ov:]
            else:
                self.row[ix][:ov] = tile[-ov:]

        return Tile(idx=(iy, ix), vec=(y, x), data=tile)


class BlendCropper:
    """
    Blends edges/angles of overlapping tiles.
    Returns non-overlapping parts of tiles.

    All parts form non-uniform rectangular grid
    (i.e. all parts with same `iy` have same `shape[0]` and same `y`).

    NOTE: Each tile can make up to 4 parts.
    """
    __slots__ = ('overlap', 'call', 'cells', 'size')

    size: int
    cells: dict[NDIndex, npt.NDArray[np.bool_]]

    def __init__(self, cells: npt.NDArray[np.bool_], overlap: int, step: int):
        if overlap > step:
            raise ValueError('Tile overlap must be less or equal to tile step')
        self.overlap = overlap

        ih, iw = cells.shape
        # -> (2h+1 2w+1)
        cells = cells.repeat(2, 0).repeat(2, 1)
        cells = np.pad(cells, (0, 1))
        # -> (2h 2w)
        cells = np.lib.stride_tricks.sliding_window_view(cells, (2, 2))
        cells = np.bitwise_and.reduce(cells, axis=(-2, -1))
        # -> (h w 2 2) [[this, next], [next row, next diag]]
        cells = cells.reshape(ih, 2, iw, 2).transpose(0, 2, 1, 3)

        c00 = cells[:, :, 0, 0]
        sizes = (c00[:, :, None, None] & cells).sum((0, 1))

        if overlap < step:
            self.call = self._3x3
            self.size = sizes.sum()
        else:
            self.call = self._2x2
            self.size = sizes[1, 1]

        # (iy, ix) -> 2x2 of bool
        iys, ixs = c00.nonzero()
        self.cells = {
            (iy, ix): m
            for iy, ix, m in zip(iys.tolist(), ixs.tolist(), cells[iys, ixs])
        }

    def __call__(self, src: Iterable[Tile]) -> Iterator[Tile]:
        """Merge and split tiles."""
        # TODO: relax input/output order
        return self.call(src)

    def _2x2(self, src: Iterable[Tile]) -> Iterator[Tile]:
        ops = defaultdict[NDIndex, list[_Op]](list)
        ov = self.overlap

        for tile in src:  # NOTE: supports C and F-ordering
            iy, ix = idx = tile.idx

            for op in ops.pop(idx, []):  # full top & left (3)
                yield from op(tile)

            # TODO: bag ops
            if self.cells[iy, ix][1, 1]:  # bottom-right (1)
                u = _angle(ov, idx, tile.data)
                next(u)
                ops[iy, ix + 1].append(u.send)
                ops[iy + 1, ix].append(u.send)
                ops[iy + 1, ix + 1].append(partial(_final_send, u))

    def _3x3(self, src: Iterable[Tile]) -> Iterator[Tile]:
        last_iy = -1
        buf = deque[Tile]()
        ops = defaultdict[NDIndex, list[_Op]](list)
        ov = self.overlap

        for tile in src:  # NOTE: only C-ordering is supported.
            iy, ix = idx = tile.idx

            if iy != last_iy:  # Start/end row
                while buf:
                    yield buf.popleft()
                last_iy = iy  # NOTE: this one locks iteration order

            for op in ops.pop(idx, []):  # full top & left (5)
                yield from op(tile)
            (_, c01), (c10, c11) = self.cells[iy, ix].tolist()

            buf.append(_center(ov, tile))  # center (1)

            if c01:  # right (1)
                ops[iy, ix + 1].append(partial(_vert_edge, ov, buf, tile.data))

            if c10:  # bottom (1)
                ops[iy + 1, ix].append(partial(_horz_edge, ov, tile.data))

            if c11:  # bottom-right (1)
                u = _angle(ov, (iy * 2 + 1, ix * 2 + 1), tile.data)
                next(u)
                ops[iy, ix + 1].append(u.send)
                ops[iy + 1, ix].append(u.send)
                ops[iy + 1, ix + 1].append(partial(_final_send, u))

        while buf:
            yield buf.popleft()  # End row


def _center(pad: int, tile: Tile) -> Tile:
    # [- - -]
    # [- R -]
    # [- - -]
    (ix, iy), (y, x), data = tile
    return Tile((2 * ix, 2 * iy), (y + pad, x + pad), data[pad:-pad, pad:-pad])


def _vert_edge(pad: int, buf: deque[Tile], left: np.ndarray,
               tile: Tile) -> tuple[()]:
    # [- - -]    [- - -]
    # [- - X] -> [R - -]
    # [- - -]    [- - -]
    (iy, ix), (y, x), right = tile
    tile = Tile((iy * 2, ix * 2 - 1), (y + pad, x),
                left[pad:-pad, -pad:] + right[pad:-pad, :pad])
    buf.append(tile)
    return ()


def _horz_edge(pad: int, top: np.ndarray, tile: Tile) -> tuple[Tile]:
    # [- - -]    [- R -]
    # [- - -] -> [- - -]
    # [- X -]    [- - -]
    (iy, ix), (y, x), bottom = tile
    tile = Tile((2 * iy - 1, 2 * ix), (y, x + pad),
                top[-pad:, pad:-pad] + bottom[:pad, pad:-pad])
    return tile,


def _angle(pad: int, idx: NDIndex,
           data: np.ndarray) -> Generator[tuple[()], Tile, tuple[Tile]]:
    # [- . -]    [- . -]    [- . X]    [R . -]
    # [. . .] -> [. . .] -> [. . .] -> [. . .]
    # [- . X]    [X . -]    [- . -]    [- . -]

    # top-left + top-right
    data = data[-pad:, -pad:] + (yield ()).data[-pad:, :pad]
    # bottom-left
    data += (yield ()).data[:pad, -pad:]
    # bottom-right
    _, vec, last = yield ()
    data += last[:pad, :pad]

    return Tile(idx, vec, data),


def _final_send(coro: Generator[tuple[()], Tile, tuple[Tile]],
                tile: Tile) -> tuple[Tile]:
    try:
        coro.send(tile)
    except StopIteration as stop:
        return stop.value
    else:
        raise RuntimeError('Coroutine not stopped')
