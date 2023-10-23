__all__ = ['Driver', 'Item', 'Lod', 'REGISTRY']

import re
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from typing import final

import cv2
import numpy as np

from bipl import env
from bipl.ops import Tile, get_fusion, normalize_loc, resize

REGISTRY: dict[re.Pattern, list[type['Driver']]] = {}
_MIN_TILE = 256


@dataclass(frozen=True)
class Item:
    shape: tuple[int, ...]

    @property
    def key(self) -> str | None:
        raise NotImplementedError

    def numpy(self) -> np.ndarray:
        return np.array(self, copy=False)

    def __array__(self) -> np.ndarray:
        raise NotImplementedError

    def apply(self, fn: Callable[[np.ndarray], np.ndarray]) -> '_PostItem':
        return _PostItem(self.shape, self, fn)


@dataclass(frozen=True)
class Lod(Item):
    mpp: float | None

    @final
    @property
    def key(self) -> None:
        return None

    def crop(self, *loc: slice) -> np.ndarray:
        """Reads crop of LOD. Overridable"""
        raise NotImplementedError

    @final
    def __getitem__(self, key: slice | tuple[slice, ...]) -> np.ndarray:
        """Reads crop of LOD"""
        y_loc, x_loc, c_loc = normalize_loc(key, self.shape)
        if not y_loc.step == x_loc.step == 1:
            raise ValueError('Y/X slice steps should be 1 for now, '
                             f'got {y_loc.step} and {x_loc.step}')
        return self.crop(y_loc, x_loc)[:, :, c_loc]

    @final
    def __array__(self) -> np.ndarray:
        """Reads whole LOD in single call"""
        return self[:, :]

    def rescale(self, scale: float) -> 'Lod':
        if scale == 1:
            return self

        base = self
        if isinstance(base, ProxyLod):
            scale = base.scale * scale
            base = base.base

        h, w, c = base.shape
        h, w = (round(h * scale), round(w * scale))  # TODO: round/ceil/floor ?
        mpp = base.mpp / scale if base.mpp else None
        if scale > 0.5:  # Downscale to less then 2x, or upsample
            return ProxyLod((h, w, c), mpp, scale, base)

        zoom = 2 ** (int(1 / scale).bit_length() - 1)
        r_tile = max(_MIN_TILE // zoom, 1)
        return TiledProxyLod((h, w, c), mpp, scale, base, zoom, r_tile)

    def _unpack_loc(
        self,
        *slices: slice,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        box = np.array([(s.start, s.stop) for s in slices])
        valid_box = box.T.clip([0, 0], self.shape[:2]).T  # (2, lo-hi)
        shape = (box[:, 1] - box[:, 0]).tolist()
        return box, valid_box, shape

    def _expand(self, rgb: np.ndarray, valid_box: np.ndarray, box: np.ndarray,
                bg_color: np.ndarray) -> np.ndarray:
        offsets = np.abs(valid_box - box)
        if offsets.any():
            tp, bm, lt, rt = offsets.ravel().tolist()
            rgb = cv2.copyMakeBorder(rgb, tp, bm, lt, rt, cv2.BORDER_CONSTANT,
                                     None, bg_color.tolist())
        return np.ascontiguousarray(rgb)

    @final
    def apply(self, fn: Callable[[np.ndarray], np.ndarray]) -> '_PostLod':
        return _PostLod(self.shape, self.mpp, self, fn)


@dataclass(frozen=True)
class _PostItem(Item):
    base: Item
    fn: Callable[[np.ndarray], np.ndarray]

    def __array__(self) -> np.ndarray:
        im = self.base.__array__()
        return self.fn(im)


@dataclass(frozen=True)
class _PostLod(Lod):
    base: Lod
    fn: Callable[[np.ndarray], np.ndarray]
    pad: int = 64

    def crop(self, *loc: slice) -> np.ndarray:
        loc_ = *(slice(s.start - self.pad, s.stop + self.pad, s.step)
                 for s in loc),
        im = self.base.crop(*loc_)
        im = self.fn(im)
        return im[self.pad:-self.pad, self.pad:-self.pad, :]


@dataclass(frozen=True)
class ProxyLod(Lod):
    scale: float
    base: Lod

    def _get_loc(self, *src_loc: slice) -> np.ndarray:
        return self.base[src_loc]

    def crop(self, *loc: slice) -> np.ndarray:
        src_loc = *[
            # TODO: round/ceil/floor ?
            slice(round(s.start / self.scale), round(s.stop / self.scale))
            for s in loc
        ],
        image = self._get_loc(*src_loc)

        h, w = ((s.stop - s.start) for s in loc)
        return resize(image, (h, w))


@dataclass(frozen=True)
class TiledProxyLod(ProxyLod):
    zoom: int
    r_tile: int

    def _get_loc(self, *src_loc: slice) -> np.ndarray:
        s_start = [s.start for s in src_loc]
        s_shape = *(s.stop - s.start for s in src_loc),
        if np.prod(s_shape) < env.BIPL_TILE_POOL_SIZE:
            return super()._get_loc(*src_loc)

        r_shape = *(ceil(size / self.zoom) for size in s_shape),
        r_tile = self.r_tile
        s_tile = r_tile * self.zoom

        ty, tx = (ceil(size / s_tile) for size in s_shape)
        tgrid = np.mgrid[:ty, :tx].reshape(2, -1).T

        t_shape = (r_tile, r_tile)
        r_tiles = map(
            Tile,
            tgrid.tolist(),
            (tgrid * r_tile).tolist(),
            (resize(self.base[sy:sy + s_tile, sx:sx + s_tile], t_shape)
             for sy, sx in (tgrid * s_tile + s_start).tolist()),
        )
        image = get_fusion(r_tiles, r_shape)
        assert image is not None
        return image


class Driver:
    @final
    @classmethod
    def register(cls, regex: str):
        """Registers type builder for extensions. Last call takes precedence"""
        REGISTRY.setdefault(re.compile(regex), []).append(cls)

    def __init__(self, path: str) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        """Count of indexed items"""
        return 0

    def __getitem__(self, index: int) -> Item:
        """Gives indexed item"""
        raise NotImplementedError

    def named_items(self) -> dict[str, Item]:
        keys = self.keys()
        return {k: self.get(k) for k in keys}

    def keys(self) -> list[str]:
        """List of names for named items"""
        return []

    def get(self, key: str) -> Item:
        """Gives named item"""
        raise NotImplementedError

    @property
    def bbox(self) -> tuple[slice, slice]:
        return slice(None), slice(None)
