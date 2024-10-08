__all__ = ['Driver', 'Image', 'ImageLevel']

import re
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, final

import cv2
import numpy as np

from bipl import env
from bipl.ops import (Shape, Span, Tile, get_fusion, normalize_loc,
                      rescale_crop, resize)

if TYPE_CHECKING:
    from ._util import Icc

_REGISTRY: dict[re.Pattern, list[type['Driver']]] = {}
_MIN_TILE = 256


@dataclass(frozen=True)
class Image:
    shape: Shape
    dtype = np.dtype(np.uint8)

    @property
    def key(self) -> str | None:
        raise NotImplementedError

    def numpy(self) -> np.ndarray:
        """Convert to ndarray"""
        raise NotImplementedError

    @final
    def __array__(self) -> np.ndarray:
        """numpy.array compatibility"""
        return self.numpy()

    def apply(self, fn: Callable[[np.ndarray], np.ndarray]) -> '_LambdaImage':
        return _LambdaImage(self.shape, self, fn)

    @property
    def icc(self) -> 'Icc | None':
        return None

    def flip(self) -> 'Image':
        return self


@dataclass(frozen=True)
class ImageLevel(Image):
    @final
    @property
    def key(self) -> None:
        return None

    def part(self, *loc: Span) -> np.ndarray:
        """Reads crop of LOD. Overridable"""
        raise NotImplementedError

    @final
    def __getitem__(self, key: slice | tuple[slice, ...]) -> np.ndarray:
        """Retrieve sub-image as array from set location"""
        y_loc, x_loc, (c_lo, c_hi) = normalize_loc(key, self.shape)
        return self.part(y_loc, x_loc)[:, :, c_lo:c_hi]

    @final
    def numpy(self) -> np.ndarray:
        """Retrieve whole image as array"""
        return self[:, :]

    def rescale(self, scale: float) -> 'ImageLevel':
        """
        Rescale image to set `scale`. Downscale if `scale` is less then 1,
        upscale otherwise.
        """
        if scale == 1:
            return self

        base = self
        if isinstance(base, ProxyLevel):
            scale = base.scale * scale
            base = base.base

        h, w, c = base.shape
        h, w = (round(h * scale), round(w * scale))  # TODO: round/ceil/floor ?
        if scale <= 0.5:  # Downscale to more then 2x

            # NOTE: to use this datapath we must
            # - have 4^k downsamples (seen only in SVS)
            # - fail `octave()` call - always succedes unless SVS is choked
            #   (0s in tile sizes)
            downsample = 2 ** (int(1 / scale).bit_length() - 1)
            r_tile = max(_MIN_TILE // downsample, 1)
            bh, bw, bc = base.shape
            bh, bw = ((bh + downsample - 1) // downsample,
                      (bw + downsample - 1) // downsample)

            scale *= downsample
            base = TiledProxyLevel((bh, bw, bc), base, downsample, r_tile)

        return ProxyLevel((h, w, c), scale, base)

    def octave(self) -> 'ImageLevel | None':
        return None

    def flip(self) -> 'ImageLevel':
        return self

    def fallback(self, lv: 'ImageLevel', ds: int) -> 'ImageLevel':
        return self

    def _unpack_2d_loc(self, *loc:
                       Span) -> tuple[np.ndarray, np.ndarray, Shape]:
        box = np.array(loc)  # box[axis, {start, stop}]

        # Slices guarantied to be within image shape
        h, w = self.shape[:2]
        valid_box = box.clip(0, [[h], [w]])

        # Full output shape
        out_shape = *(box @ [-1, 1]).tolist(),
        return box, valid_box, out_shape

    @staticmethod
    def _expand(rgb: np.ndarray, valid_box: np.ndarray, box: np.ndarray,
                bg_color: np.ndarray) -> np.ndarray:
        offsets = np.abs(valid_box - box)
        if offsets.any():
            tp, bm, lt, rt = offsets.ravel().tolist()
            rgb = cv2.copyMakeBorder(rgb, tp, bm, lt, rt, cv2.BORDER_CONSTANT,
                                     None, bg_color.tolist())
        return np.ascontiguousarray(rgb)

    @final
    def apply(  # type: ignore[override]
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        pad: int = 0,
    ) -> '_LambdaLevel':
        # _LambdaLevel is not subclass of _LambdaImage
        return _LambdaLevel(self.shape, self, fn, pad)


@dataclass(frozen=True)
class _LambdaImage(Image):
    base: Image
    fn: Callable[[np.ndarray], np.ndarray]

    def numpy(self) -> np.ndarray:
        im = self.base.numpy()
        return self.fn(im)


@dataclass(frozen=True)
class _LambdaLevel(ImageLevel):
    base: ImageLevel
    fn: Callable[[np.ndarray], np.ndarray]
    pad: int = 64

    def part(self, *loc: Span) -> np.ndarray:
        if self.pad:
            loc = *((lo - self.pad, hi + self.pad) for lo, hi in loc),
        im = self.base.part(*loc)
        im = self.fn(im)
        if self.pad:
            return im[self.pad:-self.pad, self.pad:-self.pad, :]
        return im


@dataclass(frozen=True)
class ProxyLevel(ImageLevel):
    scale: float
    base: ImageLevel

    def part(self, *loc: Span) -> np.ndarray:
        return rescale_crop(
            self.base, *loc, scale=1 / self.scale, interpolation=1)


@dataclass(frozen=True)
class TiledProxyLevel(ImageLevel):
    base: ImageLevel
    downsample: int
    r_tile: int

    def part(self, *loc: Span) -> np.ndarray:
        s_start = [lo * self.downsample for lo, _ in loc]

        r_shape = *(hi - lo for lo, hi in loc),
        s_shape = *(size * self.downsample for size in r_shape),
        if not all(s_shape):
            return np.empty((*r_shape, self.base.shape[2]), self.dtype)

        if np.prod(s_shape) < env.BIPL_TILE_POOL_SIZE:
            s_loc = *((lo * self.downsample, hi * self.downsample)
                      for lo, hi in loc),
            return resize(self.base.part(*s_loc), r_shape[:2])

        r_tile = self.r_tile
        s_tile = r_tile * self.downsample

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
    def get_mpp(self) -> float | None:
        raise NotImplementedError

    @final
    @classmethod
    def register(cls, regex: str) -> None:
        """Registers type builder for extensions. Last call takes precedence"""
        _REGISTRY.setdefault(re.compile(regex), []).append(cls)

    @final
    @staticmethod
    def find(path: str) -> list[type['Driver']]:
        tps: dict[type[Driver], None] = {}
        for pat, tps_ in _REGISTRY.items():
            if pat.match(path):
                for tp in tps_:
                    tps[tp] = None
        return [*tps]

    def __init__(self, path: str) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        """Count of indexed images, usually resolution images"""
        return 0

    def __getitem__(self, index: int) -> Image | None:
        """Gives indexed image"""
        raise NotImplementedError

    def named_items(self) -> dict[str, Image]:
        keys = self.keys()
        return {k: self.get(k) for k in keys}

    def keys(self) -> list[str]:
        """Names of associated images"""
        return []

    def get(self, key: str) -> Image:
        """Get assosiated image by key"""
        raise NotImplementedError

    @property
    def bbox(self) -> tuple[slice, ...]:
        return slice(None), slice(None)
