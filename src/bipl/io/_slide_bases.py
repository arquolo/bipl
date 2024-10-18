__all__ = ['Driver', 'Image', 'ImageLevel']

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from itertools import starmap
from math import ceil
from typing import TYPE_CHECKING, Self, final

import cv2
import numpy as np
from glow import aceil, afloor

from bipl import env
from bipl._types import HasParts, Patch, Shape, Span, Tile
from bipl.ops import get_fusion, normalize_loc, rescale_crop, resize

from ._util import round2

if TYPE_CHECKING:
    from ._util import Icc

_REGISTRY: dict[re.Pattern, list[type['Driver']]] = {}
_MIN_TILE = 256


@dataclass(frozen=True)
class Image:
    shape: Shape
    dtype = np.dtype(np.uint8)
    post: list[Callable[[np.ndarray],
                        np.ndarray]] = field(default_factory=list)

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

    def apply(self, fn: Callable[[np.ndarray], np.ndarray]) -> Self:
        return replace(self, post=[*self.post, fn])

    def _postprocess(self, im: np.ndarray) -> np.ndarray:
        for fn in self.post:
            im = fn(im)
        return im

    @property
    def icc(self) -> 'Icc | None':
        return None


@dataclass(frozen=True)
class ImageLevel(Image, HasParts):
    @final
    @property
    def key(self) -> None:
        return None

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

        if isinstance(self, ProxyLevel):
            scale = self.scale * scale
            base = replace(self.base, post=self.post)
        else:
            base = self

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
            base = TiledProxyLevel(
                (bh, bw, bc),
                post=base.post,
                base=replace(base, post=[]),
                downsample=downsample,
                r_tile=r_tile,
            )

        if scale == 1 or base.shape == (h, w, c):
            return base
        return ProxyLevel(
            (h, w, c),
            post=base.post,
            scale=scale,
            base=replace(base, post=[]),
        )

    def decimate(self, steps: int) -> 'tuple[int, ImageLevel]':
        return (0, self)

    def join(self, lv: 'ImageLevel') -> 'tuple[int, ImageLevel]':
        ds = round2(lv.shape[0] / self.shape[0])
        return ds, self

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


@dataclass(frozen=True, kw_only=True)
class ProxyLevel(ImageLevel):
    scale: float
    base: ImageLevel

    def part(self, *loc: Span) -> np.ndarray:
        im = rescale_crop(
            self.base, *loc, scale=1 / self.scale, interpolation=1)
        return self._postprocess(im)

    def parts(self,
              locs: Sequence[tuple[Span, ...]],
              max_workers: int = 0) -> Iterator[Patch]:
        scale = 1 / self.scale
        o_locs = np.asarray(locs, 'i4')  # (n yx lo/hi)
        n = o_locs.shape[0]

        i_locs_f = o_locs * scale
        i_locs = np.stack(
            [afloor(i_locs_f[:, :, 0], 'i4'),
             aceil(i_locs_f[:, :, 1], 'i4')], -1)  # (n yx lo/hi)
        sizes = o_locs @ [-1, 1]  # (n yx)

        # Transformation matrix, (n xy xyc)
        mats = np.zeros((n, 2, 3), 'f4')
        mats[:, 0, 0] = scale
        mats[:, 1, 1] = scale
        mats[:, :, 2] = (
            i_locs_f[:, ::-1, 0] - i_locs[:, ::-1, 0] + (scale - 1) / 2)

        # Map input -> output
        i_locs_lst = [tuple(map(Span, loc)) for loc in i_locs.tolist()]
        o_locs_lst = [tuple(map(Span, loc)) for loc in o_locs.tolist()]
        i2o = dict(zip(i_locs_lst, zip(o_locs_lst, sizes.tolist(), mats)))

        kwargs = {
            'flags': cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            'borderMode': cv2.BORDER_CONSTANT,
        }

        # TODO: make this parallel
        def resample(iyx: tuple[Span, ...], i: np.ndarray) -> Patch:
            oyx, (h, w), mat = i2o[iyx]
            if not i.size:
                o = np.empty((h, w, *i.shape[2:]), i.dtype)
            else:
                o = cv2.warpAffine(i, mat, (w, h), **kwargs)  # type: ignore
            return Patch(oyx, self._postprocess(o))

        return starmap(resample, self.base.parts(i_locs_lst, max_workers))


@dataclass(frozen=True, kw_only=True)
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
        return self._postprocess(image)


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
