from __future__ import annotations

__all__ = ['Slide']

import os
from bisect import bisect_right
from pathlib import Path
from typing import final

import cv2
import numpy as np

from .. import memoize
from ._openslide import Openslide
from ._slide_bases import REGISTRY, Driver, Item, Lod, normalize
from ._tiff import Tiff

# TODO: inside Slide.open import ._slide.registry,
# TODO: and in ._slide.registry do registration and DLL loading
# TODO: to make Slide export not require DLL presence

Openslide.register('bif mrxs ndpi scn svs svsslide tif tiff vms vmu')
Tiff.register('svs tif tiff')

_MAX_BYTES = int(os.environ.get('GLOW_SLIDE_BYTES') or 102_400)


@memoize(capacity=_MAX_BYTES, policy='lru')
def _cached_open(path: Path) -> Slide:
    if not path.exists():
        raise FileNotFoundError(path)

    if tps := REGISTRY.get(path.suffix):
        last_exc = BaseException()
        for tp in reversed(tps):  # Loop over types to find non-failing
            try:
                return Slide(path, tp)
            except ValueError as exc:
                last_exc = exc
        raise last_exc from None
    raise ValueError(f'Unknown file format {path}')


def _fit_to(image: np.ndarray, dsize: tuple[int, ...]):
    if image.shape[:2] == dsize:
        return image
    return cv2.resize(image, dsize[::-1], interpolation=cv2.INTER_AREA)


@final
class Slide:
    """Usage:
    ```
    slide = Slide.open('test.svs')
    shape: tuple[int, ...] = slide.shape
    scales: tuple[int, ...] = slide.scales

    # Get numpy.ndarray
    image: np.ndarray = slide[:2048, :2048]
    ```
    """
    # TODO: check if memory leak
    # TODO: add __enter__/__exit__/close to make memory management explicit
    # TODO: call .close in finalizer
    path: Path
    shape: tuple[int, ...]
    spacing: float | None
    pools: tuple[int, ...]
    lods: tuple[Lod, ...]
    extras: dict[str, Item]

    def __init__(self, path: Path, driver_cls: type[Driver]):
        self.path = path
        driver = driver_cls(path)

        num_items = len(driver)
        assert num_items > 0

        item_0, *items = (driver[idx] for idx in range(num_items))
        assert isinstance(item_0, Lod)

        lods: dict[int, Lod] = {1: item_0}
        self.extras: dict[str, Item] = {}
        for item in items:
            if isinstance(item, Lod):
                pool = round(item_0.shape[0] / item.shape[0])
                lods[pool] = item
            elif key := item.get_key():
                self.extras[key] = item

        self.pools = *sorted(lods.keys()),
        self.lods = *(lods[pool] for pool in self.pools),
        self.shape = self.lods[0].shape
        self.spacing = self.lods[0].spacing

        self.extras |= {key: driver.get(key) for key in driver.keys()}
        self.driver = driver_cls.__name__

    def __reduce__(self) -> tuple:
        return Slide.open, (self.path, )

    def __repr__(self) -> str:
        line = f"'{self.path}', shape={self.shape}, pools={self.pools}"
        if self.spacing:
            line += f', spacing={self.spacing}'
        line += f', driver={self.driver}'
        return f'{type(self).__name__}({line})'

    def best_lod_for(self, pool: float) -> tuple[int, Lod]:
        """Gives the most detailed LOD below `pool`"""
        idx = max(bisect_right(self.pools, pool) - 1, 0)
        return self.pools[idx], self.lods[idx]

    def pool(self, zoom: int) -> Lod:
        """Use like `slide.pool(4)[y0:y1, x0:x1]` call"""
        p, lod = self.best_lod_for(zoom)
        if p == zoom:
            return lod
        assert zoom % p == 0, 'fractional pooling is not supported'
        return lod.downscale(zoom // p)

    def __getitem__(self, key: slice | tuple[slice, ...]) -> np.ndarray:
        """Retrieves tile"""
        # TODO: Ignore step, always redirect to self.lods[0].__getitem__
        slices = normalize(key, self.shape)

        step0, step1 = (s.step for s in slices)
        if step0 != step1:
            raise ValueError('slice steps should be the same for each axis')
        if step0 <= 0:
            raise ValueError('slice steps should be positive')

        pool, lod = self.best_lod_for(step0)
        slices = *(slice(s.start // pool, s.stop // pool) for s in slices),
        image = lod.crop(slices)

        ratio = pool / step0
        dsize = *(round(ratio * s) for s in image.shape[:2]),
        return _fit_to(image, dsize)

    def at(self,
           z0_yx_offset: tuple[int, ...],
           dsize: int | tuple[int, ...],
           scale: float = 1) -> np.ndarray:
        """Read square region starting with offset"""
        dsize = dsize if isinstance(dsize, tuple) else (dsize, dsize)
        assert len(dsize) == 2

        pool, lod = self.best_lod_for(scale)
        slices = *(slice(int(c) // pool,
                         int(c + d * scale) // pool)
                   for c, d in zip(z0_yx_offset, dsize)),
        image = lod.crop(slices)
        return _fit_to(image, dsize)

    def extra(self, name: str) -> np.ndarray | None:
        if item := self.extras.get(name):
            return np.array(item, copy=False, order='C')
        return None

    def thumbnail(self) -> np.ndarray:
        item = self.extras.get('thumbnail')
        if not item:
            item = self.lods[-1]
        return np.asarray(item)

    @classmethod
    def open(cls, anypath: Path | str) -> Slide:
        """Open multi-scale image."""
        path = Path(anypath).resolve().absolute()
        return _cached_open(path)
