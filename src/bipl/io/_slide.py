__all__ = ['Slide']

import os
import warnings
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import final, overload
from warnings import warn

import numpy as np
from glow import memoize, shared_call, weak_memoize

from bipl import env
from bipl.ops import normalize_loc, resize

from ._openslide import Openslide
from ._slide_bases import REGISTRY, Driver, Item, Lod
from ._tiff import Tiff

# TODO: inside Slide.open import ._slide.registry,
# TODO: and in ._slide.registry do registration and DLL loading
# TODO: to make Slide export not require DLL presence

try:
    from ._gdal import Gdal
except ImportError:
    if not os.getenv('_BIPL_GDAL_NO_WARN'):
        msg = 'No GDAL is available. Please '
        if os.name == 'nt':
            msg += ('acquire it manually from '
                    'https://github.com/cgohlke/geospatial-wheels/releases ')
        else:
            msg += ('install libgdal via your system package manager, '
                    'and run "pip install gdal==`gdal-config --version`"')
        warn(msg, stacklevel=1)
        os.environ['_BIPL_GDAL_NO_WARN'] = '1'
    Gdal = None  # type: ignore[assignment,misc]

_drv: type[Driver] | None
for _drv, _regex in [  # LIFO, last driver takes priority
    (Gdal, r'^.*\.(tif|tiff)$'),
    (Openslide, r'^.*\.(bif|mrxs|ndpi|scn|svs|svsslide|tif|tiff|vms|vmu)$'),
    (Tiff, r'^.*\.(svs|tif|tiff)$'),
    (Gdal, r'^(/vsicurl.*|(http|https)://.*)$'),
]:
    if _drv is not None and _drv.__name__.lower() in env.BIPL_DRIVERS:
        _drv.register(_regex)


@shared_call  # merge duplicate calls
@weak_memoize  # reuse result if it's already exist, but used by someone else
@memoize(capacity=env.BIPL_CACHE, policy='lru')  # keep LRU for unused results
def _cached_open(path: str) -> 'Slide':
    last_exc = BaseException()
    matches = (tp for pat, tps_ in REGISTRY.items() if pat.match(path)
               for tp in tps_)
    tps = [*dict.fromkeys(matches)]
    if tps:
        for tp in reversed(tps):  # Loop over types to find non-failing
            try:
                return Slide.from_file(path, tp)
            except (ValueError, TypeError) as exc:
                last_exc = exc
        raise last_exc from None
    raise ValueError(f'Unknown file format {path}')


@final
@dataclass(frozen=True)
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
    path: str
    shape: tuple[int, ...]
    pools: tuple[int, ...]
    mpp: float | None
    driver: str
    bbox: tuple = field(repr=False)
    lods: tuple[Lod, ...] = field(repr=False)
    extras: dict[str, Item] = field(repr=False)

    @classmethod
    def from_file(
        cls,
        path: str,
        driver_fn: Callable[[str], Driver],
        mpp: float | None = None,
    ) -> 'Slide':
        driver = driver_fn(path)

        num_items = len(driver)
        if num_items == 0:
            raise ValueError('Empty file')

        item_0, *items = (driver[idx] for idx in range(num_items))
        if not isinstance(item_0, Lod):
            raise TypeError('First pyramid layer is not tiled')

        # Retrieve all sub-images
        lods: dict[int, Lod] = {1: item_0}
        extras: dict[str, Item] = {}
        for item in items:
            if isinstance(item, Lod):
                pool = round(item_0.shape[0] / item.shape[0])
                lods[pool] = item
            elif key := item.key:
                extras[key] = item
        extras |= driver.named_items()

        zoom_levels = *sorted(lods.keys()),
        # TODO: create virtual lods if zoom levels are too distant (ProxyLod?)

        if mpp is None:  # If no override is passed, use native if present
            mpp = lods[1].mpp

        return Slide(
            path=path,
            shape=lods[1].shape,
            mpp=mpp,
            pools=zoom_levels,
            driver=type(driver).__name__,
            bbox=driver.bbox,
            lods=tuple(lods[pool] for pool in zoom_levels),
            extras=extras,
        )

    def mpp_or_error(self) -> float:
        if self.mpp is None:
            raise ValueError('Slide`s MPP is unknown')
        return self.mpp

    @property
    def spacing(self) -> float | None:
        warnings.warn(
            '"Slide.spacing" is deprecated. Use "Slide.mpp"',
            category=DeprecationWarning,
            stacklevel=2)
        return self.mpp

    def __reduce__(self) -> tuple:
        return Slide.open, (self.path, )

    def best_lod_for(self,
                     pool: float,
                     *,
                     tol: float = 0.01) -> tuple[int, Lod]:
        """Gives the most detailed LOD below `pool`"""
        pool = pool * max(1, 1 + tol)
        idx = max(bisect_right(self.pools, pool) - 1, 0)
        return self.pools[idx], self.lods[idx]

    def resample(self, mpp: float, *, tol: float = 0.01) -> Lod:
        """Resample slide to specific resolution"""
        zoom = mpp / self.mpp_or_error()
        return self.pool(zoom, tol=tol)

    def pool(self, zoom: float, *, tol: float = 0.01) -> Lod:
        """Use like `slide.pool(4)[y0:y1, x0:x1]` call"""
        p, lod = self.best_lod_for(zoom, tol=tol)
        if p == zoom:
            return lod
        return lod.rescale(p / zoom)

    def __getitem__(self, key: slice | tuple[slice, ...]) -> np.ndarray:
        """Retrieves tile"""
        # TODO: Ignore step, always redirect to self.lods[0].__getitem__
        y_loc, x_loc, c_loc = normalize_loc(key, self.shape)

        step0, step1 = y_loc.step, x_loc.step
        if step0 != step1:
            raise ValueError('slice steps should be the same for each axis')
        if step0 <= 0:
            raise ValueError('slice steps should be positive')

        pool, lod = self.best_lod_for(step0)
        yx_loc = *(slice(s.start // pool, s.stop // pool)
                   for s in (y_loc, x_loc)),
        image = lod.crop(*yx_loc)

        ratio = pool / step0
        dsize = *(round(ratio * s) for s in image.shape[:2]),
        return resize(image, dsize)[:, :, c_loc]

    @overload
    def at(self,
           z0_yx_offset: tuple[int, ...],
           dsize: int | tuple[int, ...],
           *,
           scale: float,
           tol: float = ...) -> np.ndarray:
        ...

    @overload
    def at(self,
           z0_yx_offset: tuple[int, ...],
           dsize: int | tuple[int, ...],
           *,
           mpp: float,
           tol: float = ...) -> np.ndarray:
        ...

    def at(self,
           z0_yx_offset: tuple[int, ...],
           dsize: int | tuple[int, ...],
           *,
           scale: float | None = 1,
           mpp: float | None = None,
           tol: float = 0.01) -> np.ndarray:
        """Read square region starting with offset"""
        dsize = dsize if isinstance(dsize, tuple) else (dsize, dsize)
        if len(dsize) != 2:
            raise ValueError(f'dsize should be 2-tuple or int. Got {dsize}')

        if scale is None:
            if mpp is None:
                raise ValueError('Only one of scale/mpp should be None')
            scale = mpp / self.mpp_or_error()

        pool, lod = self.best_lod_for(scale, tol=tol)
        loc = *(slice(int(c) // pool,
                      int(c + d * scale) // pool)
                for c, d in zip(z0_yx_offset, dsize)),
        image = lod.crop(*loc)
        return resize(image, dsize)

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
    def open(cls, anypath: Path | str) -> 'Slide':
        """Open multi-scale image."""
        if isinstance(anypath, Path):
            anypath = Path(anypath).resolve().absolute().as_posix()
        return _cached_open(anypath)
