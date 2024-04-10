__all__ = ['Slide']

import os
import warnings
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import ceil
from pathlib import Path
from typing import final, overload
from warnings import warn

import numpy as np
from glow import memoize, shared_call, weak_memoize

from bipl import env
from bipl.ops import normalize_loc, resize

from ._openslide import Openslide
from ._slide_bases import REGISTRY, Driver, Image, ImageLevel
from ._tiff import Tiff
from ._util import clahe, round2

# TODO: inside Slide.open import ._slide.registry,
# TODO: and in ._slide.registry do registration and DLL loading
# TODO: to make Slide export not require DLL presence

_gdal_warn = 'GDAL driver was disabled because GDAL was not found. Please '
if os.name == 'nt':
    _gdal_warn += ('acquire it manually from '
                   'https://github.com/cgohlke/geospatial-wheels/releases ')
else:
    _gdal_warn += ('ensure that you have "libgdal" in your system, '
                   '"wheel" in your Python environment'
                   'and run "pip install gdal==`gdal-config --version`"')

Gdal = None
if 'gdal' in env.BIPL_DRIVERS:
    try:
        from ._gdal import Gdal
    except ImportError:  # GDAL is not available, disable
        warn(_gdal_warn, stacklevel=1)
        env.BIPL_DRIVERS.discard('gdal')
        os.environ['BIPL_DRIVERS'] = str([*env.BIPL_DRIVERS]).replace("'", '"')

_drv: type[Driver] | None
for _drv, _regex in [  # LIFO, last driver takes priority
    (Gdal, r'^.*\.(tif|tiff)$'),
    (Openslide, r'^.*\.(bif|mrxs|ndpi|scn|svs|svsslide|tif|tiff|vms|vmu)$'),
    (Tiff, r'^.*\.(svs|tif|tiff)$'),
    (Gdal, r'^(/vsicurl.*|(http|https|ftp)://.*)$'),
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
    shape: Sequence[int] = slide.shape
    downsamples: Sequence[int] = slide.downsamples

    # Get numpy.ndarray
    image: np.ndarray = slide[:2048, :2048]
    ```
    """
    # TODO: check if memory leak
    # TODO: add __enter__/__exit__/close to make memory management explicit
    # TODO: call .close in finalizer
    path: str
    shape: Sequence[int]
    downsamples: Sequence[int]
    mpp: float | None
    driver: str
    bbox: tuple = field(repr=False)
    levels: Sequence[ImageLevel] = field(repr=False)
    extras: Mapping[str, Image] = field(repr=False)
    dtype = np.dtype(np.uint8)

    @classmethod
    def from_file(
        cls,
        path: str,
        driver_fn: Callable[[str], Driver],
        mpp: float | None = None,
    ) -> 'Slide':
        driver = driver_fn(path)

        num_images = len(driver)
        if num_images == 0:
            raise ValueError('Empty file')

        im_0, *images = (driver[idx] for idx in range(num_images))
        if not isinstance(im_0, ImageLevel):
            raise TypeError('First pyramid layer is not tiled')

        # Retrieve all sub-images
        levels: dict[int, ImageLevel] = {1: im_0}
        extras: dict[str, Image] = {}
        for im in images:
            if isinstance(im, ImageLevel):
                downsample = round2(im_0.shape[0] / im.shape[0])
                levels[downsample] = im
            elif key := im.key:
                extras[key] = im
        extras |= driver.named_items()

        level_downsamples = *sorted(levels.keys()),
        # TODO: make virtual levels if downsamples are too distant (ProxyLevel)

        if mpp is None:  # If no override is passed, use native if present
            mpp = levels[1].mpp

        return Slide(
            path=path,
            shape=levels[1].shape,
            mpp=mpp,
            downsamples=level_downsamples,
            driver=type(driver).__name__,
            bbox=driver.bbox,
            levels=tuple(levels[zoom] for zoom in level_downsamples),
            extras=extras,
        )

    def icc(self) -> 'Slide':
        if icc_0 := self.levels[0].icc:
            levels = *(il.apply(icc_0) for il in self.levels),
            self = replace(self, levels=levels)

        if any(e.icc for e in self.extras.values()):
            extras = {
                t: e.apply(e.icc) if e.icc else e
                for t, e in self.extras.items()
            }
            self = replace(self, extras=extras)

        return self

    def clahe(self) -> 'Slide':
        levels = *(il.apply(clahe, pad=64) for il in self.levels),
        extras = {k: i.apply(clahe) for k, i in self.extras.items()}
        return replace(self, levels=levels, extras=extras)

    def tonemap(self, icc: bool, clahe: bool) -> 'Slide':
        r = self
        if icc:
            r = r.icc()
        if clahe:
            r = r.clahe()
        return r

    def mpp_or_error(self) -> float:
        if self.mpp is None:
            raise ValueError('Slide`s MPP is unknown')
        return self.mpp

    @property
    def pools(self) -> Sequence[int]:
        warnings.warn(
            '"Slide.pools" is deprecated. Use "Slide.downsamples"',
            category=DeprecationWarning,
            stacklevel=2)
        return self.downsamples

    @property
    def spacing(self) -> float | None:
        warnings.warn(
            '"Slide.spacing" is deprecated. Use "Slide.mpp"',
            category=DeprecationWarning,
            stacklevel=2)
        return self.mpp

    def __reduce__(self) -> tuple:
        return Slide.open, (self.path, )

    def best_level_for(
        self,
        downsample: float,
        /,
        *,
        tol: float = 0.01,
    ) -> tuple[int, ImageLevel]:
        """Gives the most detailed LOD below `downsample`"""
        downsample = downsample * max(1, 1 + tol)
        idx = max(bisect_right(self.downsamples, downsample) - 1, 0)
        return self.downsamples[idx], self.levels[idx]

    def resample(self, mpp: float, *, tol: float = 0.01) -> ImageLevel:
        """Resample slide to specific resolution"""
        downsample = mpp / self.mpp_or_error()
        return self.pool(downsample, tol=tol)

    def pool(self, downsample: float, /, *, tol: float = 0.01) -> ImageLevel:
        """Use like `slide.pool(4)[y0:y1, x0:x1]` call"""
        d, lvl = self.best_level_for(downsample, tol=tol)
        if d == downsample:
            return lvl
        return lvl.rescale(d / downsample)

    def __getitem__(self, key: slice | tuple[slice, ...]) -> np.ndarray:
        """Retrieve image patch from maximum resolution"""
        y_loc, x_loc, c_loc = normalize_loc(key, self.shape)
        if y_loc.step <= 0 or x_loc.step <= 0:
            raise ValueError('slice steps should be positive')

        r = self.levels[0].crop(
            slice(y_loc.start, y_loc.stop),
            slice(x_loc.start, x_loc.stop),
        )
        return r[::y_loc.step, ::x_loc.step, c_loc]

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
           scale: float | None = None,
           mpp: float | None = None,
           tol: float = 0.01) -> np.ndarray:
        """Read square region starting with offset"""
        dsize = dsize if isinstance(dsize, tuple) else (dsize, dsize)
        if len(dsize) != 2:
            raise ValueError(f'dsize should be 2-tuple or int. Got {dsize}')

        if scale is None:
            if mpp is None:
                raise ValueError('Only one of zoom/mpp should be None')
            scale = self.mpp_or_error() / mpp

        ds, lvl = self.best_level_for(1 / scale, tol=tol)
        loc = *(slice(int(c / ds), int((c + size / scale) / ds))
                for c, size in zip(z0_yx_offset, dsize)),
        image = lvl.crop(*loc)
        return resize(image, dsize)

    def extra(self, name: str) -> np.ndarray | None:
        if im := self.extras.get(name):
            return im.numpy()
        return None

    def thumbnail(self) -> np.ndarray:
        return self.extras.get('thumbnail', self.levels[-1]).numpy()

    @classmethod
    def open(cls,
             anypath: Path | str,
             /,
             *,
             icc: bool = env.BIPL_ICC,
             clahe: bool = env.BIPL_CLAHE) -> 'Slide':
        """Open multi-scale image."""
        if isinstance(anypath, Path):  # Filesystem
            anypath = anypath.resolve().absolute().as_posix()
        return _cached_open(anypath).tonemap(icc=icc, clahe=clahe)
