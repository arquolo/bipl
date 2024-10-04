__all__ = ['Slide']

import importlib
import os
import sys
import warnings
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import final, overload
from warnings import warn

import numpy as np
from glow import memoize, shared_call, weak_memoize

from bipl import env
from bipl.ops import Normalizer, normalize_loc, rescale_crop

from ._slide_bases import Driver, Image, ImageLevel
from ._util import clahe, round2

# TODO: inside Slide.open import ._slide.registry,
# TODO: and in ._slide.registry do registration and DLL loading
# TODO: to make Slide export not require DLL presence

_PY312 = sys.version_info >= (3, 12)


def _load_drivers() -> None:
    if os.name == 'nt':
        gdal_warn = (
            'Ensure it was properly installed or acquire it manually from '
            'https://github.com/cgohlke/geospatial-wheels/releases')
    else:
        gdal_warn = (
            'Ensure it was properly installed, and run '
            '"pip install wheel && pip install gdal==`gdal-config --version`"')

    imports = {
        'openslide': ('._openslide', 'Openslide', ''),
        'tiff': ('._tiff', 'Tiff', ''),
        'gdal': ('._gdal', 'Gdal', gdal_warn),
    }
    drivers: dict[str, type[Driver]] = {}
    for drvname in [*env.BIPL_DRIVERS]:
        # Find
        if (imp := imports.get(drvname)) is None:
            raise ValueError(f'Unknown driver: {drvname}')

        # Import
        modname, attrname, extramsg = imp
        try:
            mod = importlib.import_module(modname, __package__)
        except ImportError as exc:
            msg = f'"{drvname}" driver failed to load ({exc}). {extramsg}'
            warn(msg, stacklevel=2)
            env.BIPL_DRIVERS.discard(drvname)
        else:
            # Get driver
            drivers[drvname] = getattr(mod, attrname)

    os.environ['BIPL_DRIVERS'] = str([*env.BIPL_DRIVERS]).replace("'", '"')
    if not drivers:
        raise ImportError('No drivers loaded')

    for drvname, pat in [  # LIFO, last driver takes priority
        ('gdal', r'^.*\.(tif|tiff)$'),
        ('openslide',
         r'^.*\.(bif|mrxs|ndpi|scn|svs|svsslide|tif|tiff|vms|vmu)$'),
        ('tiff', r'^.*\.(bif|svs|tif|tiff)$'),
        ('gdal', r'^(/vsicurl|(http|https|ftp)://).*$'),
    ]:
        if drv := drivers.get(drvname):
            drv.register(pat)


_load_drivers()


@shared_call  # merge duplicate calls
@weak_memoize  # reuse result if it's already exist, but used by someone else
@memoize(capacity=env.BIPL_CACHE, policy='lru')  # keep LRU for unused results
def _cached_open(path: str) -> 'Slide':
    if tps := Driver.find(path):
        errors: list[Exception] = []
        for tp in reversed(tps):  # Loop over types to find non-failing
            try:
                return Slide.from_file(path, tp)
            except (ValueError, TypeError) as exc:
                errors.append(exc)
        if _PY312:
            raise ExceptionGroup('Cannot open file', errors) from None
        raise errors[-1] from None

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

        # Retrieve all sub-images
        levels: dict[int, ImageLevel] = {}
        extras: dict[str, Image] = {}
        for idx in range(num_images):
            match driver[idx]:
                case ImageLevel(shape=shape) as im:
                    if levels:
                        ds = round2(levels[1].shape[0] / shape[0])
                        levels[ds] = im.fallback(levels[1], ds)
                    else:
                        levels[1] = im
                case Image(key=str(key)) as im:
                    extras[key] = im
                case _:
                    continue

        extras |= driver.named_items()
        if not levels:
            raise TypeError('No tiled layers present')

        level_downsamples = *sorted(levels.keys()),

        if mpp is None:  # If no override is passed, use native if present
            mpp = driver.get_mpp()

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
        changes = {}
        if icc_0 := self.levels[0].icc:
            changes['levels'] = *(il.apply(icc_0) for il in self.levels),
        if any(e.icc for e in self.extras.values()):
            changes['extras'] = {
                t: e.apply(e.icc) if e.icc else e
                for t, e in self.extras.items()
            }
        return replace(self, **changes) if changes else self

    def norm(self, mpp: float = 64) -> 'Slide':
        ref = self.resample(mpp=mpp).numpy()
        normalize = Normalizer(ref)

        levels = *(il.apply(normalize) for il in self.levels),
        extras = {
            k: i.apply(normalize) if k == 'thumbnail' else i
            for k, i in self.extras.items()
        }
        return replace(self, levels=levels, extras=extras)

    def clahe(self) -> 'Slide':
        levels = *(il.apply(clahe, pad=64) for il in self.levels),
        extras = {k: i.apply(clahe) for k, i in self.extras.items()}
        return replace(self, levels=levels, extras=extras)

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
        k = max(1, 1 + tol)
        idx = max(bisect_right(self.downsamples, downsample * k) - 1, 0)
        ds = self.downsamples[idx]
        lv = self.levels[idx]

        # Make octave level as close as possible to target
        while (ds_ := ds * 2) < downsample and (lv_ := lv.octave()):
            ds, lv = ds_, lv_

        return ds, lv

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

        yx_offset = *(int(c * scale) for c in z0_yx_offset),
        loc = *(slice(c, c + size) for c, size in zip(yx_offset, dsize)),
        return rescale_crop(lvl, *loc, scale=1 / ds / scale, interpolation=1)

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
             norm: bool | float = env.BIPL_NORM) -> 'Slide':
        """Open multi-scale image."""
        if isinstance(anypath, Path):  # Filesystem
            anypath = anypath.resolve().absolute().as_posix()
        s = _cached_open(anypath)
        if icc:
            s = s.icc()
        if not norm:
            return s
        return s.norm() if norm is True else s.norm(mpp=norm)
