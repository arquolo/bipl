__all__ = ['Slide']

import importlib
import os
import warnings
from bisect import bisect_right
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, final, overload
from warnings import warn

import numpy as np
from glow import memoize

from bipl import cov, env
from bipl._types import Patch, Shape, Span, Vec
from bipl.ops import Normalizer, normalize_loc, rescale_crop

from ._slide_bases import Driver, Image, ImageLevel, PartMixin

# TODO: inside Slide.open import ._slide.registry,
# TODO: and in ._slide.registry do registration and DLL loading
# TODO: to make Slide export not require DLL presence


def _load_drivers() -> None:
    if os.name == 'nt':
        gdal_warn = (
            'Ensure it was properly installed or acquire it manually from '
            'https://github.com/cgohlke/geospatial-wheels/releases'
        )
    else:
        gdal_warn = (
            'Ensure it was properly installed, and run '
            '"pip install wheel && pip install gdal==`gdal-config --version`"'
        )

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
        ('gdal', r'^.*\.tiff?$'),
        ('openslide', r'^.*\.(bif|mrxs|ndpi|scn|svs(|slide)|tiff?|vms|vmu)$'),
        ('tiff', r'^.*\.(bif|ndpi|svs|tiff?)$'),
        ('gdal', r'^(/vsicurl|(https?|ftp)://).*$'),
    ]:
        if drv := drivers.get(drvname):
            drv.register(pat)


_load_drivers()


# Merge duplicate calls
# Reuse result if it's already exist, but used by someone else
# Keep LRU for unused results
@memoize(env.BIPL_CACHE, policy='lru')
def _cached_open(path: str) -> 'Slide':
    if tps := Driver.find(path):
        errors: list[Exception] = []
        for tp in reversed(tps):  # Loop over types to find non-failing
            try:
                return Slide.from_file(path, tp)
            except (ValueError, TypeError) as exc:
                errors.append(exc)
        raise ExceptionGroup('Cannot open file', errors) from None

    raise ValueError(f'Unknown file format {path}')


@final
@dataclass(frozen=True)
class Slide(PartMixin):
    """Usage:
    ```
    slide = Slide.open('test.svs')
    shape: tuple[int, ...] = slide.shape
    downsamples: tuple[int, ...] = slide.downsamples

    # Get numpy.ndarray
    image: np.ndarray = slide[:2048, :2048]
    ```
    """

    # TODO: check if memory leak
    # TODO: add __enter__/__exit__/close to make memory management explicit
    # TODO: call .close in finalizer
    path: str
    shape: Shape
    downsamples: tuple[int, ...]
    mpp: float | None
    driver: str
    bbox: tuple[slice, ...] = field(repr=False)
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
        levels: list[ImageLevel] = []
        extras: dict[str, Image] = {}
        for idx in range(num_images):
            match driver[idx]:
                case ImageLevel() as im:
                    levels.append(im)
                case Image(key=str(key)) as im:
                    extras[key] = im
                case _:
                    continue
        extras |= driver.named_items()

        downsamples, levels = driver.build_pyramid(levels)

        if mpp is None:  # If no override is passed, use native if present
            mpp = driver.get_mpp()

        if mpp is not None and env.BIPL_MPP_Q:
            mpp = 2 ** (round(np.log2(mpp) * env.BIPL_MPP_Q) / env.BIPL_MPP_Q)

        return Slide(
            path=path,
            shape=levels[0].shape,
            mpp=mpp,
            downsamples=downsamples,
            driver=type(driver).__name__,
            bbox=driver.bbox,
            levels=levels,
            extras=extras,
        )

    def icc(self) -> 'Slide':
        changes = {}
        if icc_0 := self.levels[0].icc:
            changes['levels'] = tuple(il.apply(icc_0) for il in self.levels)
        if any(e.icc for e in self.extras.values()):
            changes['extras'] = {
                t: e.apply(e.icc) if e.icc else e
                for t, e in self.extras.items()
            }
        return replace(self, **changes) if changes else self

    def norm(
        self,
        mpp: float = 64,
        weight: float = 1.0,
        channels: Literal['L', 'Lab', 'ab'] = 'ab',
        r: float = 0.5,
    ) -> 'Slide':
        ref = self.resample(mpp=mpp).numpy()
        normalize = Normalizer(ref, weight=weight, channels=channels, r=r)

        levels = tuple(il.apply(normalize) for il in self.levels)
        extras = {
            k: i.apply(normalize) if k == 'thumbnail' else i
            for k, i in self.extras.items()
        }
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
            stacklevel=2,
        )
        return self.downsamples

    @property
    def spacing(self) -> float | None:
        warnings.warn(
            '"Slide.spacing" is deprecated. Use "Slide.mpp"',
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.mpp

    def __reduce__(self) -> tuple:
        return Slide.open, (self.path,)

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

        # Make 2^k level as close as possible to target
        return lv.decimate(downsample, ds)

    def resample(self, mpp: float, *, tol: float = 0.01) -> ImageLevel:
        """Resample slide to specific resolution"""
        downsample = mpp / self.mpp_or_error()
        return self.pool(downsample, tol=tol)

    def pool(self, downsample: float, /, *, tol: float = 0.01) -> ImageLevel:
        """Use like `slide.pool(4)[y0:y1, x0:x1]` call"""
        ds, lvl = self.best_level_for(downsample, tol=tol)
        if ds == downsample:
            return lvl
        return lvl.rescale(ds / downsample)

    def __getitem__(self, key: slice | tuple[slice, ...]) -> np.ndarray:
        """Retrieve image patch from maximum resolution"""
        y_loc, x_loc, (c_lo, c_hi) = normalize_loc(key, self.shape)
        return self.part(y_loc, x_loc)[:, :, c_lo:c_hi]

    def parts(
        self, locs: Sequence[tuple[Span, ...]], max_workers: int = 0
    ) -> Iterator[Patch]:
        return self.levels[0].parts(locs, max_workers=max_workers)

    def numpy(self) -> np.ndarray:
        return self.levels[0].numpy()

    @overload
    def at(
        self,
        z0_yx_offset: Vec,
        dsize: int | Shape,
        *,
        scale: float,
        tol: float = ...,
    ) -> np.ndarray: ...

    @overload
    def at(
        self,
        z0_yx_offset: Vec,
        dsize: int | Shape,
        *,
        mpp: float,
        tol: float = ...,
    ) -> np.ndarray: ...

    def at(
        self,
        z0_yx_offset: Vec,
        dsize: int | Shape,
        *,
        scale: float | None = None,
        mpp: float | None = None,
        tol: float = 0.01,
    ) -> np.ndarray:
        """Read square region starting with offset"""
        dsize = dsize if isinstance(dsize, Sequence) else (dsize, dsize)
        if len(dsize) != 2:
            raise ValueError(f'dsize should be 2-tuple or int. Got {dsize}')

        if scale is None:
            if mpp is None:
                raise ValueError('Only one of zoom/mpp should be None')
            scale = self.mpp_or_error() / mpp

        cov.update(self.path, z0_yx_offset, dsize, scale)
        ds, lvl = self.best_level_for(1 / scale, tol=tol)

        yx_offset = tuple(int(c * scale) for c in z0_yx_offset)
        loc = tuple((c, c + size) for c, size in zip(yx_offset, dsize))
        return rescale_crop(lvl, *loc, scale=1 / ds / scale, interpolation=1)

    def extra(self, name: str) -> np.ndarray | None:
        if im := self.extras.get(name):
            return im.numpy()
        return None

    def thumbnail(self) -> np.ndarray:
        return self.extras.get('thumbnail', self.levels[-1]).numpy()

    @classmethod
    def open(
        cls,
        anypath: Path | str,
        /,
        *,
        icc: bool = env.BIPL_ICC,
        norm: bool | float = env.BIPL_NORM,
    ) -> 'Slide':
        """Open multi-scale image."""
        if isinstance(anypath, Path):  # Filesystem
            anypath = anypath.resolve().absolute().as_posix()
        s = _cached_open(anypath)
        if icc:
            s = s.icc()
        if not norm:
            return s
        return s.norm() if norm is True else s.norm(mpp=norm)
