"""
Driver based on OpenSlide
- slow
- thread-safe
- compatible with formats: tiff/tif/svs, ndpi/vms/vmu, scn, mrxs, svsslide, bif
- relies on tile caching

To adjust tile cache size use BIPL_TILE_CACHE environment variable.
"""

# TODO: handle viewport offsets

__all__ = ['Openslide']

import atexit
import weakref
from collections.abc import Iterator
from ctypes import (
    POINTER,
    addressof,
    byref,
    c_char_p,
    c_double,
    c_int32,
    c_int64,
    c_ubyte,
    c_uint64,
    c_void_p,
)
from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np
from packaging.version import Version

from bipl._env import env
from bipl._types import Span

from ._libs import load_library
from ._slide_bases import Driver, Image, ImageLevel, PartsMixin
from ._util import Icc, round2, unflatten

OSD = load_library('libopenslide', 1, 0)

sptr = POINTER(c_ubyte)
OSD.openslide_open.restype = sptr
OSD.openslide_close.argtypes = [sptr]

OSD.openslide_get_version.restype = c_char_p
OSD.openslide_get_error.restype = c_char_p
OSD.openslide_get_property_value.restype = c_char_p

OSD.openslide_get_property_names.restype = POINTER(c_char_p)
OSD.openslide_get_associated_image_names.restype = POINTER(c_char_p)

OSD.openslide_get_level_dimensions.argtypes = [
    sptr,  # openslide_t
    c_int32,  # level
    POINTER(c_int64),  # width
    POINTER(c_int64),  # height
]

OSD.openslide_get_level_downsample.argtypes = [sptr, c_int32]
OSD.openslide_get_level_downsample.restype = c_double
OSD.openslide_read_associated_image.argtypes = [sptr, c_char_p, c_void_p]

OSD.openslide_read_region.argtypes = [
    sptr,  # openslide_t
    c_void_p,  # buffer
    c_int64,  # x0
    c_int64,  # y1
    c_int32,  # level
    c_int64,  # width
    c_int64,  # height
]

_VERSION = None
_OSD_4X = False
if _version_bytes := OSD.openslide_get_version():
    _VERSION = Version(_version_bytes.decode())
    _OSD_4X = _VERSION >= Version('4.0')  # noqa: SIM300


def _init_4x_icc():
    if not _OSD_4X:
        return
    OSD.openslide_get_icc_profile_size.argtypes = [sptr]
    OSD.openslide_get_icc_profile_size.restype = c_int64
    OSD.openslide_read_icc_profile.argtypes = [sptr, c_void_p]

    OSD.openslide_get_associated_image_icc_profile_size.argtypes = [
        sptr,  # openslide_t
        c_char_p,  # name
    ]
    OSD.openslide_get_associated_image_icc_profile_size.restype = c_int64
    OSD.openslide_read_associated_image_icc_profile.argtypes = [
        sptr,  # openslide_t
        c_char_p,  # name
        c_void_p,  # buffer
    ]


def _init_4x_cache():
    if not _OSD_4X:
        return None
    OSD.openslide_cache_create.argtypes = [c_uint64]
    OSD.openslide_cache_create.restype = sptr
    OSD.openslide_cache_release.argtypes = [sptr]
    OSD.openslide_set_cache.argtypes = [sptr, sptr]

    cache = OSD.openslide_cache_create(env.BIPL_TILE_CACHE)
    atexit.register(OSD.openslide_cache_release, cache)
    return cache


_init_4x_icc()
_CACHE = _init_4x_cache()


def _ntas_to_iter(null_terminated_array_of_strings) -> Iterator[bytes]:
    if not null_terminated_array_of_strings:
        return
    i = 0
    while s := null_terminated_array_of_strings[i]:
        yield s
        i += 1


_LUT_CA = (np.outer(np.arange(256), np.arange(256)) // 255).astype('u1')


def _mbgra_to_rgb(bgra: np.ndarray, rgb_base: np.ndarray) -> np.ndarray:
    """
    Pre-multiplied BGRA to RGB. Alpha blending done with background.
    I.e. `RGB_color = mRGBA_color + (1 - alpha) x background`
    """
    b, g, r, a = cv2.split(bgra)
    if (ia := ~a).any():
        # rgb[:] += rgb_base * ia // 255
        for dst, lut in zip([r, g, b], _LUT_CA[rgb_base]):
            dst[:] += cv2.LUT(ia, lut)
    return cv2.merge([r, g, b])


# def _mbgra_to_rgb(bgra: np.ndarray, rgb_base: np.ndarray) -> np.ndarray:
#     """
#     Pre-multiplied BGRA to RGB. Only image colors are kept.
#     I.e. `RGB_color = alpha ? (mRGBA_color / alpha) : background`
#     """
#     bgra = cv2.cvtColor(bgra, cv2.COLOR_mRGBA2RGBA)  # alpha unscale
#     rgb, a = bgra[..., 2::-1], bgra[..., 3]
#     rgb[a == 0] = rgb_base  # fill fully transparent pixels
#     return rgb


@dataclass(frozen=True, kw_only=True)
class _Image(Image):
    name: bytes
    osd: 'Openslide'

    def numpy(self) -> np.ndarray:
        bgra = np.empty((*self.shape[:2], 4), 'u1')
        OSD.openslide_read_associated_image(
            self.osd.ptr, self.name, c_void_p(bgra.ctypes.data)
        )
        rgb = _mbgra_to_rgb(bgra, self.osd.bg_color)
        rgb = np.ascontiguousarray(rgb)
        return self._postprocess(rgb)

    @property
    def icc(self) -> Icc | None:
        if not _OSD_4X:
            raise NotImplementedError(
                f'Openslide 4.x is required. Got {_VERSION}'
            )

        size = OSD.openslide_get_associated_image_icc_profile_size(
            self.osd.ptr, self.name
        )
        if not size:
            return None

        buf = np.empty(size, 'u1')
        OSD.openslide_read_associated_image_icc_profile(
            self.osd.ptr, self.name, c_void_p(buf.ctypes.data)
        )
        return Icc(buf.tobytes())


@dataclass(frozen=True, kw_only=True)
class _Level(PartsMixin, ImageLevel):
    downsample: int
    index: int
    osd: 'Openslide'

    def part(self, *loc: Span) -> np.ndarray:
        box, valid_box, shape = self._unpack_2d_loc(*loc)

        (y0, y1), (x0, x1) = valid_box
        if y0 == y1 or x0 == x1:  # Patch is outside slide
            return np.broadcast_to(self.osd.bg_color, (*shape, 3))

        bgra = np.empty((y1 - y0, x1 - x0, 4), dtype='u1')
        OSD.openslide_read_region(
            self.osd.ptr,
            c_void_p(bgra.ctypes.data),
            int(x0 * self.downsample),
            int(y0 * self.downsample),
            self.index,
            int(x1 - x0),
            int(y1 - y0),
        )
        rgb = _mbgra_to_rgb(bgra, self.osd.bg_color)

        rgb = self._expand(rgb, valid_box, box, self.osd.bg_color)
        return self._postprocess(rgb)

    @property
    def icc(self) -> Icc | None:
        return self.osd.icc


class Openslide(Driver):
    def __init__(self, path: str) -> None:
        self.ptr = OSD.openslide_open(path.encode())
        if not self.ptr:
            raise ValueError('libopenslide failed to open file')

        if err := OSD.openslide_get_error(self.ptr):
            raise ValueError(err)
        weakref.finalize(self, OSD.openslide_close, self.ptr)

        if _CACHE is not None:  # Use global tile cache for all slides
            OSD.openslide_set_cache(self.ptr, _CACHE)

        meta = self._get_metadata()
        self.meta = unflatten(meta)
        self.osd_meta: dict = self.meta.get('openslide', {})

        bg_hex = self.osd_meta.get('background-color', 'FFFFFF')
        self.bg_color: np.ndarray = np.frombuffer(bytes.fromhex(bg_hex), 'u1')

    def get_mpp(self) -> float | None:
        mpp = (self.osd_meta.get(f'mpp-{t}') for t in ('y', 'x'))
        if s := [float(m) for m in mpp if m]:
            return float(np.mean(s))

        tiff_meta = self.meta.get('tiff', {})
        unit: str = tiff_meta.get('ResolutionUnit', '')
        unit_base = {'centimeter': 10_000, 'inch': 2_540}.get(unit)
        if unit_base is None:
            return None

        resolution = (tiff_meta.get(f'{t}Resolution') for t in ('Y', 'X'))
        if s := [(unit_base / float(v)) for v in resolution if v]:
            return float(np.mean(s))

        return None

    def __repr__(self) -> str:
        return f'{type(self).__name__}({addressof(self.ptr.contents):0x})'

    def _get_metadata(self) -> dict[str, str]:
        names = OSD.openslide_get_property_names(self.ptr)
        r = {}
        for name in _ntas_to_iter(names):
            if value := OSD.openslide_get_property_value(self.ptr, name):
                try:
                    r[name.decode()] = value.decode()
                except UnicodeDecodeError:
                    continue
        return r

    def __len__(self) -> int:
        return OSD.openslide_get_level_count(self.ptr)

    def __getitem__(self, index: int) -> _Level:
        h, w = c_int64(), c_int64()
        OSD.openslide_get_level_dimensions(self.ptr, index, byref(w), byref(h))
        downsample = OSD.openslide_get_level_downsample(self.ptr, index)
        if downsample <= 0:
            raise ValueError(f'Invalid level downsample: {downsample}')

        return _Level(
            (h.value, w.value, 3),
            downsample=round2(downsample),
            index=index,
            osd=self,
        )

    def keys(self) -> list[str]:
        names = OSD.openslide_get_associated_image_names(self.ptr)
        return [name.decode() for name in _ntas_to_iter(names)]

    def get(self, key: str) -> _Image:
        name = key.encode()
        w, h = c_int64(), c_int64()
        OSD.openslide_get_associated_image_dimensions(
            self.ptr, name, byref(w), byref(h)
        )
        return _Image((h.value, w.value, 3), key=key, name=name, osd=self)

    @property
    def bbox(self) -> tuple[slice, ...]:
        bbox = (
            self.osd_meta.get(f'bounds-{t}')
            for t in ('y', 'x', 'height', 'width')
        )
        y0, x0, h, w = (
            int(s) if s and s.strip().isdigit() else None for s in bbox
        )
        y1, x1 = (
            o + s if o is not None and s is not None else None
            for o, s in [(y0, h), (x0, w)]
        )
        return slice(y0, y1), slice(x0, x1)

    @cached_property
    def icc(self) -> Icc | None:
        if not _OSD_4X:
            raise NotImplementedError(
                f'Openslide 4.x is required. Got {_VERSION}'
            )

        size = OSD.openslide_get_icc_profile_size(self.ptr)
        if not size:
            return None

        buf = np.empty(size, 'u1')
        OSD.openslide_read_icc_profile(self.ptr, c_void_p(buf.ctypes.data))
        return Icc(buf.tobytes())
