"""
Driver based on OpenSlide
- slow
- thread-safe
- compatible with formats: tiff/tif/svs, ndpi/vms/vmu, scn, mrxs, svsslide, bif
"""
# TODO: handle viewport offsets
from __future__ import annotations

__all__ = ['Openslide']

import weakref
from collections.abc import Iterator
from ctypes import (POINTER, addressof, byref, c_char_p, c_double, c_int64,
                    c_ubyte, c_void_p)
from dataclasses import dataclass

import cv2
import numpy as np

from ._libs import load_library
from ._slide_bases import Driver, Item, Lod
from ._util import unflatten

OSD = load_library('libopenslide', 0)

sptr = POINTER(c_ubyte)
OSD.openslide_open.restype = sptr
OSD.openslide_close.argtypes = [sptr]

OSD.openslide_get_error.restype \
    = OSD.openslide_get_property_value.restype \
    = c_char_p

OSD.openslide_get_property_names.restype \
    = OSD.openslide_get_associated_image_names.restype \
    = POINTER(c_char_p)

OSD.openslide_get_level_downsample.restype = c_double
OSD.openslide_read_associated_image.argtypes = [sptr, c_char_p, c_void_p]
OSD.openslide_read_region.argtypes = [
    sptr, c_void_p, c_int64, c_int64, c_int64, c_int64, c_int64
]


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


@dataclass(frozen=True)
class _Item(Item):
    name: bytes
    osd: Openslide

    @property
    def key(self) -> str:
        return self.name.decode()

    def __array__(self) -> np.ndarray:
        bgra = np.empty((*self.shape[:2], 4), 'u1')
        OSD.openslide_read_associated_image(self.osd.ptr, self.name,
                                            c_void_p(bgra.ctypes.data))
        rgb = _mbgra_to_rgb(bgra, self.osd.bg_color)
        return np.ascontiguousarray(rgb)


@dataclass(frozen=True)
class _Lod(Lod):
    pool: int
    index: int
    osd: Openslide

    def crop(self, slices: tuple[slice, ...]) -> np.ndarray:
        box, valid_box, shape = self._unpack_loc(slices)

        (y0, y1), (x0, x1) = valid_box
        if y0 == y1 or x0 == x1:  # Patch is outside slide
            return np.broadcast_to(self.osd.bg_color, (*shape, 3))

        bgra = np.empty((y1 - y0, x1 - x0, 4), dtype='u1')
        OSD.openslide_read_region(
            self.osd.ptr,
            c_void_p(bgra.ctypes.data),
            int(x0 * self.pool),
            int(y0 * self.pool),
            self.index,
            int(x1 - x0),
            int(y1 - y0),
        )
        rgb = _mbgra_to_rgb(bgra, self.osd.bg_color)

        return self._expand(rgb, valid_box, box, self.osd.bg_color)


class Openslide(Driver):
    def __init__(self, path: str):
        self.ptr = OSD.openslide_open(path.encode())
        if not self.ptr:
            raise ValueError(f'File {path} cannot be opened')

        if err := OSD.openslide_get_error(self.ptr):
            raise ValueError(err)
        weakref.finalize(self, OSD.openslide_close, self.ptr)

        meta = self._get_metadata()
        self.meta = unflatten(meta)
        self.osd_meta: dict = self.meta.get('openslide', {})

        bg_hex = self.osd_meta.get('background-color', 'FFFFFF')
        self.bg_color: np.ndarray = np.frombuffer(bytes.fromhex(bg_hex), 'u1')

        self.spacing = self._spacing()

    def _spacing(self) -> float | None:
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
        return {
            name.decode(): value.decode() for name in _ntas_to_iter(names)
            if (value := OSD.openslide_get_property_value(self.ptr, name))
        }

    def __len__(self) -> int:
        return OSD.openslide_get_level_count(self.ptr)

    def __getitem__(self, index: int) -> Lod:
        h, w = c_int64(), c_int64()
        OSD.openslide_get_level_dimensions(self.ptr, index, byref(w), byref(h))
        pool: float = OSD.openslide_get_level_downsample(self.ptr, index)
        if pool <= 0:
            raise ValueError(f'invalid file, got level downsample: {pool}')

        spacing = self.spacing if index == 0 else None
        return _Lod((h.value, w.value, 3), spacing, round(pool), index, self)

    def keys(self) -> list[str]:
        names = OSD.openslide_get_associated_image_names(self.ptr)
        return [name.decode() for name in _ntas_to_iter(names)]

    def get(self, key: str) -> Item:
        name = key.encode()
        w, h = c_int64(), c_int64()
        OSD.openslide_get_associated_image_dimensions(self.ptr, name, byref(w),
                                                      byref(h))
        return _Item((h.value, w.value, 3), name, self)

    @property
    def bbox(self) -> tuple[slice, slice]:
        bbox = (
            self.osd_meta.get(f'bounds-{t}')
            for t in ('y', 'x', 'height', 'width'))
        y0, x0, h, w = (
            int(s) if s and s.strip().isdigit() else None for s in bbox)
        y1, x1 = (o + s if o is not None and s is not None else None
                  for o, s in [(y0, h), (x0, w)])
        return slice(y0, y1), slice(x0, x1)
