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
from pathlib import Path

import cv2
import numpy as np

from ._libs import load_library
from ._slide_bases import Driver, Item, Lod

OSD = load_library('libopenslide', 0)

OSD.openslide_open.restype = POINTER(c_ubyte)
OSD.openslide_get_error.restype \
    = OSD.openslide_get_property_value.restype \
    = c_char_p

OSD.openslide_get_property_names.restype \
    = OSD.openslide_get_associated_image_names.restype \
    = POINTER(c_char_p)

OSD.openslide_get_level_downsample.restype = c_double


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

    def get_key(self) -> str:
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
        (y_min2, y_max2), (x_min2, x_max2) = box_ = [[s.start, s.stop]
                                                     for s in slices]
        box = np.array(box_)
        valid_box = box.T.clip([0, 0], self.shape[:2]).T

        (y_min, y_max), (x_min, x_max) = valid_box
        if y_min == y_max or x_min == x_max:  # Patch is outside slide
            return np.broadcast_to(self.osd.bg_color,
                                   (y_max2 - y_min2, x_max2 - x_min2, 3))

        bgra = np.empty((y_max - y_min, x_max - x_min, 4), dtype='u1')
        OSD.openslide_read_region(
            self.osd.ptr,
            c_void_p(bgra.ctypes.data),
            int(x_min * self.pool),
            int(y_min * self.pool),
            self.index,
            int(x_max - x_min),
            int(y_max - y_min),
        )

        rgb = _mbgra_to_rgb(bgra, self.osd.bg_color)

        offsets = np.abs(valid_box - box).ravel().tolist()
        if any(offsets):
            bg_color = self.osd.bg_color.tolist()
            return cv2.copyMakeBorder(
                rgb, *offsets, borderType=cv2.BORDER_CONSTANT, value=bg_color)

        return np.ascontiguousarray(rgb)


class Openslide(Driver):
    def __init__(self, path: Path):
        self.ptr = OSD.openslide_open(path.as_posix().encode())
        if not self.ptr:
            raise ValueError(f'File {path} cannot be opened')

        if err := OSD.openslide_get_error(self.ptr):
            raise ValueError(err)
        weakref.finalize(self, OSD.openslide_close, self.ptr)

        self._tags = self._get_metadata()

        bg_hex = self._tags.get('openslide.background-color', 'FFFFFF')
        self.bg_color: np.ndarray = np.frombuffer(bytes.fromhex(bg_hex), 'u1')

        self.spacing = None
        mpp = (self._tags.get(f'openslide.mpp-{ax}') for ax in 'yx')
        if s := [float(m) for m in mpp if m]:
            self.spacing = float(np.mean(s))

        # TODO: handle offsets
        # self.offset = *(int(self._tags.get(f'openslide.bounds-{ax}', 0))
        #                 for ax in 'yx'),
        # self.size = *(int(self._tags.get(f'openslide.bounds-{ax}', lim))
        #               for ax, lim in zip(('height', 'width'), self.shape)),

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
        pool = OSD.openslide_get_level_downsample(self.ptr, index)
        assert pool > 0

        return _Lod((h.value, w.value, 3), self.spacing, round(pool), index,
                    self)

    def keys(self) -> list[str]:
        names = OSD.openslide_get_associated_image_names(self.ptr)
        return [name.decode() for name in _ntas_to_iter(names)]

    def get(self, key: str) -> Item:
        name = key.encode()
        w, h = c_int64(), c_int64()
        OSD.openslide_get_associated_image_dimensions(self.ptr, name, byref(w),
                                                      byref(h))
        return _Item((h.value, w.value, 3), name, self)
