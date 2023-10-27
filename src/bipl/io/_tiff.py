"""
Driver based on libtiff
- fast
- not thread safe (internally)
- compatible with TIFF and its flavours
"""

__all__ = ['Tiff']

import ctypes
import sys
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import (POINTER, addressof, byref, c_char_p, c_float, c_int,
                    c_ubyte, c_uint16, c_uint32, c_uint64, c_void_p,
                    create_string_buffer, string_at)
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from threading import Lock
from typing import NamedTuple, Protocol, TypeVar

import cv2
import imagecodecs
import numpy as np

from bipl._env import env

from ._libs import load_library
from ._slide_bases import Driver, Image, ImageLevel
from ._util import Icc, is_aperio, parse_aperio_description

_T = TypeVar('_T')

TIFF = load_library('libtiff', 6, 5)
# _TIFF.TIFFSetErrorHandler(None)

(TIFF.TIFFOpenW if sys.platform == 'win32' else TIFF.TIFFOpen).restype \
    = POINTER(c_ubyte)


# ---------------------------- TIFF tags mappings ----------------------------
class _ColorSpace(Enum):
    MINISBLACK = 1
    RGB = 2
    YCBCR = 6


class _ColorInfo(NamedTuple):
    space: _ColorSpace
    subsampling: tuple[int, int]

    def to_rgb(self, x: np.ndarray) -> np.ndarray:
        if self.space is not _ColorSpace.YCBCR or self.subsampling != (2, 2):
            return x

        h, w = x.shape[:2]
        assert h % 2 == w % 2 == 0

        # (2 h/2) w 3 -> h/2 w 3
        # h/2 (w/2 2) 3 -> h/2 w/2 2 3 -> h/2 w/2 (2 3) -> h/2 w/2 6
        hw6 = x[:h // 2, :, :].reshape(h // 2, w // 2, 6)

        # y00, y01, y10, y11, cb, cr
        y = hw6[:, :, [[0, 1], [2, 3]]].transpose(0, 2, 1, 3).reshape(h, w)
        cb_cr = hw6[:, :, [4, 5]]
        return cv2.cvtColorTwoPlane(y, cb_cr, cv2.COLOR_YUV2RGB_NV12)


class _Compression(Enum):
    RAW = 1
    CCITT = 2
    CCITTFAX3 = 3
    CCITTFAX4 = 4
    LZW = 5
    JPEG_OLD = 6
    JPEG = 7
    ADOBE_DEFLATE = 8
    RAW_16 = 32771
    PACKBITS = 32773
    THUNDERSCAN = 32809
    DEFLATE = 32946
    DCS = 32947
    JPEG2000_YUV = 33003
    JPEG2000_RGB = 33005
    JBIG = 34661
    SGILOG = 34676
    SGILOG24 = 34677
    JPEG2000 = 34712
    LZMA = 34925
    ZSTD = 50000
    WEBP = 50001


class _Tag:  # noqa: PIE795,RUF100
    JPEG_TABLES = 347
    TILE_BYTE_COUNTS = 325
    BACKGROUND_COLOR = 434


class _Tags:
    def __init__(self, ptr):
        self._ptr = ptr
        compression, = self._get(c_uint16, 259)
        self.compression = _Compression(compression)

        self.spp, = self._get(c_uint16, 277)
        self.is_planar, = self._get(c_uint16, 284)
        self.image_size = self._get(c_uint32, 257, 256)
        self.tile_size = self._get(c_uint32, 323, 322)
        self.description = self._get_str(270)
        self.resolution = self._get(c_float, 283, 282)

        # ! crashes, but should work according to openslide docs
        # self.is_hamamatsu = bool(self._get(c_uint16, 65420))
        self.make = self._get_str(271)

        self.bps, = self._get(c_uint16, 339) or [1]

        ics, = self._get(c_uint16, 262)
        colorspace = _ColorSpace(ics)
        subsampling = (1, 1)

        # TODO: use this in YCbCr conversion
        self.gray = np.array([], 'f4')  # Luma coefficients
        self.yuv_centered = True
        self.yuv_bw = np.array([], 'f4')  # BW pairs, per channel

        if colorspace is _ColorSpace.YCBCR:
            self.gray = np.array([.299, .587, .114], 'f4')
            gray_ptr = POINTER(c_float)()
            if TIFF.TIFFGetField(self._ptr, 529, byref(gray_ptr)):
                self.gray = np.ctypeslib.as_array(gray_ptr, [3]).copy()

            # YCbCr subsampling H/W, i.e:
            # 24bps:
            #  (1 1) = 4:4:4 (cccc/cccc), 100% H, 100% W
            # 16bps:
            #  (1 2) = 4:2:2 (c_c_/c_c_), 100% H, 50% W
            #  (2 1) = 4:4:0 (cccc/____), 50% H, 100% W
            # 12bps:
            #  (1 4) = 4:1:1 (c___/c___), 100% H, 25% W
            #  (2 2) = 4:2:0 (c_c_/____), 50% H, 50% W - a.k.a. YUV-NV12
            # 10bps:
            #  (2 4) = 4:1:0 (c___/____), 50% H, 25% W
            ss_w, ss_h = self._get_varargs((c_uint16, c_uint16), 530) or (2, 2)
            subsampling = ss_h, ss_w

            self.yuv_centered = 2 not in self._get(c_uint16, 531)

            bw_ptr = POINTER(c_float)()
            if TIFF.TIFFGetField(self._ptr, 532, byref(bw_ptr)):
                self.yuv_bw = np.ctypeslib.as_array(bw_ptr, [3, 2])

        self.color = _ColorInfo(colorspace, subsampling)

        self.icc = None
        icc_size = c_int()
        icc_ptr = c_char_p()
        if (TIFF.TIFFGetField(self._ptr, 34675, byref(icc_size),
                              byref(icc_ptr)) and icc_size.value > 0):
            self.icc = Icc(string_at(icc_ptr, icc_size.value))

    def _get(
        self,
        tp: type['ctypes._SimpleCData[_T]'],
        *tags: int,
    ) -> tuple[_T, ...]:
        values: list[_T] = []
        for tag in tags:
            cv = tp()
            if TIFF.TIFFGetField(self._ptr, c_uint32(tag), byref(cv)):
                values.append(cv.value)
        return *values,

    def _get_str(self, tag: int) -> str:
        values = self._get(c_char_p, tag)
        if not values:
            return ''
        return (values[0] or b'').decode()

    def _get_varargs(self, tps: tuple[type['ctypes._SimpleCData[_T]'], ...],
                     tag: int) -> tuple[_T, ...]:
        cvs = *(tp() for tp in tps),
        if TIFF.TIFFGetField(self._ptr, c_uint32(tag), *map(byref, cvs)):
            return *(cv.value for cv in cvs),
        return ()


# -------------------------- lazy decoding proxies ---------------------------
class SupportsArray(Protocol):
    def __array__(self) -> np.ndarray:
        raise NotImplementedError


class _CachedArray:
    def __array__(self) -> np.ndarray:
        return self.numpy

    @cached_property
    def numpy(self) -> np.ndarray:
        return self._impl()

    def _impl(self) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ImageArray(_CachedArray):
    data: object

    def _impl(self) -> np.ndarray:
        return imagecodecs.imread(self.data)


@dataclass
class JpegArray(_CachedArray):
    data: object
    jpt: bytes
    colorspace: int

    def _impl(self) -> np.ndarray:
        return imagecodecs.jpeg_decode(
            self.data, tables=self.jpt, colorspace=self.colorspace)


# ------------------ image, level & opener implementations -------------------


@dataclass(frozen=True)
class _BaseImage(Image):
    index: int
    icc_impl: Icc | None
    tiff: 'Tiff'

    @property
    def icc(self) -> Icc | None:
        return self.icc_impl


@dataclass(frozen=True)
class _Image(_BaseImage):
    head: list[str]
    vendor: str

    @property
    def key(self) -> str | None:
        if self.vendor == 'aperio' and self.index == 1:
            return 'thumbnail'
        for key in ('label', 'macro'):
            if any(key in s for s in self.head):
                return key
        return None

    def numpy(self) -> np.ndarray:
        h, w = self.shape[:2]
        rgba = np.empty((h, w, 4), dtype='u1')

        with self.tiff.ifd(self.index) as ptr:
            ok = TIFF.TIFFReadRGBAImageOriented(ptr, w, h,
                                                c_void_p(rgba.ctypes.data), 1,
                                                0)
            if not ok:
                raise ValueError('TIFF image read failed')

        rgba = cv2.cvtColor(rgba, cv2.COLOR_mRGBA2RGBA)
        # TODO: do we need to use bg_color to fill points where alpha = 0 ?
        return rgba[..., :3]


@dataclass(frozen=True)
class _Level(ImageLevel, _BaseImage):
    color: _ColorInfo
    bg_color: np.ndarray
    compression: _Compression
    jpt: bytes = field(repr=False)
    tile: tuple[int, ...]
    tile_sizes: np.ndarray = field(repr=False)

    def _read_tile(self, y, x, ptr) -> SupportsArray:
        offset = TIFF.TIFFComputeTile(ptr, x, y, 0, 0)
        nbytes = int(self.tile_sizes[offset])

        if not nbytes:  # If nothing to read, don't read
            raise ValueError('File has corrupted tiles with zero size')
            # TODO: read from previous level
            # * If tile is empty on level N,
            # * then all tiles on levels >N are invalid, whether empty or not
            return np.broadcast_to(self.bg_color, self.tile)

        # if not self.compression.name.startswith('JPEG2000'):
        if self.compression not in {
                _Compression.JPEG2000_RGB, _Compression.JPEG2000_YUV
        }:
            image = np.empty(self.tile, dtype='u1')
            isok = TIFF.TIFFReadTile(ptr, c_void_p(image.ctypes.data), x, y, 0,
                                     0)
            if isok == -1:
                raise ValueError('TIFF tile read failed')
            return self.color.to_rgb(image)

        data = create_string_buffer(nbytes)
        TIFF.TIFFReadRawTile(ptr, offset, data, len(data))

        if self.jpt:
            return JpegArray(data, self.jpt, self.color.space.value)
        return ImageArray(data)

    def _get_tile(self, y: int, x: int, ptr, cache_ok: bool) -> SupportsArray:
        # Runs within lock
        key = (self.index, y, x)
        cache = self.tiff.cache
        if (obj := cache.pop(key, None)) is not None:
            # Cache hit, move to the end
            cache[key] = obj
        else:
            # Cache miss
            obj = self._read_tile(y, x, ptr)

            if cache_ok and (cache_cap := env.BIPL_TILE_CACHE):
                while len(cache) >= cache_cap:  # Evict least used item
                    cache.pop(next(iter(cache)))
                cache[key] = obj
        return obj

    def crop(self, *loc: slice) -> np.ndarray:
        box = np.array([(s.start, s.stop) for s in loc])

        *tile, spp = self.tile
        dyx = box[:, 0]  # (2 lo-hi) -> (2)
        out = np.ascontiguousarray(
            np.broadcast_to(
                self.bg_color,
                np.r_[box[:, 1] - box[:, 0], spp],
            ))

        bmin, bmax = box.T.clip(0, self.shape[:2])

        axes = *map(slice, bmin // tile, -(-bmax // tile)),
        t_lo = np.mgrid[axes].reshape(2, -1).T * tile  # [N, 2]
        if not t_lo.size:
            return out

        # Cache only edges
        # TODO: profile whether it gives any perf benefits
        is_edge = (t_lo == t_lo.min(0)).any(1) | (t_lo == t_lo.max(0)).any(1)
        with self.tiff.ifd(self.index) as ptr:
            parts = [
                self._get_tile(y, x, ptr, cache_ok)
                for (y, x), cache_ok in zip(t_lo.tolist(), is_edge.tolist())
            ]

        # [N, lo-hi, yx]
        crops = np.stack([t_lo, t_lo + tile], 1).clip(bmin, bmax)

        # [N, yx, lo-hi]
        o_crops = (crops - dyx).transpose(0, 2, 1)
        t_crops = (crops - t_lo[:, None, :]).transpose(0, 2, 1)
        for part, (oy, ox), (ty, tx) in zip(parts, o_crops, t_crops):
            patch = np.array(part, copy=False)
            out[slice(*oy), slice(*ox)] = patch[slice(*ty), slice(*tx)]

        return out


# FIXME: Get around slides from choked SVS encoder
class Tiff(Driver):
    def __init__(self, path: str):
        # TODO: use memmap instead of libtiff
        self._ptr = (
            TIFF.TIFFOpenW(path, b'rm') if sys.platform == 'win32' else
            TIFF.TIFFOpen(path.encode(), b'rm'))
        if not self._ptr:
            raise ValueError(f'File {path} cannot be opened')

        weakref.finalize(self, TIFF.TIFFClose, self._ptr)

        self._lock = Lock()
        self.cache: dict[tuple, SupportsArray] = {}

    def __repr__(self) -> str:
        return f'{type(self).__name__}({addressof(self._ptr.contents):0x})'

    @contextmanager
    def ifd(self, index: int) -> Iterator:
        with self._lock:
            TIFF.TIFFSetDirectory(self._ptr, index)
            try:
                yield self._ptr
            finally:
                TIFF.TIFFFreeDirectory(self._ptr, index)

    def __len__(self) -> int:
        return TIFF.TIFFNumberOfDirectories(self._ptr)

    def _bg_color(self) -> np.ndarray:
        bg_hex = b'FFFFFF'
        bg_color_ptr = c_char_p()
        # TODO: find sample file to test this path. Never reached
        if TIFF.TIFFGetField(self._ptr, _Tag.BACKGROUND_COLOR,
                             byref(bg_color_ptr)):
            bg_hex = string_at(bg_color_ptr, 3)
            # TIFF._TIFFfree(bg_color_ptr)  # TODO: ensure no segfault
        return np.frombuffer(bytes.fromhex(bg_hex.decode()), 'u1')

    def _parse_description(self, desc: str,
                           make: str) -> tuple[str, list[str], dict[str, str]]:
        vendor = ''
        if is_aperio(desc):
            vendor = 'aperio'
            head, meta = parse_aperio_description(desc)

        else:
            if make == 'Hamamatsu':
                raise ValueError('Hamamatsu is not yet supported via libtiff')
            # TODO: put xml parser here (tiff)
            head = [desc]
            meta = {}

        return vendor, head, meta

    def _mpp(self, resolution: tuple[float, ...],
             meta: dict[str, str]) -> float | None:
        if s := [(10_000 / v) for v in resolution if v]:
            return float(np.mean(s))
        if mpp := meta.get('MPP'):
            return float(mpp)
        return None

    def _get(self, index: int) -> Image:
        tags = _Tags(self._ptr)

        if tags.is_planar != 1:
            raise TypeError(f'Level {index} is not contiguous!')

        bg_color = self._bg_color()
        vendor, head, meta = self._parse_description(tags.description,
                                                     tags.make)
        mpp = self._mpp(tags.resolution, meta) if index == 0 else None

        if (tags.color.space is _ColorSpace.YCBCR
                and tags.color.subsampling != (2, 2)):
            raise ValueError('Unsupported YUV subsampling: '
                             f'{tags.color.subsampling}')

        # Compression and JPEG tables
        jpt = b''
        if tags.compression is _Compression.JPEG:
            count = c_int()
            jpt_ptr = c_char_p()
            if TIFF.TIFFGetField(self._ptr, _Tag.JPEG_TABLES, byref(count),
                                 byref(jpt_ptr)) and count.value > 4:
                jpt = string_at(jpt_ptr, count.value)
                # TIFF._TIFFfree(jpt_ptr)  # TODO: ensure no segfault

        # Whole level shape
        shape = (*tags.image_size, tags.spp)
        if len(shape) != 3:
            raise ValueError(f'Bad shape in TIFF: {shape}')

        # Tile shape, if applicable
        if not TIFF.TIFFIsTiled(self._ptr):  # Not yet supported
            return _Image(shape, index, tags.icc, self, head, vendor)

        # Tile sizes
        tile = (*tags.tile_size, tags.spp)
        if len(tile) != 3:
            raise ValueError(f'Bad tile shape in TIFF: {tile}')

        tbc = np.empty([], 'u8')
        tbc_ptr = POINTER(c_uint64)()
        if TIFF.TIFFGetField(self._ptr, _Tag.TILE_BYTE_COUNTS, byref(tbc_ptr)):
            num_tiles = TIFF.TIFFNumberOfTiles(self._ptr)
            tbc = np.ctypeslib.as_array(tbc_ptr, [num_tiles]).copy()
            # TIFF._TIFFfree(tbc_ptr)  # TODO: ensure no segfault
            if (tbc == 0).any():
                raise ValueError('Found 0s in tile size table')

        return _Level(shape, index, tags.icc, self, mpp, tags.color, bg_color,
                      tags.compression, jpt, tile, tbc)

    def __getitem__(self, index: int) -> Image:
        with self.ifd(index):
            return self._get(index)
