"""
Driver based on libtiff
- fast
- not thread safe (internally)
- compatible with TIFF and its flavours
"""

from __future__ import annotations

__all__ = ['Tiff']

import ctypes
import sys
import weakref
from contextlib import contextmanager
from ctypes import (POINTER, addressof, byref, c_char_p, c_float, c_int,
                    c_ubyte, c_uint16, c_uint32, c_uint64, c_void_p,
                    create_string_buffer, string_at)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Iterator, NamedTuple, Protocol, TypeVar

import cv2
import imagecodecs
import numpy as np

from ._libs import load_library
from ._slide_bases import Driver, Item, Lod

_T = TypeVar('_T')

TIFF = load_library('libtiff', 5)
# _TIFF.TIFFSetErrorHandler(None)

(TIFF.TIFFOpenW if sys.platform == 'win32' else TIFF.TIFFOpen).restype \
    = POINTER(c_ubyte)


# ---------------------------- TIFF tags mappings ----------------------------
class _Colorspace(Enum):
    MINISBLACK = 1
    RGB = 2
    # YCBCR = 6


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


class _Tag:  # noqa: PIE795
    JPEG_TABLES = 347
    TILE_BYTE_COUNTS = 325
    BACKGROUND_COLOR = 434


class _Tags:
    def __init__(self, ptr):
        self._ptr = ptr
        self.compression, = self._get(c_uint16, 259)
        self.colorspace, = self._get(c_uint16, 262)
        self.spp, = self._get(c_uint16, 277)
        self.is_planar, = self._get(c_uint16, 284)
        self.image_size = self._get(c_uint32, 257, 256)
        self.tile_size = self._get(c_uint32, 323, 322)
        self.description = self._get(c_char_p, 270).pop() or b''
        self.resolution = self._get(c_float, 283, 282)

    def _get(self, tp: type[ctypes._SimpleCData[_T]], *tags: int) -> list[_T]:
        values = []
        for tag in tags:
            cv = tp()
            if TIFF.TIFFGetField(self._ptr, c_uint32(tag), byref(cv)):
                values.append(cv.value)
        return values


# -------------------------- lazy decoding proxies ---------------------------
class SupportsArray(Protocol):
    def __array__(self) -> np.ndarray:
        raise NotImplementedError


class ImageArray(NamedTuple):
    data: object

    def __array__(self) -> np.ndarray:
        return imagecodecs.imread(self.data)


class JpegArray(NamedTuple):
    data: object
    jpt: bytes
    colorspace: int

    def __array__(self) -> np.ndarray:
        return imagecodecs.jpeg_decode(
            self.data, tables=self.jpt, colorspace=self.colorspace)


# -------------------- item, lod & opener implementations --------------------
@dataclass(frozen=True)
class _ItemBase(Item):
    index: int
    tiff: Tiff
    head: list[str]
    meta: dict[str, str]  # unused
    colorspace: _Colorspace
    bg_color: np.ndarray
    compression: _Compression
    jpt: bytes = field(repr=False)


@dataclass(frozen=True)
class _Item(_ItemBase):
    def get_key(self) -> str | None:
        if self.tiff.is_svs and self.index == 1:
            return 'thumbnail'
        for key in ('label', 'macro'):
            if any(key in s for s in self.head):
                return key
        return None

    def __array__(self) -> np.ndarray:
        h, w = self.shape[:2]
        bgra = np.empty((h, w, 4), dtype='u1')

        with self.tiff.ifd(self.index) as ptr:
            ok = TIFF.TIFFReadRGBAImage(ptr, w, h, c_void_p(bgra.ctypes.data),
                                        0)
            assert ok

        bgra = cv2.cvtColor(bgra, cv2.COLOR_mRGBA2RGBA)
        # TODO: do we need to use bg_color to fill points where alpha = 0 ?
        return bgra[..., 2::-1]


@dataclass(frozen=True)
class _Lod(Lod, _ItemBase):
    tile: tuple[int, ...]
    tile_sizes: np.ndarray = field(repr=False)

    def _get_tile(self, y, x, ptr) -> SupportsArray:
        offset = TIFF.TIFFComputeTile(ptr, x, y, 0, 0)
        nbytes = int(self.tile_sizes[offset])

        assert nbytes, 'File has corrupted tiles with zero size'
        if not nbytes:  # If nothing to read, don't read
            # TODO: read from previous lod
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
            assert isok != -1
            return image

        data = create_string_buffer(nbytes)
        TIFF.TIFFReadRawTile(ptr, offset, data, len(data))

        if self.jpt:
            return JpegArray(data, self.jpt, self.colorspace.value)
        return ImageArray(data)

    def crop(self, slices: tuple[slice, ...]) -> np.ndarray:
        box = np.array([(s.start, s.stop) for s in slices])

        *tile, spp = self.tile
        dy, dx = (low for low, _ in box)
        out = np.ascontiguousarray(
            np.broadcast_to(self.bg_color,
                            [hi - lo for lo, hi in box] + [spp]))

        hw = self.shape[:2]
        bmin, bmax = np.transpose(box).clip(0, hw)

        axes = *map(slice, bmin // tile, -(-bmax // tile)),
        t_lo = np.mgrid[axes].reshape(2, -1).T * tile  # [N, 2]
        if not t_lo.size:
            return out

        with self.tiff.ifd(self.index) as ptr:
            parts = [self._get_tile(y, x, ptr) for y, x in t_lo.tolist()]

        # [N, lo-hi, yx]
        crops = np.stack([t_lo, t_lo + tile], 1).clip(bmin, bmax)

        # [N, yx, lo-hi]
        o_crops = (crops - [dy, dx]).transpose(0, 2, 1)
        t_crops = (crops - t_lo[:, None, :]).transpose(0, 2, 1)
        for part, (oy, ox), (ty, tx) in zip(parts, o_crops, t_crops):
            patch = np.array(part, copy=False)
            out[slice(*oy), slice(*ox)] = patch[slice(*ty), slice(*tx)]

        return out


# FIXME: Get around slides from choked SVS encoder
class Tiff(Driver):
    def __init__(self, path: Path):
        # TODO: use memmap instead of libtiff
        spath = path.as_posix()
        self._ptr = (
            TIFF.TIFFOpenW(spath, b'rm') if sys.platform == 'win32' else
            TIFF.TIFFOpen(spath.encode(), b'rm'))
        if not self._ptr:
            raise ValueError(f'File {path} cannot be opened')

        weakref.finalize(self, TIFF.TIFFClose, self._ptr)

        self._lock = Lock()
        self.is_svs = path.suffix == '.svs'  # TODO: parse vendor tags for this

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
        if TIFF.TIFFGetField(self._ptr, _Tag.BACKGROUND_COLOR,
                             byref(bg_color_ptr)):
            bg_hex = string_at(bg_color_ptr, 3)
            # TIFF._TIFFfree(bg_color_ptr)  # TODO: ensure no segfault
        return np.frombuffer(bytes.fromhex(bg_hex.decode()), 'u1')

    def _parse_description(self,
                           desc: str) -> tuple[list[str], dict[str, str]]:
        head, *rest = desc.split('|')
        heads = [s.strip() for s in head.splitlines()]

        meta = {}
        for s in rest:
            if len(tags := s.split('=', 1)) == 2:
                meta[tags[0].strip()] = tags[1].strip()
            else:
                raise ValueError(f'Unparseable line in description: {s!r}')

        return heads, meta

    def _spacing(self, resolution: list[float],
                 meta: dict[str, str]) -> float | None:
        if s := [(10_000 / v) for v in resolution if v]:
            return float(np.mean(s))
        if mpp := meta.get('MPP'):
            return float(mpp)
        return None

    def _get(self, index: int) -> Item:
        tags = _Tags(self._ptr)

        if tags.is_planar != 1:
            raise TypeError(f'Level {index} is not contiguous!')

        bg_color = self._bg_color()
        head, meta = self._parse_description(tags.description.decode())
        spacing = self._spacing(tags.resolution, meta)

        colorspace = _Colorspace(tags.colorspace)

        # Compression and JPEG tables
        jpt = b''
        compression = _Compression(tags.compression)
        if compression is _Compression.JPEG:
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
            return _Item(shape, index, self, head, meta, colorspace, bg_color,
                         compression, jpt)

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
                raise ValueError('Tiles are corrupted as their sizes have 0s')

        return _Lod(shape, index, self, head, meta, colorspace, bg_color,
                    compression, jpt, spacing, tile, tbc)

    def __getitem__(self, index: int) -> Item:
        with self.ifd(index):
            return self._get(index)
