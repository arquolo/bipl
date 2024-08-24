"""
Driver based on libtiff
- fast
- not thread safe (internally)
- compatible with TIFF and its flavours
"""

__all__ = ['Tiff']

import ctypes
import mmap
import sys
import warnings
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from ctypes import (POINTER, addressof, byref, c_char_p, c_float, c_int,
                    c_ubyte, c_uint16, c_uint32, c_uint64, c_void_p, string_at)
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from heapq import heappop, heappush
from itertools import product
from threading import Lock
from typing import TypeVar

import cv2
import imagecodecs
import numpy as np
import numpy.typing as npt
from glow import shared_call, si_bin, sizeof
from numpy.lib.stride_tricks import as_strided

from bipl._env import env

from ._libs import load_library
from ._slide_bases import Driver, Image, ImageLevel, ProxyLevel
from ._util import (Icc, get_aperio_properties, get_ventana_properties,
                    parse_xml, unflatten)

_T = TypeVar('_T')
_U8 = npt.NDArray[np.uint8]

TIFF = load_library('libtiff', 6, 5)
# _TIFF.TIFFSetErrorHandler(None)

(TIFF.TIFFOpenW if sys.platform == 'win32' else TIFF.TIFFOpen).restype \
    = POINTER(c_ubyte)

_RESOLUTION_UNITS = {2: 25400, 3: 10000}


# ---------------------------- TIFF tags mappings ----------------------------
class _ColorSpace(Enum):
    MINISBLACK = 1
    RGB = 2
    YCBCR = 6


def nv12_to_rgb(x: _U8) -> _U8:
    w = x.shape[1]
    assert w % 2 == 0
    w2 = w // 2

    # h/2 w 3 -> h/2 w/2 6
    # 6 channels are [Y-Y-Y-Y-Cb-Cr] of 2x2 pixel block
    hw6 = x.reshape(-1, w2, 6)

    # -> h/2 w/2 4 -> h/2 w/2 2 2 -> h/2 2 w/2 2 -> h w
    y = hw6[..., :4].reshape(-1, w2, 2, 2).transpose(0, 2, 1, 3).reshape(-1, w)
    # -> h/2 w/2 2
    cb_cr = hw6[:, :, 4:]

    r = cv2.cvtColorTwoPlane(y, cb_cr, cv2.COLOR_YUV2RGB_NV12)
    return np.asarray(r)


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


class _Planarity(Enum):
    CONTIG = 1  # Single buffer for all image planes
    # SEPARATE = 2  # Separate buffers for each R/G/B plane. TODO: support


class _Tag:  # noqa: PIE795,RUF100
    DESCRIPTION = 270
    MAKE = 271
    TILE_OFFSETS = 324
    TILE_BYTE_COUNTS = 325
    JPEG_TABLES = 347
    BACKGROUND_COLOR = 434
    XMP = 700
    ICC_PROFILE = 34675
    JPEGCOLORMODE = 65538


class _Tags:
    def __init__(self, ptr):
        self._ptr = ptr

        compression, = self._get(c_uint16, 259)
        self.compression = _Compression(compression)

        planarity, = self._get(c_uint16, 284)
        self.planarity = _Planarity(planarity)  # TODO: use later

        self.spp, = self._get(c_uint16, 277)

        # ! crashes, but should work according to openslide docs
        # self.is_hamamatsu = bool(self._get(c_uint16, 65420))

        self.bps, = self._get(c_uint16, 339) or [1]

        ics, = self._get(c_uint16, 262)
        self.color = _ColorSpace(ics)
        self.subsampling = (1, 1)

        # TODO: use this in YCbCr conversion
        self.gray = np.array([], 'f4')  # Luma coefficients
        self.yuv_centered = True
        self.yuv_bw = np.array([], 'f4')  # BW pairs, per channel

        if self.color is _ColorSpace.YCBCR:
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
            self.subsampling = ss_h, ss_w

            self.yuv_centered = 2 not in self._get(c_uint16, 531)

            bw_ptr = POINTER(c_float)()
            if TIFF.TIFFGetField(self._ptr, 532, byref(bw_ptr)):
                self.yuv_bw = np.ctypeslib.as_array(bw_ptr, [3, 2])

            self.jpcm, = self._get(c_int, _Tag.JPEGCOLORMODE)

        self.icc = None
        icc_size = c_int()
        icc_ptr = c_char_p()
        if TIFF.TIFFGetField(self._ptr, _Tag.ICC_PROFILE, byref(icc_size),
                             byref(icc_ptr)) and icc_size.value > 0:
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

    @property
    def image_shape(self) -> tuple[int, int, int]:
        shape = *self._get(c_uint32, 257, 256), self.spp
        if len(shape) != 3:
            raise ValueError(f'TIFF: Bad image shape - {shape}')
        return shape

    @property
    def tile_shape(self) -> tuple[int, int, int] | None:
        if not TIFF.TIFFIsTiled(self._ptr):
            return None
        tile = *self._get(c_uint32, 323, 322), self.spp
        if len(tile) != 3:
            raise ValueError(f'TIFF: Bad tile shape - {tile}')
        return tile

    def tile_spans(self, shape: tuple[int, ...],
                   tile: tuple[int, ...]) -> npt.NDArray[np.uint64]:
        """Returns (h w d 2) tensor of start:stop of each tile w.r.t file."""
        grid_shape = *(len(range(0, s, t)) for s, t in zip(shape, tile)),

        ptr = POINTER(c_uint64)()
        if not TIFF.TIFFGetField(self._ptr, _Tag.TILE_OFFSETS, byref(ptr)):
            raise ValueError('TIFF has tiled image, but no '
                             'tile offsets table present')
        tbs = np.ctypeslib.as_array(ptr, grid_shape).copy()
        # TIFF._TIFFfree(tbc_ptr)  # TODO: ensure no segfault

        ptr = POINTER(c_uint64)()
        if not TIFF.TIFFGetField(self._ptr, _Tag.TILE_BYTE_COUNTS, byref(ptr)):
            raise ValueError('TIFF has tiled image, but no '
                             'tile size table present')
        tbc = np.ctypeslib.as_array(ptr, grid_shape).copy()
        # TIFF._TIFFfree(tbc_ptr)  # TODO: ensure no segfault

        return np.stack((tbs, tbs + tbc), -1)

    def get_decoder(self) -> Callable[[bytes], _U8]:
        match self.compression:
            case _Compression.JPEG:
                jpt = None
                size = c_int()
                ptr = c_char_p()
                if (TIFF.TIFFGetField(self._ptr, _Tag.JPEG_TABLES, byref(size),
                                      byref(ptr)) and size.value > 4):
                    jpt = string_at(ptr, size.value)
                    # TIFF._TIFFfree(ptr)  # TODO: ensure no segfault
                return _JpegDecoder(jpt, self.color.name)

            case _Compression.JPEG2000_RGB | _Compression.JPEG2000_YUV:
                return imagecodecs.jpeg2k_decode

            case _:
                return imagecodecs.imread  # type: ignore

    def _bg_color(self, meta: dict) -> _U8:
        if c := meta.get('ScanWhitePoint'):
            return np.array(int(c), 'u1')

        bg_hex = b'FFFFFF'
        bg_color_ptr = c_char_p()
        # TODO: find sample file to test this path. Never reached
        if TIFF.TIFFGetField(self._ptr, _Tag.BACKGROUND_COLOR,
                             byref(bg_color_ptr)):
            bg_hex = string_at(bg_color_ptr, 3)
            # TIFF._TIFFfree(bg_color_ptr)  # TODO: ensure no segfault
        return np.frombuffer(bytes.fromhex(bg_hex.decode()), 'u1').copy()

    def vendor_props(
        self,
        vendor: str,
        index: int = 0,
        description: str | None = None,
    ) -> tuple[str, dict[str, str]] | None:
        if description is None:
            description = self._get_str(_Tag.DESCRIPTION)

        match vendor:
            case 'ventana':  # BIF of Roche
                xmp_size = c_int()
                xmp_ptr = c_char_p()
                if TIFF.TIFFGetField(self._ptr, _Tag.XMP, byref(xmp_size),
                                     byref(xmp_ptr)) and xmp_size.value > 0:
                    xmp = string_at(xmp_ptr, xmp_size.value)
                    if meta := get_ventana_properties(xmp, index):
                        return description, meta
                if index != 0:
                    return description, {}

            case 'aperio':  # SVS
                return get_aperio_properties(description, index)

            case 'hamamatsu':  # NDPI
                if self._get_str(_Tag.MAKE) == 'Hamamatsu':
                    raise ValueError('TIFF: Hamamatsu is not yet supported')

            case 'generic':  # TIFF
                try:
                    meta = unflatten(parse_xml(description))
                except Exception:  # noqa: BLE001
                    return description, {}
                else:
                    return '', meta

        return None

    def vendor_properties(self) -> tuple[str, str, dict[str, str]]:
        description = self._get_str(_Tag.DESCRIPTION)

        for vendor in ('ventana', 'aperio', 'hamamatsu', 'generic'):
            if r := self.vendor_props(vendor, 0, description):
                header, meta = r
                return vendor, header, meta

        raise ValueError('Unknown vendor')

    def _get_mpp(self, vendor: str, meta: dict) -> float | None:
        """Extract MPP, um/pixel, phisical pixel size"""
        if vendor == 'ventana':
            # Ventana messes with TIFF tag XResolution, YResolution
            return float(mpp_s) if (mpp_s := meta.get('ScanRes')) else None

        if pixels_per_unit := self._get(c_float, 283, 282):
            res_unit_kind, = self._get(c_uint16, 296)
            if res_unit := _RESOLUTION_UNITS.get(res_unit_kind):
                mpp_xy = [res_unit / ppu for ppu in pixels_per_unit if ppu]
                if mpp_xy and (mpp := float(np.mean(mpp_xy))):
                    return mpp

        if mpp_s := meta.get('MPP'):
            return float(mpp_s)
        return None

    def properties(self) -> tuple[str, str, dict[str, str], float | None, _U8]:
        """Extracts: (vendor, header, metadata, mpp)"""
        vendor, header, meta = self.vendor_properties()
        mpp = self._get_mpp(vendor, meta)
        bg_color = self._bg_color(meta)
        return vendor, header, meta, mpp, bg_color


# ------------------ image, level & opener implementations -------------------


@dataclass(frozen=True)
class _BaseImage(Image):
    icc_impl: Icc | None
    flipped: bool

    @property
    def icc(self) -> Icc | None:
        return self.icc_impl


@dataclass(frozen=True)
class _Image(_BaseImage):
    tiff: 'Tiff'
    head: str
    index: int

    @property
    def key(self) -> str | None:
        if self.tiff.vendor == 'aperio' and self.index == 1:
            return 'thumbnail'
        if self.tiff.vendor == 'ventana':
            match self.head:
                case 'Thumbnail':
                    return 'thumbnail'
                case 'Label Image' | 'Label_Image':
                    return 'macro'
            return None
        for key in ('label', 'macro'):
            if any(key in s for s in self.head.splitlines()):
                return key
        return None

    def numpy(self) -> _U8:
        h, w = self.shape[:2]
        rgba = np.empty((h, w, 4), dtype='u1')

        # TODO: find offset & size of such images
        with self.tiff.ifd(self.index) as ptr:
            ok = TIFF.TIFFReadRGBAImageOriented(ptr, w, h,
                                                c_void_p(rgba.ctypes.data), 1,
                                                0)
            if not ok:
                raise ValueError('TIFF image read failed')

        rgba = cv2.cvtColor(rgba, cv2.COLOR_mRGBA2RGBA)
        # TODO: do we need to use bg_color to fill points where alpha = 0 ?
        im = np.asarray(rgba[..., :3], 'u1')
        return im.swapaxes(0, 1) if self.flipped else im

    def flip(self) -> '_Image':
        h, w, c = self.shape
        return replace(self, shape=(w, h, c), flipped=not self.flipped)


@dataclass(frozen=True, slots=True)
class _JpegDecoder:
    jpt: bytes | None = field(repr=False)
    color: str | None

    def __call__(self, buf: bytes) -> _U8:
        return imagecodecs.jpeg_decode(
            buf, tables=self.jpt, colorspace=self.color, outcolorspace='RGB')


@dataclass(eq=False, frozen=True)
class _Level(ImageLevel, _BaseImage):
    memo: mmap.mmap
    spans: npt.NDArray[np.uint64] = field(repr=False)
    tile: tuple[int, ...]
    order: str | None
    decode: Callable[[bytes], _U8]
    cache: '_CacheZYXC'
    fill: _U8
    pool: int

    def octave(self) -> '_Level | None':
        th, tw, tc = self.tile
        if th % 2 or tw % 2:
            return None

        h, w, c = self.shape
        return replace(
            self,
            shape=((h + 1) // 2, (w + 1) // 2, c),
            tile=(th // 2, tw // 2, tc),
            pool=self.pool * 2,
        )

    def __eq__(self, rhs) -> bool:
        return (type(self) is type(rhs) and self.memo is rhs.memo
                and self.shape == rhs.shape)

    def __hash__(self) -> int:
        return hash(self.memo) ^ hash(self.shape)

    @shared_call  # Thread safety
    def _get_tile(self, *loc: int) -> _U8 | None:
        lo, hi = self.spans[loc].tolist()
        if lo == hi:  # If nothing to read, don't read
            return None
        return self._get_tile_raw(lo, hi, *loc)

    def _get_tile_raw(self, lo: int, hi: int, *loc: int) -> _U8:
        # NOTE:
        # Like OpenSlide does we use private cache for each slide
        #   with (level, y, x) key to cache decoded pixels.
        # But we also cache opened slides to not waste time on re-opening
        #   (that can lead to multiple caches existing at the same moment).
        key = (*self.shape, *loc)

        # Cache hit
        if (im := self.cache[key]) is not None:
            return im

        # Cache miss
        # Read tile from disk
        im = self.decode(self.memo[lo:hi])

        # Resize if level is pooled
        th, tw = self.tile[:2]
        if im.shape[:2] != (th, tw):
            if self.pool > 2:
                im = cv2.resize(im, (tw, th))
            else:
                im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        if self.flipped:
            im = im.swapaxes(0, 1)

        self.cache[key] = im  # type: ignore
        return im  # type: ignore

    def crop(self, *loc: slice) -> _U8:
        *tile, spp = self.tile

        # (y/x lo/hi)
        box = np.array([(s.start, s.stop) for s in loc])
        out_shape = np.r_[box[:, 1] - box[:, 0], spp]

        if not out_shape.all():
            return np.empty(out_shape, self.dtype)

        bmin, bmax = box.T.clip(0, self.shape[:2])  # (y/x)
        if (bmin == bmax).any():  # Crop is outside of image
            return np.broadcast_to(self.fill, out_shape)

        iloc, t_crops, o_crops, ((y0, y1), (x0, x1)) = zip(
            *map(self._make_index, box[:, 0], bmin, bmax, tile))

        parts = [
            self._get_tile(y, x, 0)
            for y, x in product(*(ids.tolist() for ids in iloc))
        ]

        out = np.empty(out_shape, self.dtype)
        out[:y0] = self.fill
        out[y0:y1, :x0] = self.fill
        out[y0:y1, x1:] = self.fill
        out[y1:] = self.fill
        for part, (oy, ox), (ty, tx) in zip(parts, product(*o_crops),
                                            product(*t_crops)):
            if part is None:
                out[slice(*oy), slice(*ox)] = self.fill
            else:
                out[slice(*oy), slice(*ox)] = part[slice(*ty), slice(*tx)]
        return out

    def _make_index(
            self, min_: int, vmin: int, vmax: int,
            tile: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        # (n + 1)
        ids1 = np.arange(vmin // tile, 1 - (-vmax // tile))
        n = ids1.size - 1
        ts = ids1 * tile
        vsep = ts.clip(vmin, vmax)

        # (n lo/hi), source & target slices
        u = vsep.itemsize
        o_crops = as_strided(vsep - min_, (n, 2), (u, u))
        t_crops = as_strided(vsep, (n, 2), (u, u)) - ts[:-1, None]

        # (lo/hi), region of `out` to fill
        o_span = o_crops[[0, -1], [0, 1]].tolist()

        return ids1[:-1], t_crops, o_crops, o_span

    def flip(self) -> '_Level':
        h, w, c = self.shape
        th, tw, tc = self.tile
        reorder = {'C': 'F', 'F': 'C', None: None}
        return replace(
            self,
            shape=(w, h, c),
            flipped=not self.flipped,
            spans=self.spans.swapaxes(0, 1),
            tile=(tw, th, tc),
            order=reorder[self.order],
        )


@dataclass(eq=False, frozen=True)
class _AperioLevel(_Level):
    def fallback(self, lv: ImageLevel, ds: int) -> '_Level':
        while ds != 1 and (lv_ := lv.octave()):
            ds //= 2
            lv = lv_
        if ds != 1:
            lv = ProxyLevel(self.shape, 1 / ds, lv)

        if not isinstance(self, _AperioSubLevel):
            r = {f.name: getattr(self, f.name) for f in fields(self)}
            return _AperioSubLevel(**r, prev=lv)

        return replace(self, prev=lv)


@dataclass(eq=False, frozen=True)
class _AperioSubLevel(_AperioLevel):
    prev: ImageLevel

    @time_this
    @shared_call  # Thread safety
    def _get_tile(self, *loc: int) -> _U8 | None:
        lo, hi = self.spans[loc].tolist()
        if lo == hi:  # Read tile from backup level
            iy, ix, _ = loc
            th, tw = self.tile[:2]
            return self.prev[iy * th:iy * th + th, ix * tw:ix * tw + tw]

        return self._get_tile_raw(lo, hi, *loc)


class Tiff(Driver):
    def __init__(self, path: str) -> None:
        # Open TIFF in read-only (`r`) mode and disable memory mapping (`m`)
        self._ptr = (
            TIFF.TIFFOpenW(path, b'rm') if sys.platform == 'win32' else
            TIFF.TIFFOpen(path.encode(), b'rm'))
        if not self._ptr:
            raise ValueError(f'File {path} cannot be opened')

        weakref.finalize(self, TIFF.TIFFClose, self._ptr)
        self._dir = 0
        self._lock = Lock()
        self.cache = _CacheZYXC()

        # Initially directory 0 is active
        self._tags = _Tags(self._ptr)
        self.vendor, self._head, self._meta, self.mpp, self._bg_color \
            = self._tags.properties()

        with open(path, 'rb') as f:
            self._memo = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({addressof(self._ptr.contents):0x})'

    def get_mpp(self):
        return self.mpp

    @contextmanager
    def ifd(self, index: int) -> Iterator:
        with self._lock:
            if self._dir != index:
                self._dir = index
                TIFF.TIFFSetDirectory(self._ptr, index)
            yield self._ptr

    def __len__(self) -> int:
        return TIFF.TIFFNumberOfDirectories(self._ptr)

    def _get(self, index: int) -> Image | None:
        if index == 0:
            tags = self._tags
            head = self._head
        else:
            tags = _Tags(self._ptr)
            r = tags.vendor_props(self.vendor, index)
            if not r:
                raise ValueError('File directories are from different vendors')
            head, _ = r

        if tags.color is _ColorSpace.YCBCR and tags.subsampling != (2, 2):
            raise ValueError('Unsupported YUV subsampling: '
                             f'{tags.subsampling}')

        shape = tags.image_shape
        tile = tags.tile_shape

        if tile is None:  # Not tiled
            return _Image(shape, tags.icc, False, self, head, index)

        spans = tags.tile_spans(shape, tile)
        order = _detect_tile_order(spans[..., 0])

        # Aperio can choke on non-L0 levels. Read those from L0 to fix.
        # `openslide` does this for us, delegate to it.
        if self.vendor == 'aperio' and (spans[..., 0] == spans[..., 1]).any():
            raise ValueError('Found 0s in tile size table')

        return _Level(shape, tags.icc, False, self._memo, spans, tile, order,
                      tags.get_decoder(), self.cache, self._bg_color, 1)

    def __getitem__(self, index: int) -> Image | None:
        with self.ifd(index):
            return self._get(index)


def _detect_tile_order(start: npt.NDArray[np.integer]) -> str | None:
    start = start.squeeze().copy()
    if start.ndim < 2:
        return 'C'

    for order in ('F', 'C'):
        a = start.ravel(order)
        a = a[a > 0]
        if (a[:-1] < a[1:]).all():
            return order

    return None


# ------------------------------- tile caching -------------------------------


@dataclass(repr=False, slots=True)
class _CacheZYXC:
    lock: Lock = field(default_factory=Lock, repr=False)
    used: int = 0
    keys: list[tuple] = field(default_factory=list, repr=False)
    # IYXC -> (size, buf)
    buf: dict[tuple, tuple[int, _U8]] = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return (f'{type(self).__name__}'
                f'(used={si_bin(self.used)}, items={len(self.buf)})')

    def __getitem__(self, key: tuple) -> _U8 | None:
        with self.lock:
            if e := self.buf.get(key):
                return e[1]
        return None

    def __setitem__(self, key: tuple, obj: _U8) -> None:
        if not (capacity := env.BIPL_TILE_CACHE):
            return
        if (size := sizeof(obj)) > capacity:
            warnings.warn(
                f'Rejecting overlarge cache entry of size {size} bytes',
                stacklevel=3)
            return
        max_size = capacity - size

        with self.lock:
            while self.keys and self.used > max_size:
                self.used -= self.buf.pop(heappop(self.keys))[0]

            if self.used <= max_size:
                heappush(self.keys, key)
                self.buf[key] = (size, obj)
                self.used += size
