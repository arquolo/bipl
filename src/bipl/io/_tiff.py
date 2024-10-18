"""
Driver based on libtiff
- fast
- not thread safe (internally)
- compatible with TIFF and its flavours
"""

__all__ = ['Tiff']

import mmap
import sys
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from ctypes import POINTER, addressof, c_ubyte, c_void_p
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from enum import Enum
from itertools import product
from threading import Lock
from typing import Any

import cv2
import imagecodecs
import numpy as np
import numpy.typing as npt
from glow import shared_call, si_bin, starmap_n

from bipl._env import env
from bipl._types import NDIndex, Patch, Shape, Span

from ._libs import load_library
from ._slide_bases import Driver, Image, ImageLevel, ProxyLevel
from ._util import (Icc, get_aperio_properties, get_ventana_properties,
                    parse_xml, unflatten)

_U8 = npt.NDArray[np.uint8]
_U64 = npt.NDArray[np.uint64]

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
    return np.asarray(r, 'u1')


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


class _Orientation(Enum):
    # fmt: off
    TOP_LEFT = 1      # [y, x]
    TOP_RIGHT = 2     # [y, -x] -> im[:, ::-1]
    BOTTOM_RIGHT = 3  # [-y, -x] -> im[::-1, ::-1]
    BOTTOM_LEFT = 4   # [-y, x]  -> im[::-1]
    LEFT_TOP = 5      # [x, y] -> im.transpose(1, 0, 2)
    RIGHT_TOP = 6     # [-x, y] -> im.transpose(1, 0, 2)[:, ::-1]
    RIGHT_BOTTOM = 7  # [-x, -y] -> im.transpose(1, 0, 2)[::-1, ::-1]
    LEFT_BOTTOM = 8   # [x, -y] -> im.transpose(1, 0, 2)[::-1]
    # fmt: on


class _Tag(Enum):
    # &1 = reduced resolution, &2 = page of multipage, &4 = transparency mask
    NEW_SUBFILE_TYPE = 254
    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 257
    BITS_PER_SAMPLE = 258
    COMPRESSION = 259
    COLORSPACE = 262
    DESCRIPTION = 270
    MAKE = 271
    MODEL = 272
    STRIP_OFFSETS = 273
    ORIENTATION = 274
    SAMPLES_PER_PIXEL = 277
    STRIP_HEIGHT = 278
    STRIP_NBYTES = 279
    RES_X = 282
    RES_Y = 283
    PLANAR = 284
    RES_UNIT = 296
    SOFTWARE = 305
    DATETIME = 306
    ARTIST = 315
    PREDICTOR = 317
    TILE_WIDTH = 322
    TILE_HEIGHT = 323
    TILE_OFFSETS = 324
    TILE_NBYTES = 325
    SAMPLE_FORMAT = 339
    SAMPLE_MIN = 340
    SAMPLE_MAX = 341
    JPEG_TABLES = 347
    BACKGROUND_COLOR = 434
    YUV_COEFFICIENTS = 529
    YUV_SUBSAMPLING = 530
    YUV_POSITIONING = 531
    REF_BLACK_WHITE = 532
    XMP = 700
    IMAGE_DEPTH = 32997
    ICC_PROFILE = 34675
    NDPI_VERSION = 65420  # 1 for any NDPI
    SOURCE_LENS = 65421  # macro = -1, map of non-empty regions = -2
    OFFSET_X = 65422
    OFFSET_Y = 65423
    OFFSET_Z = 65424
    # = 65425
    # = 65426  # Low 32 bits of optimisation entries
    REFERENCE = 65427
    AUTH_CODE = 65428
    # = 65432  # High 32 bits of optimisation entries
    # = 65433
    # = 65439
    # = 65440
    # = 65441  # 0?
    SCANNER_SERIAL_NUMBER = 65442
    # = 65443  # 0 or 16?
    # = 65444  # 80?
    # = 65445  # 0, 2 or 10?
    # = 65446  # 0?
    # = 65449  # ASCII metadata, key=value pairs
    # = 65455  # 13?
    # = 65456  # 101?
    # = 65457  # 0?
    # = 65458  # 0?


class _Tags:
    def __init__(self, ifd: Mapping[_Tag, Any]) -> None:
        # ! used only to check
        self.planarity: _Planarity = ifd.get(_Tag.PLANAR, _Planarity.CONTIG)

        # ! unused
        self.is_hamamatsu = ifd.get(_Tag.NDPI_VERSION) == 1
        self.bps = ifd.get(_Tag.SAMPLE_FORMAT, 1)

        self.color = ifd[_Tag.COLORSPACE]
        self.subsampling = (1, 1)

        # TODO: use this in YCbCr conversion
        self.gray = np.array([], 'f4')  # Luma coefficients
        self.yuv_centered = True
        self.yuv_bw = np.array([], 'f4')  # BW pairs, per channel

        if self.color is _ColorSpace.YCBCR:
            self.gray = np.array([.299, .587, .114], 'f4')
            self.gray = ifd.get(_Tag.YUV_COEFFICIENTS, self.gray)

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
            ss_w, ss_h = ifd.get(_Tag.YUV_SUBSAMPLING, (2, 2))
            self.subsampling = ss_h, ss_w

            self.yuv_centered = ifd.get(_Tag.YUV_POSITIONING) != 2
            self.yuv_bw = ifd.get(_Tag.REF_BLACK_WHITE, self.yuv_bw)

        self.ifd = ifd

    @property
    def image_shape(self) -> Shape:
        return (self.ifd[_Tag.IMAGE_HEIGHT], self.ifd[_Tag.IMAGE_WIDTH],
                self.ifd[_Tag.SAMPLES_PER_PIXEL])

    @property
    def tile_shape(self) -> Shape | None:
        tile = (
            self.ifd.get(_Tag.TILE_HEIGHT, 0),
            self.ifd.get(_Tag.TILE_WIDTH, 0),
            self.ifd[_Tag.SAMPLES_PER_PIXEL],
        )
        if not all(tile):  # Missing tile height/width tag
            return None
        return tile

    def tile_spans(self, shape: Shape, tile_shape: Shape) -> _U64:
        """Returns (h w d 2) tensor of start:stop of each tile w.r.t file."""
        grid_shape = *(len(range(0, s, t)) for s, t in zip(shape, tile_shape)),

        tbs: _U64 = np.reshape(self.ifd[_Tag.TILE_OFFSETS], grid_shape)
        tbc: _U64 = np.reshape(self.ifd[_Tag.TILE_NBYTES], grid_shape)
        return np.stack((tbs, tbs + tbc), -1)

    def get_decoder(self) -> Callable[[bytes], _U8]:
        match self.ifd[_Tag.COMPRESSION]:
            case _Compression.JPEG:
                jpt: bytes | None = self.ifd.get(_Tag.JPEG_TABLES)
                return _JpegDecoder(jpt, self.color.name)

            case _Compression.JPEG2000_RGB | _Compression.JPEG2000_YUV:
                return imagecodecs.jpeg2k_decode

            case _:
                return imagecodecs.imread  # type: ignore

    def _bg_color(self, meta: dict) -> _U8:
        if c := meta.get('ScanWhitePoint'):
            return np.array(int(c), 'u1')

        bg_hex: bytes = self.ifd.get(_Tag.BACKGROUND_COLOR, b'FFFFFF')
        return np.frombuffer(bytes.fromhex(bg_hex.decode()), 'u1').copy()

    def vendor_props(
        self,
        vendor: str,
        index: int = 0,
        description: str | None = None,
    ) -> tuple[str, dict[str, str]] | None:
        if description is None:
            description = self.ifd.get(_Tag.DESCRIPTION, '')
            assert isinstance(description, str)

        match vendor:
            case 'ventana':  # BIF of Roche
                if ((xmp := self.ifd.get(_Tag.XMP)) is not None
                        and (meta := get_ventana_properties(xmp, index))):
                    return description, meta
                if index != 0:
                    return description, {}

            case 'aperio':  # SVS
                return get_aperio_properties(description, index)

            case 'hamamatsu':  # NDPI
                if self.ifd.get(_Tag.MAKE) == 'Hamamatsu':
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
        description: str = self.ifd.get(_Tag.DESCRIPTION, '')

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

        res_unit_kind = self.ifd.get(_Tag.RES_UNIT)
        if (res_unit_kind
                and (res_unit := _RESOLUTION_UNITS.get(res_unit_kind))):
            mpp_xy = [
                res_unit / ppu
                for t in (_Tag.RES_Y, _Tag.RES_X) if (ppu := self.ifd.get(t))
            ]
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
    icc_impl: Icc | None = None

    @property
    def icc(self) -> Icc | None:
        return self.icc_impl


@dataclass(frozen=True, kw_only=True)
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
        rgb = np.asarray(rgba[..., :3], 'u1')
        return self._postprocess(rgb)


@dataclass(frozen=True, slots=True)
class _JpegDecoder:
    jpt: bytes | None = field(repr=False)
    color: str | None

    def __call__(self, buf: bytes) -> _U8:
        return imagecodecs.jpeg_decode(
            buf, tables=self.jpt, colorspace=self.color, outcolorspace='RGB')


@dataclass(eq=False, frozen=True, kw_only=True)
class _Level(ImageLevel, _BaseImage):
    memo: mmap.mmap
    spans: _U64 = field(repr=False)
    tile_shape: Shape
    decode: Callable[[bytes], _U8]
    cache: '_CacheZYXC'
    fill_orig: _U8 = field(repr=False)
    fill: _U8 = field(
        init=False, default_factory=lambda: np.full(3, 255, 'u1'))
    decimations: int = 0  # [0, 1, 2, ..., n]

    def __post_init__(self) -> None:
        self.fill[:] = self._postprocess(self.fill_orig[None, None])[0, 0, :]

    def decimate(self, num_steps: int) -> 'tuple[int, _Level]':
        th, tw, tc = self.tile_shape

        downsamples = 2 ** np.arange(num_steps + 1)
        ids, = ((th % downsamples == 0) & (tw % downsamples == 0)).nonzero()
        num_steps = ids.max()
        if num_steps == 0:  # Failed to decimate
            return (0, self)

        ds = downsamples[num_steps]
        h, w, c = self.shape
        lv = replace(
            self,
            shape=((h + ds - 1) // ds, (w + ds - 1) // ds, c),
            tile_shape=(th // ds, tw // ds, tc),
            decimations=self.decimations + num_steps,
        )
        return (num_steps, lv)

    def __eq__(self, rhs) -> bool:
        return (type(self) is type(rhs) and self.memo is rhs.memo
                and self.shape == rhs.shape)

    def __hash__(self) -> int:
        return hash(self.memo) ^ hash(self.shape)

    @shared_call  # Thread safety
    def tile(self, *idx: int) -> _U8 | None:
        lo, hi = self.spans[idx].tolist()
        if lo == hi:  # If nothing to read, don't read
            return None
        return self._get_tile_raw(lo, hi)

    def _get_tile_raw(self, lo: int, hi: int) -> _U8:
        # NOTE:
        # Like OpenSlide does we use private cache for each slide
        #   with (level, y, x) key to cache decoded pixels.
        # But we also cache opened slides to not waste time on re-opening
        #   (that can lead to multiple caches existing at the same moment).
        key = (*self.shape, id(self), lo, hi)

        # Cache hit
        if (im := self.cache[key]) is not None:
            return im

        # Cache miss
        # Read tile from disk
        im = self.decode(self.memo[lo:hi])

        # Resize if level is pooled
        th, tw = self.tile_shape[:2]
        if self.decimations == 1:
            im = cv2.resize(im, (tw, th))
        elif self.decimations >= 2:
            im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)

        self.cache[key] = im = self._postprocess(im)  # type: ignore
        return im  # type: ignore

    def part(self, *loc: Span) -> _U8:
        (_, p), = self.parts([loc])
        return p

    def parts(self,
              locs: Sequence[tuple[Span, ...]],
              max_workers: int = 0) -> Iterator[Patch]:
        if not locs:
            return
        n = len(locs)

        # (n yx lo/hi)
        boxes = np.asarray(locs, 'i4')

        # (n yxc)
        out_shapes = np.empty((n, 3), int)
        out_shapes[:, :2] = boxes @ [-1, 1]
        _, _, out_shapes[:, 2] = self.tile_shape

        # a[box idx]
        roi = np.empty((n, 2, 2), int)  # ROI to fill, [[y0 y1] [x0 x1]]
        counts = np.zeros(n, int)  # num used tiles

        # box idx -> patch, buffer to store results before they're complete
        buf: dict[int, Patch] = {}

        # tile idx -> [(loc id, o loc, t loc), ...]
        tile_to_boxes: dict[
            NDIndex,
            list[tuple[int, tuple[slice, ...], tuple[slice, ...]]],
        ] = {}

        for i, (loc, oshape, box) in enumerate(zip(locs, out_shapes, boxes)):
            coords, roi[i] = self._make_index(box)
            for iyx, oyx, tyx in coords:
                tile_to_boxes.setdefault((*iyx, 0), []).append((i, oyx, tyx))
                counts[i] += 1

            a = (
                np.broadcast_to(self.fill, oshape)
                if oshape.all() else np.empty(oshape, self.dtype))
            buf[i] = Patch(loc, a)

        # Pop initial nulls
        pos = 0
        for pos, cnt in enumerate(counts):
            if cnt > 0:
                break
            yield buf.pop(pos)

        # Read linear according to file layout
        ids = tile_to_boxes.keys()
        ids = sorted(ids, key=self.spans[:, :, :, 0].__getitem__)
        tiles = zip(ids, starmap_n(self.tile, ids, max_workers=max_workers))
        fill = self.fill

        # Unwrap & build patches
        for iyx, tile in tiles:
            # Update all parts using current tile
            for i, oyx, tyx in tile_to_boxes.get(iyx, []):
                if (out := buf[i]).data.base is fill:
                    a = np.empty_like(out.data)
                    (y0, y1), (x0, x1) = roi[i].tolist()
                    a[:y0] = fill
                    a[y0:y1, :x0] = fill
                    a[y0:y1, x1:] = fill
                    a[y1:] = fill
                    buf[i] = out = Patch(out.loc, a)

                out.data[oyx] = fill if tile is None else tile[tyx]
                counts[i] -= 1

            # Yield completed items, order preserved
            while pos < n and counts[pos] == 0:
                yield buf.pop(pos)
                pos += 1

    def _make_index(
        self, box: np.ndarray
    ) -> tuple[list[tuple[NDIndex, tuple[slice, ...], tuple[slice, ...]]],
               np.ndarray]:
        # box: (yx lo/hi)
        # (yx)
        min_ = box[:, 0]
        box_size = box @ [-1, 1]
        # (yx lo/hi)
        vbox = box.T.clip(0, self.shape[:2]).T  # box within slide
        o_loc = vbox - min_[:, None]  # region of `out` to fill

        # (yx)
        vbox_size = vbox @ [-1, 1]
        # (yx lo/hi)
        o_loc = o_loc.clip(0, box_size[:, None])

        if not vbox_size.all():  # Empty valid box
            return [], o_loc

        # (yx)
        tile_size = self.tile_shape[:2]
        i_lo = vbox[:, 0] // tile_size  # i_lo * tile <= vmin
        i_hi = -(-vbox[:, 1] // tile_size)  # i_hi * tile >= vmax

        # (ny) (nx)
        *ids, = map(range, i_lo, i_hi)
        oxs: list[list[slice]] = []
        txs: list[list[slice]] = []
        for i_lo_, i_hi_1, tsize, vspan, min__ in zip(i_lo, i_hi + 1,
                                                      tile_size, vbox, min_):
            # (n + 1)
            ts = np.arange(i_lo_, i_hi_1) * tsize
            vts = ts.clip(*vspan)
            vts_src = vts - min__

            # (n), result slices
            *o_crops, = map(slice, vts_src[:-1], vts_src[1:])
            oxs.append(o_crops)

            # (n), tile slices
            *t_crops, = map(slice, vts[:-1] - ts[:-1], vts[1:] - ts[:-1])
            txs.append(t_crops)

        return (
            [*zip(product(*ids), product(*oxs), product(*txs), strict=True)],
            o_loc,
        )


@dataclass(eq=False, frozen=True)
class _AperioLevel(_Level):
    def join(self, lv: ImageLevel) -> 'tuple[int, _Level]':
        ds, _ = super().join(lv)

        steps = max(ds.bit_length() - 1, 0)
        steps, lv = lv.decimate(steps)
        lv_ds = 2 ** steps

        if lv_ds != ds:
            lv = ProxyLevel(self.shape, scale=lv_ds / ds, base=lv)

        if not isinstance(self, _AperioSubLevel):
            r = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
            return ds, _AperioSubLevel(**r, prev=lv)

        return ds, replace(self, prev=lv)


@dataclass(eq=False, frozen=True, kw_only=True)
class _AperioSubLevel(_AperioLevel):
    prev: ImageLevel

    @shared_call  # Thread safety
    def tile(self, *idx: int) -> _U8 | None:
        lo, hi = self.spans[idx].tolist()
        if lo == hi:  # Read tile from backup level
            iy, ix, _ = idx
            th, tw = self.tile_shape[:2]
            return self.prev.part((iy * th, iy * th + th),
                                  (ix * tw, ix * tw + tw))

        return self._get_tile_raw(lo, hi)


class Tiff(Driver):
    def __init__(self, path: str) -> None:
        # Open TIFF in read-only (`r`) mode and disable memory mapping (`m`)
        self._ptr = (
            TIFF.TIFFOpenW(path, b'rm') if sys.platform == 'win32' else
            TIFF.TIFFOpen(path.encode(), b'rm'))
        if not self._ptr:
            raise ValueError('libtiff failed to open file')

        weakref.finalize(self, TIFF.TIFFClose, self._ptr)
        self._dir = 0
        self._lock = Lock()
        self.cache = _CacheZYXC()

        with open(path, 'rb') as f:
            self._memo = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        self._ifds = list(self._iter_ifds())

        # Directory 0 is main
        self.vendor, self._head, self._meta, self.mpp, self._bg_color \
            = self._ifds[0].properties()

    def _iter_ifds(self) -> Iterator[_Tags]:
        f = self._memo
        magic = f[:8]

        if magic[:4] in (b'II*\0', b'MM\0*'):
            usize = 4
            t_head = np.dtype('u2')
        elif magic in (b'II+\0\x08\0\0\0', b'MM\0+\0\x08\0\0'):
            usize = 8
            t_head = np.dtype('u8')
        else:
            raise ValueError(f'Unknown magic: {magic.hex("-", 2)}')

        t_body = np.dtype([('tag', 'u2'), ('type', 'u2'),
                           ('count', f'u{usize}'), ('value', f'{usize}u1')])
        o_dt = np.dtype(f'u{usize}')
        dtypes = _DTYPES
        if sys.byteorder != {b'II': 'little', b'MM': 'big'}[magic[:2]]:
            t_head = t_head.newbyteorder()
            t_body = t_body.newbyteorder()
            o_dt = o_dt.newbyteorder()
            dtypes = {i: dt.newbyteorder() for i, dt in dtypes.items()}

        pos = np.frombuffer(f[usize:usize * 2], o_dt).item()
        while pos:
            # Read IFD header
            num_tags = np.frombuffer(f[pos:pos + t_head.itemsize],
                                     t_head).item()
            pos += t_head.itemsize

            # Read tags
            ts = np.frombuffer(f[pos:pos + t_body.itemsize * num_tags], t_body)
            pos += t_body.itemsize * num_tags

            yield self._parse_ifd(f, ts, o_dt=o_dt, dtypes=dtypes)

            # Jump to next IFD
            pos = np.frombuffer(f[pos:pos + usize], o_dt).item()

    def _parse_ifd(self, f: '_File', ts: np.ndarray, o_dt: np.dtype,
                   dtypes: Mapping[int, np.dtype]) -> _Tags:

        m = np.isin(ts['tag'], _TAG_NAMES_A) & np.isin(ts['type'], _DTYPES_A)
        if (unknown := ts[~m][['tag', 'type']]).size:
            unknown = {t: _DTYPES.get(i, i) for t, i in unknown}
            raise ValueError(f'Unknown tags: {unknown}')
        ts = ts[m]

        tags: dict[_Tag, Any] = {}
        for tag_, type_, count, value in ts:
            dt = dtypes[type_]
            size = int(count * dt.itemsize)

            if size > o_dt.itemsize:
                o = value.view(o_dt).item()
                v = np.frombuffer(f[o:o + size], dt.base)
            else:
                v = value[:size].view(dt.base)

            if dt.shape:  # 2u4, 2i4
                v = v[::2] / v[1::2]

            if type_ == 2:  # char
                v = v.tobytes().removesuffix(b'\0')
                try:
                    v = v.decode()
                except UnicodeDecodeError:
                    v = ascii(v)
            elif type_ == 7:  # undefined
                v = v.tobytes()
            elif v.size == 1:
                v = v.item()

            tags[_Tag(tag_)] = v

        for k, t in [
            (_Tag.COMPRESSION, _Compression),
            (_Tag.COLORSPACE, _ColorSpace),
            (_Tag.ORIENTATION, _Orientation),
            (_Tag.PLANAR, _Planarity),
            (_Tag.DATETIME,
             lambda x: datetime.strptime(x, '%Y:%m:%d %H:%M:%S')),
            (_Tag.REF_BLACK_WHITE, lambda x: x.reshape(3, 2)),
            (_Tag.ICC_PROFILE, Icc),
        ]:
            if (v := tags.get(k)) is not None:
                tags[k] = t(v)

        return _Tags(tags)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({addressof(self._ptr.contents):0x})'

    def get_mpp(self) -> float | None:
        return self.mpp

    @contextmanager
    def ifd(self, index: int) -> Iterator:
        with self._lock:
            if self._dir != index:
                self._dir = index
                TIFF.TIFFSetDirectory(self._ptr, index)
            yield self._ptr

    def __len__(self) -> int:
        n = len(self._ifds)
        if n < 3:
            raise ValueError(f'Not enough levels (<3): {n}')
        return n

    def __getitem__(self, index: int) -> Image | None:
        tags = self._ifds[index]
        if index == 0:
            head = self._head
        else:
            r = tags.vendor_props(self.vendor, index)
            if not r:
                raise ValueError('File directories are from different vendors')
            head, _ = r

        if tags.color is _ColorSpace.YCBCR and tags.subsampling != (2, 2):
            raise ValueError('Unsupported YUV subsampling: '
                             f'{tags.subsampling}')

        shape = tags.image_shape
        tile_shape = tags.tile_shape

        if tile_shape is None:  # Not tiled
            return _Image(
                shape,
                icc_impl=tags.ifd.get(_Tag.ICC_PROFILE),
                tiff=self,
                head=head,
                index=index,
            )

        spans = tags.tile_spans(shape, tile_shape)

        # Aperio can choke on non-L0 levels. Read those from L0 to fix.
        # `openslide` does this for us, delegate to it.
        if self.vendor == 'aperio' and (spans[..., 0] == spans[..., 1]).any():
            raise ValueError('Found 0s in tile size table')

        return _Level(
            shape,
            icc_impl=tags.ifd.get(_Tag.ICC_PROFILE),
            memo=self._memo,
            spans=spans,
            tile_shape=tile_shape,
            decode=tags.get_decoder(),
            cache=self.cache,
            fill_orig=self._bg_color,
        )


# ------------------------------- tile caching -------------------------------


class _CacheZYXC:
    __slots__ = ('lock', 'used', 'buf')

    def __init__(self) -> None:
        self.lock = Lock()
        self.used: int = 0
        self.buf: dict[tuple, tuple[int, _U8]] = {}  # IYXC -> (size, buf)

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
        size = obj.nbytes
        if size > capacity:
            warnings.warn(
                f'Rejecting overlarge cache entry of size {size} bytes',
                stacklevel=3)
            return
        max_size = capacity - size
        with self.lock:
            while self.buf and self.used > max_size:
                key = next(iter(self.buf))
                self.used -= self.buf.pop(key)[0]
            if self.used <= max_size:
                self.buf[key] = (size, obj)
                self.used += size


_DTYPES: dict[int, np.dtype] = {
    1: np.dtype('u1'),
    2: np.dtype('u1'),  # Null-terminated ascii string
    3: np.dtype('u2'),
    4: np.dtype('u4'),
    5: np.dtype('2u4'),
    6: np.dtype('i1'),
    7: np.dtype('u1'),  # Undefined
    8: np.dtype('i2'),
    9: np.dtype('i4'),
    10: np.dtype('2i4'),
    11: np.dtype('f4'),
    12: np.dtype('f8'),
    16: np.dtype('u8'),
}

_DTYPES_A = np.array([*_DTYPES], 'u2')
_TAG_NAMES_A = np.array([*_Tag._value2member_map_], 'u2')
