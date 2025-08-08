"""
Pure python driver inspired by libtiff
- fast
- thread safe
- compatible with different variants of TIFF like AperioSVS, Ventana, e.t.c.
- not compatible with Hamamatsu NDPI
"""

__all__ = ['Tiff']

import struct
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from functools import partial
from itertools import chain, product
from math import gcd
from threading import Lock
from typing import Any, Self, assert_never

import cv2
import imagecodecs
import numpy as np
import numpy.typing as npt
from glow import memoize, si_bin, starmap_n

from bipl._env import env
from bipl._fileio import Paged, fopen
from bipl._types import NDIndex, Patch, Shape, Span

from ._slide_bases import Driver, Image, ImageLevel, PartMixin, ProxyLevel
from ._util import (
    Icc,
    get_aperio_properties,
    get_ventana_properties,
    parse_xml,
    unflatten,
)

_U8 = npt.NDArray[np.uint8]
_U64 = npt.NDArray[np.uint64]

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
    # SUBIFDS = 330  # offset to child IFDs
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
    CZ_LSMINFO = 34412
    ICC_PROFILE = 34675
    # EER metadata
    BITS_SKIP_POS = 65007
    BITS_HORZ_SUB = 65008
    BITS_VERT_SUB = 65009
    # NDPI stiff
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


class _ImageFileDirectory(dict[_Tag, Any]):
    def __init__(self, tags: dict[_Tag, Any], index: int) -> None:
        # Use strong types
        for k, t in [
            (_Tag.COLORSPACE, _ColorSpace),
            (_Tag.ORIENTATION, _Orientation),
            (_Tag.PLANAR, _Planarity),
            (
                _Tag.DATETIME,
                lambda x: datetime.strptime(x, '%Y:%m:%d %H:%M:%S'),
            ),
            (_Tag.PREDICTOR, _unpredictors.get),
            (_Tag.REF_BLACK_WHITE, lambda x: x.reshape(3, 2)),
            (_Tag.ICC_PROFILE, Icc),
        ]:
            if (v := tags.get(k)) is not None:
                tags[k] = t(v)

        super().__init__(tags)
        self.index = index

        if (
            self.get(_Tag.NDPI_VERSION) == 1
            or self.get(_Tag.MAKE) == 'Hamamatsu'
        ):
            raise ValueError('TIFF: Hamamatsu is not yet supported')
        # ! unused
        self.bps = self.get(_Tag.BITS_PER_SAMPLE, 1)
        self.sample_format = self.get(_Tag.SAMPLE_FORMAT, 1)

        self.subsampling = (1, 1)

        # TODO: use this in YCbCr conversion
        self.gray = np.array([], 'f4')  # Luma coefficients
        self.yuv_centered = True
        self.yuv_bw = np.array([], 'f4')  # BW pairs, per channel

        if self[_Tag.COLORSPACE] is _ColorSpace.YCBCR:
            self.gray = np.array([0.299, 0.587, 0.114], 'f4')
            self.gray = self.get(_Tag.YUV_COEFFICIENTS, self.gray)

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
            ss_w, ss_h = self.get(_Tag.YUV_SUBSAMPLING, (2, 2))
            self.subsampling = ss_h, ss_w

            self.yuv_centered = self.get(_Tag.YUV_POSITIONING) != 2
            self.yuv_bw = self.get(_Tag.REF_BLACK_WHITE, self.yuv_bw)

        self.image_shape = (
            self[_Tag.IMAGE_HEIGHT],
            self[_Tag.IMAGE_WIDTH],
            self[_Tag.SAMPLES_PER_PIXEL],
        )
        self.is_tiled, self.unit_shape, self.spans = self._tiling_schema()

    def _tiling_schema(self) -> tuple[bool, Shape, _U64]:
        """
        Returns tile shape &
        (h w d 2) tensor of start:stop of each tile w.r.t file.
        """
        spp = self[_Tag.SAMPLES_PER_PIXEL]

        if th := self.get(_Tag.TILE_HEIGHT, 0):  # Tiles
            tw = self[_Tag.TILE_WIDTH]
            offset_tag = _Tag.TILE_OFFSETS
            byte_tag = _Tag.TILE_NBYTES
            is_tiled = True

        else:  # Strips
            th = self[_Tag.STRIP_HEIGHT]
            tw = self[_Tag.IMAGE_WIDTH]
            offset_tag = _Tag.STRIP_OFFSETS
            byte_tag = _Tag.STRIP_NBYTES
            is_tiled = False

        unit_shape = (th, tw, spp)
        grid_shape = tuple(
            len(range(0, s, t)) for s, t in zip(self.image_shape, unit_shape)
        )
        tbs: _U64 = np.reshape(self[offset_tag], grid_shape)
        tbc: _U64 = np.reshape(self[byte_tag], grid_shape)

        grid = np.stack((tbs, tbs + tbc), -1)

        return is_tiled, unit_shape, grid

    def get_decoder(self) -> Callable[[bytes], _U8]:
        compression = self[_Tag.COMPRESSION]
        try:
            decompress = _decompressors[compression]
        except KeyError:
            raise ValueError(
                f'IFD: {self.index}: Unknown compression {compression}'
            ) from None

        if decompress is None:  # RAW
            tile_size = np.prod(self.unit_shape)
            self.spans[..., 1] = self.spans[..., 0] + tile_size

        if decompress in {imagecodecs.jpeg_decode, imagecodecs.jpeg8_decode}:
            return partial(
                decompress,
                tables=self.get(_Tag.JPEG_TABLES),
                colorspace=self[_Tag.COLORSPACE].name,
                outcolorspace='RGB',
            )

        if decompress == imagecodecs.eer_decode:
            match compression:
                case 65001:
                    rlebits, horzbits, vertbits = 8, 2, 2
                case 65002:
                    rlebits, horzbits, vertbits = 7, 2, 2
                case 65002:
                    rlebits = self.get(_Tag.BITS_SKIP_POS, 7)
                    horzbits = self.get(_Tag.BITS_HORZ_SUB, 2)
                    vertbits = self.get(_Tag.BITS_VERT_SUB, 2)
                case _ as unreachable:
                    assert_never(unreachable)

            return partial(
                imagecodecs.eer_decode,
                shape=(self.unit_shape[0], self.unit_shape[1]),
                rlebits=rlebits,
                horzbits=horzbits,
                vertbits=vertbits,
                superres=False,
            )

        if decompress is _jetraw_decode:
            return partial(_jetraw_decode, shape=self.unit_shape)

        if decompress in {
            imagecodecs.jpeg2k_decode,
            imagecodecs.jpegxl_decode,
            imagecodecs.jpegxr_decode,
            imagecodecs.png_decode,
            imagecodecs.webp_decode,
        }:
            return decompress

        unpredict: Callable | None = self.get(_Tag.PREDICTOR)
        return _LambdaDecoder(decompress, unpredict, self.unit_shape)

    def _bg_color(self, meta: dict) -> _U8:
        if c := meta.get('ScanWhitePoint'):
            return np.array(int(c), 'u1')

        bg_hex: bytes = self.get(_Tag.BACKGROUND_COLOR, b'FFFFFF')
        return np.frombuffer(bytes.fromhex(bg_hex.decode()), 'u1').copy()

    def vendor_props(self, vendor: str) -> tuple[str, dict[str, str]] | None:
        description = self.get(_Tag.DESCRIPTION, '')
        xmp = self.get(_Tag.XMP)
        assert isinstance(description, str)

        match vendor:
            case 'ventana':  # BIF of Roche
                if xmp is not None and (
                    meta := get_ventana_properties(xmp, self.index)
                ):
                    return description, meta
                if self.index != 0:
                    return description, {}

            case 'aperio':  # SVS
                return get_aperio_properties(description, self.index)

            case 'generic':  # TIFF
                try:
                    meta = unflatten(parse_xml(description))
                except Exception:  # noqa: BLE001
                    return description, {}
                else:
                    return '', meta

        return None

    def _get_mpp(self, vendor: str, meta: dict) -> float | None:
        """Extract MPP, um/pixel, phisical pixel size"""
        if vendor == 'ventana':
            # Ventana messes with TIFF tag XResolution, YResolution
            return float(mpp_s) if (mpp_s := meta.get('ScanRes')) else None

        res_unit_kind = self.get(_Tag.RES_UNIT)
        if res_unit_kind and (
            res_unit := _RESOLUTION_UNITS.get(res_unit_kind)
        ):
            mpp_xy = [
                res_unit / ppu
                for t in (_Tag.RES_Y, _Tag.RES_X)
                if (ppu := self.get(t))
            ]
            if mpp_xy and (mpp := float(np.mean(mpp_xy))):
                return mpp

        if mpp_s := meta.get('MPP'):
            return float(mpp_s)
        return None

    def properties(self) -> tuple[str, str, dict[str, str], float | None, _U8]:
        """Extracts: (vendor, header, metadata, mpp, bg color)"""
        for vendor in ('ventana', 'aperio', 'generic'):
            if r := self.vendor_props(vendor):
                header, meta = r
                break
        else:
            raise ValueError('Unknown vendor')

        mpp = self._get_mpp(vendor, meta)
        bg_color = self._bg_color(meta)
        return vendor, header, meta, mpp, bg_color


# ------------------ image, level & opener implementations -------------------


@dataclass(frozen=True, slots=True)
class _LambdaDecoder:
    decompress: Callable[[bytes], bytes] | None
    unpredict: Callable | None
    shape: Shape

    def __call__(self, buf: bytes) -> _U8:
        if self.decompress:
            buf = self.decompress(buf)

        h, w, c = self.shape
        arr = np.frombuffer(buf, 'u1')
        arr = arr[: h * w * c].reshape(-1, w, c)

        if self.unpredict:
            arr = self.unpredict(arr, axis=-2, out=arr)

        if arr.shape[0] == h:
            return arr
        return np.pad(arr, ((0, h - arr.shape[0]), (0, 0), (0, 0)))


def _jetraw_decode(buf: bytes, *, shape: Shape) -> _U8:
    out = np.zeros(shape, 'u2')
    imagecodecs.jetraw_decode(buf, out=out.ravel())
    return out


# For IDs see bipl.io._tiff_compressions
_decomp_to_ids: dict[Callable | None, set[int]] = {
    # bytes -> bytes
    None: {1},
    imagecodecs.deflate_decode: {8, 32946, 50013},
    imagecodecs.lzma_decode: {34925},
    imagecodecs.lzw_decode: {5},
    imagecodecs.packbits_decode: {32773},
    imagecodecs.zstd_decode: {34926, 50000},
    # bytes -> ndarray, no predictor
    _jetraw_decode: {48124},
    imagecodecs.eer_decode: {65000, 65001, 65002},
    imagecodecs.jpeg_decode: {6, 7, 33007},
    imagecodecs.jpeg2k_decode: {33003, 33004, 33005, 34712},
    imagecodecs.jpeg8_decode: {34892},
    imagecodecs.jpegxl_decode: {50002, 52546},
    imagecodecs.jpegxr_decode: {22610, 34934},
    imagecodecs.lerc_decode: {34887},
    imagecodecs.png_decode: {34933},
    imagecodecs.webp_decode: {34927, 50001},
}
_decompressors: dict[int, Callable | None] = {}
for _decomp, _codec_ids in _decomp_to_ids.items():
    for _codec_id in _codec_ids:
        assert _codec_id not in _decompressors, f'Duplicates: {_decomp_to_ids}'
        _decompressors[_codec_id] = _decomp


_unpredictors: dict[int, Callable] = {
    2: imagecodecs.delta_decode,
    3: imagecodecs.floatpred_decode,
    34892: partial(imagecodecs.delta_decode, dist=2),
    34893: partial(imagecodecs.delta_decode, dist=4),
    34894: partial(imagecodecs.floatpred_decode, dist=2),
    34895: partial(imagecodecs.floatpred_decode, dist=4),
}


@dataclass(eq=False, frozen=True, kw_only=True)
class _BaseImage(Image):
    icc_impl: Icc | None = None
    buf: Paged
    path: str
    spans: _U64 = field(repr=False)
    tile_shape: Shape
    decode: Callable[[bytes], _U8]
    cache: '_CacheZYXC'
    fill: _U8
    decimations: int = 0  # [0, 1, 2, ..., n]
    prev: ImageLevel | None = None

    @property
    def icc(self) -> Icc | None:
        return self.icc_impl

    def decimate(self, dst: float, src: int = 1) -> tuple[int, Self]:
        assert dst >= 1
        assert src >= 1
        th, tw, tc = self.tile_shape
        t = gcd(th, tw)

        max_ds = (-t) & t  # See: https://stackoverflow.com/q/1551775
        steps = min(int(dst / src), max_ds).bit_length() - 1

        if steps <= 0:  # No need or cannot decimate
            return (src, self)

        ds = 2**steps
        h, w, c = self.shape

        prev = self.prev
        shape = ((h + ds - 1) // ds, (w + ds - 1) // ds, c)

        if prev is not None:
            prev_ds, prev = prev.decimate(ds, 1)
            if prev_ds != ds:
                prev = prev.rescale(prev_ds / ds)

        lv = replace(
            self,
            shape=shape,
            tile_shape=(th // ds, tw // ds, tc),
            decimations=self.decimations + steps,
            prev=prev,
        )
        return (src * ds, lv)

    def __eq__(self, rhs) -> bool:  # Used by _Level.tile:shared_call
        return (
            type(self) is type(rhs)
            and self.buf is rhs.buf
            and self.shape == rhs.shape
            and self.decimations == rhs.decimations
        )

    def __hash__(self) -> int:  # Used by _Level.tile:shared_call
        return hash((self.buf, self.shape, self.decimations))

    @memoize(0)  # Thread safety
    def tile(self, *idx: int) -> _U8 | None:
        lo, hi = self.spans[idx].tolist()
        if lo == hi:  # If nothing to read, don't read
            return None

        # NOTE:
        # Like OpenSlide does we use private cache for each slide
        #   with (level, y, x) key to cache decoded pixels.
        # But we also cache opened slides to not waste time on re-opening
        #   (that can lead to multiple caches existing at the same moment).
        key = (self.decimations, lo, hi)

        # Cache hit
        if (im := self.cache[key]) is not None:
            return self._postprocess(im)

        # Cache miss
        # Read tile from disk
        im = self.decode(self.buf.pread(hi - lo, lo))

        # Resize if level is pooled
        th, tw = self.tile_shape[:2]
        if self.decimations == 1:
            im = cv2.resize(im, (tw, th))
        elif self.decimations >= 2:
            im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)

        self.cache[key] = im
        return self._postprocess(im)

    def parts(
        self, locs: Sequence[tuple[Span, ...]], max_workers: int = 0
    ) -> Iterator[Patch]:
        if not locs:
            return
        n = len(locs)

        # (n yx lo/hi)
        boxes = np.asarray(locs, 'i4')
        th, tw, spp = self.tile_shape

        # (n yxc)
        out_shapes = np.empty((n, 3), int)
        out_shapes[:, :2] = boxes @ [-1, 1]
        out_shapes[:, 2] = spp

        # tile idx -> [(loc id, o loc, t loc), ...]
        tile_to_boxes: dict[
            NDIndex,
            list[tuple[int, tuple[slice, ...], tuple[slice, ...]]],
        ] = {}

        counts = np.zeros(n, int)  # num used tiles
        box_ids, mins_, vboxes, i_lows, i_highs, counts_, o_locs = (
            self._init_index(boxes)
        )
        counts[box_ids] = counts_
        for i, *args in zip(box_ids.tolist(), mins_, vboxes, i_lows, i_highs):
            for iyx, oyx, tyx in self._make_subindex(*args):
                tile_to_boxes.setdefault((*iyx, 0), []).append((i, oyx, tyx))

        # box idx -> patch, buffer to store results before they're complete
        fill = self._postprocess(self.fill[None, None])[0, 0, :]
        buf: dict[int, Patch] = {
            i: Patch(loc, np.broadcast_to(fill, oshape))
            for i, (loc, oshape) in enumerate(zip(locs, out_shapes))
        }
        # Pop initial nulls
        pos = box_ids[0] if box_ids.size else n
        for i in range(pos):
            yield buf.pop(i)
        if not buf:
            return

        # Read linear according to file layout
        ids: list[NDIndex] = []
        nulls: list[NDIndex] = []
        for iyx in tile_to_boxes:
            nbytes = self.spans[iyx] @ [-1, 1]
            (ids, nulls)[int(nbytes) == 0].append(iyx)

        prev_tiles: Iterator[tuple[NDIndex, np.ndarray | None]]
        if self.prev is not None and nulls:
            prev_parts = self.prev.parts(
                [
                    ((iy * th, iy * th + th), (ix * tw, ix * tw + tw))
                    for iy, ix, _ in nulls
                ],
                max_workers=max_workers,
            )
            prev_tiles = ((i, a) for i, (_, a) in zip(nulls, prev_parts))
        else:
            prev_tiles = ((i, None) for i in nulls)

        ids = sorted(ids, key=self.spans[:, :, :, 0].__getitem__)
        tiles = zip(ids, starmap_n(self.tile, ids, max_workers=max_workers))

        # Unwrap & build patches
        rois: dict[int, tuple[Span, ...]] = dict(
            zip(box_ids.tolist(), o_locs.tolist())
        )
        for iyx, tile in chain(prev_tiles, tiles):

            # Update all parts using current tile
            for i, oyx, tyx in tile_to_boxes.get(iyx, []):
                out = buf[i]

                if roi := rois.pop(i, None):
                    a = np.empty_like(out.data)
                    (y0, y1), (x0, x1) = roi
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

    def _init_index(self, boxes: np.ndarray) -> tuple[  # (n yx lo/hi)
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        shape = np.reshape(self.shape[:2], (2, 1))
        tile_size = np.reshape(self.tile_shape[:2], (1, 2))

        # (n yx lo/hi), box within slide
        vboxes = boxes.clip(0, shape)

        # (n yx)
        mins_ = boxes[:, :, 0]
        box_sizes = boxes @ [-1, 1]
        vbox_sizes = vboxes @ [-1, 1]

        # (n), drop empty "valid" boxes
        mask = vbox_sizes.all(-1)
        [box_ids] = mask.nonzero()
        mins_, box_sizes, vboxes = mins_[mask], box_sizes[mask], vboxes[mask]

        # (n yx)
        i_lows = vboxes[:, :, 0] // tile_size  # i_lo * tile <= vmin
        i_highs = -(-vboxes[:, :, 1] // tile_size)  # i_hi * tile >= vmax

        # (n)
        counts = (i_highs - i_lows).prod(-1)
        assert counts.all()

        # (n yx lo/hi)
        o_locs = vboxes - mins_[:, :, None]  # region of `out` to fill
        o_locs = o_locs.clip(0, box_sizes[:, :, None])

        # (n) (n yx) (n yx lo/hi) (n yx) (n yx) (n) {i -> (yx lo/hi)}
        return box_ids, mins_, vboxes, i_lows, i_highs, counts, o_locs

    def _make_subindex(
        self,
        min_: np.ndarray,  # (yx)
        vbox: np.ndarray,  # (yx lo/hi)
        i_lo: np.ndarray,  # (yx)
        i_hi: np.ndarray,  # (yx)
    ) -> list[tuple[NDIndex, tuple[slice, ...], tuple[slice, ...]]]:
        # (yx)
        tile_size = self.tile_shape[:2]

        # (ny) (nx)
        ids = map(range, i_lo, i_hi)
        oxs: list[list[slice]] = []
        txs: list[list[slice]] = []
        for i_lo_, i_hi_1, tsize, vspan, amin_ in zip(
            i_lo, i_hi + 1, tile_size, vbox, min_
        ):
            # (n + 1)
            ts = np.arange(i_lo_, i_hi_1) * tsize
            vts = ts.clip(*vspan)
            vts_src = vts - amin_

            # (n), result slices
            o_crops = map(slice, vts_src[:-1], vts_src[1:])
            oxs.append(list(o_crops))

            # (n), tile slices
            t_crops = map(slice, vts[:-1] - ts[:-1], vts[1:] - ts[:-1])
            txs.append(list(t_crops))

        return [*zip(product(*ids), product(*oxs), product(*txs), strict=True)]

    def apply(self, fn: Callable[[np.ndarray], np.ndarray]) -> Self:
        post = [*self.post, fn]

        prev = self.prev
        if prev is not None:
            prev = replace(prev, post=post)

        return replace(self, post=post, prev=prev)


def _get_key(
    ifd: _ImageFileDirectory, vendor: str, head: str, index: int
) -> str | None:
    if ifd.is_tiled:
        return None

    match (vendor, index, head):
        case ('aperio', 1, _) | ('ventana', _, 'Thumbnail'):
            return 'thumbnail'
        case ('ventana', _, 'Label Image') | ('ventana', _, 'Label_Image'):
            return 'macro'
        case ('ventana', _, _):
            return None

    for key in ('label', 'macro'):
        if any(key in s for s in head.splitlines()):
            return key
    return None


@dataclass(eq=False, frozen=True, kw_only=True)
class _Image(_BaseImage):
    def __eq__(self, rhs) -> bool:  # Used by _Level.tile:shared_call
        return (
            type(self) is type(rhs)
            and self.buf is rhs.buf
            and self.shape == rhs.shape
            and self.decimations == rhs.decimations
            and self.key == rhs.key
        )

    def __hash__(self) -> int:  # Used by _Level.tile:shared_call
        return hash((self.buf, self.shape, self.decimations, self.key))

    def numpy(self) -> np.ndarray:
        """Retrieve whole image as array"""
        h, w, _ = self.shape
        [(_, a)] = self.parts([((0, int(h)), (0, int(w)))])
        return a


@dataclass(eq=False, frozen=True, kw_only=True)
class _Level(PartMixin, ImageLevel, _BaseImage):
    def __eq__(self, rhs) -> bool:  # Used by _Level.tile:shared_call
        return (
            type(self) is type(rhs)
            and self.buf is rhs.buf
            and self.shape == rhs.shape
            and self.decimations == rhs.decimations
        )

    def __hash__(self) -> int:  # Used by _Level.tile:shared_call
        return hash((self.buf, self.shape, self.decimations))


class Tiff(Driver):
    def __init__(self, path: str) -> None:
        self._buf = fopen(path)
        self._path = path
        self.cache = _CacheZYXC()
        self._ifds = list(self._iter_ifds())

        # Directory 0 is main
        self.vendor, self._head, self._meta, self.mpp, self._bg_color = (
            self._ifds[0].properties()
        )

    def _iter_ifds(self) -> Iterator[_ImageFileDirectory]:
        header = self._buf.pread(4, 0)
        try:
            byteorder = {b'II': '<', b'MM': '>', b'EP': '<'}[header[:2]]
        except KeyError as exc:
            raise ValueError(f'not a TIFF file {header!r}') from exc
        assert byteorder in ('<', '>')

        [version] = struct.unpack(byteorder + 'H', header[2:4])
        if version == 42:  # TIFF
            usize = 4
            head_fmt = 'H'  # u2
            head_size = 2
            o_fmt = 'I'  # u4
        elif version == 43:  # BigTIFF
            usize, zero = struct.unpack(
                byteorder + 'HH', self._buf.pread(4, 4)
            )
            if usize != 8 or zero != 0:
                raise ValueError(
                    f'invalid BigTIFF offset size {(usize, zero)}'
                )
            head_fmt = o_fmt = 'Q'  # u8
            head_size = 8
        else:
            raise ValueError(f'invalid TIFF version: {version}')

        t_body = np.dtype(
            [
                ('tag', 'H'),
                ('type', 'H'),
                ('count', o_fmt),
                ('value', f'{usize}B'),
            ]
        )
        dtypes = _DTYPES
        if sys.byteorder != {'<': 'little', '>': 'big'}[byteorder]:
            head_fmt = byteorder + head_fmt
            o_fmt = byteorder + o_fmt
            t_body = t_body.newbyteorder()
            dtypes = {i: dt.newbyteorder() for i, dt in dtypes.items()}

        [pos] = struct.unpack(o_fmt, self._buf.pread(usize, usize))
        idx = 0
        while pos:
            # Read IFD header
            [num_tags] = struct.unpack(
                head_fmt, self._buf.pread(head_size, pos)
            )
            pos += head_size

            # Read tags
            ts = np.frombuffer(
                self._buf.pread(t_body.itemsize * num_tags, pos), t_body
            )
            pos += ts.nbytes

            tags = self._get_ifd_tags(ts, o_fmt=o_fmt, dtypes=dtypes)
            yield _ImageFileDirectory(tags, idx)

            # Jump to next IFD
            [pos] = struct.unpack(o_fmt, self._buf.pread(usize, pos))
            idx += 1

    def _get_ifd_tags(
        self, ts: np.ndarray, o_fmt: str, dtypes: Mapping[int, np.dtype]
    ) -> dict[_Tag, Any]:
        m = np.isin(ts['tag'], _TAG_NAMES_A) & np.isin(ts['type'], _DTYPES_A)
        if (unknown := ts[~m][['tag', 'type']]).size:
            unknown = {t: _DTYPES.get(i, i) for t, i in unknown}
            raise ValueError(f'Unknown tags: {unknown}')
        ts = ts[m]

        tags: dict[_Tag, Any] = {}
        for tag_, type_, count, value in ts:
            dt = dtypes[type_]
            size = int(count * dt.itemsize)

            if size > struct.calcsize(o_fmt):
                [o] = struct.unpack(o_fmt, value)
                v = np.frombuffer(self._buf.pread(size, o), dt.base)
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

        return tags

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._path})'

    def get_mpp(self) -> float | None:
        return self.mpp

    def __len__(self) -> int:
        n = len(self._ifds)
        if n < 3:
            raise ValueError(f'Not enough levels (<3): {n}')
        return n

    def __getitem__(self, index: int) -> Image | None:
        ifd = self._ifds[index]
        if index == 0:
            head = self._head
        else:
            r = ifd.vendor_props(self.vendor)
            if not r:
                raise ValueError('File directories are from different vendors')
            head, _ = r

        if ifd[_Tag.COLORSPACE] is _ColorSpace.YCBCR and (
            ifd.subsampling != (2, 2)
        ):
            raise ValueError(f'Unsupported YUV subsampling: {ifd.subsampling}')

        key = _get_key(ifd, self.vendor, head, index)
        tp = _Level if ifd.is_tiled else _Image
        return tp(
            ifd.image_shape,
            key=key,
            icc_impl=ifd.get(_Tag.ICC_PROFILE),
            buf=self._buf,
            path=self._path,
            spans=ifd.spans,
            tile_shape=ifd.unit_shape,
            decode=ifd.get_decoder(),
            cache=self.cache,
            fill=self._bg_color,
        )

    def build_pyramid(
        self, levels: Sequence[ImageLevel]
    ) -> tuple[tuple[int, ...], list[ImageLevel]]:
        downsamples, levels = super().build_pyramid(levels)

        if self.vendor != 'aperio':
            return downsamples, levels

        # Aperio can choke on non-L0 levels. Read those from L0 to fix.
        # `openslide` does this for us, delegate to it.
        for i, (ds, lv) in enumerate(zip(downsamples, levels)):
            assert isinstance(lv, _Level)
            empty = lv.spans[..., 0] == lv.spans[..., 1]

            if i == 0:
                if empty.any():
                    raise ValueError('Found 0s in tile size table')
                continue

            if empty.any():  # Cut tile grid right after first empty tile
                first_empty = empty.argmax()
                lv.spans.reshape(-1, 2)[first_empty:] = 0

            prev = levels[0]
            prev_ds, prev = prev.decimate(ds, 1)

            if prev_ds != ds:
                prev = ProxyLevel(
                    lv.shape,
                    prev.post,
                    scale=prev_ds / ds,
                    base=replace(prev, post=[]),
                )

            levels[i] = replace(lv, prev=prev)

        return downsamples, levels


# ------------------------------- tile caching -------------------------------


class _CacheZYXC:
    __slots__ = ('items', 'lock', 'used')

    def __init__(self) -> None:
        self.lock = Lock()
        self.used: int = 0
        self.items: dict[tuple, tuple[int, _U8]] = {}  # IYXC -> (size, buf)

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}'
            f'(used={si_bin(self.used)}, items={len(self.items)})'
        )

    def __getitem__(self, key: tuple) -> _U8 | None:
        with self.lock:
            if e := self.items.get(key):
                return e[1]
        return None

    def __setitem__(self, key: tuple, obj: _U8) -> None:
        if not (capacity := env.BIPL_TILE_CACHE):
            return
        size = obj.nbytes
        if size > capacity:
            warnings.warn(
                f'Rejecting overlarge cache entry of size {size} bytes',
                stacklevel=3,
            )
            return
        max_size = capacity - size
        with self.lock:
            while self.items and self.used > max_size:
                k = next(iter(self.items))
                self.used -= self.items.pop(k)[0]
            if self.used <= max_size:
                self.items[key] = (size, obj)
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
