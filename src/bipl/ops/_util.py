__all__ = [
    'at',
    'crop_to',
    'get_fusion',
    'get_trapz',
    'merge_intervals',
    'normalize_loc',
    'probs_to_rgb_heatmap',
    'resize',
]

from collections.abc import Iterable, Sequence
from itertools import starmap
from math import ceil, floor

import cv2
import numpy as np

from bipl import env
from bipl._types import NumpyLike, Span, Tile, Vec


def probs_to_rgb_heatmap(prob: np.ndarray) -> np.ndarray:
    """Converts probability array of (H W C) shape to (H W 3) RGB image"""
    if prob.ndim != 3:
        raise ValueError(f'prob should be 3d, got: {prob.shape}')
    h, w, c = prob.shape
    if not prob.size:
        return np.empty((h, w, 3), dtype='u1')

    hue = prob.argmax(-1).astype('u1')  # NOTE: valid only for simple Indexer
    value = prob.max(-1)
    if value.dtype != 'u1':
        value *= 255
        value = value.round().astype('u1')

    hue_lut = np.zeros(256, 'u1')
    hue_lut[:c] = np.linspace(0, 127, c, endpoint=False, dtype='u1')
    hsv = cv2.merge(
        [
            cv2.LUT(hue, hue_lut),
            np.broadcast_to(np.uint8(255), (h, w)),
            value,
        ]
    )
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def get_trapz(step: int, overlap: int) -> np.ndarray:
    """Returns trapezional window kernel to apply stiching"""
    if overlap == 0:
        raise ValueError('not applicable for overlap 0')

    ramp = np.arange(1, overlap + 1).astype('f4')
    ramp /= overlap + 1

    r = np.empty(step + overlap, dtype='f4')
    r[:overlap] = ramp
    r[overlap:step] = 1
    r[step:] = ramp[::-1]
    return r


def normalize_loc(
    loc: Sequence[slice] | slice, shape: Sequence[int]
) -> tuple[Span, ...]:
    """Ensures slices match ndim and have noo `None` endpoints"""
    if isinstance(loc, slice):
        loc = (loc,)
    if any(s.step not in (None, 1) for s in loc):
        raise ValueError(f'Slice step is not supported, got: {loc}')
    loc_ = [
        (s.start or 0, size if s.stop is None else s.stop)
        for s, size in zip(loc, shape)
    ]
    return _pad_loc(loc_, shape)


def _pad_loc(loc: Sequence[Span], shape: Sequence[int]) -> tuple[Span, ...]:
    n = len(loc)
    ndim = len(shape)
    if n > ndim:
        raise ValueError(f'loc is too deep for {ndim}D, got: {loc}')
    return *loc, *((0, s) for s in shape[n:])


def padslice(a: NumpyLike, *loc: Span) -> np.ndarray:
    """
    Do `a[loc]`, but extend `a` (with 0s) if `loc` indices beyond `a.shape`.
    """
    loc = _pad_loc(loc, a.shape)
    r_shape = [hi - lo for lo, hi in loc]

    # Nothing to pad
    if not all(a.shape):
        return np.zeros(r_shape, a.dtype)

    # No need to pad
    if all(0 <= lo <= hi <= size for (lo, hi), size in zip(loc, a.shape)):
        return at(a, *loc)

    iloc = tuple(slice(*np.clip(s, 0, size)) for s, size in zip(loc, a.shape))
    rloc = tuple(
        slice(i.start - lo, i.stop - lo) for (lo, _), i in zip(loc, iloc)
    )

    r = np.empty(r_shape, a.dtype)
    pad: tuple[slice, ...] = ()
    for o in rloc:
        r[*pad, : o.start] = 0
        r[*pad, o.stop :] = 0
        pad += (o,)
    r[rloc] = a[iloc]
    return r


def crop_to(
    vec: Vec, a: NumpyLike, shape: Sequence[int]
) -> tuple[Vec, np.ndarray]:
    """
    Crop `a` to be completely within shape, i.e.
    `0 <= vec[i] <= vec[i] + a.shape[i] <= shape[i]`.

    Arguments:
    - vec - top-left offset of `a` w.r.t. target.
    - a - data to crop.
    - shape - target shape to fit into.

    Returns new offset & data.
    """
    assert len(a.shape) >= len(shape), 'Desired shape has more dims than `a`'
    loc = tuple(
        tuple(np.clip([0, ts], -t0, s - t0))
        for t0, ts, s in zip(vec, a.shape, shape)
    )
    a = padslice(a, *loc)

    vec = tuple(max(v, 0) for v in vec)
    return vec, a


def resize(
    image: np.ndarray, hw: Sequence[int], *, antialias: bool | None = None
) -> np.ndarray:
    if image.shape[:2] == hw[:2]:
        return image

    f_max = max(s0 / s1 for s0, s1 in zip(image.shape, hw))
    if f_max < 1:  # Upscaling
        antialias = False
    elif antialias is None:  # Enable only if we downscale more than 2.5x
        antialias = f_max > 2.5

    h, w = hw
    interpolation = cv2.INTER_LINEAR
    if antialias:  # Downsampling
        match env.BIPL_DOWN:
            case 'box':  # Pyramid, box filter. Fastest
                for _ in range(int(f_max).bit_length() - 1):
                    image = cv2.resize(image, None, fx=0.5, fy=0.5)

            case 'gauss':  # Pyramid, gauss filter. No aliasing. 2X slower
                for _ in range(int(f_max).bit_length() - 1):
                    image = cv2.pyrDown(image)

            case 'area':  # No pyramid, box filter. Slowest
                interpolation = cv2.INTER_AREA

    return cv2.resize(image, (w, h), interpolation=interpolation)


def get_fusion(
    tiles: Iterable[Tile], shape: Sequence[int] | None = None
) -> np.ndarray | None:
    """Stack tiles to large image. Tiles lying outside of bounds are cropped"""
    r: np.ndarray | None = None

    if shape is None:  # Collect all the tiles to compute destination size
        tiles = [*tiles]
        if not tiles:
            return None
        # N arrays of (yx + hw)
        yx_hw = np.array([[[t.vec, t.data.shape[:2]] for t in tiles]]).sum(1)
        # bottom left most edge
        shape = tuple(yx_hw.max(0).tolist())
    elif len(shape) != 2:
        raise ValueError(f'shape should be 2-tuple, got {shape}')

    for _, vec, tile in tiles:
        (y, x), a = crop_to(vec, tile, shape)
        if not a.size:
            continue

        h, w, c = a.shape
        if r is None:  # First iteration, initialize
            r = np.zeros((*shape, c), a.dtype)
        if c != r.shape[2]:
            raise RuntimeError('tile channel counts changed during iteration')
        r[y : y + h, x : x + w] = a

    return r


def rescale_crop(
    a: NumpyLike, *loc: Span, scale: float = 1, interpolation: int = 0
):
    """
    Rescale image, then crop with respect to subpixels.

    Interpolation is Nearest (0) by default, but Linear (1), Bicubic (2),
    Area (3) & Lanczos-4 (4) are also supported (OpenCV codes).

    Effectively is same as:
    ```
    (y0, y1), (x0, x1) = ys * scale, xs * scale
    h, w = ys[1] - ys[0], xs[1] - xs[0]
    return cv2.resize(a[y0: y1, x0: x1], (w, h), interpolation=interpolation)
    ```
    """
    assert 0 <= interpolation <= 4
    if scale == 1:
        return padslice(a, *loc)

    box = np.array(loc, 'i4')  # (y/x, start/stop)
    h, w = (box[:, 1] - box[:, 0]).tolist()
    if not h or not w:  # 0-size output
        return np.empty((h, w, *a.shape[2:]), a.dtype)

    lo_f, hi_f = np.multiply(box, scale, dtype='f4').T

    loc = tuple((floor(lo_), ceil(hi_)) for lo_, hi_ in zip(lo_f, hi_f))
    if not env.BIPL_SUBPIX:
        return resize(padslice(a, *loc), (h, w))

    # Extra margin to accomodate interpolation kernel
    # 2x2 (0 extra) - nearest (0), bilinear (1), area (3)
    # 4x4 (1 extra) - bicubic (2)
    # 8x8 (3 extra) - lanczos4 (4)
    eps = (0, 0, 1, 0, 3)[interpolation]

    # Tight slice to have all necessary pixels
    loc = tuple(
        tuple(np.clip([lo - eps, hi + eps], 0, size))
        for (lo, hi), size in zip(loc, a.shape)
    )
    r = at(a, *loc)
    if not r.size:  # 0-size tight slice
        return np.zeros((h, w, *a.shape[2:]), a.dtype)

    # Resample image crop to destination grid
    dy, dx = (lo_ - lo + (scale - 1) / 2 for lo_, (lo, _) in zip(lo_f, loc))
    return cv2.warpAffine(
        r,
        np.array([[scale, 0, dx], [0, scale, dy]], 'f4'),
        (w, h),
        flags=interpolation | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )


def at(a: NumpyLike, *loc: Span) -> np.ndarray:
    s = tuple(starmap(slice, loc))
    return a[s]


def merge_intervals(spans: list[Span]) -> tuple[list[int], list[Span]]:
    n = len(spans)
    if n == 1:
        return [*range(n)], spans

    # Ascening order of span.start
    spans_a = np.asarray(spans)
    pos = spans_a[:, 0].argsort()  # (n)
    spans_a = spans_a[pos]  # (n 2)

    # "Merge intervals" algorithm for non-overlapping intervals
    # See https://stackoverflow.com/a/58976449/9868257
    edges = spans_a[:-1, 1] < spans_a[1:, 0]  # (n-1)
    edges = np.r_[True, edges, True]  # (n+1)
    mask = np.ndarray((n, 2), '?', buffer=edges, strides=edges.strides * 2)
    spans_a = spans_a[mask].reshape(-1, 2)

    return pos.tolist(), spans_a.tolist()
