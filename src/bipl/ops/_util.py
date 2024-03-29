__all__ = [
    'crop_to', 'get_fusion', 'get_trapz', 'normalize_loc',
    'probs_to_rgb_heatmap', 'resize'
]

from collections.abc import Iterable, Sequence
from itertools import zip_longest

import cv2
import numpy as np

from bipl import env

from ._types import NumpyLike, Tile, Vec


def probs_to_rgb_heatmap(prob: np.ndarray) -> np.ndarray:
    """Converts probability array of (H W C) shape to (H W 3) RGB image"""
    if prob.ndim != 3:
        raise ValueError(f'prob should be 3d, got: {prob.shape}')
    h, w, c = prob.shape
    if not prob.size:
        return np.empty((h, w, 3), dtype='u1')

    hue = prob.argmax(-1).astype('u1')
    value = prob.max(-1)
    if value.dtype != 'u1':
        value *= 255
        value = value.round().astype('u1')

    hue_lut = np.zeros(256, 'u1')
    hue_lut[:c] = np.linspace(0, 127, c, endpoint=False, dtype='u1')
    hsv = cv2.merge([
        cv2.LUT(hue, hue_lut),
        np.broadcast_to(np.uint8(255), (h, w)),
        value,
    ])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def get_trapz(step: int, overlap: int) -> np.ndarray:
    """Returns trapezional window kernel to apply stiching"""
    if overlap == 0:
        raise ValueError('not applicable for overlap 0')
    pad = np.linspace(0, 1, overlap + 2)[1:-1]  # strip initial 0 and final 1
    return np.r_[pad, np.ones(step - overlap), pad[::-1]].astype('f4')


def normalize_loc(loc: Sequence[slice] | slice,
                  shape: Sequence[int]) -> tuple[slice, ...]:
    """Ensures slices match ndim and have not none endpoints"""
    if isinstance(loc, slice):
        loc = loc,
    ndim = len(shape)
    if len(loc) > ndim:
        raise ValueError(f'loc is too deep for {ndim}D, got: {loc}')
    it = (slice(
        s.start or 0,
        s.stop if s.stop is not None else axis_len,
        s.step if s.step is not None else 1,
    ) for s, axis_len in zip_longest(loc, shape, fillvalue=slice(None)))
    return *it,


def padslice(a: NumpyLike, *loc: slice) -> np.ndarray:
    """
    Do `a[loc]`, but extend `a` (with 0s) if `loc` indices beyond `a.shape`.
    """
    loc = normalize_loc(loc, a.shape)

    pos_loc = *(slice(max(0, s.start), max(0, s.stop)) for s in loc),
    a = a[pos_loc]

    pad = [(pos.start - raw.start, pos.stop - pos.start - size)
           for raw, pos, size in zip(loc, pos_loc, a.shape)]
    return np.pad(a, pad) if np.any(pad) else a


def crop_to(vec: Vec, a: NumpyLike,
            shape: Sequence[int]) -> tuple[Vec, np.ndarray]:
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
    loc = *(slice(*np.clip([0, ts], -t0, s - t0))
            for t0, ts, s in zip(vec, a.shape, shape)),
    a = padslice(a, *loc)

    vec = *(max(v, 0) for v in vec),
    return vec, a


def resize(image: np.ndarray,
           hw: Sequence[int],
           *,
           antialias: bool | None = None) -> np.ndarray:
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


def get_fusion(tiles: Iterable[Tile],
               shape: Sequence[int] | None = None) -> np.ndarray | None:
    """Stack tiles to large image. Tiles lying outside of bounds are cropped"""
    r: np.ndarray | None = None

    if shape is None:  # Collect all the tiles to compute destination size
        tiles = [*tiles]
        if not tiles:
            return None
        # N arrays of (yx + hw)
        yx_hw = np.array([[[t.vec, t.data.shape[:2]] for t in tiles]]).sum(1)
        # bottom left most edge
        shape = *yx_hw.max(0).tolist(),
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
        r[y:y + h, x:x + w] = a

    return r
