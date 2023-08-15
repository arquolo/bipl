__all__ = ['get_trapz', 'normalize_loc', 'probs_to_rgb_heatmap']

import cv2
import numpy as np


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


def normalize_loc(slices: tuple[slice, ...] | slice,
                  shape: tuple[int, ...]) -> tuple[slice, ...]:
    """Ensures slices match shape and have non-none endpoints"""
    if isinstance(slices, slice):
        slices = slices,
    slices += (slice(None), ) * max(0, len(shape) - len(slices))
    if len(slices) != len(shape):
        raise ValueError(f'loc is too deep, got: {slices}')
    return *(slice(
        s.start if s.start is not None else 0,
        s.stop if s.stop is not None else axis_len,
        s.step if s.step is not None else 1,
    ) for s, axis_len in zip(slices, shape)),
