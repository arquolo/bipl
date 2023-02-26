__all__ = ['get_trapz', 'probs_to_hsv']

import cv2
import numpy as np


def probs_to_hsv(prob: np.ndarray) -> np.ndarray:
    """Converts probability array of (H W C) shape to (H W 3) RGB image"""
    assert prob.ndim == 3
    if not prob.size:
        return np.empty((*prob.shape[:2], 3), dtype='u1')

    hue = prob.argmax(-1).astype('u1')
    value = prob.max(-1)
    if value.dtype != 'u1':
        value *= 255
        value = value.round().astype('u1')

    hue_lut = (np.arange(256) * (127 / prob.shape[2])).round().astype('u1')
    hsv = cv2.merge([
        cv2.LUT(hue, hue_lut),
        np.broadcast_to(np.uint8(255), prob.shape[:2]),
        value,
    ])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def get_trapz(step: int, overlap: int) -> np.ndarray:
    """Returns trapezional window kernel to apply stiching"""
    assert overlap
    pad = np.linspace(0, 1, overlap + 2)[1:-1]  # strip initial 0 and final 1
    return np.r_[pad, np.ones(step - overlap), pad[::-1]].astype('f4')
