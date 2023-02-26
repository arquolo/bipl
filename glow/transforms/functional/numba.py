__all__ = ['dither']

from types import MappingProxyType
from typing import Literal

import cv2
import numba
import numpy as np

_MATRICES = MappingProxyType({
    key: cv2.normalize(np.array(mat, 'f4'), None, norm_type=cv2.NORM_L1)
    for key, mat in {
        'jarvis-judice-ninke': [
            [0, 0, 0, 7, 5],
            [3, 5, 7, 5, 3],
            [1, 3, 5, 3, 1],
        ],
        'sierra': [
            [0, 0, 0, 5, 3],
            [2, 4, 5, 4, 2],
            [0, 2, 3, 2, 0],
        ],
        'stucki': [
            [0, 0, 0, 8, 4],
            [2, 4, 8, 4, 2],
            [1, 2, 4, 2, 1],
        ],
    }.items()
})
_DitherKind = Literal['jarvis-judice-ninke', 'stucki', 'sierra']


@numba.jit
def _dither(image: np.ndarray, mat: np.ndarray, quant: int) -> np.ndarray:
    for y in range(image.shape[0] - 2):
        for x in range(2, image.shape[1] - 2):
            old = image[y, x]
            new = (old // quant) * quant
            delta = ((old - new) * mat).astype(image.dtype)
            image[y:y + 3, x - 2:x + 3] += delta
            image[y, x] = new

    return image[:-2, 2:-2]


def dither(image: np.ndarray,
           bits: int = 3,
           kind: _DitherKind = 'stucki') -> np.ndarray:
    mat = _MATRICES[kind]
    assert image.dtype == 'u1'
    if image.ndim == 3:
        mat = mat[..., None]
        channel_pad = [(0, 0)]
    else:
        channel_pad = []
    image = np.pad(image, [(0, 2), (2, 2)] + channel_pad, mode='constant')

    dtype = image.dtype
    if dtype == 'uint8':
        max_value = 256
        image = image.astype('i2')
    else:
        max_value = 1
    quant = max_value / 2 ** bits

    image = _dither(image, mat, quant)
    return image.clip(0, max_value - quant).astype(dtype)
