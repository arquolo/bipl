__all__ = ['LumaScaler', 'Normalizer']

from typing import Literal

import cv2
import numpy as np
from glow import around

LUMA_MAX = 240
LUT_THRESHOLD = 160  # Slide dimmer this considered as low light


class LumaScaler:
    def __init__(self, rgb: np.ndarray) -> None:
        gray = np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        luma95 = np.percentile(gray, 95)  # type: ignore

        if luma95 <= LUT_THRESHOLD:
            lut = np.arange(256).astype('f4')
            lut *= LUMA_MAX / luma95
            self.lut = around(lut.clip(0, 255), 'u1')
        else:
            self.lut = None

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if self.lut is not None:
            return cv2.LUT(rgb.ravel(), self.lut).reshape(rgb.shape)
        return rgb


class Normalizer:
    """
    Normalizes slide's thumbnail & tiles.

    To calibrate use 64um slide level.
    Defaults 64um, r=0.5 and 'ab' channels were acquired during testing.
    """
    def __init__(
        self,
        rgb: np.ndarray,
        weight: float = 1.0,
        r: float = 0.5,
        channels: Literal['L', 'ab', 'Lab'] = 'ab',
    ) -> None:
        assert 0 < weight <= 1
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)  # (h w 3)

        f4 = lab.astype('f4')
        h, w = f4.shape[:2]

        match channels:
            case 'L':
                f4 = f4[:, :, :1]
                f4 *= 1 / 255
            case 'ab':
                f4 = f4[:, :, 1:]
                f4 *= 1 / 100
            case 'Lab':
                f4 *= [1 / 255, 1 / 100, 1 / 100]

        kern = np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]], 'f4') / 16
        var = cv2.filter2D(f4, -1, kern, borderType=cv2.BORDER_REPLICATE)
        var = var.reshape(h, w, -1)

        var = np.square(var).sum(-1).ravel()  # L2, (h w)
        lstar = lab[:, :, 0].ravel()  # (h w)

        i = lstar.argsort()
        lstar, var = lstar[i], var[i]

        # 256 total value counts
        pdf = np.bincount(lstar, minlength=256)
        self.cdf = around(pdf.cumsum() * (255 / pdf.sum()), 'u1')

        # 256 total variances
        m = np.r_[True, lstar[:-1] != lstar[1:]]
        sep, = m.nonzero()
        v256 = np.zeros(256, 'f4')
        v256[lstar[m]] = np.add.reduceat(var, sep)

        # 256 mean errors
        v256 /= np.maximum(pdf, 1)  # sum(E^2) -> mean(E^2)
        v256 **= r / 2  # sqrt(mean(E^2)) ** r
        v256 /= v256.sum()

        if weight < 1:
            v256 *= weight
            v256 += (1 - weight) / 256

        self.vcdf = around(v256.cumsum() * 255, 'u1')

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if not rgb.size:
            return rgb
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.LUT(lab[:, :, 0], self.vcdf)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
