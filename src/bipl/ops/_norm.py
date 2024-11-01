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
        rgb: np.ndarray | None = None,
        weight: float = 1.0,
        channels: Literal['L', 'ab', 'Lab'] = 'ab',
        r: float = 0.5,
    ) -> None:
        assert 0 < weight <= 1
        self.channels = channels
        self.r = r
        self.weight = weight
        self.lut = None
        if rgb is not None:
            self.lut = self._make_lut(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB))

    def _make_lut(self, lab: np.ndarray) -> np.ndarray:
        cdf = _minmax_cdf(lab) if self.r == 0 else self._unsharp_cdf(lab)

        if self.weight < 1:
            cdf *= self.weight
            cdf += np.linspace(0, 1 - self.weight, 256, dtype='f4')

        return around(cdf * 255, 'u1')

    def _unsharp_cdf(self, lab: np.ndarray) -> np.ndarray:
        f4 = lab.astype('f4')
        h, w = f4.shape[:2]

        # 256 total value counts
        lstar = lab[:, :, 0].ravel()  # (h w) -> n
        counts = np.bincount(lstar, minlength=256)  # 256

        match self.channels:
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
        var = var[lstar.argsort()]

        # 256 variances
        m = counts > 0
        counts_nz = counts[m]
        sep = np.r_[0, counts_nz[:-1]].cumsum()
        df = np.zeros(256, 'f4')
        df[m] = np.add.reduceat(var, sep) / counts_nz  # sum(E^2) -> mean(E^2)
        df **= self.r / 2  # sqrt(mean(E^2)) ^ r
        df /= df.sum()

        return _make_cdf(df)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if not rgb.size:
            return rgb
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

        if (lut := self.lut) is None:
            lut = self._make_lut(lab)

        lab[:, :, 0] = cv2.LUT(lab[:, :, 0], lut)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _make_cdf(pdf: np.ndarray) -> np.ndarray:
    cdf = np.ma.masked_equal(pdf.cumsum(), 0).astype('f4')
    cdf -= cdf.min()
    cdf /= cdf.max()
    return cdf.filled(0)


def _minmax_cdf(lab: np.ndarray) -> np.ndarray:
    lstar = lab[:, :, 0]  # (h w)
    lo, hi = int(lstar.min()), int(lstar.max())  # Don't overflow
    return np.r_[
        np.zeros(lo, dtype='f4'),
        np.linspace(0, 1, hi - lo + 1, dtype='f4'),
        np.ones(255 - hi, 'f4'),
    ]
