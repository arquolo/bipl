from __future__ import annotations

__all__ = [
    'BitFlipNoise', 'ChannelMix', 'ChannelShuffle', 'CutOut', 'DegradeJpeg',
    'DegradeQuality', 'Elastic', 'FlipAxis', 'HsvShift', 'LumaJitter',
    'MaskDropout', 'MultiNoise', 'WarpAffine'
]

from dataclasses import InitVar, dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.stats import ortho_group

from . import functional as F
from .core import DualStageTransform, ImageTransform, MaskTransform

# ---------------------------------- mixins ----------------------------------


class _LutTransform(ImageTransform):
    def get_lut(self,
                rng: np.random.Generator) -> np.ndarray | list[np.ndarray]:
        raise NotImplementedError

    def image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        assert image.dtype == np.uint8

        lut = self.get_lut(rng)
        if isinstance(lut, np.ndarray):
            return cv2.LUT(image.ravel(), lut).reshape(image.shape)

        assert len(lut) == image.shape[2]
        planes = map(cv2.LUT, cv2.split(image), lut)
        return cv2.merge([*planes]).reshape(image.shape)


# ---------------------------------- noise ----------------------------------


class AddNoise(ImageTransform):
    """Add uniform[-strength ... +strength] to each item"""
    def __init__(self, strength: float = 0.2) -> None:
        self.strength = int(strength * 255)

    def __repr__(self) -> str:
        return f'{type(self).__name__}(strength={self.strength})'

    def image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        assert image.dtype == np.uint8

        res = rng.integers(
            -self.strength, self.strength, size=image.shape, dtype='i2')
        res += image

        return res.clip(0, 255).astype('u1')


class MultiNoise(ImageTransform):
    """Multiply uniform[1 - strength ... 1 + strength] to each item"""
    def __init__(self, strength: float = 0.5) -> None:
        self.low = max(0, 1 - strength)
        self.high = 1 + strength

    def __repr__(self) -> str:
        return f'{type(self).__name__}(low={self.low}, high={self.high})'

    def image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        assert image.dtype == np.uint8

        res = rng.random(image.shape, dtype='f4')
        res *= self.high - self.low
        res += self.low
        res *= image  # Multiply

        return res.clip(0, 255).astype('u1')


@dataclass
class BitFlipNoise(ImageTransform):
    bitdepth: int = 4

    def image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        assert image.dtype.kind == 'u'
        planes = 8 * image.dtype.itemsize

        if self.bitdepth >= planes:
            return image

        high_flip = 1 << (planes - self.bitdepth)
        bitmask = (1 << planes) - high_flip

        res = image & bitmask
        res += rng.integers(high_flip, size=image.shape, dtype=image.dtype)
        return res


# ----------------------------- color alteration -----------------------------


class ChannelShuffle(ImageTransform):
    def __repr__(self) -> str:
        return f'{type(self).__name__}()'

    def image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        assert image.ndim == 3
        return rng.permutation(image, axis=-1)


@dataclass
class ChannelMix(ImageTransform):
    intensity: tuple[float, float] = (0.5, 1.5)

    def image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        assert image.ndim == 3
        assert image.dtype == np.uint8
        image = image.astype('f4')

        num_channels = image.shape[-1]
        mat = ortho_group.rvs(num_channels, random_state=rng).astype('f4')

        mat *= rng.uniform(*self.intensity)
        lumat = np.full((num_channels, num_channels), 1 / num_channels)

        image = image @ ((np.eye(num_channels) - lumat) @ mat + lumat)

        return image.clip(0, 255).astype('u1')


@dataclass
class LumaJitter(_LutTransform):
    brightness: tuple[float, float] = (-0.2, 0.2)
    contrast: tuple[float, float] = (0.8, 1.2)

    def get_lut(self, rng: np.random.Generator) -> np.ndarray:
        lut = np.arange(256, dtype='f4')

        lut += 256 * rng.uniform(*self.brightness)
        lut = (lut - 128) * rng.uniform(*self.contrast) + 128

        return lut.clip(0, 255).astype('u1')


@dataclass
class GammaJitter(_LutTransform):
    """Alters gamma from [1/(1+gamma) ... 1+gamma]"""
    gamma: float = 0.2

    def __post_init__(self):
        assert self.gamma >= 0

    def get_lut(self, rng: np.random.Generator) -> np.ndarray:
        lut = np.linspace(0, 1, num=256, dtype='f4')

        max_gamma = 1 + self.gamma
        lut **= rng.uniform(1 / max_gamma, max_gamma)
        lut *= 255

        return lut.clip(0, 255).astype('u1')


@dataclass
class HsvShift(_LutTransform):
    max_shift: int = 20

    def get_lut(self, rng: np.random.Generator) -> list[np.ndarray]:
        hue, sat, val = rng.uniform(-self.max_shift, self.max_shift, size=3)
        ramp = np.arange(256, dtype='i2')
        luts = (
            (ramp + hue) % 180,
            (ramp + sat).clip(0, 255),
            (ramp + val).clip(0, 255),
        )
        return [lut.astype('u1') for lut in luts]

    def image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        assert image.ndim == image.shape[-1] == 3

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image = super().image(image, rng)
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


# ------------------------------- compression -------------------------------


@dataclass
class DegradeJpeg(ImageTransform):
    quality: tuple[int, int] = (0, 15)

    def image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        quality = int(rng.integers(*self.quality))
        _, buf = cv2.imencode('.jpg', image,
                              (cv2.IMWRITE_JPEG_QUALITY, quality))
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).reshape(image.shape)


@dataclass
class DegradeQuality(ImageTransform):
    scale: tuple[float, float] = (0.25, 0.5)
    modes: tuple[str, ...] = ('NEAREST', 'LINEAR', 'INTER_CUBIC', 'AREA')

    def image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        shape = image.shape
        scale = rng.uniform(*self.scale)

        # downscale
        mode = getattr(cv2, f'INTER_{rng.choice(self.modes)}')
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=mode)

        # upscale
        mode = getattr(cv2, f'INTER_{rng.choice(self.modes)}')
        image = cv2.resize(image, shape[1::-1], interpolation=mode)
        return image.reshape(shape)


# ----------------------------- mask alteration -----------------------------


@dataclass
class MaskDropout(MaskTransform):
    """
    Drops redundant pixels for each class,
    so that np.bincount(mask.ravel()) <= alpha * mask.size
    """

    alpha: float
    ignore_index: int = -1

    def mask(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return F.mask_dropout(
            mask, rng, alpha=self.alpha, ignore_index=self.ignore_index)


# --------------------------------- geometry ---------------------------------


@dataclass
class FlipAxis(DualStageTransform):
    """
    Flips image/mask vertically/horizontally & rotate by 90 at random.
    In non-isotropic mode (default) flips only horizontally
    """

    isotropic: bool = False

    def prepare(self, rng: np.random.Generator, /, **_) -> dict[str, Any]:
        ud, lr, rot90 = rng.integers(2, size=3)
        if not self.isotropic:
            ud = rot90 = 0
        return {'ud': ud, 'lr': lr, 'rot90': rot90}

    def image(self, image: np.ndarray, **params) -> np.ndarray:
        return F.flip(image, **params)

    def mask(self, mask: np.ndarray, **params) -> np.ndarray:
        return F.flip(mask, **params)


@dataclass
class WarpAffine(DualStageTransform):
    angle: float = 180
    skew: float = 0.5
    scale: tuple[float, float] = (1.0, 1.0)
    inter: InitVar[str] = 'LINEAR'
    _inter: int = field(init=False)

    def __post_init__(self, inter: str):
        self._inter = getattr(cv2, f'INTER_{inter}')

    def prepare(self, rng: np.random.Generator, /, **_) -> dict[str, Any]:
        return {
            'skew': rng.uniform(-self.skew, self.skew),
            'angle': rng.uniform(-self.angle, self.angle),
            'scale': rng.uniform(*self.scale),
        }

    def image(self, image: np.ndarray, **params) -> np.ndarray:
        return F.affine(image, **params, inter=self._inter)

    def mask(self, mask: np.ndarray, **params) -> np.ndarray:
        return F.affine(mask, **params, inter=cv2.INTER_NEAREST)


@dataclass
class Elastic(DualStageTransform):
    """Elastic deformation of image

    Parameters:
    - scale - max shift for each pixel
    - sigma - size of gaussian kernel
    """

    scale: float = 1
    sigma: float = 50
    inter: InitVar[str] = 'LINEAR'
    _inter: int = field(init=False)

    def __post_init__(self, inter: str):
        self._inter = getattr(cv2, f'INTER_{inter}')

    def prepare(self,
                rng: np.random.Generator,
                /,
                image: np.ndarray | None = None,
                **_) -> dict[str, Any] | None:
        if image is None:
            return None
        offsets = rng.random((2, *image.shape[:2]), dtype='f4')
        offsets *= self.scale * 2
        offsets -= self.scale

        for dim, (off, size) in enumerate(zip(offsets, image.shape[:2])):
            shape = np.where(np.arange(2) == dim, size, 1)
            off += np.arange(size).reshape(shape)
            cv2.GaussianBlur(off, (17, 17), self.sigma, dst=off)
        return {'offsets': offsets[::-1]}

    def _apply(self, image: np.ndarray, inter: int, **params) -> np.ndarray:
        map_x, map_y = params['offsets']
        return cv2.remap(
            image, map_x, map_y, inter, borderMode=cv2.BORDER_REFLECT_101)

    def image(self, image: np.ndarray, **params) -> np.ndarray:
        return self._apply(image, self._inter, **params)

    def mask(self, mask: np.ndarray, **params) -> np.ndarray:
        return self._apply(mask, cv2.INTER_NEAREST, **params)


@dataclass
class CutOut(ImageTransform):
    max_holes: int = 80
    size: int = 8
    fill_value: int = 0

    def image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        num_holes = rng.integers(self.max_holes)
        if not num_holes:
            return image

        anchors = rng.integers(0, image.shape[:2], size=(num_holes, 2))

        # [N, dims, (min, max)]
        holes = anchors[:, :, None] + [-self.size // 2, self.size // 2]
        holes = holes.clip(0, np.array(image.shape[:2])[:, None])

        image = image.copy()
        for (y0, y1), (x0, x1) in holes:
            image[y0:y1, x0:x1] = self.fill_value
        return image
