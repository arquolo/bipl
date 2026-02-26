__all__ = ['Dzi']

from typing import TYPE_CHECKING, Literal, NamedTuple

import cv2
import numpy as np

from bipl._types import NDIndex
from bipl.ops._util import keep3d

if TYPE_CHECKING:
    from ._slide import Slide


class Dzi(NamedTuple):
    tile_size: int
    quality: int = 90
    fmt: Literal['jpeg', 'webp', 'avif'] = 'webp'

    def head(self, slide: 'Slide') -> dict:
        h, w, _ = slide.shape
        mpp = slide.mpp
        image = {
            'xmlns': 'http://schemas.microsoft.com/deepzoom/2008',
            'Size': {'Height': h, 'Width': w},
            'MPP': mpp,
            'TileSize': self.tile_size,
            'Overlap': '0',
            'Format': self.fmt,
        }
        return {'Image': image}

    def tile_at(
        self, slide: 'Slide', level: int, iy_ix: NDIndex
    ) -> np.ndarray:
        scale = 2 ** (level - max(slide.shape[:2]).bit_length())
        tile_0 = self.tile_size / scale
        offset_0 = tuple(ip * tile_0 for ip in iy_ix)
        return slide.at(offset_0, self.tile_size, scale=scale)

    def compress(self, im: np.ndarray) -> bytes:
        qtag = {
            'jpeg': cv2.IMWRITE_JPEG_QUALITY,
            'webp': cv2.IMWRITE_WEBP_QUALITY,
        }.get(self.fmt)
        if qtag is None:
            raise RuntimeError(f'Unknown format: {self.fmt}')

        im = keep3d(im)
        if im.shape[2] == 1:
            bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        _, data = cv2.imencode(f'.{self.fmt}', bgr, [qtag, self.quality])
        return data.tobytes()
