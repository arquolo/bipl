__all__ = ['Dzi']

import json
from typing import TYPE_CHECKING, Literal, NamedTuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from ._slide import Slide


class Dzi(NamedTuple):
    tile: int
    quality: int = 90
    fmt: Literal['jpeg', 'webp', 'avif'] = 'webp'

    def head(self, slide: 'Slide') -> str:
        h, w, _ = slide.shape
        mpp = slide.mpp
        image = {
            'xmlns': 'http://schemas.microsoft.com/deepzoom/2008',
            'Size': {
                'Height': h,
                'Width': w,
            },
            'MPP': mpp,
            'TileSize': self.tile,
            'Overlap': '0',
            'Format': self.fmt,
        }
        return json.dumps({'Image': image})

    def tile_at(self, slide: 'Slide', level: int,
                iy_ix: tuple[int, ...]) -> np.ndarray:
        scale = 2 ** (level - max(slide.shape[:2]).bit_length())
        tile_0 = self.tile / scale
        offset_0 = *(ip * tile_0 for ip in iy_ix),
        return slide.at(offset_0, self.tile, scale=scale)

    def compress(self, rgb: np.ndarray) -> bytes:
        qtag = {
            'jpeg': cv2.IMWRITE_JPEG_QUALITY,
            'webp': cv2.IMWRITE_WEBP_QUALITY,
        }.get(self.fmt)
        if qtag is None:
            raise RuntimeError(f'Unknown format: {self.fmt}')

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, data = cv2.imencode(f'.{self.fmt}', bgr, [qtag, self.quality])
        return data.tobytes()
