__all__ = ['Dzi']

from typing import TYPE_CHECKING, Literal, NamedTuple

import cv2
import numpy as np
from lxml.builder import ElementMaker
from lxml.etree import tostring

if TYPE_CHECKING:
    from ._slide import Slide


class Dzi(NamedTuple):
    tile: int
    quality: int = 90
    fmt: Literal['jpeg', 'webp', 'avif'] = 'webp'

    def head(self, slide: 'Slide') -> str:
        e = ElementMaker()
        return tostring(
            e.Image(
                e.Size(Height=str(slide.shape[0]), Width=str(slide.shape[1])),
                TileSize=str(self.tile),
                Overlap='0',
                Format=self.fmt,
                xmlns='http://schemas.microsoft.com/deepzoom/2008'))

    def tile_at(self, slide: 'Slide', level: int,
                iy_ix: tuple[int, ...]) -> np.ndarray:
        scale = 2 ** (level - max(slide.shape[:2]).bit_length())
        tile_0 = self.tile * scale
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
