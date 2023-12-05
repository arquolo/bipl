__all__ = ['env']

from typing import Literal

from pydantic import BaseSettings, ByteSize, parse_obj_as


class Env(BaseSettings):
    BIPL_DRIVERS: set[str] = {'gdal', 'tiff', 'openslide'}
    # Max slides opened
    BIPL_CACHE: ByteSize = parse_obj_as(ByteSize, '10 MiB')
    # Max tiles cached per tiff slide
    BIPL_TILE_CACHE: ByteSize = parse_obj_as(ByteSize, '16 MiB')

    # area - downsample in single `resize(inter=area)` call (slow)
    # box2d - if downsample more than 2X, do gradual `resize(f=0.5)` first
    # gauss - do pyramid downsampling (2x slow than box2d), no aliasing
    BIPL_DOWN: Literal['area', 'box2x', 'gauss'] = 'gauss'  # Downsampling mode

    BIPL_TILE_POOL_SIZE: int = 64_000_000  # Min resolution for tiled pooling
    BIPL_ICC: bool = False  # Apply image color correction
    BIPL_CLAHE: bool = False


env = Env()
