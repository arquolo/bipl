__all__ = ['env']

from pydantic import BaseSettings, ByteSize, parse_obj_as


class Env(BaseSettings):
    BIPL_DRIVERS: set[str] = {'gdal', 'tiff', 'openslide'}
    BIPL_CACHE: ByteSize = parse_obj_as(ByteSize, '10 MiB')
    BIPL_TILE_CACHE: int = 100  # max tiles cached per tiff slide
    BIPL_INTER_PYRAMID: bool = True  # if not set, uses cv2.INTER_AREA
    BIPL_TILE_POOL_SIZE: int = 64_000_000  # Min resolution for tiled pooling
    BIPL_RGB_RAMPS: str | None = None
    BIPL_CLAHE: bool = False


env = Env()
