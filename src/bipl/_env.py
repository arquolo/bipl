__all__ = ['env']

from pydantic import BaseSettings, ByteSize, parse_obj_as


class Env(BaseSettings):
    BIPL_DRIVERS: set[str] = {'gdal', 'tiff', 'openslide'}
    BIPL_CACHE: ByteSize = parse_obj_as(ByteSize, '10 MiB')


env = Env()
