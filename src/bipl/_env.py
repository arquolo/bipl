__all__ = ['env']

from typing import Literal

from pydantic import ByteSize, HttpUrl, TypeAdapter
from pydantic_settings import BaseSettings

_bs_adapter = TypeAdapter(ByteSize)


class Env(BaseSettings):
    BIPL_DRIVERS: set[str] = {'gdal', 'tiff', 'openslide'}
    # Max slides opened
    BIPL_CACHE: int = 64
    # Max tiles cached per tiff slide
    BIPL_TILE_CACHE: ByteSize = _bs_adapter.validate_python('16 MiB')

    # `area`
    # - Downsample in single `cv2.resize(inter=AREA)` op.
    # - Uses `box` filter.
    # + No aliasing.
    # - Slow for large downsample factors.
    # box
    # - If downsample is more than 2X,
    #   do `cv2.resize(f=0.5)` till remaining downsample becomes less then 2X.
    # - Each step effectively filters image with [1 1] window,
    #   keeping 1/4 of pixels.
    # + Faster than `area` at least as 2x.
    # + Preserves alignment.
    # - Can lead to some aliasing.
    # gauss
    # - same as `box` but uses `cv2.pyrDown`
    # - Each step effectively filters image with [1 4 6 4 1] window,
    #   keeping 1/4 of pixels.
    # + No aliasing.
    # + Middle-ground in perf.
    # - Shifts image by 1/2 of pixel on each step.
    # TODO: implement [1 3 3 1] filtering
    BIPL_DOWN: Literal['area', 'box', 'gauss'] = 'gauss'  # Downsampling mode
    BIPL_SUBPIX: bool = True  # Subpixel resampling

    BIPL_TILED_POOLING: bool = False  # Do tiled pooling
    BIPL_ICC: bool = False  # Apply image color correction
    BIPL_NORM: bool = False  # Normalize luminance

    BIPL_MPP_Q: int = 12  # 2^(1/Q) MPP rounding (default 1.06x step)

    # Remote TIFF block size & cache size
    BIPL_TIFF_BLOCK_SIZE: ByteSize = _bs_adapter.validate_python('16 KiB')
    BIPL_TIFF_NUM_BLOCKS: int = 1000

    BIPL_COV_URL: HttpUrl | None = None


env = Env()
