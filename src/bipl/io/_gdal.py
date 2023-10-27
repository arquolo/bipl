__all__ = ['Gdal', 'url_to_gdal', 'url_to_safe_gdal']

import os
from dataclasses import dataclass
from threading import Lock
from urllib.parse import urlencode

import numpy as np
from osgeo import gdal, gdal_array
from pydantic import AnyHttpUrl, ValidationError, parse_obj_as

from ._slide_bases import Driver, ImageLevel
from ._util import gdal_parse_mpp

gdal.SetConfigOption('AWS_VIRTUAL_HOSTING', 'FALSE')
gdal.SetConfigOption('CPL_VSIL_CURL_USE_HEAD', 'NO')  # use_head=no
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN',
                     'EMPTY_DIR')  # list_dir=no
gdal.SetConfigOption('GDAL_HTTP_UNSAFESSL', 'YES')
gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '3')  # max_retry=3
gdal.UseExceptions()


def url_to_gdal(url: str, /, **query) -> str:
    if not query:
        return f'/vsicurl/{url}'
    return '/vsicurl?{params}'.format(params=urlencode(query | {'url': url}))


def url_to_safe_gdal(url: str, /) -> str:
    return url_to_gdal(url, list_dir='no', use_head='no', max_retry='3')


def _fix_if_url(s: str, /) -> str:
    if os.path.isfile(s) or s.startswith('/vsicurl'):
        return s
    try:
        return url_to_safe_gdal(parse_obj_as(AnyHttpUrl, s))
    except ValidationError:
        raise FileNotFoundError(f'Neither a fs path nor URL: {s!r}') from None


# TODO: handle associated images via `base.Image`


@dataclass(frozen=True)
class _Level(ImageLevel):
    index: int
    g: 'Gdal'
    bands: tuple[gdal.Band, ...]

    def crop(self, *loc: slice) -> np.ndarray:
        box, valid_box, shape = self._unpack_loc(*loc)

        (y0, y1), (x0, x1) = valid_box.tolist()
        if y0 == y1 or x0 == x1:  # Patch is outside slide
            return np.broadcast_to(self.g.bg_color, (*shape, 3))

        h, w = y1 - y0, x1 - x0
        c = self.g.num_channels
        chw = np.empty((c, h, w), 'u1')

        with self.g.lock:
            for b, hw in zip(self.bands, np.split(chw, c, 0)):
                gdal_array.BandReadAsArray(b, x0, y0, w, h, buf_obj=hw)

        # TODO: add pad if necessary
        rgb = chw.transpose(1, 2, 0)
        return self._expand(rgb, valid_box, box, self.g.bg_color)


class Gdal(Driver):
    def __init__(self, path: str):
        path = _fix_if_url(path)
        self.ds: gdal.Dataset = gdal.OpenEx(path, gdal.GA_ReadOnly)

        drv: gdal.Driver = self.ds.GetDriver()
        if drv.ShortName != 'GTiff':
            raise ValueError(f'Unsupported driver {drv.ShortName}')

        self.num_channels = self.ds.RasterCount
        if self.num_channels not in (3, 4):
            raise ValueError('Unknown colorspace')

        self._bands: tuple[gdal.Band, ...] = tuple(
            self.ds.GetRasterBand(i + 1) for i in range(self.num_channels))
        self.dtype = np.dtype(gdal_array.flip_code(self._bands[0].DataType))
        if self.dtype != 'u1':
            raise ValueError(f'Unsupported dtype: {self.dtype}')

        self.meta = self.ds.GetMetadata().copy()

        self.bg_color = np.full(3, 255, 'u1')  # TODO: parse tags
        self.mpp = self._mpp()
        self.lock = Lock()

    def __repr__(self) -> str:
        return f'{type(self).__name__}({id(self.ds):0x})'

    def _mpp(self) -> float | None:
        if mpp := gdal_parse_mpp(self.meta):
            return float(np.mean(mpp))
        return None

    def __len__(self) -> int:
        return 1 + self.ds.GetRasterBand(1).GetOverviewCount()

    def __getitem__(self, index: int) -> _Level:
        bands: tuple[gdal.Band, ...] = tuple(
            b if index == 0 else b.GetOverview(index - 1) for b in self._bands)

        shape = (bands[0].YSize, bands[0].XSize, self.num_channels)
        mpp = self.mpp if index == 0 else None
        return _Level(shape, mpp, index, self, bands)

    def keys(self) -> list[str]:
        return []  # TODO: fill if GDAL can detect auxilary images
