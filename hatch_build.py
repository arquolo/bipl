import io
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from tqdm import tqdm

_BASEURL = ('https://github.com/openslide/openslide-winbuild/releases/download'
            '/v{version}/openslide-win64-{version}.zip')
_FILTERS = {
    '20221217': r'(libjpeg-62|libtiff-6|zlib1).dll',
    '20231011': r'.*\.dll',
}
_URLS = {
    _BASEURL.format(version=version): re.compile(filter_)
    for version, filter_ in _FILTERS.items()
}
_TARGET = 'src/bipl/io/libs'


def _url_to_io(url: str) -> io.BytesIO:
    r = urlopen(url)

    total = int(value) if (value := r.headers['content-length']) else None
    buf = io.BytesIO()
    with tqdm.wrapattr(r, 'read', total, desc='retrieve libraries') as fp:
        shutil.copyfileobj(fp, buf)
    return buf


def _download_dlls(folder: Path) -> None:
    """Download archive, extract DLLs and place them into target folder"""
    if sys.platform != 'win32' or folder.exists():
        return

    folder.mkdir(parents=True, exist_ok=True)
    try:
        for url, regex in _URLS.items():
            with ZipFile(_url_to_io(url)) as zf:
                for zfpath in zf.namelist():
                    if not regex.fullmatch(Path(zfpath).name):
                        continue
                    with zf.open(zfpath) as src, \
                            (folder / Path(zfpath).name).open('wb') as dst:
                        shutil.copyfileobj(src, dst)
    except BaseException:
        for p in folder.glob('*.dll'):
            p.unlink()
        folder.rmdir()
        raise


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != 'wheel':
            return

        build_data |= {
            'tag': 'py3-none-win_amd64',  # lock platform
            'pure_python': False,  # got DLLs inside
        }
        _download_dlls(Path(self.root, _TARGET))
