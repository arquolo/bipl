import io
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from tqdm import tqdm

_VERSION = '20221217'
_URL = ('https://github.com/openslide/openslide-winbuild/releases/download'
        f'/v{_VERSION}/openslide-win64-{_VERSION}.zip')
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
        with ZipFile(_url_to_io(_URL)) as zf:
            for zfpath in zf.namelist():
                if not zfpath.endswith('.dll'):
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
