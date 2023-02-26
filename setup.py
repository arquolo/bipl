#!/bin/python3
"""
Usage:
    python setup.py sdist
    python setup.py bdist_wheel --plat-name=win-amd64
    twine upload dist/*
"""

import io
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install

_NAME = 'glow'
_URL = ('https://github.com/openslide/openslide-winbuild/releases/download'
        '/v20171122/openslide-win64-20171122.zip')


def _url_to_io(url: str) -> io.BytesIO:
    from tqdm import tqdm
    r = urlopen(url)

    total = int(value) if (value := r.headers['content-length']) else None
    buf = io.BytesIO()
    with tqdm.wrapattr(r, 'read', total, desc='retrieve libraries') as fp:
        shutil.copyfileobj(fp, buf)
    return buf


def _download_deps(cmd: setuptools.Command, path: Path) -> None:
    if sys.platform != 'win32' or cmd.dry_run or path.exists():
        return

    cmd.mkpath(path.as_posix())
    try:
        with ZipFile(_url_to_io(_URL)) as zf:
            for name in zf.namelist():
                if name.endswith('.dll'):
                    (path / Path(name).name).write_bytes(zf.read(name))
    except BaseException:
        for p in path.glob('*.dll'):
            p.unlink()
        path.unlink()
        raise


class PostInstall(install):
    def run(self):
        _download_deps(self, Path(self.build_lib, _NAME, 'io/libs'))
        install.run(self)


class PostDevelop(develop):
    def run(self):
        _download_deps(self, Path(self.egg_path, _NAME, 'io/libs'))
        develop.run(self)


setuptools.setup(cmdclass={'install': PostInstall, 'develop': PostDevelop})
