__all__ = ['load_library']

import os
import sys
from ctypes import CDLL
from pathlib import Path

_root = (Path(__file__).parent / 'libs').as_posix()


def _load_library(name: str, version: int) -> CDLL:
    if sys.platform != 'win32':
        return CDLL(f'{name}.so.{version}')

    with os.add_dll_directory(_root):
        return CDLL(f'{_root}/{name}-{version}.dll')


def load_library(name: str, *versions: int) -> CDLL:
    errors = []
    for v in versions:
        try:
            return _load_library(name, v)
        except (OSError, FileNotFoundError) as exc:
            errors.append(exc)
    raise errors[-1]
