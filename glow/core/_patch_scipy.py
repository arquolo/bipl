"""
Fix for strange bug in SciPy on Anaconda for Windows
https://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats

Combines both:
.. [https://stackoverflow.com/a/39021051]
.. [https://stackoverflow.com/a/44822794]
"""
__all__ = ['apply']

import ctypes
import os
import sys
import warnings
from collections.abc import Iterator
from importlib.util import find_spec
from pathlib import Path

_FORTRAN_FLAG = 'FOR_DISABLE_CONSOLE_CTRL_HANDLER'


def _get_conda_libs() -> Iterator[ctypes.CDLL]:
    """Preload DLLs from icc_rt conda package"""
    root = Path(sys.prefix)
    if (root / 'conda-meta').exists() and find_spec('scipy'):
        for dllname in ('libmmd.dll', 'libifcoremd.dll'):
            if (dll := root / f'Library/bin/{dllname}').exists():
                yield ctypes.CDLL(dll.as_posix())


def _patch_handler_and_load_scipy() -> None:
    # Source - (stackoverflow)[https://stackoverflow.com/a/39021051/9868257]
    ptr = ctypes.c_void_p()
    ok = ctypes.windll.kernel32.VirtualProtect(
        ptr, ctypes.c_size_t(1), 0x40, ctypes.byref(ctypes.c_uint32(0)))
    if not ok or (addr := ptr.value) is None:
        return

    code: bytearray = (ctypes.c_char * 3).from_address(
        addr)  # type: ignore[assignment]
    if not code:
        return

    new = b'\xC2\x08\x00' if ctypes.sizeof(ctypes.c_void_p) == 4 else b'\xC3'
    patch_size = len(new)
    old, code[:patch_size] = code[:patch_size], new
    try:
        import scipy.stats  # noqa: F401
    finally:
        code[:patch_size] = old


def apply() -> None:
    if (sys.platform != 'win32'
            or _FORTRAN_FLAG in os.environ  # Patch is already applied
            or not [*_get_conda_libs()]):  # Only Anaconda's SciPy is affected
        return

    os.environ[_FORTRAN_FLAG] = '1'  # Child will inherit this, and work fine

    warnings.warn('Ctrl-C on Windows is broken when scipy is from conda. '
                  'Please use scipy from PyPI')
    if 'scipy.stats' in sys.modules:
        warnings.warn('Cannot fix handling of Ctrl-C in current process. '
                      'Import glow before scipy.stats to fix this.')
    else:
        _patch_handler_and_load_scipy()
