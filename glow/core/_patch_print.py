"""
Patches builtin `print` function to be compatible with `tqdm`.
Adds some thread safety.
"""
__all__ = ['apply']

import builtins
from functools import update_wrapper, wraps
from threading import RLock

from ._import_hook import register_post_import_hook

_print = builtins.print
_lock = RLock()


@wraps(_print)
def locked_print(*args, **kwargs):
    with _lock:
        _print(*args, **kwargs)


def tqdm_print(module, *values, sep=' ', end='\n', file=None, flush=False):
    module.tqdm.write(sep.join(map(str, values)), end=end, file=file)


def patch_print(module) -> None:
    # Create blank to force initialization of cls._lock and cls._instances
    tqdm = module.tqdm
    tqdm(disable=True)

    def tqdm_print(*values, sep=' ', end='\n', file=None, flush=False):
        tqdm.write(sep.join(map(str, values)), end=end, file=file)

    builtins.print = update_wrapper(tqdm_print, _print)


def apply() -> None:
    builtins.print = locked_print
    register_post_import_hook(patch_print, 'tqdm')
