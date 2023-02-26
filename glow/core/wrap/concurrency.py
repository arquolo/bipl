from __future__ import annotations

__all__ = ['call_once', 'shared_call', 'streaming', 'threadlocal']

import sys
import threading
from collections.abc import Callable, Hashable, Iterable, Sequence
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from functools import partial, update_wrapper
from queue import Empty, SimpleQueue
from threading import Lock, Thread
from time import monotonic, sleep
from typing import TypeVar, cast
from weakref import WeakValueDictionary

from .util import make_key

_T = TypeVar('_T')
_R = TypeVar('_R')
_F = TypeVar('_F', bound=Callable)
_BatchFn = Callable[[list[_T]], Iterable[_R]]
_ZeroArgsF = TypeVar('_ZeroArgsF', bound=Callable[[], object])

_PATIENCE = 0.01
_unset = object()


def threadlocal(fn: Callable[..., _T], *args: object,
                **kwargs: object) -> Callable[[], _T]:
    """Thread-local singleton factory, mimics `functools.partial`"""
    local_ = threading.local()

    def wrapper() -> _T:
        try:
            return local_.obj
        except AttributeError:
            local_.obj = fn(*args, **kwargs)
            return local_.obj

    return update_wrapper(wrapper, fn)


@dataclass
class _UFuture:
    _fn: Callable[[], object]
    _lock: Lock = field(default_factory=Lock)
    _result: object = _unset
    _exception: BaseException | None = None

    def result(self):
        with self._lock:
            if self._exception:
                raise self._exception
            if self._result is not _unset:
                return self._result

            try:
                self._result = r = self._fn()
                return r
            except BaseException as e:
                self._exception = e
                raise


def call_once(fn: _ZeroArgsF) -> _ZeroArgsF:
    """Makes `fn()` callable a singleton.
    DO NOT USE with recursive functions"""
    def wrapper():
        return f.result()

    fn._future = f = _UFuture(fn)  # type: ignore[attr-defined]
    return cast(_ZeroArgsF, update_wrapper(wrapper, fn))


def shared_call(fn: _F) -> _F:
    """Merges concurrent calls to `fn` with the same `args` to single one.
    DO NOT USE with recursive functions"""
    fs = WeakValueDictionary[Hashable, _UFuture]()
    lock = Lock()

    def wrapper(*args, **kwargs):
        key = make_key(*args, **kwargs)

        with lock:  # Create only one task per args-kwargs set
            if not (f := fs.get(key)):
                fs[key] = f = _UFuture(partial(fn, *args, **kwargs))

        return f.result()

    return cast(_F, update_wrapper(wrapper, fn))


# ----------------------------- batch collation ------------------------------


def _fetch_batch(q: SimpleQueue[_T], batch_size: int,
                 timeout: float) -> list[_T]:
    batch: list[_T] = []

    # Wait indefinitely until the first item is received
    if sys.platform == 'win32':
        # On Windows lock.acquire called without a timeout is not interruptible
        # See:
        # https://bugs.python.org/issue29971
        # https://github.com/dask/dask/pull/2144#issuecomment-290556996
        # https://github.com/dask/dask/pull/2144/files
        while not batch:
            try:
                batch.append(q.get(timeout=_PATIENCE))
            except Empty:
                sleep(0)  # Allow other thread to fill the batch
    else:
        batch.append(q.get())

    endtime = monotonic() + timeout
    while len(batch) < batch_size and (waittime := endtime - monotonic()) > 0:
        try:
            batch.append(q.get(timeout=waittime))
        except Empty:
            break

    if len(batch) < batch_size:
        print(f'timeout({len(batch)})')
    return batch


def _batch_invoke(
    func: _BatchFn[_T, _R],
    batch: Sequence[tuple[Future[_R], _T]],
):
    batch = [(f, x) for f, x in batch if f.set_running_or_notify_cancel()]
    if not batch:
        return

    try:
        *results, = func([x for _, x in batch])
        assert len(results) == len(batch)
    except BaseException as exc:  # noqa: PIE786
        for f, _ in batch:
            f.set_exception(exc)
    else:
        # TODO: use zip(strict=True) for python3.10+
        for (f, _), r in zip(batch, results):
            f.set_result(r)


def _start_fetch_compute(func, workers, batch_size, timeout):
    q = SimpleQueue()
    lock = Lock()

    def loop():
        while True:
            # Because of lock, _fetch_batch could be inlined into wrapper,
            # and dispatch to thread pool could be done from there,
            # thus allowing usage of scalable ThreadPool
            # TODO: implement above
            with lock:  # Ensurance that none worker steals tasks from other
                batch = _fetch_batch(q, batch_size, timeout)
            if batch:
                _batch_invoke(func, batch)
            else:
                sleep(0.001)

    for _ in range(workers):
        Thread(target=loop, daemon=True).start()
    return q


def streaming(func=None,
              /,
              *,
              batch_size,
              timeout=0.1,
              workers=1,
              pool_timeout=20.):
    """
    Delays start of computation to until batch is collected.
    Accepts two timeouts (in seconds):
    - `timeout` is a time to wait till the batch is full, i.e. latency.
    - `pool_timeout` is time to wait for results.

    Uses ideas from
    - https://github.com/ShannonAI/service-streamer
    - https://github.com/leon0707/batch_processor
    - ray.serve.batch
      https://github.com/ray-project/ray/blob/master/python/ray/serve/batching.py

    Note: currently supports only functions and bound methods.

    Implementation details:
    - constantly keeps alive N workers
    - any caller enqueues jobs and starts waiting
    - on any failure during waiting caller cancels all jobs it submitted
    - single worker at a time fetches jobs from shared queue, resolves them,
      and notifies all waiters
    """
    if func is None:
        return partial(
            streaming,
            batch_size=batch_size,
            timeout=timeout,
            workers=workers,
            pool_timeout=pool_timeout)

    assert callable(func)
    assert workers >= 1
    q = _start_fetch_compute(func, workers, batch_size, timeout)

    def wrapper(items):
        fs = {Future(): item for item in items}
        try:
            for f_x in fs.items():
                q.put(f_x)
            if wait(fs, pool_timeout, return_when='FIRST_EXCEPTION').not_done:
                raise TimeoutError

        except BaseException:  # Cancel all not yet submitted futures
            while fs:
                fs.popitem()[0].cancel()
            raise

        return [f.result() for f in fs]

    # TODO: if func is instance method - recreate wrapper per instance
    # TODO: find how to distinguish between
    # TODO:  not yet bound method and plain function
    # TODO:  maybe implement __get__ on wrapper
    return update_wrapper(wrapper, func)
