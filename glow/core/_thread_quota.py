"""
? At each moment each thread serve one executor.
+ Threads are reused over all executors.
+ Idle threads are stopped on _TIMEOUT.
? Most recently used threads used first (a.k.a. stack, more dropout).
"""
from __future__ import annotations

__all__ = ['ThreadQuota']

from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future
from concurrent.futures._base import LOGGER
from concurrent.futures.thread import _WorkItem
from queue import Empty, SimpleQueue
from threading import Lock, Thread, _register_atexit
from typing import TypeVar
from weakref import WeakKeyDictionary, WeakSet

# TODO: investigate hangups when _TIMEOUT <= .01
_TIMEOUT = 1
_MIN_IDLE = 10

# ------------------------------- generics -----------------------------------

_T = TypeVar('_T')


def _safe_call(fn: Callable[..., _T], *args, **kwargs) -> _T | None:
    try:
        return fn(*args, **kwargs)
    except (Empty, IndexError, ValueError):
        return None


# ------------------------------ implementation ------------------------------

_Pipe = SimpleQueue['ThreadQuota | None']

_shutdown = False
_shutdown_lock = Lock()
_executors = WeakSet['ThreadQuota']()
_workers = WeakKeyDictionary[Thread, _Pipe]()
_idle = deque[_Pipe]()


def _python_exit():
    global _shutdown
    with _shutdown_lock:
        _shutdown = True

    for e in _executors:
        e.shutdown(cancel_futures=True)
    *items, = _workers.items()
    for _, q in items:
        q.put(None)
    for w, _ in items:
        w.join()


_register_atexit(_python_exit)


def _worker(q: _Pipe) -> None:
    try:
        while executor := _safe_call(q.get, timeout=_TIMEOUT):
            while work_item := _safe_call(executor._work_queue.popleft):
                work_item.run()  # Process task
                if _shutdown:
                    return

            executor._idle.append(1)  # Decrease worker usage for executor
            _idle.append(q)  # Mark worker as idle, LIFO/stack
            if _shutdown:
                return

    except BaseException:  # noqa: PIE786
        LOGGER.critical('Exception in worker', exc_info=True)
    finally:
        if _TIMEOUT:
            _safe_call(_idle.remove, q)  # Omit when '_idle' tracks weakrefs


class ThreadQuota(Executor):
    __slots__ = ('_work_queue', '_idle', '_shutdown_lock', '_shutdown')

    def __init__(self, max_workers: int):
        self._work_queue = deque[_WorkItem]()
        self._idle = [1] * max_workers  # semaphore

        self._shutdown_lock = Lock()
        self._shutdown = False

        with _shutdown_lock:
            _executors.add(self)

    def submit(self, fn, /, *args, **kwargs):
        with self._shutdown_lock, _shutdown_lock:
            if self._shutdown or _shutdown:
                raise RuntimeError('cannot schedule futures after shutdown')

            f = Future()
            self._work_queue.append(_WorkItem(f, fn, args, kwargs))

            if _safe_call(self._idle.pop):  # Pool is not maximized yet
                if not (q := _safe_call(_idle.pop)):
                    q = _Pipe()
                    w = Thread(target=_worker, args=(q, ))
                    w.start()
                    _workers[w] = q
                q.put(self)

            return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True

            if cancel_futures:
                while work_item := _safe_call(self._work_queue.pop):
                    work_item.future.cancel()

            if not _TIMEOUT:
                # Keep at most 25% of workers idle
                while (len(_idle) > max(len(_workers) / 4, _MIN_IDLE)
                       and (q := _safe_call(_idle.popleft))):
                    q.put(None)
