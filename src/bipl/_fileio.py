__all__ = ['Paged', 'fopen']

import asyncio
import atexit
import mmap
import os
from collections.abc import Coroutine, Sequence
from io import BytesIO
from threading import Thread
from typing import Any, Protocol

import httpx
from glow import memoize
from pydantic import HttpUrl, TypeAdapter, ValidationError

from bipl._env import env
from bipl._types import Span
from bipl.ops._util import merge_intervals

_URL_ADAPTER = TypeAdapter(HttpUrl)


class Paged(Protocol):
    def pread(self, n: int, start: int, /) -> bytes: ...


def fopen(path: str) -> Paged:
    try:  # Process as URL
        return _RemoteIO(path)

    except ValidationError:  # Got local filename
        return _MappedIO(path)


class _MappedIO:
    __slots__ = ('m',)

    def __init__(self, path: str) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            self.m = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)

    def __hash__(self) -> int:  # Used by _Level.tile:shared_call
        return hash(self.m)

    def __eq__(self, rhs) -> bool:  # Used by _Level.tile:shared_call
        return type(self) is type(rhs) and self.m == rhs.m

    def pread(self, n: int, start: int, /) -> bytes:
        data = self.m[start : start + n]
        return data


class _RemoteIO:
    __slots__ = ('_await', '_cached', '_client', '_semlk', 'url')

    def __init__(self, url: str) -> None:
        _URL_ADAPTER.validate_python(url)
        self.url = httpx.URL(url)

        self._await, self._client = _get_await_n_client()
        self._await(self._check_http_range_support())

        self._semlk = asyncio.Semaphore(env.BIPL_TIFF_REQS)

        self._cached = memoize(
            env.BIPL_TIFF_NUM_BLOCKS, batched=True, policy='lru'
        )(self._get_many)

    def __hash__(self) -> int:  # Used by _Level.tile:shared_call
        return hash(self.url)

    def __eq__(self, rhs) -> bool:  # Used by _Level.tile:shared_call
        return type(self) is type(rhs) and self.url == rhs.url

    def pread(self, n: int, start: int, /) -> bytes:
        bs = env.BIPL_TIFF_BLOCK_SIZE
        stop = start + n
        head = start % bs
        tail = (-stop) % bs

        s = BytesIO()
        spans = [(o, o + bs) for o in range(start - head, stop + tail, bs)]
        for blk in self._await(self._cached(spans)):
            s.write(blk)

        s.seek(head)
        return s.read(n)

    async def _get_many(self, spans: Sequence[Span]) -> list[bytes]:
        # `spans` is a list of block ids without duplicates, yet unsorted.
        # Combine spans with common border, and reposition to match them
        pos, spans = merge_intervals(spans)

        # Get long chunks
        fp = BytesIO()
        coros = (self._get_block(start, stop) for start, stop in spans)
        for blk in await asyncio.gather(*coros):
            fp.write(blk)

        # Split back to blocks and order
        fp.seek(0)
        bs = env.BIPL_TIFF_BLOCK_SIZE
        rs = [b''] * len(pos)
        for i in pos:
            rs[i] = fp.read(bs)
        return rs

    async def _check_http_range_support(self) -> None:
        # Check if HTTP Range is supported
        rsp = await self._client.head(self.url, headers={'Accept': '*/*'})
        rsp.raise_for_status()
        if rsp.headers.get('Accept-Ranges') != 'bytes':
            raise ValueError(
                f'Remote "{self.url}" does not support Range header'
            )

    async def _get_block(self, start: int, stop: int) -> bytes:
        async with (
            self._semlk,
            self._client.stream(
                'GET', self.url, headers={'Range': f'bytes={start}-{stop - 1}'}
            ) as rsp,
        ):
            rsp.raise_for_status()
            if rsp.status_code != 206:
                raise ValueError(
                    f'Unexpected status: {rsp.status_code} for {self.url}'
                )
            return await rsp.aread()


class _Awaiter(Protocol):
    def __call__[R](self, aw: Coroutine[Any, Any, R]) -> R: ...


@memoize()
def _get_await_n_client() -> tuple[_Awaiter, httpx.AsyncClient]:
    loop = asyncio.new_event_loop()
    Thread(target=loop.run_forever, daemon=True).start()

    def await_[R](aw: Coroutine[Any, Any, R]) -> R:
        return asyncio.run_coroutine_threadsafe(aw, loop).result()

    aclient = httpx.AsyncClient(http1=False, http2=True, follow_redirects=True)
    await_(aclient.__aenter__())
    atexit.register(await_, aclient.__aexit__(None, None, None))

    return await_, aclient
