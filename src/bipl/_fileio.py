__all__ = ['Paged', 'fopen']

import asyncio
import atexit
import mmap
import os
from contextlib import AsyncExitStack
from io import BytesIO
from threading import Thread
from typing import Protocol

import httpx
from glow import memoize
from pydantic import HttpUrl, TypeAdapter, ValidationError

from bipl._env import env
from bipl._types import Span
from bipl.ops._util import merge_intervals

_URL_ADAPTER = TypeAdapter(HttpUrl)


class Paged(Protocol):
    def pread(self, n: int, offset: int, /) -> bytes: ...


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

    def pread(self, n: int, offset: int, /) -> bytes:
        data = self.m[offset : offset + n]
        return data


class _RemoteIO:
    __slots__ = ('_cached', '_client', '_loop', 'url')

    def __init__(self, url: str) -> None:
        _URL_ADAPTER.validate_python(url)
        self.url = httpx.URL(url)

        self._loop, self._client = _aget_client()
        asyncio.run_coroutine_threadsafe(self._peek(), self._loop).result()

        self._cached = memoize(env.BIPL_TIFF_NUM_BLOCKS, batched=True)(
            self._get_many
        )

    def __hash__(self) -> int:  # Used by _Level.tile:shared_call
        return hash(self.url)

    def __eq__(self, rhs) -> bool:  # Used by _Level.tile:shared_call
        return type(self) is type(rhs) and self.url == rhs.url

    def pread(self, n: int, offset: int, /) -> bytes:
        blk_size = env.BIPL_TIFF_BLOCK_SIZE
        lo = offset // blk_size
        hi = (offset + n + blk_size - 1) // blk_size

        s = BytesIO()
        for blk in self._cached([(o, o + 1) for o in range(lo, hi)]):
            s.write(blk)

        s.seek(offset - lo * blk_size)
        return s.read(n)

    def _get_many(self, spans: list[Span]) -> list[bytes]:
        # Combine spans with common border
        pos, spans = merge_intervals(spans)

        # Get long chunks
        fp = BytesIO()
        asyncio.run_coroutine_threadsafe(
            self._update_all(fp, spans), self._loop
        ).result()

        # Split back to blocks and order
        fp.seek(0)
        rs = [b''] * len(pos)
        for i in pos:
            rs[i] = fp.read(env.BIPL_TIFF_BLOCK_SIZE)
        return rs

    async def _peek(self) -> None:
        # Ensure range ok
        r = await self._client.head(self.url, headers={'Accept': '*/*'})
        r.raise_for_status()
        if r.headers.get('Accept-Ranges') != 'bytes':
            raise ValueError(
                f'Remote "{self.url}" does not support Range header'
            )

    async def _update_all(self, s: BytesIO, spans: list[Span]) -> None:
        fs = (self._update(start, stop) for start, stop in spans)
        for blk in await asyncio.gather(*fs):
            s.write(blk)

    async def _update(self, start: int, stop: int) -> bytes:
        blk_size = env.BIPL_TIFF_BLOCK_SIZE
        start *= blk_size
        stop *= blk_size

        async with self._client.stream(
            'GET', self.url, headers={'Range': f'bytes={start}-{stop - 1}'}
        ) as r:
            r.raise_for_status()
            if r.status_code != 206:
                raise ValueError(
                    f'Unexpected status: {r.status_code} for {self.url}'
                )
            return await r.aread()


@memoize()
def _aget_client() -> tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]:
    loop = asyncio.new_event_loop()
    Thread(target=loop.run_forever, daemon=True).start()

    aes = AsyncExitStack()
    atexit.register(_afinalize, aes, loop)

    aclient = httpx.AsyncClient(http1=False, http2=True, follow_redirects=True)
    asyncio.run_coroutine_threadsafe(
        aes.enter_async_context(aclient), loop
    ).result()

    return loop, aclient


def _afinalize(aes: AsyncExitStack, loop: asyncio.AbstractEventLoop) -> None:
    asyncio.run_coroutine_threadsafe(aes.aclose(), loop).result()
