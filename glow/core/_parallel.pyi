from collections.abc import Iterable, Iterator
from concurrent.futures import Executor
from typing import Generic, Protocol, TypeVar, overload

_T = TypeVar('_T')
_T1_ctr = TypeVar('_T1_ctr', contravariant=True)
_T2_ctr = TypeVar('_T2_ctr', contravariant=True)
_T3_ctr = TypeVar('_T3_ctr', contravariant=True)
_R_co = TypeVar('_R_co', covariant=True)


class _Callable1(Generic[_T1_ctr, _R_co], Protocol):
    def __call__(self, __1: _T1_ctr, /) -> _R_co:
        ...


class _Callable2(Generic[_T1_ctr, _T2_ctr, _R_co], Protocol):
    def __call__(self, __1: _T1_ctr, __2: _T2_ctr, /) -> _R_co:
        ...


class _Callable3(Generic[_T1_ctr, _T2_ctr, _T3_ctr, _R_co], Protocol):
    def __call__(self, __1: _T1_ctr, __2: _T2_ctr, __3: _T3_ctr, /) -> _R_co:
        ...


class _Callable4(Generic[_R_co], Protocol):
    def __call__(self, __1, __2, __3, __4, *args) -> _R_co:
        ...


def max_cpu_count(upper_bound: int = ..., mp: bool = ...) -> int:
    ...


def buffered(__iter: Iterable[_T],
             /,
             *,
             latency: int = ...,
             mp: bool | Executor = ...) -> Iterator[_T]:
    ...


@overload
def starmap_n(__func: _Callable1[_T1_ctr, _R_co],
              __iter: Iterable[tuple[_T1_ctr]],
              /,
              *,
              max_workers: int | None = ...,
              prefetch: int | None = ...,
              mp: bool = ...,
              chunksize: int | None = ...,
              order: bool = ...) -> Iterator[_R_co]:
    ...


@overload
def starmap_n(__func: _Callable2[_T1_ctr, _T2_ctr, _R_co],
              __iter: Iterable[tuple[_T1_ctr, _T2_ctr]],
              /,
              *,
              max_workers: int | None = ...,
              prefetch: int | None = ...,
              mp: bool = ...,
              chunksize: int | None = ...,
              order: bool = ...) -> Iterator[_R_co]:
    ...


@overload
def starmap_n(__func: _Callable3[_T1_ctr, _T2_ctr, _T3_ctr, _R_co],
              __iter: Iterable[tuple[_T1_ctr, _T2_ctr, _T3_ctr]],
              /,
              *,
              max_workers: int | None = ...,
              prefetch: int | None = ...,
              mp: bool = ...,
              chunksize: int | None = ...,
              order: bool = ...) -> Iterator[_R_co]:
    ...


@overload
def starmap_n(__func: _Callable4[_R_co],
              __iter: Iterable[Iterable],
              /,
              *,
              max_workers: int | None = ...,
              prefetch: int | None = ...,
              mp: bool = ...,
              chunksize: int | None = ...,
              order: bool = ...) -> Iterator[_R_co]:
    ...


@overload
def map_n(__func: _Callable1[_T1_ctr, _R_co],
          __iter1: Iterable[_T1_ctr],
          /,
          *,
          max_workers: int | None = ...,
          prefetch: int | None = ...,
          mp: bool = ...,
          chunksize: int | None = ...,
          order: bool = ...) -> Iterator[_R_co]:
    ...


@overload
def map_n(__f: _Callable2[_T1_ctr, _T2_ctr, _R_co],
          __iter1: Iterable[_T1_ctr],
          __iter2: Iterable[_T2_ctr],
          /,
          *,
          max_workers: int | None = ...,
          prefetch: int | None = ...,
          mp: bool = ...,
          chunksize: int | None = ...,
          order: bool = ...) -> Iterator[_R_co]:
    ...


@overload
def map_n(__f: _Callable3[_T1_ctr, _T2_ctr, _T3_ctr, _R_co],
          __iter1: Iterable[_T1_ctr],
          __iter2: Iterable[_T2_ctr],
          __iter3: Iterable[_T3_ctr],
          /,
          *,
          max_workers: int | None = ...,
          prefetch: int | None = ...,
          mp: bool = ...,
          chunksize: int | None = ...,
          order: bool = ...) -> Iterator[_R_co]:
    ...


@overload
def map_n(__func: _Callable4[_R_co],
          /,
          __iter1: Iterable,
          __iter2: Iterable,
          __iter3: Iterable,
          __iter4: Iterable,
          *__iters: Iterable,
          max_workers: int | None = ...,
          prefetch: int | None = ...,
          mp: bool = ...,
          chunksize: int | None = ...,
          order: bool = ...) -> Iterator[_R_co]:
    ...
