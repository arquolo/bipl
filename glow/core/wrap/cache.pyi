from collections.abc import Callable, Hashable, Iterable
from typing import Literal, TypeVar, overload

from typing_extensions import TypeAlias

_F = TypeVar('_F', bound=Callable)
_BatchedFn = TypeVar('_BatchedFn', bound=Callable[[list], Iterable])
_Policy: TypeAlias = Literal['raw', 'lru', 'mru']
_KeyFn: TypeAlias = Callable[..., Hashable]


@overload
def memoize(capacity: int,
            *,
            policy: _Policy = ...,
            key_fn: _KeyFn = ...) -> Callable[[_F], _F]:
    ...


@overload
def memoize(capacity: int,
            *,
            batched: Literal[True],
            policy: _Policy = ...,
            key_fn: _KeyFn = ...) -> Callable[[_BatchedFn], _BatchedFn]:
    ...
