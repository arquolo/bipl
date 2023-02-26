from __future__ import annotations

__all__ = [
    'Lambda', 'Metric', 'Scores', 'Staged', 'compose', 'to_index',
    'to_index_sparse', 'to_prob', 'to_prob_sparse'
]

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import Protocol, overload

import torch

from .. import coroutine


@dataclass(frozen=True)
class Scores:
    scalars: dict[str, float | int] = field(default_factory=dict)
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, mapping: dict[str, torch.Tensor]) -> Scores:
        obj = cls()
        for k, v in mapping.items():
            if v.numel() == 1:
                obj.scalars[k] = v.item()
            else:
                obj.tensors[k] = v
        return obj


class MetricFn(Protocol):
    def __call__(self, pred, true) -> torch.Tensor:
        ...


class Metric(ABC):
    """Base class for metric"""
    @abstractmethod
    def __call__(self, pred, true) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def collect(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class Lambda(Metric):
    """Wraps arbitary loss function to metric"""
    fn: MetricFn

    @overload
    def __init__(self, fn: Callable, name: str):
        ...

    @overload
    def __init__(self, fn: MetricFn, name: None = ...):
        ...

    def __init__(self, fn, name=None):
        self.fn = fn
        self.name = fn.__name__ if name is None else name

    def __call__(self, pred, true) -> torch.Tensor:
        return self.fn(pred, true)

    def collect(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        return {self.name: state}


class Staged(Metric):
    """Makes metric a "producer": applies multiple functions to its "state" """
    def __init__(self, **funcs: Callable[[torch.Tensor], torch.Tensor]):
        self.funcs = funcs

    def collect(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        return {key: fn(state) for key, fn in self.funcs.items()}


def to_index(pred: torch.Tensor,
             true: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    """
    Convert `pred` of logits with shape [B, C, ...] to [B, ...] of indices,
    i.e. tensors of long.
    """
    assert pred.shape[0] == true.shape[0]
    assert pred.shape[2:] == true.shape[1:]

    c = pred.shape[1]
    pred = pred.argmax(dim=1)

    return c, pred, true


def to_index_sparse(
        pred: torch.Tensor,
        true: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    """
    Convert `pred` of logits with shape [B, C, ...] to [B, ...] of indices,
    i.e. tensors of long. Drops bad indices.
    Result is flattened.
    """
    c, pred, true = to_index(pred, true)

    pred = pred.view(-1)
    true = true.view(-1)

    mask = (true >= 0) & (true < c)
    return c, pred[mask], true[mask]


def to_prob(pred: torch.Tensor,
            true: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    """
    Convert `pred` of logits with shape [B, C, ...] to probs,
    i.e. tensors of float.
    """
    assert pred.shape[0] == true.shape[0]
    assert pred.shape[2:] == true.shape[1:]

    c = pred.shape[2]
    pred = pred.softmax(dim=1)

    return c, pred, true


def to_prob_sparse(
        pred: torch.Tensor,
        true: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    """
    Convert `pred` of logits with shape [B, C, ...] to probs,
    i.e. tensors of float.
    Drops bad indices, i.e. those that are out of range(C).
    Results have shape of [N, C] and [N].
    """
    c, pred, true = to_prob(pred, true)

    b = true.shape[0]
    true = true.view(-1)  # (b n)
    pred = pred.view(b, c, -1).permute(0, 2, 1).view(-1, c)  # (b n) c

    mask = (true >= 0) & (true < c)
    return c, pred[mask], true[mask]


@coroutine
def _batch_averaged(
    fn: Metric
) -> Generator[dict[str, torch.Tensor], Sequence[torch.Tensor], None]:
    assert isinstance(fn, Metric)
    args = yield {}
    state = fn(*args)
    for n in count(2):
        args = yield fn.collect(state)
        state.lerp_(fn(*args), 1 / n)


@coroutine
def compose(*fns: Metric) -> Generator[Scores, Sequence[torch.Tensor], None]:
    updates = *(_batch_averaged(fn) for fn in fns),
    args = yield Scores()
    while True:
        scores: dict[str, torch.Tensor] = {}
        for u in updates:
            scores |= u.send(args)
        args = yield Scores.from_dict(scores)
