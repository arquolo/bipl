from __future__ import annotations

__all__ = ['Trainer']

from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from dataclasses import InitVar, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim
from tqdm.auto import tqdm

from .. import ichunked
from .. import metrics as m
from ._loader import _Loader
from .amp import Grads, get_grads
from .util import eval_


@dataclass
class Stage:
    net: nn.Module
    device: torch.device
    dtype: InitVar[torch.dtype | None]

    def __post_init__(self, dtype: torch.dtype | None):
        self._autocast: Any = (
            torch.autocast(self.device.type, dtype)
            if dtype in (torch.float16, torch.bfloat16) else nullcontext())

    def _move(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device, non_blocking=True)

    def __call__(self, loader: _Loader) -> Iterator[tuple[torch.Tensor, ...]]:
        raise NotImplementedError


@dataclass
class EvalStage(Stage):
    def _step(self, data: torch.Tensor,
              target: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with self._autocast:
            out = self.net(self._move(data))

        return out, target

    def __call__(self, loader: _Loader) -> Iterator[tuple[torch.Tensor, ...]]:
        with eval_(self.net), torch.inference_mode():
            for data, target in loader:
                yield self._step(self._move(data), self._move(target))


@dataclass
class TrainStage(Stage):
    criterion: Callable[..., torch.Tensor]
    grads: Grads
    grad_steps: int = 1

    def _step(self, data: torch.Tensor,
              target: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with self._autocast:
            out = self.net(self._move(data))
            loss = self.criterion(out, target)

        self.grads.backward(loss)
        return out.detach(), target

    def __call__(self, loader: _Loader) -> Iterator[tuple[torch.Tensor, ...]]:
        for batches in ichunked(loader, self.grad_steps):
            with self.grads:
                for data, target in batches:
                    yield self._step(self._move(data), self._move(target))
                # Clip norm here if needed


class Trainer:
    def __init__(self,
                 net: nn.Module,
                 opt: torch.optim.Optimizer,
                 criterion: Callable[..., torch.Tensor],
                 metrics: Iterable[m.Metric],
                 device: torch.device,
                 sched: torch.optim.lr_scheduler._LRScheduler | None = None,
                 dtype: torch.dtype | None = None,
                 grad_steps: int = 1) -> None:
        self.metrics = [m.Lambda(criterion, name='loss'), *metrics]

        grads = get_grads(opt, sched, dtype, max_retries=0)
        self.stages = (
            TrainStage(net, device, dtype, criterion, grads, grad_steps),
            EvalStage(net, device, dtype),
        )

    def _run(self, stage: Stage, loader: _Loader, pbar: tqdm) -> m.Scores:
        meter = m.compose(*self.metrics)
        scores = m.Scores()

        for out in stage(loader):
            scores = meter.send(out)
            pbar.set_postfix(scores.scalars)
            pbar.update()

        return scores

    def train(self, loader: _Loader, pbar: tqdm) -> m.Scores:
        return self._run(self.stages[0], loader, pbar)

    def eval(self, loader: _Loader, pbar: tqdm) -> m.Scores:
        return self._run(self.stages[1], loader, pbar)

    def run(self,
            train_loader: _Loader,
            eval_loader: _Loader,
            epochs: int = 1):
        for i in tqdm(range(1, 1 + epochs), smoothing=0):
            with tqdm(train_loader, desc='train', leave=False) as bar:
                tscalars = self.train(bar, bar).scalars

            with tqdm(eval_loader, desc='val', leave=False) as bar:
                vscalars = self.eval(bar, bar).scalars

            assert tscalars.keys() == vscalars.keys()
            tags = sorted(tscalars.keys() | vscalars.keys())

            # TODO: those lines should be moved outsize into loggers
            line = ','.join(
                f'{tag}: ' + '/'.join(f'{s[tag]:.3f}'
                                      for s in (tscalars, vscalars))
                for tag in tags)
            print(f'[{i:03d}] {line}')


# TODO: define Logger/Handler classes
# TODO:  i.e. Logger which handles tensors and scalars
# TODO:  and Handler flavours like StdoutHandler, TensorBoardHandler, etc.
