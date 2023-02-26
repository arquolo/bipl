from __future__ import annotations

__all__ = [
    'auto_ddp', 'auto_model', 'barrier', 'get_ddp_info', 'reduce_if_needed'
]

import pickle
from collections.abc import Callable
from functools import partial, update_wrapper
from pathlib import Path
from typing import Any, NamedTuple, Protocol, TypeVar, cast

import torch
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as tmp
from torch import nn

# -------------------------------- primitives --------------------------------


class _DdpInfo(NamedTuple):
    world: int
    rank: int


def get_ddp_info() -> _DdpInfo | None:
    if not dist.is_initialized():
        return None
    return _DdpInfo(dist.get_rank(), dist.get_world_size())


def barrier(rank: int | None = None) -> None:
    """Synchronize all processes"""
    if (info := get_ddp_info()) and (rank is None or rank == info.rank):
        dist.barrier()


def reduce_if_needed(*tensors: torch.Tensor,
                     mean: bool = False) -> tuple[torch.Tensor, ...]:
    """Reduce tensors across all machines"""
    if (ddp := get_ddp_info()) and ddp.world > 1:
        tensors = *(t.clone() for t in tensors),
        dist.all_reduce_multigpu(tensors)
        if mean:
            tensors = *(t / ddp.world for t in tensors),
    return tensors


# --------------------------------- wrappers ---------------------------------


def auto_model(net: nn.Module, sync_bn: bool = True) -> nn.Module:
    if (ddp := get_ddp_info()) and ddp.world > 1:
        torch.cuda.set_device(ddp.rank)

        net.to(ddp.rank)
        if sync_bn:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        return nn.parallel.DistributedDataParallel(net, device_ids=[ddp.rank])

    net.cuda()
    return (nn.parallel.DataParallel(net)
            if torch.cuda.device_count() > 1 else net)


class _TrainFn(Protocol):
    def __call__(self, __net: nn.Module, *args, **kwargs) -> Any:
        ...


_F = TypeVar('_F', bound=Callable)
_TrainFnType = TypeVar('_TrainFnType', bound=_TrainFn)


class _AutoDdp:
    def __init__(self, train_fn: _TrainFn, net: nn.Module, *args, **kwargs):
        self.train_fn = train_fn
        self.net = net
        self.args = args
        self.kwargs = kwargs
        self.ngpus = torch.cuda.device_count()

        if self.ngpus == 1:
            self._worker(None)
            return

        # ! Not tested
        # * Actually, here we can use loky.ProcessPoolExecutor, like this:
        # from . import map_n
        # ngpus = self.ngpus
        # jobs = map_n(self._worker, range(ngpus), max_workers=ngpus, mp=True)
        # list(jobs)
        # * Left as safe measure
        tmp.spawn(self._worker, nprocs=self.ngpus)

    def _worker(self, rank: int | None) -> None:
        if rank is None:
            return self.train_fn(self.net, *self.args, **self.kwargs)

        dist.init_process_group(
            backend='nccl', rank=rank, world_size=self.ngpus)
        try:
            self.train_fn(auto_model(self.net), *self.args, **self.kwargs)
        finally:
            dist.destroy_process_group()


def auto_ddp(train_fn: _TrainFnType) -> _TrainFnType:
    return cast(_TrainFnType,
                update_wrapper(partial(_AutoDdp, train_fn), train_fn))


def once_per_world(fn: _F) -> _F:
    """Call function only in rank=0 process, and share result for others"""
    def wrapper(*args, **kwargs):
        ddp = get_ddp_info()
        if not ddp or ddp.world == 1:
            # Master process, so no neighbors to share results with
            return fn(*args, **kwargs)

        # Generate random fname and share it among whole world
        idx = torch.empty((), dtype=torch.int64).random_()
        if ddp.rank == 0:
            dist.broadcast(idx, 0)

        tmp = Path(f'/tmp/_ddp_share_{idx.item():x}.pkl')
        result = None

        if ddp.rank == 0:  # 0th child
            result = fn(*args, **kwargs)
            with tmp.open('wb') as fp:
                pickle.dump(result, fp)

        barrier()

        if ddp.rank > 0:  # Gather results from 0th child
            with tmp.open('rb') as fp:
                result = pickle.load(fp)

        barrier()

        if ddp.rank == 0:  # 0th child
            tmp.unlink()

        return result

    return cast(_F, update_wrapper(wrapper, fn))
