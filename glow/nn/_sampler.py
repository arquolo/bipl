from __future__ import annotations

__all__ = ['DdpSampler']

from collections.abc import Iterator, Sized
from itertools import chain, cycle, islice
from typing import Generic, Protocol, TypeVar

import torch
from torch.utils.data import (RandomSampler, Sampler, SequentialSampler,
                              SubsetRandomSampler, WeightedRandomSampler)

from ..distributed import get_ddp_info

_T_co = TypeVar('_T_co', covariant=True)
_TORCH_SAMPLERS = (RandomSampler, SubsetRandomSampler, WeightedRandomSampler)

# If we need resume-like reproducibility, then .step() if necessary
# (i.e. using BASE_SEED + epoch as seed for each epoch).
# Though this way, any alteration of epoch len will break reproducibility.
# This is useful when epoch len = dataset len, and sampling is done
# without replacement.

# If we only need from-scratch-like reproducibility, then we can drop .step()
# and rely on generator state.
# This way we can decouple on epoch len.
# Epoch transitions will be blurred, so the whole train loop will be like on
# the infinite dataset.
# Useful with replacement sampling and custom epoch len.


def generate_seed() -> int:
    return int(torch.empty((), dtype=torch.int64).random_().item())


class SamplerLike(Protocol, Generic[_T_co]):
    def __iter__(self) -> Iterator[_T_co]:
        ...

    def __len__(self) -> int:
        ...


class DdpSampler(Sampler[_T_co]):
    def __init__(self,
                 sampler: Sampler[_T_co] | SamplerLike[_T_co],
                 drop_last: bool = False) -> None:
        assert isinstance(sampler, Sized)
        self.base = sampler
        self.drop_last = drop_last
        self.seed = generate_seed()

        if (isinstance(sampler, _TORCH_SAMPLERS)
                and sampler.generator is None):
            # Enforce to use local generator
            sampler.generator = torch.Generator()

    def step(self, n: int = 1) -> None:
        self.seed += n

        # If torch random sampler, reseed its generator
        if isinstance(self.base, _TORCH_SAMPLERS):
            assert isinstance(self.base.generator, torch.Generator)
            self.base.generator.manual_seed(self.seed)

        # Non-sequential sampler, treat as random sampler
        # But as we don't know whether it uses generator reseed globally
        elif not isinstance(self.base, SequentialSampler):
            torch.manual_seed(self.seed)

    def __iter__(self) -> Iterator[_T_co]:
        self.step()

        if ddp := get_ddp_info():
            total = len(self) * ddp.world
            indices = iter(self.base)

            if (padding := total - len(self.base)) > 0:
                if len(self.base) < ddp.world:
                    indices = cycle(indices)
                else:
                    *head, = islice(indices, padding)
                    indices = chain(head, indices, head)

            return islice(indices, ddp.rank, total, ddp.world)

        return iter(self.base)

    def __len__(self) -> int:
        if ddp := get_ddp_info():
            if self.drop_last:
                return (len(self.base) + ddp.world - 1) // ddp.world
            return len(self.base) // ddp.world
        return len(self.base)
