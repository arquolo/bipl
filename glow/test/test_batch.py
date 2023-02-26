import random
from unittest.mock import MagicMock

import pytest

import glow


def test_as_is():
    load = MagicMock(side_effect=lambda x: x)

    @glow.memoize(8192, batched=True)
    def load_batch(xs):
        return [load(x) for x in xs]

    for _ in range(5):
        numbers = random.choices(range(20), k=5)
        results = load_batch(numbers)
        assert numbers == results

    called_with = (x for (x, ), _ in load.call_args_list)
    assert sorted(called_with) == sorted(load_batch.cache)


@pytest.mark.parametrize('max_workers', [2, 4] * 10)
def test_thread_safe(max_workers):
    @glow.memoize(8192, batched=True)
    def load_batch(xs):
        return [*xs]

    numbers = random.choices(range(100), k=100)
    nchunks = glow.chunked(numbers, 5)
    rchunks = glow.map_n(load_batch, nchunks, max_workers=max_workers)
    assert [r for c in rchunks for r in c] == numbers
