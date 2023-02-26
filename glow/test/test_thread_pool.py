from typing import NamedTuple

import numpy as np
import pytest

import glow
from glow.core._parallel import max_cpu_count

DEATH_RATE = 0
SIZE = 100
NUM_STEPS = 10
DTYPE = np.dtype(np.float32)

MAX_PROCS = 8
NUM_PROCS = max_cpu_count(MAX_PROCS, True)


def _make_array(n):
    return np.arange(int(n), dtype=DTYPE)


class AsInit:
    def __init__(self, n):
        self.data = _make_array(n)

    def args(self):
        return [[np.mean(self.data)]] * NUM_STEPS

    def __call__(self, mean):
        assert np.mean(self.data) == mean


class AsArg(NamedTuple):
    n: int

    def args(self):
        for _ in range(NUM_STEPS):
            data = _make_array(self.n)
            yield data, np.mean(data)

    def __call__(self, data, mean):
        return np.mean(data) == mean


class AsArgRepeated(AsArg):
    def args(self):
        args, = super().args()
        return [[*args]] * NUM_STEPS


class AsResult(NamedTuple):
    n: int

    def args(self):
        return np.arange(NUM_STEPS).reshape(-1, 1)

    def __call__(self, _):
        return _make_array(self.n)


def run_glow(task, *args):
    return glow.map_n(task, *args, max_workers=2, mp=True)


def run_joblib(task, *args):
    from glow.joblib import Parallel, delayed
    return Parallel(
        n_jobs=2, backend='multiprocessing')(
            delayed(task)(*a) for a in zip(*args))


def run_joblib_mp(task, *args):
    from glow.joblib import Parallel, delayed
    return Parallel(n_jobs=2)(delayed(task)(*a) for a in zip(*args))


def bench_ipc_speed(order=25, steps=100):
    from matplotlib import pyplot as plt

    sizes = np.asarray([2 ** order, *np.logspace(order, 2, num=steps, base=2)])
    to_bytes = DTYPE.itemsize * 2  # x2, because copy+read

    fig = plt.figure(figsize=(10, 4))
    workers = [AsInit, AsArgRepeated, AsArg, AsResult]
    for i, worker in enumerate(workers, 1):
        ax = fig.add_subplot(
            1,
            len(workers),
            i,
            ylabel='bytes/s',
            xlabel='size',
            xscale='log',
            yscale='log',
            ylim=(1, 1e12),
            title=worker.__name__)
        for runner in [run_glow, run_joblib, run_joblib_mp]:
            label = f'{worker.__name__}-{runner.__name__}'
            times = []
            for size in sizes:
                task = worker(size)
                args = zip(*task.args())
                with glow.timer(times.append):
                    [*runner(task, *args)]

            bps = to_bytes * NUM_STEPS * sizes / np.asarray(times)
            print(f'max {glow.si_bin(bps.max())}/s - {label}')

            ax.plot(to_bytes * sizes, bps, label=runner.__name__)
            ax.legend()

    plt.tight_layout()
    plt.show()


def source(size):
    deads = np.random.uniform(size=size).astype('f4')
    print(np.where(deads < DEATH_RATE)[0].tolist()[:10])
    for seed, death in enumerate(deads):
        if death < DEATH_RATE:
            raise ValueError(f'Source died: {seed}')
        yield seed


def do_work(seed, offset):
    rg = np.random.default_rng(seed + offset)
    n = 10
    a = rg.random((2 ** n, 2 ** n), dtype='f4')
    b = rg.random((2 ** n, 2 ** n), dtype='f4')
    (a @ b).sum()
    if rg.uniform() < DEATH_RATE:
        raise ValueError(f'Worker died: {seed}') from None
    return seed


def _test_interrupt():
    """Should die gracefully on Ctrl-C"""
    sources = (
        source(SIZE),
        np.random.randint(2 ** 10, size=SIZE),
    )
    # sources = map(glow.buffered, sources)
    res = glow.map_n(do_work, *sources, max_workers=NUM_PROCS, mp=True)
    print('start main', end='')
    for r in res:
        print(end=f'\rmain {r} computes...')
        rg = np.random.default_rng(r)
        n = 10
        a = rg.random((2 ** n, 2 ** n), dtype='f4')
        b = rg.random((2 ** n, 2 ** n), dtype='f4')
        (a @ b).sum()
        yield r
        print(end=f'\rmain {r} waits...')
    print('\rmain done')


@pytest.mark.skipif(NUM_PROCS < 2, reason='not enough memory')
def test_interrupt():
    rs = _test_interrupt()
    assert [*rs] == [*range(SIZE)]


@pytest.mark.skipif(NUM_PROCS < 2, reason='not enough memory')
def test_interrupt_with_buffer():
    rs = glow.buffered(_test_interrupt())
    assert [*rs] == [*range(SIZE)]


if __name__ == '__main__':
    bench_ipc_speed()
