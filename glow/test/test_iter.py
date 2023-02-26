from weakref import WeakValueDictionary

import numpy as np

from glow import chunked, ichunked, windowed


def test_windowed():
    it = windowed(range(5), 3)
    assert len(it) == 3
    assert [*it] == [range(0, 3), range(1, 4), range(2, 5)]
    assert len(it) == 0

    it = windowed(iter(range(5)), 3)
    assert [*it] == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]


def test_chunked():
    it = chunked(range(5), 3)
    assert len(it) == 2
    assert [*it] == [range(0, 3), range(3, 5)]
    assert len(it) == 0

    it = chunked(iter(range(5)), 3)
    assert [*it] == [(0, 1, 2), (3, 4)]


def test_ichunked():
    it = ichunked(range(5), 3)
    assert len([*it]) == 2

    it = ichunked(range(5), 3)
    assert [tuple(c) for c in it] == [(0, 1, 2), (3, 4)]

    it = ichunked(range(5), 3)
    assert [tuple(c) for c in [*it]] == [(0, 1, 2), (3, 4)]


def generator(n, refs):
    heap = {}
    for i, x in enumerate(np.random.rand(n, 4)):
        refs[i] = heap[i] = x
        del x
        # Careful not to keep reference to x
        yield heap.popitem()


def test_windowed_refs():
    refs = WeakValueDictionary()
    assert not refs

    it = generator(10, refs)
    for c in windowed(it, 3):
        assert len(refs) <= 3
        del c
    assert not refs


def test_chunked_refs():
    refs = WeakValueDictionary()
    assert not refs

    it = generator(10, refs)
    for c in chunked(it, 3):
        assert len(refs) <= 3
        del c
    assert not refs


def test_ichunked_refs():
    refs = WeakValueDictionary()
    assert not refs

    it = generator(10, refs)
    for c in ichunked(it, 3):
        for x in c:
            assert len(refs) <= 1
            del x
    assert not refs

    it = generator(10, refs)
    chunks = [*ichunked(it, 3)]
    assert len(refs) == 10
    for i, x in zip(range(10, 0, -1), (x for c in chunks for x in c)):
        assert len(refs) <= i
        del x
    assert not refs
