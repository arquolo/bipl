from collections import Counter
from uuid import UUID, uuid4

import pytest

from glow import Uid


def test_generation():
    assert 20 < len(str(Uid.v4())) < 24


def test_encoding():
    u = UUID('{3b1f8b40-222c-4a6e-b77e-779d5a94e21c}')
    assert str(Uid(u)) == 'CXc85b4rqinB7s5J52TRYb'


def test_decoding():
    u = UUID('{3b1f8b40-222c-4a6e-b77e-779d5a94e21c}')
    assert Uid('CXc85b4rqinB7s5J52TRYb') == u


def test_padding():
    uid = uuid4()
    uid_smallest = UUID(int=0)

    encoded = str(Uid(uid))
    assert Uid(encoded) == uid

    encoded_small = str(Uid(uid_smallest))
    assert Uid(encoded_small) == uid_smallest

    assert len(encoded) == len(encoded_small)


def test_stability():
    u = uuid4()
    assert u == Uid(str(Uid(u)))

    u = Uid.v4()
    assert isinstance(u, Uid)
    assert u == Uid(u)
    assert u == Uid(str(u))

    with pytest.raises(ValueError):
        Uid('0')


def test_consistency():
    num_iterations = 1000
    lengths = Counter()

    for _ in range(num_iterations):
        uid = Uid.v4()

        encoded = str(uid)
        lengths[len(encoded)] += 1
        uid_decoded = Uid(encoded)

        assert uid == uid_decoded

    assert len(lengths) == 1

    (_, count), = lengths.most_common()
    assert count == num_iterations


def test_edge_cases():
    with pytest.raises(ValueError):
        Uid([])
    with pytest.raises(ValueError):
        Uid({})
    with pytest.raises(ValueError):
        Uid((2, ))
