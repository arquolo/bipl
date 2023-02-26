import time
from dataclasses import dataclass
from itertools import count

import glow


@dataclass
class Value:
    value: int


def test_base():
    counter = count()
    ref = glow.Reusable(lambda: Value(next(counter)), delay=0.05)
    assert ref


def test_fail():
    counter = count()
    ref = glow.Reusable(lambda: Value(next(counter)), delay=0.05)
    assert ref().value == 0

    time.sleep(0.1)
    assert ref().value == 1


def test_success() -> None:
    counter = count()
    ref = glow.Reusable(lambda: Value(next(counter)), delay=0.1)
    assert ref().value == 0

    time.sleep(0.05)
    assert ref().value == 0


def test_success_double():
    counter = count()
    ref = glow.Reusable(lambda: Value(next(counter)), delay=0.1)
    assert ref().value == 0

    time.sleep(0.05)
    assert ref().value == 0

    time.sleep(0.05)
    assert ref().value == 0
