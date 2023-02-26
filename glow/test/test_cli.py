from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Optional, TypeVar

import pytest

from glow.cli import parse_args

T = TypeVar('T')


@dataclass
class Arg:
    arg: str


@dataclass
class List_:
    args: list[str]


@dataclass
class UntypedList:  # Forbidden, as list field should always be typed
    args: list


@dataclass
class UnsupportedSet:
    args: set[str]


@dataclass
class BadBoolean:  # Forbidden, as boolean field should always have default
    arg: bool


@dataclass
class Boolean:
    param: bool = False


@dataclass
class Nullable:
    param: Optional[str] = None


@dataclass
class Optional_:
    param: str = 'hello'


@dataclass
class Nested:
    arg: str
    nested: Optional_


@dataclass
class NestedArg:  # Forbidden, as only top level args can be positional
    arg2: str
    nested: Arg


@dataclass
class Aliased:
    arg: str = 'hello'


@dataclass
class NestedAliased:  # Forbidden as all field names must be unique
    arg: str
    nested: Aliased


@pytest.mark.parametrize(('argv', 'expected'), [
    (['value'], Arg('value')),
    ([], List_([])),
    (['a'], List_(['a'])),
    (['a', 'b'], List_(['a', 'b'])),
    ([], Boolean()),
    (['--no-param'], Boolean()),
    (['--param'], Boolean(True)),
    ([], Nullable()),
    (['--param', 'value'], Nullable('value')),
    ([], Optional_()),
    (['--param', 'world'], Optional_('world')),
    (['value'], Nested('value', Optional_())),
    (['value', '--param', 'pvalue'], Nested('value', Optional_('pvalue'))),
])
def test_good_class(argv: list[str], expected: Any):
    cls = type(expected)
    result, _ = parse_args(cls, argv)
    assert isinstance(result, cls)
    assert result == expected


@pytest.mark.parametrize(('cls', 'exc_type'), [
    (Arg, SystemExit),
    (BadBoolean, ValueError),
    (UnsupportedSet, ValueError),
    (UntypedList, ValueError),
    (Nested, SystemExit),
    (NestedArg, ValueError),
    (NestedAliased, ValueError),
])
def test_bad_class(cls: type[Any], exc_type: type[BaseException]):
    with pytest.raises(exc_type):
        parse_args(cls, [])


def _no_op():
    return ()


def _arg(a: int):
    return a


def _kwarg(a: int = 4):
    return a


def _kwarg_opt(a: int = None):  # type: ignore[assignment]
    return a


def _kwarg_literal(a: Literal[1, 2] = 1):
    return a


def _kwarg_bool(a: bool = False):
    return a


def _kwarg_list(a: list[int] = []):  # noqa: B006
    return a


def _kwarg_opt_list(a: list[int] = None):  # type: ignore[assignment]
    return a


def _arg_kwarg(a: int, b: str = 'hello'):
    return a, b


@pytest.mark.parametrize(('argv', 'func', 'expected'), [
    ([], _no_op, ()),
    (['42'], _arg, 42),
    ([], _kwarg, 4),
    (['--a', '58'], _kwarg, 58),
    ([], _kwarg_opt, None),
    (['--a', '73'], _kwarg_opt, 73),
    ([], _kwarg_literal, 1),
    (['--a', '2'], _kwarg_literal, 2),
    ([], _kwarg_bool, False),
    (['--no-a'], _kwarg_bool, False),
    (['--a'], _kwarg_bool, True),
    ([], _kwarg_list, []),
    (['--a'], _kwarg_list, []),
    (['--a', '1'], _kwarg_list, [1]),
    (['--a', '1', '2'], _kwarg_list, [1, 2]),
    ([], _kwarg_opt_list, None),
    (['--a'], _kwarg_opt_list, []),
    (['--a', '1'], _kwarg_opt_list, [1]),
    (['--a', '1', '2'], _kwarg_opt_list, [1, 2]),
    (['53'], _arg_kwarg, (53, 'hello')),
    (['87', '--b', 'bye'], _arg_kwarg, (87, 'bye')),
])
def test_good_func(argv: list[str], func: Callable[..., T], expected: T):
    result, _ = parse_args(func, argv)
    assert result == expected
