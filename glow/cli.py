"""argparse and dataclasses, married.

Example:
```
@dataclass
class Args:
    a: int
    b: str = 'hello'

args, parser = parse_args(Args)
```
Or with plain function:
```
@parse_args
def main(name: str = 'user'):
    print(f'Hello {name}')
```

Reasons not to use alternatives:
- [simple-parsing](https://github.com/lebrice/SimpleParsing):
  - Has underscores (`--like_this`) instead of dashes (`--like-this`)
  - Erases type on parser result, thus making typed prototype useless
    (what is the point of using dataclasses if it is passed
     through function returning typing.Any/argparse.Namespace?)

- [datargs](https://github.com/roee30/datargs):
  - No nesting support
  - No function's support

- [pydantic](https://github.com/samuelcolvin/pydantic):
  - supports CLI via BaseSettings and environment variables parsing
  - no nesting, as requires mixing only via multiple inheritance

- [typer](https://github.com/tiangolo/typer):
  - No support on dataclasses
    (https://github.com/tiangolo/typer/issues/154).
  - No fine way to extract parsed options without invoking because of
    decorator/callback based implementation. Thus enforces wrapping of the
    whole app into `typer.run`.
    (https://github.com/tiangolo/typer/issues/197).
"""

from __future__ import annotations

__all__ = ['arg', 'parse_args']

import argparse
import sys
import types
from argparse import ArgumentParser, BooleanOptionalAction, _ArgumentGroup
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import MISSING, Field, field, fields, is_dataclass
from inspect import getmodule, signature, stack
from typing import (Any, Literal, TypeVar, Union, get_args, get_origin,
                    get_type_hints)

_T = TypeVar('_T')
_Node = Union[str, tuple[str, type, list['_Node']]]
_NoneType = type(None)
_UNION_TYPES: list = [Union]

if sys.version_info >= (3, 10):
    _UNION_TYPES += [types.UnionType]


def arg(
    default=MISSING,
    /,
    *,
    flag=None,
    factory=MISSING,
    init=True,
    repr=True,  # noqa: A002
    hash=None,  # noqa: A002
    help=None,  # noqa: A002
    compare=True,
    metadata=None,
):
    """Convinient alias for dataclass.field with extra metadata (like help)"""
    metadata = metadata or {}
    for k, v in {'flag': flag, 'help': help}.items():
        if v:
            metadata = metadata | {k: v}
    return field(  # type: ignore[call-overload]
        default=default,
        default_factory=factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata)


def _unwrap_type(tp: type) -> tuple[type, dict[str, Any]]:
    if tp is list:
        raise ValueError('Type list should be parametrized')

    origin = get_origin(tp)
    *args, = get_args(tp)
    if not origin or not args:
        return tp, {'type': tp}

    if origin is list:
        cls, opts = _unwrap_type(args[0])
        return cls, opts | {'nargs': argparse.ZERO_OR_MORE}

    if origin in _UNION_TYPES and len(args) == 2 and _NoneType in args:
        args.remove(_NoneType)
        cls, opts = _unwrap_type(args[0])
        if opts.get('nargs') == argparse.ZERO_OR_MORE:
            return cls, opts
        return cls, opts | {'nargs': argparse.OPTIONAL}

    if origin is Literal:
        choices = get_args(tp)
        if len(tps := {type(c) for c in choices}) != 1:
            raise ValueError('Literal parameters should have '
                             f'the same type. Got: {tps}')
        cls, = tps
        return cls, {'type': cls, 'choices': choices}

    raise ValueError('Only list, Optional and Literal are supported '
                     f'as generic types. Got: {tp}')


def _get_fields(fn: Callable) -> Iterator[Field]:
    if is_dataclass(fn):  # Shortcut
        yield from fields(fn)
        return

    for p in signature(fn).parameters.values():
        if p.kind is p.KEYWORD_ONLY and p.default is p.empty:
            raise ValueError(f'Keyword "{p.name}" must have default')
        if p.kind in {p.POSITIONAL_ONLY, p.VAR_POSITIONAL, p.VAR_KEYWORD}:
            raise ValueError(f'Unsupported parameter type: {p.kind}')

        if isinstance(p.default, Field):
            fd = p.default
        else:
            fd = arg(MISSING if p.default is p.empty else p.default)
        fd.name = p.name
        yield fd


def _visit_nested(parser: ArgumentParser | _ArgumentGroup, fn: Callable,
                  seen: dict[str, list]) -> list[_Node]:
    try:
        hints = get_type_hints(fn)
    except NameError:
        if fn.__module__ != '__main__':
            raise
        for finfo in stack():
            if not getmodule(finfo.frame):
                hints = get_type_hints(fn, finfo.frame.f_globals)
                break
        else:
            raise

    nodes: list[_Node] = []
    for fd in _get_fields(fn):
        if fd.init:
            seen.setdefault(fd.name, []).append(fn)
            nodes.append(_visit_field(parser, hints[fd.name], fd, seen))

    for name, usages in seen.items():
        if len(usages) > 1:
            raise ValueError(f'Field name "{name}" occured multiple times: '
                             + ', '.join(f'{c.__module__}.{c.__qualname__}'
                                         for c in usages)
                             + '. All field names should be unique')
    return nodes


def _visit_field(parser: ArgumentParser | _ArgumentGroup, tp: type, fd: Field,
                 seen: dict[str, list]) -> _Node:
    cls, opts = _unwrap_type(tp)

    help_ = fd.metadata.get('help') or ''
    if cls is not bool and fd.default is not MISSING:
        help_ += f' (default: {fd.default})'

    if is_dataclass(cls):  # Nested dataclass
        arg_group = parser.add_argument_group(fd.name)
        return fd.name, cls, _visit_nested(arg_group, cls, seen)

    snake = fd.name.replace('_', '-')
    flags = [f] if (f := fd.metadata.get('flag')) else []

    if cls is bool:  # Optional
        if fd.default is MISSING:
            raise ValueError(f'Boolean field "{fd.name}" should have default')
        parser.add_argument(
            f'--{snake}',
            *flags,
            action=BooleanOptionalAction,
            default=fd.default,
            help=help_)

    elif fd.default is not MISSING:  # Generic optional
        if opts.get('nargs') == argparse.OPTIONAL:
            del opts['nargs']
        parser.add_argument(
            f'--{snake}', *flags, **opts, default=fd.default, help=help_)

    elif isinstance(parser, ArgumentParser):  # Allow only for root parser
        if flags:
            raise ValueError(f'Positional-only field "{fd.name}" '
                             'should not have flag')
        parser.add_argument(snake, **opts, help=help_)

    else:
        raise ValueError('Positional-only fields are forbidden '
                         'for nested types. Please set default value '
                         f'for "{fd.name}"')

    return fd.name


def _construct(src: dict[str, Any], fn: Callable[..., _T],
               args: Collection[_Node]) -> _T:
    kwargs = {}
    for a in args:
        if isinstance(a, str):
            kwargs[a] = src.pop(a)
        else:
            kwargs[a[0]] = _construct(src, a[1], a[2])
    return fn(**kwargs)


def parse_args(fn: Callable[..., _T],
               args: Sequence[str] | None = None,
               prog: str | None = None) -> tuple[_T, ArgumentParser]:
    """Create parser from type hints of callable, parse args and do call"""
    # TODO: Rename to `run`
    parser = ArgumentParser(prog)
    nodes = _visit_nested(parser, fn, {})

    if args is not None:  # fool's protection
        args = line.split(' ') if (line := ' '.join(args).strip()) else []

    namespace = parser.parse_args(args)
    obj = _construct(vars(namespace), fn, nodes)
    return obj, parser
