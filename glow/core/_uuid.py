from __future__ import annotations

__all__ = ['Uid']

import math
import re
import string
from functools import lru_cache
from typing import SupportsInt
from uuid import UUID, uuid4

ALPHABET = string.digits + string.ascii_letters
ALPHABET = ''.join(sorted({*ALPHABET} - {*'0O1Il'}))

_BASE = len(ALPHABET)  # 57
_LEN = math.ceil(128 / math.log2(_BASE))  # 22

_TABLE = ALPHABET.encode('ascii').ljust(256, b'\0')
_NUMBERS = {s: i for i, s in enumerate(ALPHABET)}
_REGEX = re.compile(f'^[{ALPHABET}]{{{_LEN}}}$')


@lru_cache  # Small performance optimization
def base57_encode(number: int) -> str:
    out = bytearray(_LEN)
    for i in range(_LEN - 1, -1, -1):
        number, out[i] = divmod(number, _BASE)
    return out.translate(_TABLE).decode('ascii')


def base57_decode(shortuuid: str) -> int:
    if not _REGEX.fullmatch(shortuuid):
        raise ValueError('invalid shortuuid format')
    out = 0
    for char in shortuuid:
        out = out * _BASE + _NUMBERS[char]
    return out


class Uid(UUID):
    """Subclass of UUID with support of short-uuid serialization format.

    Uses base57 instead of hex for serialization.

    base57 uses lowercase and uppercase letters and digits,
    excluding similar-looking characters such as l, 1, I, O and 0,
    and it doesn't use URL-unsafe +, /, = characters (opposed to base64).

    UUIDs encoded with base57 have length of 22 characters, while
    with hex (default) - 32 characters.

    Uid can be created directly from UUID:

        >>> u = UUID('3b1f8b40-222c-4a6e-b77e-779d5a94e21c')
        >>> Uid(u)
        Uid('CXc85b4rqinB7s5J52TRYb')
        >>> str(Uid(u))
        'CXc85b4rqinB7s5J52TRYb'

    Or from string representation of short-uuid:

        >>> Uid('CXc85b4rqinB7s5J52TRYb')
        Uid('CXc85b4rqinB7s5J52TRYb')

    Simplified and more optimized (2-3x faster on average) fork of
    [shortuuid](https://github.com/skorokithakis/shortuuid)
    """
    def __init__(self, obj: str | SupportsInt):
        """
        Creates Uid either from str (parsing it as short-uuid) or
        from int()-compatible type
        """
        if not isinstance(obj, (str, SupportsInt)):
            raise ValueError('Either int, string or UUID required. '
                             f'Got {type(obj)}')

        value = base57_decode(obj) if isinstance(obj, str) else int(obj)
        super().__init__(int=value)

    def __str__(self) -> str:
        return base57_encode(int(self))

    @classmethod
    def __get_validators__(cls):  # Required for Pydantic
        yield cls

    @classmethod
    def __modify_schema__(cls, field_schema: dict):  # Required for OpenAPI
        field_schema.update(
            examples=[str(cls.v4()) for _ in range(2)],
            type='string',
            format=None,
            pattern=_REGEX.pattern,
        )

    @classmethod
    def v4(cls) -> Uid:
        """Alias for Uid(uuid.uuid4())"""
        return cls(uuid4())
