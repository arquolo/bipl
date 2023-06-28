from collections.abc import Mapping
from itertools import zip_longest
from typing import Any

# ------------------------- flat list of properties --------------------------


def unflatten(d: Mapping[str, str]) -> dict[str, Any]:
    if not d:
        return {}

    tab = {ord('['): '.', ord(']'): None}
    pairs = sorted((_make_key(k, tab), v) for k, v in d.items())
    r: dict = {}

    for parts, v in pairs:
        rr = r
        for curr, next_ in zip_longest(parts, parts[1:], fillvalue=None):
            # Put list if next is list idx, otherwise put dict
            default: list | dict | str
            default = v if next_ is None else (
                [] if isinstance(next_, int) else {})

            if isinstance(curr, int):  # last was list idx
                if not isinstance(rr, list):  # Ambigous structure
                    break
                if curr >= len(rr):
                    rr.extend(None for _ in range(len(rr), curr + 1))
                    rr[curr] = default
                rr = rr[curr]
            else:  # last was dict key
                if not isinstance(rr, dict):  # Ambigous structure
                    break
                rr = rr.setdefault(curr, default)
    return r


def _make_key(s: str, tab: Mapping[int, str | None]) -> tuple[str | int, ...]:
    parts = s.translate(tab).split('.')
    return tuple(int(word) if word.isdigit() else word for word in parts)


# ---------------------------- aperio description ----------------------------


def is_aperio(s: str) -> bool:
    return s.startswith('Aperio')


def parse_aperio_description(s: str) -> tuple[list[str], dict[str, str]]:
    header, *kv_pairs = s.split('|')

    head = [s.strip() for s in header.splitlines()]

    meta = {}
    for s in kv_pairs:
        tags = s.split('=', 1)
        if len(tags) == 2:
            meta[tags[0].strip()] = tags[1].strip()
        else:
            raise ValueError(f'Unparseable line in description: {s!r}')

    return head, meta
