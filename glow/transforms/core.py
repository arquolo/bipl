from __future__ import annotations

__all__ = [
    'Chain', 'DualStageTransform', 'ImageTransform', 'MaskTransform',
    'Transform'
]

from collections.abc import Iterable
from typing import Any, Protocol, final, runtime_checkable

import numpy as np


@runtime_checkable
class Transform(Protocol):
    def __call__(self, rng: np.random.Generator, /, **data) -> dict[str, Any]:
        raise NotImplementedError

    def __mul__(self, prob: float) -> Transform:
        if not isinstance(prob, (int, float)):
            return NotImplemented
        if not (0 <= prob <= 1):
            raise ValueError('Probability should be in [0.0 .. 1.0] range')

        if prob == 1:
            return self

        if isinstance(self, _Maybe):
            prob *= self.prob
            self = self.func
        return _Maybe(self, prob)

    def __rmul__(self, prob: float) -> Transform:
        return self * prob

    def __or__(self, rhs: Transform) -> _OneOf:
        if not isinstance(rhs, Transform):
            return NotImplemented
        ts = (
            t_ for t in (self, rhs)
            for t_ in (t.transforms if isinstance(t, _OneOf) else [t]))
        return _OneOf(*ts)


class _SingleTransform(Transform):
    _key: str

    @final
    def __call__(self, rng: np.random.Generator, /, **data) -> dict[str, Any]:
        func = getattr(self, self._key)
        return {**data, self._key: func(data[self._key], rng)}


class ImageTransform(_SingleTransform):
    _key = 'image'

    def image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class MaskTransform(_SingleTransform):
    _key = 'mask'

    def mask(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class DualStageTransform(Transform):
    _keys = frozenset[str]({'image', 'mask'})

    def prepare(self, rng: np.random.Generator, /,
                **data) -> dict[str, Any] | None:
        return {}

    def image(self, image: np.ndarray, **params) -> np.ndarray:
        return image

    def mask(self, mask: np.ndarray, **params) -> np.ndarray:
        return mask

    @final
    def __call__(self, rng: np.random.Generator, /, **data) -> dict[str, Any]:
        if unknown_keys := {*data} - self._keys:
            raise ValueError(f'Got unknown keys in data: {unknown_keys}')

        params = self.prepare(rng, **data)
        if params is None:
            return data
        return {
            key: getattr(self, key)(value, **params)
            for key, value in data.items() if value is not None
        }


@final
class _Maybe(Transform):
    def __init__(self, func: Transform, prob: float = 1.0):
        self.func = func
        self.prob = prob

    def __repr__(self):
        return f'{self.prob:.2f} * {self.func!r}'

    def __call__(self, rng: np.random.Generator, /, **data) -> dict[str, Any]:
        return self.func(rng, **data) if rng.random() <= self.prob else data


class _Compose(Transform):
    transforms: tuple[Transform, ...]

    def _repr(self, words: Iterable[str]) -> str:
        if parts := ',\n'.join(words):
            parts = '\n'.join((f'    {p}' if p.strip() else p)
                              for p in parts.splitlines(True))
            parts = f'\n{parts}\n'
        return f'{type(self).__name__}({parts})'


@final
class Chain(_Compose):
    def __init__(self, *transforms: Transform):
        self.transforms = *(t for t in transforms
                            if not isinstance(t, _Maybe) or t.prob > 0),

    def __repr__(self) -> str:
        return self._repr(f'{t!r}' for t in self.transforms)

    def __call__(self, rng: np.random.Generator, /, **data) -> dict[str, Any]:
        for call in self.transforms:
            data = call(rng, **data)
        return data


@final
class _OneOf(_Compose):
    transforms: tuple[_Maybe, ...]
    probs: np.ndarray

    def __init__(self, *transforms: Transform):
        probs, self.transforms = zip(
            *((t.prob, t) if isinstance(t, _Maybe) else (1.0, _Maybe(t))
              for t in transforms))
        self.probs = np.array(probs)
        self.probs /= self.probs.sum()

    def __repr__(self) -> str:
        words = (f'{p:.2f} * {t.func}'
                 for p, t in zip(self.probs, self.transforms))
        return self._repr(words)

    def __call__(self, rng: np.random.Generator, /, **data) -> dict[str, Any]:
        func = rng.choice(self.transforms, p=self.probs)  # type: ignore
        return func.func(rng, **data)
