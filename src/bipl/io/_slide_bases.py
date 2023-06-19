from __future__ import annotations

__all__ = ['Driver', 'Item', 'Lod', 'REGISTRY']

from dataclasses import dataclass
from pathlib import Path
from typing import final

import cv2
import numpy as np

REGISTRY: dict[str, list[type[Driver]]] = {}


def normalize(slices: tuple[slice, ...] | slice,
              shape: tuple[int, ...]) -> tuple[slice, ...]:
    """Ensures slices to be exactly 2 slice with non-none endpoints"""
    if isinstance(slices, slice):
        slices = slices, slice(None)
    assert len(slices) == 2
    return *(slice(
        s.start if s.start is not None else 0,
        s.stop if s.stop is not None else axis_len,
        s.step if s.step is not None else 1,
    ) for s, axis_len in zip(slices, shape)),


@dataclass(frozen=True)
class Item:
    shape: tuple[int, ...]

    @property
    def key(self) -> str | None:
        raise NotImplementedError

    def __array__(self) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class Lod(Item):
    spacing: float | None

    @final
    @property
    def key(self) -> None:
        return None

    def crop(self, slices: tuple[slice, ...]) -> np.ndarray:
        """Reads crop of LOD. Overridable"""
        raise NotImplementedError

    @final
    def __getitem__(self, key: slice | tuple[slice, ...]) -> np.ndarray:
        """Reads crop of LOD"""
        slices = normalize(key, self.shape)
        assert all(s.step == 1 for s in slices)
        return self.crop(slices)

    @final
    def __array__(self) -> np.ndarray:
        """Reads whole LOD in single call"""
        return self[:, :]

    def rescale(self, scale: float) -> Lod:
        if scale == 1:
            return self
        h, w, c = self.shape
        return ProxyLod(
            # TODO: round/ceil/floor ?
            (round(h * scale), round(w * scale), c),
            (self.spacing / scale if self.spacing else None),
            scale,
            self.base if isinstance(self, ProxyLod) else self,
        )


@dataclass(frozen=True)
class ProxyLod(Lod):
    scale: float
    base: Lod

    def crop(self, slices: tuple[slice, ...]) -> np.ndarray:
        src_slices = *[
            # TODO: round/ceil/floor ?
            slice(round(s.start / self.scale), round(s.stop / self.scale))
            for s in slices
        ],
        image = self.base[src_slices]

        shape = *((s.stop - s.start) for s in slices),
        return cv2.resize(image, shape[::-1], interpolation=cv2.INTER_AREA)


class Driver:
    @final
    @classmethod
    def register(cls, extensions: str):
        """Registers type builder for extensions. Last call takes precedence"""
        for ext in extensions.split():
            REGISTRY.setdefault(f'.{ext}', []).append(cls)

    def __init__(self, path: Path) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        """Count of indexed items"""
        return 0

    def __getitem__(self, index: int) -> Item:
        """Gives indexed item"""
        raise NotImplementedError

    def named_items(self) -> dict[str, Item]:
        keys = self.keys()
        return {k: self.get(k) for k in keys}

    def keys(self) -> list[str]:
        """List of names for named items"""
        return []

    def get(self, key: str) -> Item:
        """Gives named item"""
        raise NotImplementedError

    @property
    def bbox(self) -> tuple[slice, slice]:
        return slice(None), slice(None)
