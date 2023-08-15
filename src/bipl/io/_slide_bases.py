from __future__ import annotations

__all__ = ['Driver', 'Item', 'Lod', 'REGISTRY']

import re
from dataclasses import dataclass
from typing import final

import cv2
import numpy as np

from bipl.ops import normalize_loc

REGISTRY: dict[re.Pattern, list[type[Driver]]] = {}


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
        y_loc, x_loc, c_loc = normalize_loc(key, self.shape)
        if not y_loc.step == x_loc.step == 1:
            raise ValueError('Y/X slice steps should be 1 for now, '
                             f'got {y_loc.step} and {x_loc.step}')
        return self.crop((y_loc, x_loc))[:, :, c_loc]

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

    def _unpack_loc(
        self,
        slices: tuple[slice, ...],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        box = np.array([(s.start, s.stop) for s in slices])
        valid_box = box.T.clip([0, 0], self.shape[:2]).T  # (2, lo-hi)
        shape = (box[:, 1] - box[:, 0]).tolist()
        return box, valid_box, shape

    def _expand(self, rgb: np.ndarray, valid_box: np.ndarray, box: np.ndarray,
                bg_color: np.ndarray) -> np.ndarray:
        offsets = np.abs(valid_box - box)
        if offsets.any():
            tp, bm, lt, rt = offsets.ravel().tolist()
            rgb = cv2.copyMakeBorder(rgb, tp, bm, lt, rt, cv2.BORDER_CONSTANT,
                                     None, bg_color.tolist())
        return np.ascontiguousarray(rgb)


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
        # TODO: read base part-by-part, not all at once, if scale > 2(?)
        image = self.base[src_slices]

        h, w = ((s.stop - s.start) for s in slices)
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


class Driver:
    @final
    @classmethod
    def register(cls, regex: str):
        """Registers type builder for extensions. Last call takes precedence"""
        REGISTRY.setdefault(re.compile(regex), []).append(cls)

    def __init__(self, path: str) -> None:
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
