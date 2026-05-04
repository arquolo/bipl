__all__ = ['convert', 'list_colorspaces']

# TODO: torch/numpy kernels
# TODO: separate package ("xcolors" ?)
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, Self

import cv2
import numpy as np
import numpy.typing as npt
from glow import timer
from scipy.sparse import coo_array
from scipy.sparse.csgraph import shortest_path

from bipl._dev import prof

_F32 = npt.NDArray[np.float32]
_Call = Callable[[_F32], _F32]
_log = 1


class _Op(Protocol):
    def __call__(self, x: _F32) -> _F32: ...

    def __invert__(self) -> Self: ...

    def children(self) -> list[str]:
        return [type(self).__name__.removeprefix('_')]


# op[to, from]
_table: dict[tuple[str, str], _Op] = {}

# --------------------------------- utility ----------------------------------


@dataclass(frozen=True, slots=True)
class _Func(_Op):
    fwd: _Call
    bwd: _Call

    def __call__(self, x: _F32) -> _F32:
        if _log:
            print(f' {type(self).__name__}({self.fwd})')
        return self.fwd(x)

    def __invert__(self) -> '_Func':
        return _Func(self.bwd, self.fwd)


@dataclass(frozen=True, slots=True)
class _Pow(_Op):
    exp: float

    @prof
    def __call__(self, x: _F32) -> _F32:
        if _log:
            print(f' {type(self).__name__}')
        x.clip(min=0, out=x)
        cv2.pow(x, self.exp, dst=x)
        return x

    def __invert__(self) -> '_Pow':
        return _Pow(1 / self.exp)


_cv2map = {getattr(cv2, k): k for k in dir(cv2) if k.startswith('COLOR_')}


@dataclass(frozen=True, slots=True)
class _CvtColor(_Op):
    fwd: int
    bwd: int

    def __call__(self, x: _F32) -> _F32:
        if _log:
            print(f' {type(self).__name__}({_cv2map[self.fwd]})')
        r = cv2.cvtColor(x.reshape(1, -1, 3), self.fwd)
        return r.reshape(x.shape)  # type: ignore

    def __invert__(self) -> '_CvtColor':
        return _CvtColor(self.bwd, self.fwd)


@dataclass(frozen=True, slots=True)
class _Dot(_Op):
    m: _F32

    @classmethod
    def new(
        cls,
        m: np.ndarray | list[list[float]] | list[float] | float,
    ) -> '_Dot':
        m = np.asarray(m, 'f')
        if m.ndim == 2:
            # Strip pad -> (3 4) or (3 3)
            m = m[:3]
            # Strip bias -> (3 4) or (3 3)
            if not m[:, 3:].any():
                m = m[:, :3]
            # Drop off-diagonal -> (3 4), (3 3) or (3)
            diag = m.diagonal()
            if np.count_nonzero(diag) == np.count_nonzero(m):
                m = diag

        return _Dot(m) if m.ndim == 2 else _Mul(m)

    def __array__(self) -> _F32:
        r = np.eye(4, dtype='f')
        r[:3, : self.m.shape[1]] = self.m
        return r

    def __add__(self, rhs: '_Dot') -> '_Dot':
        # (b a) (c b)
        return _Dot.new(np.matmul(rhs, self))

    @prof
    def __call__(self, x: _F32) -> _F32:
        # (* c1) (c2 c1+1) -> (* c2)
        if _log:
            print(f' {type(self).__name__}{np.shape(self.m)}')
        c = x.shape[-1]
        r = np.empty(x.shape, x.dtype)
        cv2.transform(x.reshape(-1, 1, c), self.m, dst=r.reshape(-1, 1, c))
        return r

    def __invert__(self) -> '_Dot':
        return _Dot.new(np.linalg.inv(self))


@dataclass(frozen=True, slots=True)
class _Mul(_Dot):
    m: _F32 | float

    def __array__(self) -> np.ndarray:
        r = np.eye(4, dtype='f')
        np.fill_diagonal(r[:3, :3], self.m)
        return r

    @prof
    def __call__(self, x: _F32) -> _F32:
        # (* c1) () -> (* c1)
        # (* c1) (c1) -> (* c1)
        if _log:
            print(f' {type(self).__name__}{np.shape(self.m)}')
        return x * self.m

    def __invert__(self) -> '_Mul':
        return _Mul(1 / self.m)


class _Chain(list[_Op], _Op):
    def __init__(self, ops: Iterable[_Op]):
        super().__init__()
        ops = [  # flatten nested
            op
            for op_ in ops
            for op in (op_ if isinstance(op_, _Chain) else [op_])
        ]
        for op in ops:
            # collapse consecutive matrices
            if self and isinstance(self[-1], _Dot) and isinstance(op, _Dot):
                self[-1] += op
            else:
                self.append(op)

    def __call__(self, x: _F32) -> _F32:
        for op in self:
            x = op(x)
        return x

    def __invert__(self) -> '_Chain':
        return _Chain(~op for op in reversed(self))

    def children(self) -> list[str]:
        return [type(x).__name__.removeprefix('_') for x in self]


# ------------------------- white points & primaries -------------------------


def _xy_to_xyz(xy: _F32) -> _F32:
    *volume, _ = xy.shape
    x, y = np.moveaxis(xy, -1, 0)
    m = y > 0
    r = np.empty((*volume, 3), 'f')
    np.divide(x, y, where=m, out=r[..., 0])
    r[..., 1] = 1
    np.divide(1 - x - y, y, where=m, out=r[..., 2])
    r[~m, 0] = np.nan
    r[~m, 2] = np.nan
    return r


_whites = ['C', 'D50', 'D65', 'E']
_whites_xy = np.array(
    [
        # (n xy)
        [0.31006, 0.31616],  # C, obsolete
        [0.34567, 0.35850],  # D50
        [0.31272, 0.32903],  # D65
        [0.33333, 0.33333],  # E
    ]
).astype('f')

_primaries = ['HDTV', 'Adobe RGB', 'P3', 'UHDTV', 'CIE RGB']
_primaries_xy = np.array(
    [
        # (n RGB xy)
        [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]],  # HDTV, sRGB
        [[0.64, 0.33], [0.21, 0.71], [0.15, 0.06]],  # Adobe RGB
        [[0.68, 0.32], [0.265, 0.69], [0.15, 0.06]],  # DCI/Display P3
        [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],  # UHDTV
        [
            [0.73474284, 0.26525716],
            [0.27377903, 0.7174777],
            [0.16655563, 0.00891073],
        ],  # CIE RGB
    ]
).astype('f')

_whites_xyz = _xy_to_xyz(_whites_xy)  # (n XYZ)
_primaries_xyz = _xy_to_xyz(_primaries_xy)  # (n RGB XYZ)


def _make_rgb(rgb: _F32, white: _F32) -> _F32:
    return rgb.T * (white @ np.linalg.inv(rgb))


# ----------------------------- sRGB <-> HLS/HSV -----------------------------

_table['HLS', 'sRGB'] = _CvtColor(cv2.COLOR_RGB2HLS, cv2.COLOR_HLS2RGB)
_table['HSV', 'sRGB'] = _CvtColor(cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB)

# ------------------------ sRGB <-> linear RGB @ D65 -------------------------


@prof
def _srgb_linearize(x: _F32) -> _F32:
    x.clip(0, 1, out=x)
    m = x > 0.04045

    np.multiply(x, 1 / 12.92, where=~m, out=x)

    v = x[m]
    v += 0.055
    v *= 1 / 1.055
    cv2.pow(v, 2.4, dst=v)
    x[m] = v

    return x


@prof
def _srgb_gamma(x: _F32) -> _F32:
    x.clip(0, 1, out=x)
    m = x >= 0.0031308

    np.multiply(x, 12.92, where=~m, out=x)

    v = x[m]
    cv2.pow(v, 1 / 2.4, dst=v)
    v *= 1.055
    v -= 0.055
    x[m] = v

    return x


_table['RGB', 'sRGB'] = _Func(_srgb_linearize, _srgb_gamma)

# ----------------------- linear RGB @ D65 <-> CIE XYZ -----------------------

# RGB -> XYZ = cv2.COLOR_RGB2XYZ
# XYZ -> RGB = cv2.COLOR_XYZ2RGB

_table['CIEXYZ', 'RGB'] = _Dot.new(
    [  # mat[CIE XYZ, linear sRGB @ D65]
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_table['CIEXYZ', 'CIERGB'] = _Dot.new(
    [  # mat[CIE RGB, CIE XYZ]
        [0.49000, 0.31000, 0.20000],
        [0.17697, 0.81240, 0.01063],
        [0.00000, 0.01000, 0.99000],
    ]
)

# ------------------------ CIE XYZ <-> CIE L*a*b* D65 ------------------------

_4_29 = 4 / 29
_6_29 = 6 / 29
_108_841 = 3 * (_6_29**2)
_216_24389 = _6_29**3


@prof
def _cielab_gamma(x: _F32) -> _F32:
    x.clip(0, 1, out=x)
    m = x > _216_24389
    nm = ~m

    v = x[nm]
    v *= 1 / _108_841
    v += _4_29
    x[nm] = v

    v = x[m]
    cv2.pow(v, 1 / 3, dst=v)
    x[m] = v

    return x


@prof
def _cielab_linearize(x: _F32) -> _F32:
    x.clip(0, 1, out=x)
    m = x > _6_29
    nm = ~m

    v = x[nm]
    v -= _4_29
    v *= _108_841
    x[nm] = v

    v = x[m]
    cv2.pow(v, 3, dst=v)
    x[m] = v

    return x


_table['CIELab', 'CIEXYZ'] = _Chain(
    [
        _Mul(1 / _whites_xyz[_whites.index('D65')]),
        _Func(_cielab_gamma, _cielab_linearize),
        _Dot.new(
            [
                [0, 1.16, 0, -0.16],
                [5, -5, 0, 0],
                [0, 2, -2, 0],
            ]
        ),
    ]
)
_table['CIELab', 'RGB'] = _Chain(
    [
        _CvtColor(cv2.COLOR_LRGB2Lab, cv2.COLOR_Lab2LRGB),
        _Mul(0.01),  # cv2.LAB: 0..100, ours: 0..1
    ]
)
_table['CIELab', 'sRGB'] = _Chain(
    [
        _CvtColor(cv2.COLOR_RGB2Lab, cv2.COLOR_Lab2RGB),
        _Mul(0.01),  # cv2.LAB: 0..100, ours: 0..1
    ]
)

# ---------------------------------- Oklab -----------------------------------

_table['Oklab', 'CIEXYZ'] = _Chain(
    [
        _Dot.new(
            [
                [+0.8189330101, +0.3618667424, -0.1288597137],
                [+0.0329845436, +0.9293118715, +0.0361456387],
                [+0.0482003018, +0.2643662691, +0.6338517070],
            ]
        ),
        _Pow(1 / 3),
        _Dot.new(
            [
                [+0.2104542553, +0.7936177850, -0.0040720468],
                [+1.9779984951, -2.4285922050, +0.4505937099],
                [+0.0259040371, +0.7827717662, -0.8086757660],
            ]
        ),
    ]
)

# -------------------------------- L* <-> LCh --------------------------------


def _cube(cylinder: _F32) -> _F32:
    h, mag, angle = cylinder.reshape(1, -1, 3).T
    x, y = cv2.polarToCart(mag, angle)
    return np.stack((h, x, y), -1).reshape(cylinder.shape)


def _cylinder(cube: _F32) -> _F32:
    h, x, y = cube.reshape(1, -1, 3).T
    mag, angle = cv2.cartToPolar(x, y)
    return np.stack((h, mag, angle), -1).reshape(cube.shape)


_table['CIELChab', 'CIELab'] = _Func(_cylinder, _cube)

# ----------------------------------------------------------------------------


def _score_links(table: Mapping[tuple[str, str], _Op]) -> _F32:
    cmp = {
        'Dot': 1,
        'CvtColor': 3,
        'Pow': 3,
        'Mul': 9,
        'Func': 20,
    }
    if 1:
        weights = [1] * len(table)
        weights = [len(op.children()) for op in table.values()]
        weights = [sum(cmp[c] for c in op.children()) for op in table.values()]
        weights = np.divide(weights, min(weights))

    else:
        global _log
        _log = False

        rg = np.random.default_rng(1234)
        tmp = rg.random((65536, 3), dtype='f')

        weights = []  # (t)
        mat = []
        for op in table.values():
            tmp_ = tmp.copy()
            with timer(weights.append):
                for _ in range(100):
                    op(tmp_)
            mat.append(op.children())
        weights = np.divide(weights, min(weights))

        # Approx part weights
        uc = sorted({c for cc in mat for c in cc})
        w = np.zeros((len(table), len(uc)), int)  # (t c)
        for cc, w_ in zip(mat, w):
            for c in cc:
                w_[uc.index(c)] += 1
        # x = np.linalg.pinv(w) @ weights
        # print(dict(zip(uc, x / x.min(), strict=True)))
        # print(np.abs(w @ x - weights))
        # print(np.abs(w @ x - weights).max())

        _log = True

    return weights.round(2)


def _add_inverses(
    table: Mapping[tuple[str, str], _Op],
) -> dict[tuple[str, str], _Op]:
    table = {**table} | {(k1, k0): ~op for (k0, k1), op in table.items()}
    return dict(sorted(table.items()))


@prof
def _build_routes(
    table: Mapping[tuple[str, str], _Op],
) -> dict[str, dict[str, tuple[_Op, str]]]:
    # Find shortest paths
    keys = sorted({k for ks in table for k in ks})
    weights = _score_links(table)

    # from pprint import pprint
    # pprint({
    #     (w, k0, k1): op.children()
    #     for ((k1, k0), op), w in zip(table.items(), weights)
    # })
    digraph = coo_array(
        (
            # np.broadcast_to(1, len(table)),
            weights,
            tuple([keys.index(k) for k in ks] for ks in zip(*table)),
        ),
        (len(keys), len(keys)),
    )
    _, prevs = shortest_path(digraph, return_predecessors=True)

    # {dst -> {src -> (op, next_src)}}
    paths: dict[str, dict[str, tuple[_Op, str]]] = {}
    for end, next_ in zip(keys, prevs):
        sub: dict[str, tuple[_Op, str]] = {}

        for start, i in zip(keys, next_):
            if i < 0:
                continue
            k = keys[i]
            sub[start] = (table[k, start], k)

        paths[end] = sub

    return dict(sorted(paths.items()))


_table = _add_inverses(_table)
_paths = _build_routes(_table)


def convert(x: _F32, dst: str, src: str = 'sRGB') -> _F32:
    """Convert image from `src` colorspace to `dst` colorspace"""
    subpaths = _paths[dst]
    ops = []
    while src != dst:
        op, src = subpaths[src]
        ops.append(op)
    if len(ops) == 1:
        return ops[0](x)
    return _Chain(ops)(x)


def list_colorspaces() -> list[str]:
    return sorted({k for ks in _table for k in ks})
