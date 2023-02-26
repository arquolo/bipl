from __future__ import annotations

__all__ = [
    'detach_', 'device', 'dump_to_onnx', 'eval_', 'frozen', 'inference',
    'materialize', 'param_count', 'profile'
]

import functools
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from io import BytesIO
from itertools import islice
from typing import Any, TypeVar, cast

import numpy as np
import torch
from torch import nn

from .. import si
from .modules.lazy import _materialize_cls

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable[..., Iterator])


def _apply(xs: _T, fn: Callable[[torch.Tensor], Any]) -> _T:
    if isinstance(xs, torch.Tensor):
        return fn(xs)

    if isinstance(xs, (str, bytes, np.ndarray)):
        return xs  # type: ignore

    if isinstance(xs, tuple) and hasattr(xs, '_fields'):  # namedtuple
        return type(xs)(*(_apply(x, fn) for x in xs))  # type: ignore
    if isinstance(xs, Mapping):
        return dict(_apply(kv, fn) for kv in xs.items())  # type: ignore
    if isinstance(xs, Iterable):
        return type(xs)(_apply(x, fn) for x in xs)  # type: ignore
    return xs


def device() -> torch.device:
    """Gets current device, including CPU"""
    return torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda
                        .is_available() else 'cpu')


def param_count(module: nn.Module) -> int:
    """Count of parameters in net, both training and not"""
    params = {p for p in module.parameters() if not nn.parameter.is_lazy(p)}
    return si(sum(p.numel() for p in params))


@contextmanager
def eval_(module: nn.Module) -> Iterator[None]:
    """
    Switches all children to eval mode.
    Restores train/eval distribution at exit.
    """
    were_train = {m for m in module.modules() if m.training}
    try:
        module.eval()
        yield
    finally:
        for m in module.modules():
            if m in were_train:
                m.training = True  # Don't call .train() as it's recursive.


@contextmanager
def detach_(module: nn.Module) -> Iterator[None]:
    """Prevents module from changing its parameters.

    Forbids autograd to record operations on parameters in this module, thus
    excluding them from gradient computation.

    This method is helpful for freezing part of the module for finetuning or
    training parts of a model individually (e.g., GAN training).

    NEITHER disable gradient flow NOR prevents buffers to change.
    """
    required_grad = {
        p.detach_() for p in module.parameters()
        if not nn.parameter.is_lazy(p) and p.requires_grad
    }
    try:
        yield
    finally:
        for p in required_grad:
            p.requires_grad_()


@contextmanager
def frozen(module: nn.Module) -> Iterator[None]:
    """Blocks module from changing state of its parameters and buffers.

    Switches all children to eval mode and detaches all parameters.
    DOES NOT disable gradient flow.
    """
    with eval_(module), detach_(module):
        yield


@contextmanager
def inference(module: nn.Module) -> Iterator[None]:
    """Enables inference mode for module.

    Switches all children to eval mode.
    Disables gradient flow.

    All the tensors created in this mode are marked as inference,
    and they are NOT COMPATIBLE WITH AUTOGRAD AT ALL
    (used in JIT, backward, etc.).

    DON'T use this mode to initialize lazy modules.
    """
    with eval_(module), torch.inference_mode():
        yield


# ----------------------------- profile CUDA ops -----------------------------


def profile(fn: _F) -> _F:
    """Decorator to profile CUDA ops. Use with `nvprof`

    Use in script launched via:
    ```bash
    nvprof --profile-from-start off -o trace.prof -- python main.py
    ```
    Usage:
    >>> @profile
    ... def train_loop():
    ...     for data in loader:
    ...         yield step(data)

    """
    def wrapper(*args, **kwargs):
        results = fn(*args, **kwargs)
        with torch.cuda.profiler.profile():
            yield from islice(results, 1)
            with torch.autograd.profiler.emit_nvtx():
                yield from results

    return cast(_F, functools.update_wrapper(wrapper, fn))


def dump_to_onnx(model: nn.Module,
                 *shapes: tuple[int, ...],
                 device: str | torch.device = 'cpu') -> bytes:
    """Converts model to ONNX graph, represented as bytes

    Parameters:
    - model - torch.nn.Module to convert
    - shapes - Shapes of input data, all except batch dimension

    Example usage:
    >>> module = torch.nn.Linear(4, 4)
    >>> bytes_ = dump_to_onnx(module, [4])

    To restore graph:
    >>> from onnxruntime import backend
    >>> rep = backend.prepare(bytes_or_filename, device='cpu')
    >>> rep.run([np.zeros(4, 4)])[0]

    """
    dynamic_axes = {
        f'inp_{i}': {
            0: 'batch',
            **{dim: f'inp_{i}_dim_{dim}' for dim in range(2, 1 + len(shape))}
        } for i, shape in enumerate(shapes)
    }
    buf = BytesIO()
    torch.onnx.export(
        model.to(device).eval(),
        tuple(
            torch.rand(1, *s, requires_grad=True, device=device)
            for s in shapes),
        buf,
        input_names=[*dynamic_axes],
        dynamic_axes=dynamic_axes,
        opset_version=13,
        do_constant_folding=True)
    return buf.getvalue()


# --------------------------- fix for lazy module ----------------------------


def materialize(model: nn.Module, *args, **kwargs):
    """
    Materialize all the lazy modules within model.
    Safely call forward() if args or kwargs are passed.
    """
    lazy = {
        name: m for name, m in model.named_modules()
        if isinstance(m, nn.modules.lazy.LazyModuleMixin)
    }
    if not lazy:
        return

    uninitialized = {
        name: m for name, m in lazy.items()
        if m.has_uninitialized_params()  # type: ignore
    }
    if not uninitialized:  # Complete initialization without forward() call
        for m in lazy.values():
            _materialize_cls(m)  # type: ignore
        return

    if args or kwargs:  # Initialize from forward() call
        with eval_(model), torch.no_grad():
            model(*args, **kwargs)
        return

    raise RuntimeError(
        'Found uninitialized lazy modules but no example inputs were passed '
        'to initialize them:\n'
        f'{[*uninitialized]}')
