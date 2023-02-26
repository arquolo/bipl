from __future__ import annotations

__all__ = ['get_grads']

import warnings
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager

import torch
from torch import nn, optim

try:
    from .driver import get_gpu_capability
except ImportError:
    get_gpu_capability = None  # type: ignore[assignment]

_MIN_SCALE = 2.0 ** -16
_MAX_SCALE = 2.0 ** +16
_PATIENCE = 2000

_MIN_FP16_CUDA_CAPABILITY = (7, 0)


def _fp16_is_definitely_not_available() -> bool:
    """
    Checks if all GPUs are not CUDA 7.x+ capable (to enable FP16).
    Doesn't trigger CUDA init.

    Possible results:
    - CUDA capability <7.0 -> FP32 mode
    - CUDA capability >=7.0,<8.0 -> FP32 mode, FP16 with scaler
    - CUDA capability >=8.0 -> FP32/TF32/BF16 as is, FP16 with scaler
    """
    if not torch.cuda.is_available():
        # No CUDA, so definitely no FP16
        return True

    if not torch.cuda.is_initialized():
        # Using Torch to check CUDA capability here triggers CUDA init,
        # (what is heavy if done from Torch side due to lots of kernels)
        # but we have no NVML alternative,
        # so consider that *maybe* some of GPUs can FP16
        if get_gpu_capability is None:
            return False

        # Rely on NVML driver
        caps = get_gpu_capability()

    else:
        # CUDA is initialized already, no need to be careful
        device_ids = range(torch.cuda.device_count())
        caps = [torch.cuda.get_device_capability(dev) for dev in device_ids]

    return all(cap < _MIN_FP16_CUDA_CAPABILITY for cap in caps)


# ------------------------- basic context, no scaler -------------------------


class Grads:
    _num_backwards: int = 0

    def __init__(self,
                 opt: optim.Optimizer,
                 sched: optim.lr_scheduler._LRScheduler | None = None):
        self._opt = opt
        self._sched = sched

    def zero_grad(self) -> None:
        self._opt.zero_grad(set_to_none=True)

    def backward(self, tensor: torch.Tensor):
        tensor.backward()
        self._num_backwards += 1

    def unscale(self) -> None:
        return

    def step(self) -> None:
        if self._num_backwards:
            self._opt.step()
            if self._sched:
                self._sched.step()
        self._num_backwards = 0

    def __enter__(self) -> None:
        self.zero_grad()

    def __exit__(self, type_, *_) -> None:
        if type_ is None:
            self.step()

    def state_dict(self) -> dict:
        s = {
            'num_backwards': self._num_backwards,
            'optimizer': self._opt.state_dict(),
        }
        if self._sched is not None:
            s['scheduler'] = self._sched.state_dict()
        return s

    def load_state_dict(self, state: dict):
        self._num_backwards = state['num_backwards']
        self._opt.load_state_dict(state['optimizer'])
        if self._sched is not None:
            self._sched.load_state_dict(state['scheduler'])


# --------------------------- default grad scaler ----------------------------
_PRIVATE = True
_NAN = float('nan')  # nan != nan


class _ScalingGrads(Grads):
    """
    Grads around torch's GradScaler.

    - Accumulates scaled grads in .grad fields
    - Unscaling and scale update done once per optim.step
    - If any grad fraction has inf/nan, then optim.step is skipped,
      thus making whole sequence of N fwd-bwd calls (N = grad steps)
      gone to nowhere.
    """
    def __init__(self,
                 opt: optim.Optimizer,
                 sched: optim.lr_scheduler._LRScheduler | None = None,
                 scale: float = _MAX_SCALE):
        super().__init__(opt, sched)
        self._scaler = torch.cuda.amp.grad_scaler.GradScaler(scale)

        if _PRIVATE:
            self._steps = getattr(self._opt, '_step_count', _NAN)
        else:
            self._scale = self._scaler.get_scale()

    def backward(self, tensor: torch.Tensor) -> None:
        self._scaler.scale(tensor).backward()

    def unscale_(self) -> None:
        self._scaler.unscale_(self._opt)

    def step(self) -> None:
        self._scaler.step(self._opt)
        self._scaler.update()

        if not self._sched:
            return

        if _PRIVATE:
            steps, self._steps = self._steps, getattr(self._opt, '_step_count',
                                                      _NAN)
            if steps == self._steps:  # No step done
                return
        else:
            # ! CPU-GPU sync
            old, self._scale = self._scale, self._scaler.get_scale()
            if old > self._scale:  # Scale was decreased due to overflow
                return

        self._sched.step()

    def state_dict(self) -> dict:
        return super().state_dict() | {'scaler': self._scaler.state_dict()}

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)
        self._scaler.load_state_dict(state['scaler'])


# ------------------------- grad scaler with retries -------------------------


@contextmanager
@torch.no_grad()
def accumulating(params: Iterable[torch.Tensor], done: int):
    def _borrow(p: torch.Tensor) -> torch.Tensor | None:
        # Save old state
        grad, p.grad = p.grad, None
        return None if grad is None else grad.detach_()

    def _update(p: torch.Tensor, grad: torch.Tensor | None):
        # In case the fail happened, or new grad is missing, use old grad
        if inf[0] or p.grad is None:
            p.grad = grad

        # If both grads exist, do running mean update
        elif grad is not None:
            # mu' = (new + mu * n) / (1 + n)
            #     = lerp(new, mu, n / (n + 1))
            p.grad.lerp_(grad, done / (1 + done))

    stash = {p: _borrow(p) for p in params}
    inf = [False]
    try:
        with torch.enable_grad():
            yield inf
    finally:
        for p, grad in stash.items():
            _update(p, grad)


class _GenericScalingGrads(Grads):
    """
    Grads around custom GradScaler (inlined and optimized).

    Differences from torch.GradScaler:

    - Stores scaled grads from single bwd in .grad fields
    - Unscales grads and updates scale
    - If any .grad got inf/nan, does backward() again to get correct grads,
      (configurable), or drops bad grads.
    - Mixes new grads with the grads from the previous calls
    - After N such fwd-bwd calls does step if all grads in sequence were ok.

    Thus, this scaler always tries to re-compute failed backward(),
    reducing chance of skipping optimizer step.
    But does gradient unscaling more frequiently, and maybe this can add some
    CPU-GPU sync penalty.

    This scaler takes care of consecutive bwd calls, averaging gradients
    over time.
    """
    def __init__(self,
                 opt: optim.Optimizer,
                 sched: optim.lr_scheduler._LRScheduler | None = None,
                 max_retries: int = 1,
                 scale: float = _MAX_SCALE,
                 min_scale: float | None = _MIN_SCALE):
        self._max_retries = max_retries
        self._min_scale = min_scale

        self._growth_tracker = torch.zeros(1).int()
        self._scale = torch.empty(1).fill_(scale)

        self._params: set[nn.Parameter] = {
            p for param_group in opt.param_groups
            for p in param_group['params'] if p.requires_grad
        }
        super().__init__(opt, sched)

    def _unscale_grads_(self) -> tuple[torch.Tensor, ...]:
        devs = set[torch.device]()
        grad_groups: dict[tuple[torch.device, torch.dtype],
                          list[torch.Tensor]] = defaultdict(list)
        for p in self._params:
            if p.grad is None:
                continue
            if p.grad.dtype == torch.float16:
                raise ValueError('Attempting to unscale FP16 gradients.')
            devs.add(p.device)
            grad_groups[p.device, p.dtype].append(p.grad)

        # Compute in FP64 for more precision
        inv_scale = self._scale.double().reciprocal().float()

        # Inf flags
        infs = {dev: torch.zeros(1, device=dev) for dev in devs}

        # Unscale grads and update infs
        for (dev, _), grads in grad_groups.items():
            torch._amp_foreach_non_finite_check_and_unscale_(
                grads, infs[dev], inv_scale.to(dev, non_blocking=True))
        return *infs.values(),

    def _update_(self, infs: Iterable[torch.Tensor]) -> None:
        # Collect infs from all devices
        found_inf, *rest = (
            inf.to(self._scale.device, non_blocking=True) for inf in infs)
        for inf in rest:
            found_inf += inf

        torch._amp_update_scale_(self._scale, self._growth_tracker, found_inf,
                                 2.0, 0.5, _PATIENCE)

    def backward(self, tensor: torch.Tensor) -> None:
        self._growth_tracker = self._growth_tracker.to(
            tensor.device, non_blocking=True)
        self._scale = self._scale.to(tensor.device, non_blocking=True)

        # Borrow grads, and zero source
        with accumulating(self._params, self._num_backwards) as inf:
            for remaining in reversed(range(1 + self._max_retries)):
                # Collect grads from backward
                (tensor * self._scale).backward(retain_graph=bool(remaining))

                # Check for inf/nan
                infs = self._unscale_grads_()
                self._update_(infs)
                inf[0] = any(v.item() for v in infs)  # ! cpu-gpu sync

                if not inf[0]:  # Success
                    self._num_backwards += 1
                    return

                # Second round (or complete failure), zero grads to fill again
                self.zero_grad()
                print('Got inf in grads,',
                      'try another scale' if remaining else 'skip step')

        # ! Optional cpu-gpu sync
        if self._min_scale is None or self._scale.item() > self._min_scale:
            return
        raise OverflowError(f'Scale underflow {self._min_scale}')

    def state_dict(self) -> dict:
        return super().state_dict() | {
            'max_retries': self._max_retries,
            'min_scale': self._min_scale,
            'scale': self._scale.item(),
            'growth_tracker': self._growth_tracker.item()
        }

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)
        self._max_retries = state['max_retries']
        self._min_scale = state['min_scale']
        self._growth_tracker.fill_(state['growth_tracker'])
        self._scale.fill_(state['scale'])
        self._params = {
            p for param_group in self._opt.param_groups
            for p in param_group['params'] if p.requires_grad
        }


def get_grads(opt: optim.Optimizer,
              sched: optim.lr_scheduler._LRScheduler | None = None,
              dtype: torch.dtype | None = None,
              max_retries: int = 1) -> Grads:
    """Get gradients context with specified precision

    Parameters:
    - dtype - dtype used for computation. If float16 used, uses GradScaler
      to prevent nan/inf during backward passes.
      By default determines whether to enable fp16
      depending on CUDA capability of available CUDA devices.
    - max_retries - number of additional attempts to use another scale
      for the same loss when backward() results to NaN/Inf. Only for fp16 mode.

    Example:
    ```
    # Creates model and optimizer in default precision
    model = Net()
    optimizer = optim.SGD(model.parameters(), ...)

    # Enable amp
    grads = gnn.get_grads(optimizer, dtype=torch.half)
    autocast = torch.autocast(device.type, torch.half)

    with grads:
        for input, target in batches:
            with autocast:
                output = model(input)
                loss = loss_fn(output, target)

            grads.backward(loss)

        grads.unscale_()  # No-op for some Grads subclasses
        torch.nn.utils.clip_grad_norm_(opt.parameters(), MAX_NORM)
    ```
    Notes:
    - never call model.half() as autocast doesn't need FP16 weights/buffers
      unless inference mode is needed.
    - for inference mode, do weight-norm fusion before using FP16
    - by default torch (1.6-1.12) doesn't promote batchnorm/instancenorm to
      FP32 (unlike layernorm/groupnorm), thus it may cause stability issues.
      Unless you'll find a way to register batchnorm as op promoting to FP32
      in autocast mode.
      https://pytorch.org/docs/stable/amp.html#ops-that-can-autocast-to-float32
    """
    if _fp16_is_definitely_not_available():  # None can FP16, disable it anyway
        if dtype != torch.float:
            warnings.warn('Neither of active devices support FP16, disable it')
        dtype = torch.float
    elif dtype is None:  # Some GPU's can do FP16, automatic choice, enable
        dtype = torch.half

    if dtype != torch.half:
        return Grads(opt, sched)
    if not max_retries:
        return _ScalingGrads(opt, sched)
    return _GenericScalingGrads(opt, sched, max_retries)
