# General

- Add docs for all exported functions

### `__init__`

- Add explicit imports from `core.*`

## `core` contents

### `{map,starmap}_n` (from `.core._parallel`)

Implement proper serialization of np.ndarray/np.memmap via anonymous mmap on Windows, and tmpfs mmap on Linux.

- `parent` -> `child` = move to shared memory if size allows
- `child` -> `parent` = keep in shared if already there, otherwise move as usual.
- Drop shared data at pool shutdown.

### `core.{wrap -> cache}`

Add case `capacity=None` for unbound cache like in `functools`.

Use `evict: _Eviction` instead of `polycy` argument.

Combine all underlying modules to single module one, or find a better split.

Decorators for any callable with hashable args and kwargs:

- `call_once` - converts function to singleton (memoization of parameter-less function).
- `memoize` - cache calls with coalencing (unite with `shared_call`)

Decorators for callables accepting sequences of hashable items `(items: Sequence[Hashable]) -> list[Result]`:

- `stream_batched` - group calls to batches
- `memoize_batched` - cache and coalence calls

Improve test coverage.

### `whereami`

- Improve function signature to show/hide stack frames from `site` modules.
  If 100% detection of foreign functions is not possible, skip only stdlib ones.

### `core._patch_len`

- `len_hint(_object: Any) -> int: ...`
- Keep signature of wrapped function
- Make `len()` patching optional
- Add wrapper for `tqdm` to use there `len_hint(...)` instead of `total=len(...)`

### `core._repr._Si`

Add proper string formatting using `format_spec`

## `io` contents

### `io._TiffImage`

- Enable fallthrough for bad tiles
- Use mmap and tile offsets from `libtiff` to decompose I/O from decoding to allow concurrent decoding.

### `io.Mosaic` (old `cv.Mosaic`)

## `nn` contents

- Store only modules and ops for them

### `nn.modules`

- Use glow.env as storage for options

## `zoo`

- Add `get_model() -> torch.nn.Module(stem=Stem(...), levels=[Level(...), ...], head=Head())`
- Add `VGG`, `ResNet`, `ResNet-D`, `ResNeXt`, `ResNeSt`, `Inception`, `DenseNet`, `EfficientNet`, `ViT`, `SWiN`
- Add `LinkNet`, `Unet`, `DeepLab`, `SkipNet`, `Tiramisu`, `MAnet`, `FPN`, `PAN`, `PSPNet`

## `optim` contents (old `nn.optimizers`)

- Subtype of `Iterable[float]` for lr policy.
- Class-adaptor for lr scheduler (batch/epoch-wise).
- Dataclass for optimizer of single parameter group.
- Class-adaptor to combine optimizers.

## `util` contents

### `util.get_loader`

- Seed as argument to toggle patching of dataset and iterable to provide batchsize- and workers-invariant data generation

### `util.plot` (from `nn.plot`)

- Fix plotting to collapse standard modules, instead of falling through into them.
- Refactor visitor to be more readable.
