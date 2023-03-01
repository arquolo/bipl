# BIPL is a Big Image Python Library

Library to read big pyramidal images like in formats like BigTiff, Aperio SVS, Leica MRXS.

## `bipl.Slide` - ndarray-like reader for multiscale images (svs, tiff, etc...)
<details>

```python
import numpy as np
from bipl import Slide

slide = Slide.open('test.svs')
shape: tuple[int, ...] = slide.shape
pools: tuple[int, ...] = slide.pools
spacing: float = slide.spacing  # X um per pixel
image: np.ndarray = slide[:2048, :2048]  # Get numpy.ndarray of 2048x2048 from 1:1 level

mini = slide.pool(4)  # 1:4 scale, shape is 4x smaller then full resolution
image: np.ndarray = mini[:512, :512]  # Same view as `image`, but lower resolution
```
</details>

## `bipl.Mosaic` - apply function for each tile of big image on desired scale.
<details>

```python
import numpy as np
from bipl import Mosaic, Slide

m = Mosaic(step=512, overlap=0)  # Read at [0:512], [512:1024], ...

# Open slide at 1:1 scale
s = Slide.open('test.svs')

# Get view at 1:4 scale of slide. `s4.shape` = `s.shape` / 4.
# If `test.svs` has some pyramid in it (i.e. 1:1, 1:4, 1:16), it will be used to speed up reads.
s4 = s.pool(4)

# Get iterator over tiles.
# Reads will be at [0:512], [512:1024] ... @ 1:4 scale
# or [0:2048], [2048:4096], ... @ 1:1, each downscaled to 512px
tiles = m.iterate(s4)

# Read only subset of tiles according to binary mask (1s are read, 0s are not).
# `mask.shape` * `scale` = `s4.shape`, `scale` >= 1
tiles = tiles.select(mask, scale)

# Read all data, trigger I/O. All the previous calls do not trigger any disk reads beyond metadata.
images: list[np.ndarray] = [*tiles]
```
</details>

## Installation

```bash
pip install bipl
```
bipl is compatible with: Python 3.9+.
Tested on ArchLinux, Ubuntu 18.04/20.04/22.04, Windows 10/11.
