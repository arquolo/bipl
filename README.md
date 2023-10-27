# BIPL is a Big Image Python Library

Library to read big pyramidal images like in formats like BigTiff, Aperio SVS, Leica MRXS.

## `bipl.Slide` - ndarray-like reader for multiscale images (svs, tiff, etc...)
<details>

```python
import numpy as np
from bipl import Slide

slide = Slide.open('test.svs')
shape: tuple[int, ...] = slide.shape  # Native shape
pools: tuple[int, ...] = slide.pools  # List of pre-existing sub-resolution levels

# Get native miniature
tmb: np.ndarray = slide.thumbnail()

mpp: float = slide.mpp  # X um per pixel, native resolution
image: np.ndarray = slide[:2048, :2048]  # Get numpy.ndarray of 2048x2048 from full resolution

MPP = 16.  # Let's say we want slide at 16 um/px resolution
downsample = MPP / slide.mpp
mini = slide.pool(downsample)  # Gives `downsample`-times smaller image
mini = slide.resample(MPP)  # Gives the same result

# Those ones trigger ndarray conversion
image: np.ndarray
image = mini[:512, :512]  # Take a crop of
image = mini.numpy()  # Take a whole resolution level
image = np.array(mini, copy=False)  # Use __array__ API
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

# Target at 4 um/px resolution
# If `test.svs` has some pyramid in it (i.e. 1:1, 1:4, 1:16), it will be used to speed up reads.
s4 = s.resample(mpp=4.0)

# Get iterator over tiles.
# Reads will be at [0:512], [512:1024] ... @ MPP
tiles = m.iterate(s4)

# Read only subset of tiles according to binary mask (1s are read, 0s are not).
# `s4.shape * scale = mask.shape`, `scale <= 1`
tiles = tiles.select(mask, scale)

# Read all data, trigger I/O. All the previous calls do not trigger any disk reads beyond metadata.
images: list[np.ndarray] = [*tiles]
```
</details>

## Installation

```bash
pip install bipl
```
bipl is compatible with: Python 3.10+.
Tested on ArchLinux, Ubuntu 20.04/22.04, Windows 10/11.
