# Glow Library
Collection of tools for easier prototyping with deep learning extensions (PyTorch framework)

## Overview
...

## Installation

For basic installation use:

```bash
pip install glow
```
<details>
<summary>Specific versions with additional requirements</summary>

```bash
pip install glow[nn]  # For cv/neural network extras
pip install glow[io]  # For I/O extras
pip install glow[all]  # For all
```
</details>
Glow is compatible with: Python 3.9+, PyTorch 1.11+.
Tested on ArchLinux, Ubuntu 18.04/20.04, Windows 10/11.

## Structure
- `glow.*` - Core parts, available out the box
- `glow.cv.*` - Tools for computer vision tasks
- `glow.io.*` - I/O wrappers to access data in convenient formats
- `glow.transforms` - Some custom-made augmentations for data
- `glow.nn` - Neural nets and building blocks for them
- `glow.metrics` - Metric to use while training your neural network

## Core features
- `glow.mapped` - convenient tool to parallelize computations
- `glow.memoize` - use if you want to reduce number of calls for any function

## IO features

### `glow.io.Slide` - ndarray-like reader for multiscale images (svs, tiff, etc...)
<details>

```python
from glow.io import Slide

slide = Slide.open('test.svs')
shape: tuple[int, ...] = slide.shape
scales: tuple[int, ...] = slide.scales
image: np.ndarray = slide[:2048, :2048]  # Get numpy.ndarray
```
</details>

### `glow.io.Sound` - playable sound wrapper
<details>

```python
from datetime import timedelta

import numpy as np
from glow.io import Sound

array: np.ndarray
sound = Sound(array, rate=44100)  # Wrap np.ndarray
sound = Sound.load('test.flac')  # Load sound into memory from file

# Get properties
rate: int = sound.rate
duration: timedelta = sound.duration
dtype: np.dtype = sound.dtype

 # Plays sound through default device, supports Ctrl-C for interruption
sound.play()
```
</details>
