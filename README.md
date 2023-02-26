# BIPL is a Big Image Python Library

Library to read big pyramidal images like in formats like BigTiff, Aperio SVS, Leica MRXS.

## `bipl.Slide` - ndarray-like reader for multiscale images (svs, tiff, etc...)
<details>

```python
from bipl import Slide

slide = Slide.open('test.svs')
shape: tuple[int, ...] = slide.shape
scales: tuple[int, ...] = slide.scales
image: np.ndarray = slide[:2048, :2048]  # Get numpy.ndarray
```
</details>

## `bipl.Mosaic` - apply function for each tile of big image on desired scale.
<details>
...
</details>

## Installation

```bash
pip install bipl
```
bipl is compatible with: Python 3.9+.
Tested on ArchLinux, Ubuntu 18.04/20.04/22.04, Windows 10/11.
