from __future__ import annotations

__all__ = ['Sound']

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from .. import chunked

_Scalar = TypeVar('_Scalar', bound=np.generic, covariant=True)


def _play(arr: np.ndarray,
          rate: int,
          blocksize: int = 1024,
          bufsize: int = 20):
    """Plays audio from array. Supports interruption via Crtl-C."""
    import sounddevice as sd

    q: Queue[np.ndarray | None] = Queue(bufsize)
    ev = Event()

    def callback(out: np.ndarray, *_) -> None:
        if (data := q.get()) is None:
            raise sd.CallbackAbort

        size = len(data)
        out[:size] = data
        if size < len(out):
            out[size:] = 0
            raise sd.CallbackStop

    stream = sd.OutputStream(
        rate, blocksize, callback=callback, finished_callback=ev.set)

    fmt = '{percentage:3.0f}% |{bar}| [{elapsed}<{remaining}]'
    blocks = chunked(arr, blocksize)

    with ExitStack() as s:
        s.enter_context(stream)  # Close stream
        s.callback(ev.wait)  # Wait for completion
        s.callback(q.put, None)  # Close queue

        for data in s.enter_context(
                tqdm(blocks, leave=False, smoothing=0, bar_format=fmt)):
            q.put(data)


@dataclass(repr=False, frozen=True)
class Sound(Generic[_Scalar]):
    """Wraps numpy.array to be playable as sound

    Parameters:
    - rate - sample rate to use for playback

    Usage:
    ```
    import numpy as np

    sound = Sound.load('test.flac')

    # Get properties
    rate: int = sound.rate
    dtype: np.dtype = sound.dtype

    # Could be played like:
    import sounddevice as sd
    sd.play(sound, sound.rate)

    # Or like this, if you need Ctrl-C support
    sound.play()

    # Extract underlying array
    raw = sound.raw

    # Same result
    raw = np.array(sound)
    ```
    """
    data: npt.NDArray[_Scalar]
    rate: int = 44_100

    def __post_init__(self):
        assert self.data.ndim == 2
        assert self.data.shape[-1] in (1, 2)
        assert self.data.dtype in ('i1', 'i2', 'i4', 'f4')

    @property
    def channels(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> timedelta:
        return timedelta(seconds=self.data.shape[0] / self.rate)

    def __repr__(self) -> str:
        duration = self.duration
        channels = self.channels
        dtype = self.data.dtype
        return f'{type(self).__name__}({duration=!s}, {channels=}, {dtype=!s})'

    def __array__(self) -> npt.NDArray[_Scalar]:
        return self.data

    def play(self, blocksize=1024) -> None:
        """Plays audio from array. Supports interruption via Crtl-C."""
        _play(self.data, self.rate, blocksize=blocksize)

    @classmethod
    def load(cls, path: Path | str) -> Sound:
        spath = str(path)
        assert spath.endswith('.flac')
        import soundfile

        data, rate = soundfile.read(spath)
        return cls(data.astype('f4'), rate)
