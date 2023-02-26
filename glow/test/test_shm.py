import time
import weakref
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from loky import get_reusable_executor

# import sys
# from loky.backend import resource_tracker
# sys.modules['multiprocessing.resource_tracker'] = resource_tracker


class SharedArray:
    def __init__(self, arr):
        self.shm = SharedMemory(create=True, size=arr.nbytes)
        self.shm.buf[:] = arr.view(np.uint8)
        self.shape = arr.shape
        self.dtype = arr.dtype
        weakref.finalize(self, self.shm.unlink)

    def rebuild(self):
        return np.asarray(self.shm.buf).view(self.dtype).reshape(self.shape)

    def mean(self):
        time.sleep(2)
        return self.rebuild().mean()


if __name__ == '__main__':
    # executor = get_reusable_executor(max_workers=1, context='loky_init_main')
    executor = get_reusable_executor(max_workers=1)

    # Create 1GB shared array
    arr = SharedArray(np.random.rand(int(1.25e8)))
    mean = arr.mean()

    f = executor.submit(arr.mean)
    assert f.result() == mean
