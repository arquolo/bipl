import os

import numpy as np

from glow import buffered


class Job:
    size = 20

    def __iter__(self):
        for i in range(self.size):
            m = np.random.rand(2048, 2048)
            (m @ m).sum()
            print(f'child[{os.getpid():5d}]: {i}')
            # if i == len(self) // 2:
            #     raise ValueError
            yield i

    def __len__(self):
        return self.size


if __name__ == '__main__':
    g = Job()
    # for i in buffered(g, mp=True):
    for i in buffered(g):
        if i == len(g) // 2:
            raise ValueError
        print(f'main [{os.getpid():5d}]: {i}')
