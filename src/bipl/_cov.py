__all__ = ['update']

from urllib.request import Request, urlopen

from pydantic import BaseModel

from bipl._env import env
from bipl._types import Shape, Vec


class Update(BaseModel):
    path: str
    y0: int
    y1: int
    x0: int
    x1: int


def update(path: str, z0_yx_offset: Vec, dsize: Shape, scale: float) -> None:
    if not env.BIPL_COV_URL:
        return
    y0, x0 = z0_yx_offset
    y1 = y0 + round(dsize[0] / scale)
    x1 = x0 + round(dsize[1] / scale)

    request = Request(
        f'{env.BIPL_COV_URL}/update',
        method='PUT',
        data=(
            Update(path=path, y0=y0, y1=y1, x0=x0, x1=x1)
            .model_dump_json()
            .encode()
        ),
        headers={'Content-Type': 'application/json'},
    )
    with urlopen(request) as response:
        response.read()
