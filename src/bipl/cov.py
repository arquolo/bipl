"""
Start coverage server via:
    python -m bipl.cov --roodir /slides/anns/roi --host 0.0.0.0 --port 7575

In client set environment variable
    BIPL_COV_URL=http://{server_host}:7575
"""

__all__ = ['update']

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import geojson
import uvicorn
from fastapi import FastAPI
from glow.cli import parse_args
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


@dataclass
class _Coverage:
    rootdir: Path
    polygons: dict[str, list[list]] = field(default_factory=dict)
    atime: datetime = field(default_factory=datetime.now)
    ndiffs: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def flush(self) -> None:
        for fname, pgs in self.polygons.items():
            path = self.rootdir / f'{fname}.geojson'
            if path.is_file():
                fc: geojson.FeatureCollection = geojson.loads(path.read_text())
                fc[0]['geometry'] += pgs
            else:
                fc = geojson.FeatureCollection(
                    [
                        geojson.Feature(
                            geometry=geojson.MultiPolygon(pgs),
                            properties={'class_name': 'RoI'},
                        )
                    ]
                )
            path.write_text(geojson.dumps(fc))

        self.polygons.clear()

    async def update(self, u: Update) -> None:
        fname = Path(u.path).name
        pg = [[[u.x0, u.y0], [u.x0, u.y1], [u.x1, u.y1], [u.x1, u.y0]]]

        async with self.lock:
            self.polygons.setdefault(fname, []).append(pg)
            self.ndiffs += 1

            # Flush if last flush was long time ago, or lots of changes exist
            now = datetime.now()
            if (self.atime - now).total_seconds() > 60 or self.ndiffs > 1000:
                await self.flush()
                self.atime = now
                self.ndiffs = 0


def _main(rootdir: Path, host: str, port: int = 7575) -> None:
    cov = _Coverage(rootdir=rootdir)
    app = FastAPI()
    app.add_api_route('/update', cov.update, methods=['PUT'], status_code=204)
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    parse_args(_main)
