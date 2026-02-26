__all__ = ['app', 'env', 'router']

from functools import lru_cache
from os.path import normpath
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import unquote

import cv2
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import Field
from pydantic_settings import BaseSettings

from .io import Dzi, Slide


class _Env(BaseSettings):
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'

    # Deepzoom
    slides: Path = Path()
    tile_size: int = 256
    quality: int = 75
    max_slides: int = 10

    # JWT
    token_algo: str = 'HS256'
    token_secret: Annotated[
        str | None, Field(min_length=64, max_length=64)
    ] = None
    token_url: str = '/api/users/authentificate'

    mpp_default: float | None = None  # 0.5, MPP for slides without it
    mpp_max: float = float('inf')  # 100, MPP to name slide broken

    def model_post_init(self, context) -> None:
        self.slides = self.slides.resolve().absolute()


env = _Env()


# --------------------------- endpoint definitions ---------------------------

router = APIRouter()
dzi = Dzi(tile_size=env.tile_size, quality=env.quality, fmt='jpeg')


@lru_cache(env.max_slides)
def _get_slide(fname: str) -> Slide:
    fname = unquote(fname)

    # Ensure path is local to the "root"
    root = env.slides
    path = Path(normpath(root / fname))  # Resolve "..", don't touch symlinks
    if root not in path.parents:
        raise HTTPException(403, 'Directory traversal is forbidden')

    if not path.is_file():
        raise HTTPException(404, 'File not found')

    try:
        s = Slide.open(path)
    except ValueError as e:
        raise HTTPException(415, f'File cannot be opened. Error: {e}') from e

    if env.mpp_default is not None and (s.mpp is None or s.mpp > env.mpp_max):
        object.__setattr__(s, 'mpp', env.mpp_default)
    return s


# GET /preview/slide-1.svs?size=500
# TODO: strip .jpeg suffix
@router.get('/preview/{fname:path}.jpeg')
def get_thumbnail(
    s: Annotated[Slide, Depends(_get_slide)],
    size: int | None = None,
    orient: str | None = None,
) -> Response:
    image = s.thumbnail()

    if size is not None:
        image = _fit_to(image, size)
    if orient is not None:
        h, w, _ = image.shape
        if (orient == 'portrait' and w > h) or (orient == 'album' and w < h):
            image = image.transpose(1, 0, 2)

    return Response(dzi.compress(image), media_type='image/jpeg')


# GET /extras/label/slide-1.svs?size=500
@router.get('/extras/{tag}/{fname:path}')
def get_extra_image(
    tag: str,
    s: Annotated[Slide, Depends(_get_slide)],
    size: int | None = None,
) -> Response:
    image = s.extra(tag)
    if image is None:
        raise HTTPException(404, f'Image does not provide extra {tag!r}')

    if size is not None:
        image = _fit_to(image, size)

    return Response(dzi.compress(image), media_type='image/jpeg')


@router.get('/dzi/{fname:path}.dzi')  # /dzi/slide-1.svs -> JSON
def get_dzi_header(s: Annotated[Slide, Depends(_get_slide)]) -> dict:
    return dzi.head(s)


# GET /dzi/slide-1.svs_files/0/1_2.jpeg
@router.get('/dzi/{fname:path}_files/{level}/{col}_{row}.jpeg')
def get_image_tile(
    s: Annotated[Slide, Depends(_get_slide)],
    *,
    level: int,
    col: int,
    row: int,
) -> Response:
    tile = dzi.tile_at(s, level, (row, col))
    return Response(dzi.compress(tile), media_type='image/jpeg')


# -------------------------------- utilities ---------------------------------


def _keep3d(im: np.ndarray) -> np.ndarray:
    return im[:, :, None] if im.ndim == 2 else im


def _fit_to(image: np.ndarray, size: int) -> np.ndarray:
    shape = image.shape[:2]
    largest_size = max(shape)
    h, w = (round(s * size / largest_size) for s in shape)
    return _keep3d(cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA))


# -------------------------------- create app --------------------------------

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
if env.token_secret:
    oauth2_token_bearer = OAuth2PasswordBearer(env.token_url)

    async def _verify_token(
        token: Annotated[str, Depends(oauth2_token_bearer)],
    ) -> None:
        assert env.token_secret is not None
        try:
            jwt.decode(token, env.token_secret, [env.token_algo])
        except JWTError as exc:
            raise HTTPException(
                401,  # Unauthorized
                'Could not validate credentials',
                {'WWW-Authenticate': 'Bearer'},
            ) from exc

    app.router.dependencies.append(Depends(_verify_token))

app.include_router(router)
