from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any

import cv2
import numpy as np
from glow import around
from lxml import etree
from lxml.etree import XMLParser, fromstring

from bipl import env

# ------------------------- flat list of properties --------------------------


def unflatten(d: Mapping[str, str]) -> dict[str, Any]:
    if not d:
        return {}

    tab = {ord('['): '.', ord(']'): None}
    pairs = sorted((_make_key(k, tab), v) for k, v in d.items())
    r: dict = {}

    for parts, v in pairs:
        rr = r
        for curr, next_ in zip_longest(parts, parts[1:], fillvalue=None):
            # Put list if next is list idx, otherwise put dict
            default: list | dict | str
            default = v if next_ is None else (
                [] if isinstance(next_, int) else {})

            if isinstance(curr, int):  # last was list idx
                if not isinstance(rr, list):  # Ambigous structure
                    break
                if curr >= len(rr):
                    rr.extend(None for _ in range(len(rr), curr + 1))
                    rr[curr] = default
                rr = rr[curr]
            else:  # last was dict key
                if not isinstance(rr, dict):  # Ambigous structure
                    break
                rr = rr.setdefault(curr, default)
    return r


def _make_key(s: str, tab: Mapping[int, str | None]) -> tuple[str | int, ...]:
    parts = s.translate(tab).split('.')
    return tuple(int(word) if word.isdigit() else word for word in parts)


# ---------------------------- aperio description ----------------------------


def is_aperio(s: str) -> bool:
    return s.startswith('Aperio')


def parse_aperio_description(s: str) -> tuple[list[str], dict[str, str]]:
    header, *kv_pairs = s.split('|')

    head = [s.strip() for s in header.splitlines()]

    meta = {}
    for s in kv_pairs:
        tags = s.split('=', 1)
        if len(tags) == 2:
            meta[tags[0].strip()] = tags[1].strip()
        else:
            raise ValueError(f'Unparseable line in description: {s!r}')

    return head, meta


# ----------------------------- xml description ------------------------------


def parse_xml(s: str,
              /,
              *,
              group: str = 'property',
              name: str = 'name',
              value: str = 'value') -> dict[str, str]:
    t = fromstring(s, XMLParser(resolve_entities=False, no_network=True))

    # Remove a namespace URI in the element's name
    for elem in t.getiterator():
        if not isinstance(elem, etree._Comment | etree._ProcessingInstruction):
            elem.tag = etree.QName(elem).localname

    # Remove unused namespace declarations
    etree.cleanup_namespaces(t, top_nsmap=None, keep_ns_prefixes=None)

    return {e.find(name).text: e.find(value).text for e in t.iter(group)}


# ----------------------------------------------------------------------------


def _gdal_parse_description(meta: Mapping[str, str]) -> dict[str, Any]:
    desc = meta.get('TIFFTAG_IMAGEDESCRIPTION')
    if not desc:
        return {}
    if is_aperio(desc):
        raise ValueError('Aperio is not yet supported by GDAL driver')
    try:
        props = parse_xml(desc)
    except Exception:  # noqa: BLE001
        return {}
    return unflatten(props)


def gdal_parse_mpp(meta: Mapping) -> list[float]:
    props = _gdal_parse_description(meta)

    # TIFF tags
    tiff_res_tags = [f'TIFFTAG_{a}RESOLUTION' for a in 'XY']
    if res := [float(meta[trt]) for trt in tiff_res_tags if trt in meta]:
        if meta.get('TIFFTAG_RESOLUTIONUNIT') == '3 (pixels/cm)':
            return [10_000 / r for r in res]
        raise NotImplementedError

    # VIPS tags
    res = [float(props[tag]) for tag in ('xres', 'yres') if tag in props]
    if res:
        return [1_000 / r for r in res]

    # Openslide tags
    if osd := props.get('openslide'):
        mpp = [float(osd[tag]) for tag in ('mpp-x', 'mpp-y') if tag in osd]
        if mpp:
            return mpp

    # Aperio tags
    if mpp_ := props.get('MPP'):
        return [float(mpp_), float(mpp_)]

    return []


def get_transform(
    image: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray] | None:
    if env.BIPL_CLAHE:
        return clahe

    if _RGB_RAMPS is not None:
        *luts, = map(_get_lut, cv2.split(image), _RGB_RAMPS)
        return _Lut(luts)

    return None


def _get_lut(src: np.ndarray, dst_ramp: np.ndarray) -> np.ndarray:
    src_ramp = _get_ramp(src)  # [0..1]
    lut = np.interp(src_ramp, dst_ramp, np.arange(256))  # [0..255]
    return around(lut.clip(0, 255), dtype='u1')


def _get_ramp(im: np.ndarray) -> np.ndarray:
    # 0..255 to 0..1
    hist = np.bincount(im.ravel(), minlength=256).astype('f8')
    hist /= hist.sum()

    bins, = np.where(hist)
    ramp = hist[bins].cumsum()
    return np.interp(np.arange(256), bins, ramp)  # 256 of [0..1] @ f8


@dataclass(frozen=True)
class _Lut:
    luts: Sequence[np.ndarray]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.merge([
            cv2.LUT(plane, lut)
            for plane, lut in zip(cv2.split(image), self.luts)
        ])


def clahe(im: np.ndarray) -> np.ndarray:
    h, w = im.shape[:2]
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(w // 64, h // 64))

    ls, a, b = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2LAB))
    ls = cl.apply(ls)
    return cv2.cvtColor(cv2.merge([ls, a, b]), cv2.COLOR_LAB2RGB)


_RGB_RAMPS = None
if env.BIPL_RGB_RAMPS:
    _RGB_RAMPS = np.load(env.BIPL_RGB_RAMPS).reshape(3, 256)
