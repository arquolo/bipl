from collections.abc import Mapping
from io import BytesIO
from itertools import zip_longest
from math import log2
from typing import Any

import cv2
import numpy as np
from lxml import etree
from lxml.etree import XMLParser, fromstring
from PIL.Image import fromarray
from PIL.ImageCms import buildTransform, createProfile

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


def get_aperio_properties(description: str,
                          index: int = 0) -> tuple[str, dict[str, str]] | None:
    if index == 0 and not description.startswith('Aperio'):
        return None
    header, *kv_pairs = description.split('|')

    header = '\n'.join(s.strip() for s in header.splitlines())

    meta = {}
    for i, kv in enumerate(kv_pairs):
        match kv.split('=', 1):
            case [str(k), str(v)]:
                meta[k.strip()] = v.strip()
            case ['']:
                continue
            case _:
                raise ValueError(f'Cannot parse TIFF description line #{i}: '
                                 f'{kv!r}, {description!r}')

    return header, meta


# ----------------------------- xml description ------------------------------


def get_ventana_properties(s: bytes, index: int = 0) -> dict[str, str]:
    s = s.strip(b'\00')  # For safety
    t = fromstring(s, XMLParser(resolve_entities=False, no_network=True))
    if index == 0:
        if (root := t.find('iScan')) is not None:
            return dict(root.items())
        return {}
    return {'xmp': s.decode()}


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
    if get_aperio_properties(desc):
        raise ValueError('Aperio is not yet supported by GDAL driver')
    try:
        props = parse_xml(desc)
    except Exception:  # noqa: BLE001
        return {}
    return unflatten(props)


def gdal_parse_mpp(meta: Mapping) -> list[float]:
    props = _gdal_parse_description(meta)

    # TIFF tags
    res = [float(v) for a in 'XY' if (v := meta.get(f'TIFFTAG_{a}RESOLUTION'))]
    if res:
        match meta.get('TIFFTAG_RESOLUTIONUNIT'):
            case '2 (pixels/inch)':
                return [25_400 / r for r in res]
            case '3 (pixels/cm)':
                return [10_000 / r for r in res]

    # VIPS tags, always px/mm
    if res := [float(v) for tag in ('xres', 'yres') if (v := props.get(tag))]:
        return [1_000 / r for r in res]

    # Openslide tags
    if osd := props.get('openslide'):
        mpp = [float(v) for tag in ('mpp-x', 'mpp-y') if (v := osd.get(tag))]
        if mpp:
            return mpp

    # Aperio tags
    if mpp_ := props.get('MPP'):
        return [float(mpp_), float(mpp_)]

    return []


# ------------- contrast-limited adaptive histogram equalization -------------


def clahe(im: np.ndarray) -> np.ndarray:
    h, w = im.shape[:2]
    gh, gw = max(h // 64, 1), max(w // 64, 1)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gw, gh))

    ls, a, b = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2LAB))
    ls = cl.apply(ls)
    return cv2.cvtColor(cv2.merge([ls, a, b]), cv2.COLOR_LAB2RGB)


# -------------------------- image color correction --------------------------


class Icc:
    def __init__(self, icc: bytes) -> None:
        f = BytesIO(icc)
        srgb = createProfile('sRGB')
        self._tf = buildTransform(f, srgb, inMode='RGB', outMode='RGB')

    def __call__(self, image: np.ndarray) -> np.ndarray:
        pil = fromarray(image)
        pil = self._tf.apply(pil)
        return np.array(pil, copy=False)


# -------------------------------- etc---------------------------------------


def round2(x: float) -> int:
    """Round to power of 2"""
    assert x > 0
    power = round(log2(x))
    return 1 << power


def floor2(x: float) -> int:
    """Floor to power of 2"""
    power = max(0, int(log2(x)))
    return 1 << power
