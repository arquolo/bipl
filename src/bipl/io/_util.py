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
    """Round to power to 2"""
    assert x > 0
    power = round(log2(x))
    return 1 << power
