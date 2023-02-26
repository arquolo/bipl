__all__ = ['Svg']

from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
from lxml.builder import ElementMaker
from lxml.etree import ElementTree, indent, tostring

_SVG_NS = 'http://www.w3.org/2000/svg'
_SCRIPT = """\
const path = location.pathname.split("/").pop();
const url = path.substr(0, path.lastIndexOf(".")) + ".jpg";

let svg = document.documentElement;
svg.getElementsByTagName("image")[0].setAttribute("href", url);

let i = new Image();
i.onload = () => {
    svg.setAttribute("height", i.height);
    svg.setAttribute("width", i.width);
};
i.src = url;

for (let group of svg.getElementsByTagName("g")) {
    group.setAttribute("fill", "none");
};
"""


def hsv_colors(count: int) -> Iterator[str]:
    for hue in np.linspace(0, 360, num=count, endpoint=False, dtype='int32'):
        yield f'hsl({hue},100%,50%)'


class Svg:
    """Converts raster mask (2d numpy.ndarray of integers) to SVG-file.

    Parameters:
    - labels - list of all label names (0th is unlabeled),
      i-th name will be assigned to (i+1)th index.

    Usage:
    ```
    mask = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)
    Svg(mask, ['pos', 'neg']).save('sample.svg')
    ```
    """
    def __init__(self, mask: np.ndarray, labels: Sequence[str]):
        e = ElementMaker()

        groups = []
        for uniq in np.unique(mask.ravel()):
            if uniq == 0:  # skip background
                continue
            m = (mask == uniq).astype('u1')
            contours = cv2.findContours(m, cv2.RETR_CCOMP,
                                        cv2.CHAIN_APPROX_TC89_L1)[-2]
            polygons = (
                e.polygon(points=' '.join(map(str, contour.ravel())))
                for contour in contours if len(contour) >= 3)
            groups.append(e.g(*polygons, label=labels[uniq - 1]))

        root = e.svg(
            e.image(),
            *groups,
            e.script(href='main.js'),
            e.style('@import url(main.css)'),
            xmlns=_SVG_NS)

        indent(root, space='\t')
        self.body = tostring(root, encoding='utf-8')

        maxlen = max(len(c) for c in labels)
        self.labels = [f'{c:{maxlen}s}' for c in labels]

    def save(self, path: Path) -> None:
        path = Path(path)

        if not (script := path.parent / 'main.js').exists():
            script.write_text(_SCRIPT)

        if not (style := path.parent / 'main.css').exists():
            labels = zip(hsv_colors(len(self.labels)), self.labels)
            style.write_text('\n'.join(
                f'.{name} {{ stroke: {color} }}' for color, name in labels))

        path.with_suffix('.svg').write_text(self.body)

    @staticmethod
    def load(path: Path) -> dict[str, list[np.ndarray]]:
        """
        Yields contours, contour is 2d numpy array of shape [count, (x, y)]
        """
        tree = ElementTree()
        tree.parse(path.with_suffix('.svg').as_posix())

        fields = ((node.attrib['label'], (p.attrib['points'] for p in node))
                  for node in tree.getiterator(f'{{{_SVG_NS}}}g'))
        return {
            name:
            [np.fromstring(ll, 'i4', sep=' ').reshape(-1, 2) for ll in lines]
            for name, lines in fields
        }
