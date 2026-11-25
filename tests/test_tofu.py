"""ECI_TOFU (eciocr/ocr/tofu.py): missing-glyph boxes found, digits and letters not; select() maps them to fields."""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "eciocr", "ocr"))
import tofu  # noqa: E402


def page():
    """A 2806 px page (Haryana SIR scale): a missing-glyph box 9 x 26 px with 2 px strokes touching a headline on its left,
    a free-standing one, an oval '0' 10 x 17 px, a 'D'-like shape and a large ruled cell."""
    g = np.full((2806, 1983), 255, np.uint8)
    cv2.line(g, (300, 300), (402, 300), 0, 2)                    # headline that touches the box
    cv2.rectangle(g, (402, 290), (410, 315), 0, 2)               # box (touching)
    cv2.rectangle(g, (600, 290), (608, 315), 0, 2)               # box (free)
    cv2.ellipse(g, (800, 300), (5, 8), 0, 0, 360, 0, 2)          # '0'
    cv2.rectangle(g, (1000, 100), (1300, 400), 0, 2)             # ruled cell
    return g


def test_find_tofu():
    found = tofu.find_tofu(page())
    xs = sorted(x for x, y, w, h in found)
    assert len(found) == 2 and abs(xs[0] - 402) <= 2 and abs(xs[1] - 600) <= 2


def test_select_and_erase():
    g = page()
    boxes = [(250, 220, 400, 253)]                               # one voter box: the boxes sit at 0.31 of its height (name line)
    keep, at = tofu.select(tofu.find_tofu(g), boxes, 1.0)
    assert len(keep) == 1 and at == {0: {"name": 1}}            # the free box at x 600 is right of the text column (0.70 w)
    e = tofu.erase(g, keep)
    assert (e[285:320, 401:412] == 255).all() and (e[300, 300:395] == 0).all()   # box gone, headline kept


def test_no_boxes_no_change():
    g = np.full((1679, 1200), 255, np.uint8)
    cv2.putText(g, "100 HR/04/40", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    assert tofu.find_tofu(g) == []
