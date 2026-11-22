"""Find voter boxes on a roll page image (SIR roll or supplement).

Boxes are drawn rectangles in a 3-column grid. We find them from the ruled
lines, so the result does not depend on resolution (115 or 144 dpi).
"""
import cv2
import numpy as np


def _lines(gray):
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    h, w = bw.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 40)))
    hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    return bw, hl, vl


def find_boxes(img):
    """Return voter boxes as (x, y, w, h), sorted row by row, left to right."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    bw, hl, vl = _lines(gray)
    grid = cv2.bitwise_or(hl, vl)
    cnts, hier = cv2.findContours(grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # a voter box is about 1/3 of the page wide and 1/12-1/7 of the page high
        if 0.25 * W < w < 0.36 * W and 0.06 * H < h < 0.16 * H:
            boxes.append((x, y, w, h))
    # remove duplicates (outer/inner edge of the same ruled box)
    boxes.sort(key=lambda b: (b[1], b[0]))
    out = []
    for b in boxes:
        if any(abs(b[0] - o[0]) < 8 and abs(b[1] - o[1]) < 8 for o in out):
            continue
        out.append(b)
    # order: rows by y (tolerance), then x
    rows = []
    for b in sorted(out, key=lambda b: b[1]):
        if rows and abs(rows[-1][0][1] - b[1]) < 0.02 * H:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for r in rows for b in sorted(r, key=lambda b: b[0])]


def crop(img, box, pad=2):
    x, y, w, h = box
    return img[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
