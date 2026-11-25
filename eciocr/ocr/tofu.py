"""Missing-glyph boxes ('tofu') in voter names (job2 2026-11-25, switch ECI_TOFU=1 in eciocr/prod/eci_ocr.py, default off).

Haryana SIR 2026 rolls (and the 2024 rolls of the same ACs) print many names and relatives' names with an empty upright
rectangle after the name: the roll font has no glyph for a character stored there, so the PDF shows the font's .notdef box.
Survey (part 10 of every S07 AC, 81,290 boxes): 26 of 90 ACs print it, in 8.5 % of all boxes (up to 57 % of the boxes of an AC),
only on the name and relative-name lines, at the end of the name (2 of 14,189 boxes were followed by another word: 'रीना□ देवी').
The engines read it as '0', ']', ')', 'ा' or 'ी' or lose the whole name, so the field is flagged (or, rarely, read as a vowel sign
without a flag: 'राम कवर□' -> 'राम कवरी'). The box is no letter: the same names are printed without it elsewhere in the same roll.

find_tofu() finds the box by its hole: a tall, thin, rectangular white hole closed on all four sides by thin strokes with square
outer corners, 0.0078-0.0118 x the page height (a printed '0' is ~0.0061 and round at the corners). The box can touch the headline
of the letter before it, so holes are used, not connected components. select() keeps only boxes in a voter box's text column
(left of the photo, 0.20-0.72 of the box height) and names the field of each (nearest SIR line centre); erase() paints them white
on the page image as stored (before eci_ocr's rescale), so no grey rest of the box is left at the smaller scale.
"""
import cv2
import numpy as np

LINE_FIELDS = ('name', 'relation_name', 'house_no', 'age')
_CENTRES = (0.2937, 0.4097, 0.5226, 0.6484)     # = segment.TEMPLATE_SIR


def _run(seq):
    n = 0
    for v in seq:
        if not v: break
        n += 1
    return n


def find_tofu(gray, thr=170):
    """[(x, y, w, h)] of missing-glyph boxes on a page image (gray, any resolution; sizes are relative to the page height)."""
    H = gray.shape[0]
    bw = (gray < thr).astype(np.uint8)
    cnts, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None: return out
    hmin, hmax = 0.0050 * H, 0.0105 * H
    for i, c in enumerate(cnts):
        if hier[0][i][3] < 0: continue                     # an outer contour, not a hole
        x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = x + 1, y + 1, w - 2, h - 2              # a hole contour runs on the ink pixels around the hole
        if not (hmin <= h <= hmax and 0.10 * h <= w <= 0.5 * h and w >= 2) or x < 1 or y < 1: continue
        if (bw[y:y + h, x:x + w] == 0).mean() < 0.75: continue
        core = bw[y + 1:y + h - 1, x + 1:x + w - 1]
        if core.size == 0 or (core == 0).mean() < 0.95: continue   # rectangular hole (an oval '0' has ink in its corners)
        L = [_run(bw[yy, x - 1::-1][:6] > 0) for yy in range(y, y + h)]
        R = [_run(bw[yy, x + w:x + w + 6] > 0) for yy in range(y, y + h)]
        T = [_run(bw[y - 1::-1, xx][:6] > 0) for xx in range(x, x + w)]
        B = [_run(bw[y + h:y + h + 6, xx] > 0) for xx in range(x, x + w)]
        sl, sr, st, sb = (int(np.median(v)) for v in (L, R, T, B))
        if min(sl, sr, st, sb) < 1 or max(sl, sr, st, sb) > max(3, int(0.2 * h)): continue   # thin strokes on all 4 sides
        if min(np.mean([v >= 1 for v in L]), np.mean([v >= 1 for v in R]), np.mean([v >= 1 for v in T]),
               np.mean([v >= 1 for v in B])) < 0.9: continue
        X, Y, W, Hh = x - sl, y - st, w + sl + sr, h + st + sb
        if not (0.0078 * H <= Hh <= 0.0118 * H): continue
        if not all(bw[yy:yy + 2, xx:xx + 2].any() for yy, xx in ((Y, X), (Y, X + W - 2), (Y + Hh - 2, X), (Y + Hh - 2, X + W - 2))):
            continue                                       # square outer corners
        out.append((int(X), int(Y), int(W), int(Hh)))
    return out


def select(tofus, boxes, scale=1.0):
    """tofus: find_tofu() of the page before a rescale by `scale`; boxes: layout boxes of the rescaled page. Returns (the tofus
    that lie in a voter box's text column, {box index: {field: count}})."""
    keep, at = [], {}
    for t in tofus:
        x, y, w, h = t
        cx, cy = (x + w / 2) * scale, (y + h / 2) * scale
        for i, b in enumerate(boxes):
            bx, by, bw_, bh = b[:4]
            if bx + 0.015 * bw_ <= cx <= bx + 0.70 * bw_ and by + 0.20 * bh <= cy <= by + 0.72 * bh:
                f = LINE_FIELDS[int(np.argmin([abs((cy - by) / bh - c) for c in _CENTRES]))]
                at.setdefault(i, {}); at[i][f] = at[i].get(f, 0) + 1
                keep.append(t)
                break
    return keep, at


def erase(gray, rects, pad=1):
    """Copy of the page image with the rects (+ pad) painted white (on the page before the rescale)."""
    g = gray.copy()
    for x, y, w, h in rects:
        g[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad] = 255
    return g
