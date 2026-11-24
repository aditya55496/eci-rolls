"""Cut one voter box into field regions.

Box layout (SIR roll and supplement):
  top band : [serial cell] ([part cell] on supplements)          EPIC
  body     : 4 text lines (name / relative / house / age+gender) | photo cell
We find the inner ruled cells from the line mask, then text lines by a
horizontal ink profile. Works at any resolution.
"""
import os, cv2
import numpy as np


def _mask(gray):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)


def cells(gray):
    h, w = gray.shape
    bw = _mask(gray)
    hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 12), 1)))
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, h // 10))))
    cnts, _ = cv2.findContours(cv2.bitwise_or(hl, vl), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > 0.9 * w: continue  # the outer box
        if cw > 0.08 * w and ch > 0.08 * h:
            if not any(abs(x - o[0]) < 5 and abs(y - o[1]) < 5 for o in out):
                out.append((x, y, cw, ch))
    photo = [c for c in out if c[3] > 0.4 * h and c[0] > 0.5 * w]
    small = sorted([c for c in out if c[3] < 0.3 * h and c[1] < 0.3 * h and c[0] < 0.6 * w], key=lambda c: c[0])
    return bw, (photo[0] if photo else None), small


def trim(im, margin=4, dark=None):
    """Cut a field crop to its ink (plus a margin); ruled lines removed first.
    dark (job4, 2026-11-25, English lines): only pixels darker than this count as ink for the cut. English SIR pages are JPEG
    with a light grey haze (values 200-250) next to some lines; the mask took it as ink, so name crops kept a long empty tail and
    the English CRNNs read a '1' / "'" / '7' there (gold: 132 of 2,058 names of one model, all in the same box positions)."""
    if im.size == 0: return im
    bw = _mask(im)
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, im.shape[1] // 3), 1)))
    lines |= cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(6, int(im.shape[0] * 0.8)))))
    ink = cv2.subtract(bw, lines)
    if dark is not None:
        ink[im >= dark] = 0
    cols = np.where(ink.sum(0) > 0)[0]; rows = np.where(ink.sum(1) > 0)[0]
    if not len(cols): return im
    # 2026-11-23: a thin sliver of ink (<= 6 px wide) at either end, >= 20 px away from the rest, is a frame edge or noise
    # (the photo frame's edge in the 2024 print kept name crops at the full line width): leave it out
    for _ in range(3):
        gaps = np.diff(cols)
        if len(cols) > 1 and gaps[-1] >= 20 and cols[-1] - cols[np.where(gaps >= 20)[0][-1] + 1] <= 5:
            cols = cols[: np.where(gaps >= 20)[0][-1] + 1]
        elif len(cols) > 1 and gaps[0] >= 20 and np.where(gaps >= 20)[0][0] <= 5 and cols[np.where(gaps >= 20)[0][0]] - cols[0] <= 5:
            cols = cols[np.where(gaps >= 20)[0][0] + 1:]
        else:
            break
    ink = ink[:, cols[0]: cols[-1] + 1]; rows = np.where(ink.sum(1) > 0)[0]
    x0, x1 = max(0, cols[0] - margin), min(im.shape[1], cols[-1] + 1 + margin)
    y0, y1 = max(0, rows[0] - margin), min(im.shape[0], rows[-1] + 1 + margin)
    return im[y0:y1, x0:x1]


# line centres as a fraction of box height (median of normally split gold boxes)
TEMPLATE_SIR = [0.2937, 0.4097, 0.5226, 0.6484]
TEMPLATE_SUP = [0.2835, 0.4085, 0.5247, 0.6524]
# General Election 2024 print: median line centres of 1,499 unstamped gold_v2 boxes split by the profile (2026-11-23)
TEMPLATE_GER = [0.2896, 0.4105, 0.5244, 0.6503]
# Set by eci_ocr.py for rolls before 2026 (GE 2024): the DELETED stamp's lower letters reach the empty band under the 4th
# line, which no unstamped box has ink in (gold_v2: 82/82 stamped boxes >= 0.0007, 0/1,518 unstamped). Such boxes are cut
# at TEMPLATE_OVERRIDE even when the ink profile gives 4 plausible bands (the stamp shifts them by a line).
STAMP_CHECK = False
TEMPLATE_OVERRIDE = None
LAST_CENTRES = None
LAST_STAMP = None


def stamp_below(gray, xl, px):
    """Fraction of ink (< 160, long horizontal rules removed) in the band 0.72-0.92 of box height, text column only."""
    h, w = gray.shape
    lo = gray[int(0.72 * h): int(0.92 * h), xl: px]
    d = (lo < 160).astype(np.uint8)
    d = cv2.subtract(d, cv2.morphologyEx(d, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 6), 1))))
    return float(d.sum()) / max(1, d.size)


def clean_edges(band):
    """Whiten thin blobs that touch the band's top or bottom edge (the neighbour line's
    headline or descenders). Marks of this line (dots, reph) do not touch the edges."""
    if band.size == 0: return band
    bw = _mask(band); H = bw.shape[0]
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = band.copy()
    for j in range(1, n):
        x, y, w, h, a = st[j]
        touches = y == 0 or y + h >= H
        if touches and h <= max(2, int(0.3 * H)) and w >= 3 * h and w >= 0.08 * bw.shape[1]:  # headline-like only, never dots
            out[lab == j] = 255
    return out


def segment(gray):
    """Return dict of field-name -> crop (grayscale). Keys: serial, part, epic, line0..line3."""
    h, w = gray.shape
    bw, photo, small = cells(gray)
    out = {}
    top_bottom = max([c[1] + c[3] for c in small], default=int(0.18 * h))
    for i, (x, y, cw, ch) in enumerate(small[:2]):
        cell = gray[y + 3: y + ch - 3, x + 3: x + cw - 3]
        if i == 0:
            # serial cell: optional status code at the left (#, S, R2 ...), number right-aligned
            ink = (_mask(cell) > 0).sum(0) > 0
            left = ink[: int(0.4 * len(ink))]
            if left.sum() > 2:
                cols = np.where(ink)[0]
                gaps = [(cols[j + 1] - cols[j], cols[j]) for j in range(len(cols) - 1)]
                g, at = max(gaps) if gaps else (0, 0)
                if g > 0.12 * len(ink):
                    out['code'] = trim(cell[:, : at + 2]); out['serial'] = trim(cell[:, at + 2:])
                    continue
            out['serial'] = trim(cell)
        else:
            out['part'] = trim(cell)
    ex0 = (small[-1][0] + small[-1][2] + 4) if small else int(0.55 * w)
    ex0 = max(ex0, int(0.45 * w))
    out['epic'] = trim(gray[3: top_bottom + 1, ex0: w - 4])
    px = photo[0] - 3 if photo else int(0.72 * w)
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 5), 1)))
    lines = cv2.dilate(lines, np.ones((3, 1), np.uint8))
    xl = max(6, int(0.018 * w))  # skip the box's left border
    body = cv2.subtract(bw, lines)[top_bottom + 2: h - 3, xl: px]
    # remove ruled-line remnants
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))
    prof = (body > 0).sum(1).astype(float)
    k = max(1, int(0.01 * h)); sm = np.convolve(prof, np.ones(k) / k, mode='same')
    ink = np.where(sm > max(1.0, 0.01 * body.shape[1]))[0]
    merged = []
    if len(ink):
        top, bot = ink[0], ink[-1] + 1
        nl = 4
        pitch = (bot - top) / nl
        cuts = [top]
        for i in range(1, nl):
            c0 = int(top + (i - 0.35) * pitch); c1 = int(top + (i + 0.35) * pitch)
            if c1 > c0:  # middle of the longest run of minimal ink (raw profile) inside the window
                win = prof[c0:c1]; m = win.min(); runs, cur = [], None
                for j, v in enumerate(win):
                    if v <= m + 1: cur = (cur[0], j) if cur else (j, j)
                    elif cur: runs.append(cur); cur = None
                if cur: runs.append(cur)
                r0, r1 = max(runs, key=lambda r: r[1] - r[0])
                cuts.append(c0 + (r0 + r1 + 1) // 2)
            else:
                cuts.append(int(top + i * pitch))
        cuts.append(bot)
        # outer bands reach past the first/last ink rows: anusvara, reph and ऊ-matras are small marks
        # that can fall below the ink threshold, so they must not define the band edge
        cuts[0] = max(0, top - int(0.07 * h)); cuts[-1] = min(len(sm), bot + int(0.06 * h))
        for i in range(nl):
            seg = np.where(sm[cuts[i]:cuts[i + 1]] > 0.5)[0]
            if len(seg) and (seg[-1] - seg[0]) > 0.03 * h:
                merged.append((cuts[i] + seg[0], cuts[i] + seg[-1] + 1, cuts[i], cuts[i + 1]))
    pad = max(2, int(0.012 * h))
    body_h = h - 3 - (top_bottom + 2)
    hs = [b - a for a, b, lo, hi in merged]
    global LAST_CENTRES, LAST_STAMP
    LAST_CENTRES = None
    LAST_STAMP = STAMP_CHECK and stamp_below(gray, xl, px) >= 0.0007
    if len(merged) == 4 and not (hs and max(hs) > 1.6 * sorted(hs)[len(hs) // 2]) and not LAST_STAMP:
        LAST_CENTRES = [((a + b) / 2 + top_bottom + 2) / h for a, b, lo, hi in merged]
    if len(merged) < 4 or (hs and max(hs) > 1.6 * sorted(hs)[len(hs) // 2]) or LAST_STAMP:
        # stamp (DELETED) or other ink broke the profile split: cut at template line positions
        T = TEMPLATE_OVERRIDE or (TEMPLATE_SUP if len(small) >= 2 else TEMPLATE_SIR)
        cs = [c * h - (top_bottom + 2) for c in T]  # line centres in body coords
        p = (cs[3] - cs[0]) / 3
        edges = [max(0, int(cs[0] - p / 2))] + [int((cs[j] + cs[j + 1]) / 2) for j in range(3)] + [min(body_h, int(cs[3] + p / 2))]
        merged = [(edges[j], edges[j + 1], edges[j], edges[j + 1]) for j in range(4)]
    for i, (a, b, lo, hi) in enumerate(merged):
        # stay inside this line's band so the next line's matras do not leak in
        # whole band between the cut points (the cuts sit in the gaps between lines), then trim to ink
        y0 = top_bottom + 2 + lo; y1 = top_bottom + 2 + hi
        out[f'line{i}'] = trim(clean_edges(gray[y0:y1, xl:px]))
    return out


# ---------------------------------------------------------------- English rolls (2026-11-25, job engpipe)
# English SIR boxes print 4 labelled lines (Name / <relative> Name / House Number / Age .. Gender) in a smaller font than the
# Hindi print, and a long name, relative name or house number WRAPS to an unlabelled next line: a box has 4 to 7 text lines.
# The lines sit on a fixed grid (measured on 2,064 English gold boxes of 14 states: first line top at 0.23-0.245 of the box
# height, pitch 0.102-0.105). Neighbouring lines often touch (descender of one, cap of the next: 288 of 2,064 boxes had
# lines 1 and 2 as one ink run), so the cut points come from the grid, each moved to the emptiest rows near it.
ENG_PITCH = 0.1035      # line pitch / box height
ENG_MAX_LINES = 8
ENG_DARK = int(os.environ.get('ECI_ENG_DARK', '170')) or None   # job4: ink threshold of the English line cut (0 = off)


def _runs_min(prof, c0, c1):
    """Middle of the longest run of minimal profile values in prof[c0:c1]."""
    win = prof[c0:c1]
    if not len(win): return (c0 + c1) // 2
    m = win.min(); runs, cur = [], None
    for j, v in enumerate(win):
        if v <= m + 1: cur = (cur[0], j) if cur else (j, j)
        elif cur: runs.append(cur); cur = None
    if cur: runs.append(cur)
    r0, r1 = max(runs, key=lambda r: r[1] - r[0])
    return c0 + (r0 + r1 + 1) // 2


def segment_eng(gray):
    """English box -> dict: serial (+ code), part (supplements), epic, line0..lineN (N = 3..7, top to bottom)."""
    h, w = gray.shape
    bw, photo, small = cells(gray)
    out = {}
    top_bottom = max([c[1] + c[3] for c in small], default=int(0.18 * h))
    for i, (x, y, cw, ch) in enumerate(small[:2]):
        cell = gray[y + 3: y + ch - 3, x + 3: x + cw - 3]
        if i == 0:
            ink = (_mask(cell) > 0).sum(0) > 0
            left = ink[: int(0.4 * len(ink))]
            if left.sum() > 2:
                cols = np.where(ink)[0]
                gaps = [(cols[j + 1] - cols[j], cols[j]) for j in range(len(cols) - 1)]
                g, at = max(gaps) if gaps else (0, 0)
                if g > 0.12 * len(ink):
                    out['code'] = trim(cell[:, : at + 2]); out['serial'] = trim(cell[:, at + 2:])
                    continue
            out['serial'] = trim(cell)
        else:
            out['part'] = trim(cell)
    ex0 = (small[-1][0] + small[-1][2] + 4) if small else int(0.55 * w)
    ex0 = max(ex0, int(0.45 * w))
    out['epic'] = trim(gray[3: top_bottom + 1, ex0: w - 4])
    px = photo[0] - 3 if photo else int(0.72 * w)
    xl = max(6, int(0.018 * w))
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 5), 1)))
    lines = cv2.dilate(lines, np.ones((3, 1), np.uint8))
    y_body = top_bottom + 2
    body = cv2.subtract(bw, lines)[y_body: h - 3, xl:px]
    prof = (body > 0).sum(1).astype(int)
    if not len(prof): return out
    P = ENG_PITCH * h
    thr = max(2, int(0.01 * body.shape[1]))
    ink = np.where(prof > thr)[0]
    if not len(ink): return out
    top = int(ink[0])
    # grid boundaries: top - P/4 .. ; each inner boundary moved to the emptiest rows within +-0.3 P of the grid point
    cuts = [max(0, top - int(0.2 * P))]
    for k in range(1, ENG_MAX_LINES + 1):
        g = top + k * P - 0.12 * P   # the gap between lines sits a little above the next line's top
        c0, c1 = int(g - 0.3 * P), int(g + 0.3 * P)
        if c0 >= len(prof): break
        cuts.append(_runs_min(prof, max(cuts[-1] + int(0.5 * P), c0), min(len(prof), c1)) if c1 > c0 else int(g))
    cuts = [c for i, c in enumerate(cuts) if i == 0 or c > cuts[i - 1]]
    k = 0
    for a, b in zip(cuts, cuts[1:] + [len(prof)]):
        band = prof[a:b]
        # a text line: enough ink rows (>= 30 % of the pitch) - stamps and stray marks in the empty lower part are thinner
        if (band > thr).sum() < 0.3 * P:
            if k >= 4: break     # empty band after >= 4 lines: end of the text block
            continue
        y0, y1 = y_body + a, y_body + min(b, len(prof))
        out[f'line{k}'] = trim(clean_edges(gray[y0:y1, xl:px]), dark=ENG_DARK)
        k += 1
        if k >= ENG_MAX_LINES: break
    return out
