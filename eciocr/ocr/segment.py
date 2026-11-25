"""Cut one voter box into field regions.

Box layout (SIR roll and supplement):
  top band : [serial cell] ([part cell] on supplements)          EPIC
  body     : 4 text lines (name / relative / house / age+gender) | photo cell
We find the inner ruled cells from the line mask, then text lines by a
horizontal ink profile. Works at any resolution.
"""
import os
import cv2
import numpy as np


# Punjabi (Gurmukhi) rolls (2026-11-25, lang-pan): set by segment_pan only (None = all small cells, as before).
CELL_TOP_MAX = None


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
    if CELL_TOP_MAX is not None: small = [c for c in small if c[1] < CELL_TOP_MAX * h]   # segment_pan
    return bw, (photo[0] if photo else None), small


def trim(im, margin=4):
    """Cut a field crop to its ink (plus a margin); ruled lines removed first."""
    if im.size == 0: return im
    bw = _mask(im)
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, im.shape[1] // 3), 1)))
    lines |= cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(6, int(im.shape[0] * 0.8)))))
    ink = cv2.subtract(bw, lines)
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
        out[f'line{k}'] = trim(clean_edges(gray[y0:y1, xl:px]))
        k += 1
        if k >= ENG_MAX_LINES: break
    return out


# ---------------------------------------------------------------- Punjabi rolls (2026-11-25, job lang-pan)
# Punjabi SIR boxes use the Hindi box template (serial cell, EPIC, 4 labelled lines, photo cell; no part cell). The Gurmukhi
# head line of a long name plus a tall letter can close a contour that cells() takes for a small top-band cell (the part cell
# of supplements) at 0.22 h; segment() then starts the body below the name line and the EPIC crop takes in the photo frame
# (seen on S19 AC 43 part 10: 5 of 8 boxes). Real top-band cells start at 0.04 h: segment_pan keeps only small cells that
# start above 0.15 h. The rest is segment() unchanged.
def segment_pan(gray):
    global CELL_TOP_MAX
    CELL_TOP_MAX = 0.15
    try:
        return segment(gray)
    finally:
        CELL_TOP_MAX = None


# 2026-11-25 (lang-pan round 1): Gurmukhi lines touch. The u / uu signs (hook + 1 or 2 bars), halant and subjoined letters of
# one line hang into the band of the next line, and the upper marks of the next line (ੌ ੈ ੇ tippi bindi) reach up into the
# gap. segment() cuts in the emptiest rows between the lines: the bars of ੁ / ੂ went to the next line's crop (gold:
# ਕੁਮਾਰ read ਕਮਾਰ on 40+ fields). segment_pan2 assigns connected components: each line's head line (the densest ink row in its
# band) is found; components that cross a head line row are that line's letters; any other component (mark) between line i and i+1 goes to
# line i+1 when its bottom is within 0.15 pitch above line i+1's head line (upper marks sit on the head line), else to line i
# (u / uu bars, halant, subjoined letters end 7+ px above it; measured on S19 AC 36 part 61). Each line
# crop is its own components only (other ink painted white). Fallback to segment() lines when 4 head lines are not found.
PAN_PITCH = 30.3 / 257   # measured head line pitch on the SIR boxes (1x crop)
PAN_UPPER_GAP = float(os.environ.get('ECI_PAN_UPPER_GAP', '0.15'))   # x pitch: max gap above the next head line of an upper mark
PAN_COMP = os.environ.get('ECI_PAN_COMP', '1') == '1'


def segment_pan2(gray):
    out = segment_pan(gray)
    if not PAN_COMP or LAST_STAMP: return out
    h, w = gray.shape
    global CELL_TOP_MAX
    CELL_TOP_MAX = 0.15
    try:
        bw, photo, small = cells(gray)
    finally:
        CELL_TOP_MAX = None
    top_bottom = max([c[1] + c[3] for c in small], default=int(0.18 * h))
    px = photo[0] - 3 if photo else int(0.72 * w)
    xl = max(6, int(0.018 * w))
    y_off = top_bottom + 2
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 5), 1)))
    lines = cv2.dilate(lines, np.ones((3, 1), np.uint8))
    body = cv2.subtract(bw, lines)[y_off: h - 3, xl: px]
    if body.size == 0: return out
    prof = (body > 0).sum(1).astype(float)
    ink = np.where(prof > max(1.0, 0.01 * body.shape[1]))[0]
    if not len(ink): return out
    P = PAN_PITCH * h   # template line pitch 
    heads = []
    for r in np.argsort(-prof, kind='stable'):   # head lines = the 4 densest rows, >= 0.7 pitch apart
        if prof[r] <= 0: break
        if all(abs(int(r) - q) >= 0.7 * P for q in heads): heads.append(int(r))
        if len(heads) == 4: break
    heads.sort()
    if len(heads) < 4 or any(not (0.75 * P <= heads[i + 1] - heads[i] <= 1.3 * P) for i in range(3)): return out
    P = float(np.median(np.diff(heads)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(body, 8)
    cut = False
    for j in range(1, n):   # a component across two head lines (a uu sign touching the word below): cut at its thinnest row
        y, ch = st[j][1], st[j][3]
        hs = [i for i in range(4) if y <= heads[i] < y + ch]
        for i in hs[:-1]:
            a, b = int(heads[i] + 0.35 * P), int(heads[i + 1] - 1)
            if b <= a: continue
            cnt = (lab[a:b] == j).sum(1)
            r = a + int(np.argmin(cnt)); body[r][lab[r] == j] = 0; cut = True
    if cut: n, lab, st, cen = cv2.connectedComponentsWithStats(body, 8)
    owner = np.zeros(n, int) - 1
    core = [[] for _ in range(4)]
    for j in range(1, n):   # letters: components that cross a head line row
        y, ch = st[j][1], st[j][3]
        for i in range(4):
            if y <= heads[i] < y + ch: owner[j] = i; core[i].append(y + ch); break
    for j in range(1, n):   # marks: to the line whose baseline (below) or head line (above) is nearer
        if owner[j] >= 0 or st[j][4] < 2: continue
        y, ch = st[j][1], st[j][3]; cy = y + ch / 2.0
        if ch >= 0.45 * P: owner[j] = int(np.argmin([abs(y - q) for q in heads])); continue   # a word off its line's head row
        if cy < heads[0]: owner[j] = 0; continue
        if cy >= heads[3]: owner[j] = 3; continue
        i = max(k for k in range(3) if heads[k] <= cy)
        d_up = heads[i + 1] - (y + ch)   # upper marks of line i+1 sit on its head line; lower marks of line i end well above it
        flat = ch <= 0.16 * P and st[j][2] >= 2.5 * ch and d_up >= 2   # lower bar of a uu sign (ends near the next head line)
        owner[j] = i + 1 if d_up <= PAN_UPPER_GAP * P and not flat else i
    g = gray[y_off: h - 3, xl: px]
    for i in range(4):
        js = [j for j in range(1, n) if owner[j] == i]
        if not js: continue
        y0 = max(0, min(st[j][1] for j in js) - 3); y1 = min(g.shape[0], max(st[j][1] + st[j][3] for j in js) + 3)
        crop = g[y0:y1].copy()
        keep = np.isin(lab[y0:y1], js)
        other = (lab[y0:y1] > 0) & ~keep
        crop[cv2.dilate(other.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 255
        crop[cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0] = 255   # also faint pixels next to others
        out[f'line{i}'] = trim(crop)
    return out
