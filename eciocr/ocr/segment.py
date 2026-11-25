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


# job3 (2026-11-25): the grid cut between two touching English lines often goes through the descenders of the upper line (j, y,
# g, p, q): 'Indrajit' -> 'Indraiit', 'Piyarun' -> 'Pivarun', 'Begum' -> 'Bequm', and CRNN + Tesseract then agree on the wrong
# text (held-out check, 12 S05/S14 parts: 8 of 12 wrong 'agreeing' values). ECI_ENG_DESC = F (> 0): each line crop reaches F x the
# line pitch below its cut; ink components that start below the cut (the next line) are whitened, components that cross the cut
# (this line's descenders) are kept. Default 0 = engpipe crops (the CRNNs are trained on the crops of the setting in use).
ENG_DESC = float(os.environ.get('ECI_ENG_DESC', '0') or 0)


def _with_descenders(gray, y0, y1, x0, x1, ext):
    y2 = min(gray.shape[0] - 3, y1 + ext)
    if y2 <= y1 or y1 <= y0: return gray[y0:y1, x0:x1]
    m = min(y0, max(2, ext // 2))                       # rows above the band: to see what continues from the line above
    band = gray[y0 - m:y2, x0:x1].copy()
    bw = _mask(band); cut = y1 - y0 + m
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    keep = np.zeros(bw.shape, np.uint8)
    for j in range(1, n):
        x, y, w, h, _a = st[j]
        if y >= cut - 1: continue                       # starts below the cut: the next line
        if y == 0 and y + h <= m + 0.35 * (cut - m): continue  # reaches above the margin and ends high: the upper line's descender
        if y + h <= m: continue                         # entirely above the band
        keep[lab == j] = 1
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8)) > 0
    band[~keep & ((bw > 0) | (band > 140))] = 255     # removed components and their grey edges -> white
    ext_ = band[cut:]; ext_[~keep[cut:]] = 255          # below the cut: only this line's descenders (+ their grey edge)
    return band[m:]


BEN_COMP_CUT = float(os.environ.get('ECI_BEN_COMP_CUT', '0.25'))  # > 0: touching lines split at headline - CUT x P (0: narrowest row)
BEN_COMP_ZONE = float(os.environ.get('ECI_BEN_COMP_ZONE', '0.22'))  # line zone centre = headline + ZONE x P (marks go to the nearest)
BEN_COMP_UP = os.environ.get('ECI_BEN_COMP_UP', '1') == '1'       # split a line's component that reaches up into the line above
BEN_STAMP = float(os.environ.get('ECI_BEN_STAMP', '0'))          # > 0: grey-stamp removal (median grey of a stamp letter above this)
BEN_STAMP_CORE = int(os.environ.get('ECI_BEN_STAMP_CORE', '75'))  # text keeps the pixels within 1 px of a pixel darker than this
BEN_COMP_DARK = int(os.environ.get('ECI_BEN_COMP_DARK', '0'))   # ink threshold of the component labels (antialias edges touch)


def _assign_lines(gray, body, bands, y_body, x0, P):
    """lang-ben round 2: one crop per text band; every ink component goes to a line: a component that holds a headline row
    (densest row of a band) to that line; a mark without one to the line whose zone centre (headline + 0.22 P) is nearest.
    So a u-kar hook below line k stays with line k, and i-kar tops, reph and chandrabindu above line k+1 go to line k+1.
    P = median headline distance (small-print pages have a smaller pitch than the box grid). Two lines that touch (one
    component with both headlines, or a line's component that reaches the next headline) are split at the narrowest row of
    the gap (row headline - 0.25 P; gold r2 models: 0.2 / 0.25 / 0.3 / narrowest row = 89.24 / 89.24 / 89.39 / 88.23 % record;
    labels on dark pixels only (ECI_BEN_COMP_DARK 150) gave 87.06 %). Components taller than 1.3 P otherwise (stamps, frame) are kept clipped to the band, as the row cut did.
    Other components are whitened (with a 1 px grey edge)."""
    H = body.shape[0]
    prof = (body > 0).sum(1)
    heads = [a + int(np.argmax(prof[a:b])) for a, b in bands]
    if len(heads) >= 3: P = float(np.median(np.diff(heads)))   # the real pitch (small-print pages: smaller than the box grid)
    W = body.shape[1]
    dark = ((gray[y_body: y_body + H, x0: x0 + W] < BEN_COMP_DARK) & (body > 0)).astype(np.uint8) if BEN_COMP_DARK > 0 else body
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    stamp = None
    if BEN_STAMP > 0:   # grey stamp ('UNDER ADJUDICATION', job4 finding): >= 2 tall components with a light median grey ->
        G = gray[y_body: y_body + H, x0: x0 + W]   # keep only ink within 1 px of a dark core pixel, label again
        greys = []
        for j in range(1, n):
            x, y, w, h, _a = st[j]
            if h > 1.3 * P: greys.append(np.median(G[y: y + h, x: x + w][lab[y: y + h, x: x + w] == j]))
        if sum(1 for v in greys if v > BEN_STAMP) >= 2:
            core = cv2.dilate((G < BEN_STAMP_CORE).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            stamp = ~core
            dark = (dark > 0) & core
            n, lab, st, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    owner = np.full(n, -1, int); tall = np.zeros(n, bool); split = []
    def cut_row(j, k):   # the narrowest row of component j in the gap above line k + 1 (where two lines' marks touch)
        r0 = max(0, int(round(heads[k + 1] - 0.33 * P))); r1 = max(r0 + 1, int(round(heads[k + 1] - 0.1 * P)) + 1)
        x, y, w, h, _a = st[j]
        cnt = [(int((lab[r, x:x + w] == j).sum()), abs(r - (heads[k + 1] - 0.22 * P)), r) for r in range(r0, min(r1, H))]
        if BEN_COMP_CUT > 0: return int(round(heads[k + 1] - BEN_COMP_CUT * P))
        return min(cnt)[2] if cnt else int(heads[k + 1] - 0.22 * P)

    for j in range(1, n):
        x, y, w, h, _a = st[j]
        ks = [k for k, hd in enumerate(heads) if y - 1 <= hd <= y + h]
        if h > 1.3 * P and not (len(ks) == 2 and ks[1] == ks[0] + 1 and h <= 2.3 * P):
            tall[j] = True; continue             # stamp, frame: clipped to the bands
        if ks:                                   # the component holds a headline row: that line
            k = owner[j] = ks[0]
            if k + 1 < len(heads) and y + h - 1 > heads[k + 1] - 0.1 * P:   # it touches the next line's marks (small print)
                split.append((j, cut_row(j, k), k))
            elif BEN_COMP_UP and k > 0 and y < heads[k] - 0.3 * P and w <= 0.8 * P:   # a narrow glyph reaching up (a u-kar hook
                split.append((j, cut_row(j, k - 1), k - 1))                     # of line k-1 touching a digit of line k)
            continue
        c = y + h / 2.0                          # a mark (reph, chandrabindu, i-kar top, u-kar hook): nearest line zone
        d = [abs(c - (hd + BEN_COMP_ZONE * P)) for hd in heads]
        k = int(np.argmin(d))
        if d[k] >= 0.8 * P: continue
        owner[j] = k
        for k2 in range(len(heads) - 1):         # a mark that spans the gap (u-kar hook touching the next line's i-kar top):
            sp = heads[k2 + 1] - 0.3 * P         # rows above sp -> line k2, rows below -> line k2 + 1
            if y < sp - 1 and y + h > sp + 1 and h > 0.3 * P:
                split.append((j, int(round(sp)), k2))
    out = []
    for k, (a, b) in enumerate(bands):
        sel = owner == k
        rows = [(st[j][1], st[j][1] + st[j][3]) for j in np.where(sel)[0]]
        rows += [(st[j][1], sp) if k == k2 else (sp, st[j][1] + st[j][3]) for j, sp, k2 in split if k in (k2, k2 + 1)]
        y0 = min([r[0] for r in rows] + [a]); y1 = max([r[1] for r in rows] + [b])
        y0 = max(0, y0 - 2); y1 = min(H, y1 + 2)
        L = lab[y0:y1]
        keep = sel[L]
        for j, sp, k2 in split:                  # split marks: only this line's rows of them
            if k not in (k2, k2 + 1): continue
            m = L == j; r = np.arange(y0, y1)[:, None]
            keep = np.where(m, (r < sp) if k == k2 else (r >= sp), keep)
        tb = tall[L].copy(); tb[: max(0, a - y0)] = False; tb[max(0, b - y0):] = False
        keep = cv2.dilate((keep | tb).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        g = gray[y_body + y0: y_body + y1, x0: x0 + W].copy()
        g[~keep & ((L > 0) | (g > 140))] = 255
        if stamp is not None: g[stamp[y0:y1]] = 255
        out.append(g)
    return out


def segment_eng(gray, top_cells=None, pitch=None, comp=False):
    """English box -> dict: serial (+ code), part (supplements), epic, line0..lineN (N = 3..7, top to bottom).
    top_cells (lang-ben, 2026-11-25, Bengali rolls): keep only small cells that start in the top top_cells x box height. The
    Bengali headline (matra) plus the vertical strokes of a name form cell-like rectangles at 0.23 h; cells() took them as the
    part cell ('part' = the name, line0 = the relative line). The serial cell starts at ~0.05 h.
    pitch (lang-ben): line pitch / box height. The Bengali WB print has 0.1086 (median of 132 gold boxes with 4 clean ink runs);
    with the English 0.1035 the grid drifts ~5 px by the 4th line and cut the lower half of the age/gender line.
    comp (lang-ben round 2, 2026-11-25): ink components are assigned to lines (_assign_lines) instead of cutting at the
    emptiest row. FINDING: the row cut dropped the u-kar hook below a line (gold: 425 dropped u-kar) and the i-kar / ii-kar
    tops, reph and chandrabindu above the next line ('করিম' read 'কারম', 'নির্মল' read 'নিমল')."""
    h, w = gray.shape
    bw, photo, small = cells(gray)
    if top_cells:
        small = [c for c in small if c[1] < top_cells * h]
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
    P = (pitch or ENG_PITCH) * h
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
    bands = []
    for a, b in zip(cuts, cuts[1:] + [len(prof)]):
        band = prof[a:b]
        if (band > thr).sum() < 0.3 * P:
            if len(bands) >= 4: break
            continue
        bands.append((a, min(b, len(prof))))
        if len(bands) >= ENG_MAX_LINES: break
    if comp and bands:
        for k, im in enumerate(_assign_lines(gray, body, bands, y_body, xl, P)):
            out[f'line{k}'] = trim(im, dark=ENG_DARK)
        return out
    for a, b in zip(cuts, cuts[1:] + [len(prof)]):
        band = prof[a:b]
        # a text line: enough ink rows (>= 30 % of the pitch) - stamps and stray marks in the empty lower part are thinner
        if (band > thr).sum() < 0.3 * P:
            if k >= 4: break     # empty band after >= 4 lines: end of the text block
            continue
        y0, y1 = y_body + a, y_body + min(b, len(prof))
        if ENG_DESC:
            out[f'line{k}'] = trim(clean_edges(_with_descenders(gray, y0, y1, xl, px, int(ENG_DESC * P))), dark=ENG_DARK)
        else:
            out[f'line{k}'] = trim(clean_edges(gray[y0:y1, xl:px]), dark=ENG_DARK)
        k += 1
        if k >= ENG_MAX_LINES: break
    return out
