#!/usr/bin/env python3
"""OCR of image-only electoral roll PDFs (Hindi and English): SIR rolls and supplements -> one JSONL record per voter box.

States: built on UP; the printed label variants of Bihar, MP, Rajasthan, Chhattisgarh, Uttarakhand, Haryana, Jharkhand, Delhi,
Chandigarh and the Nepali parts of West Bengal are parsed by parse.from_lines (2026-11-25, see parse.py).

Usage:
  eci_ocr.py INPUT.pdf [INPUT2.pdf ...] --out OUT.jsonl [--crnn DIR ...] [--hint-2027 hints.jsonl] [--workers 4] [--lang auto]

Language (2026-11-25, engpipe): --lang auto (default) takes ENG for PDFs named *-ENG-*, else HIN. ENG: segment.segment_eng
(4 labelled lines + wrap lines), Tesseract 'eng' for the text lines, parse_eng (labels Name / Fathers Name / Husbands Name /
Mothers Name / Others / House Number / Age .. Gender), English validators in ensemble2. The CRNNs given with --crnn must be
English models for ENG rolls (charset.json Latin). --crnn none: Tesseract only (with --tess2 a second Tesseract reading as voter).

Pipeline per page (one page = one task in a process pool; ORT and Tesseract use 1 thread per worker):
  1. embedded page image, extracted without re-rendering (pymupdf, else pypdfium2)
  2. layout.find_boxes -> voter boxes; pages without boxes (cover, map, summary) are skipped
  3. segment.segment -> field crops; code pixel test on the serial cell
  4. engines: CRNN (ONNX) on every crop; Tesseract eng (serial/part, EPIC, code) and hin (4 text lines)
     at 3x Lanczos + 24 px pad, the same preprocessing as ocr/engines.py (configs crnn_<run>_line, tess_std_line_x3)
  5. parse.from_lines per engine -> ensemble2.decide (vote, validators, optional 2027 confirmation) -> flags
  6. per PDF: SIR serial numbers must run 1..N without gaps; breaks are flagged
No network calls. Uncertain fields keep the engine value chosen by decide() and are listed in 'flags'.
"""
import argparse, collections, hashlib, json, os, pickle, re, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path[:0] = [BENCH, os.path.join(BENCH, 'ocr')]
os.environ.setdefault('OMP_THREAD_LIMIT', '1')   # Tesseract: one thread per call (parallelism comes from the pool)

import cv2
import numpy as np

from layout import find_boxes, crop                 # noqa: E402
from segment import segment, segment_eng, _mask, cells   # noqa: E402
import segment as _segmod                           # noqa: E402
import tofu as _tofu                                # noqa: E402
from parse import from_lines                        # noqa: E402
import parse_eng                                    # noqa: E402
import parse_tam                                    # noqa: E402
_argv, sys.argv = sys.argv, sys.argv[:1]            # ensemble2 reads sys.argv at import time
import ensemble2                                    # noqa: E402
sys.argv = _argv
from ensemble2 import decide                        # noqa: E402
from score import FIELDS, visual                    # noqa: E402

STD_TESSDATA = '/usr/share/tesseract-ocr/5/tessdata'
TESS_SCALE = 3
# raw crop that each output field comes from (for confidences / readings)
SRC = {'code': 'code', 'serial_no': 'serial', 'part_no': 'part', 'epic': 'epic', 'name': 'line0', 'relation_label': 'line1',
       'relation_name': 'line1', 'house_no': 'line2', 'age': 'line3', 'gender': 'line3'}
OUT_FIELDS = ['code', 'serial_no', 'part_no', 'epic', 'name', 'relation_label', 'relation_name', 'house_no', 'age', 'gender']


# ---------------------------------------------------------------- page images

def page_count(pdf):
    try:
        import pymupdf
        with pymupdf.open(pdf) as d: return len(d)
    except ImportError:
        import pypdfium2 as pdfium
        d = pdfium.PdfDocument(pdf); n = len(d); d.close(); return n


_DOCS = {}


def page_gray(pdf, pno):
    """Largest embedded image of the page as 8-bit gray, pixels as stored in the PDF (no rendering).
    Returns None for a page without an image."""
    try:
        import pymupdf
        d = _DOCS.get(pdf) or _DOCS.setdefault(pdf, pymupdf.open(pdf))
        imgs = d[pno].get_images(full=True)
        if not imgs: return None
        xref = max(imgs, key=lambda i: i[2] * i[3])[0]
        r = d.extract_image(xref)
        im = cv2.imdecode(np.frombuffer(r['image'], np.uint8), cv2.IMREAD_UNCHANGED)
        if im is None:  # exotic encodings (JBIG2, JPX...): let MuPDF decode
            pm = pymupdf.Pixmap(d, xref)
            if pm.n - pm.alpha not in (1, 3): pm = pymupdf.Pixmap(pymupdf.csRGB, pm)
            im = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)
            im = im[..., :3][..., ::-1] if pm.n >= 3 else im[..., 0]
    except ImportError:
        import pypdfium2 as pdfium
        d = _DOCS.get(pdf) or _DOCS.setdefault(pdf, pdfium.PdfDocument(pdf))
        objs = list(d[pno].get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]))
        if not objs: return None
        bms = [o.get_bitmap(render=False) for o in objs]
        bm = max(bms, key=lambda b: b.width * b.height)
        im = bm.to_numpy()
        if bm.mode in ('RGB', 'RGBA'): im = im[..., :3][..., ::-1]  # -> BGR
        elif bm.mode in ('BGRA', 'BGRX'): im = im[..., :3]
    if im.ndim == 3:
        im = cv2.cvtColor(np.ascontiguousarray(im[..., :3]), cv2.COLOR_BGR2GRAY)
    if im.dtype != np.uint8: im = (im / (im.max() or 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(im)


# ---------------------------------------------------------------- engines (same preprocessing as ocr/engines.py)

class TessAPI:
    """Minimal ctypes binding of the libtesseract C API (libtesseract.so.5, Tesseract 5.x)."""
    def __init__(self, tessdata):
        import ctypes, ctypes.util
        self.c = ctypes
        lib = None
        for n in ('libtesseract.so.5', ctypes.util.find_library('tesseract')):
            if not n: continue
            try: lib = ctypes.CDLL(n); break
            except OSError: pass
        if lib is None: raise OSError('libtesseract not found')
        V, P, I, S = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, None
        lib.TessBaseAPICreate.restype = V
        lib.TessBaseAPIInit2.argtypes = [V, P, P, I]
        lib.TessBaseAPISetPageSegMode.argtypes = [V, I]
        lib.TessBaseAPISetVariable.argtypes = [V, P, P]
        lib.TessBaseAPISetImage.argtypes = [V, V, I, I, I, I]
        lib.TessBaseAPIGetUTF8Text.argtypes = [V]; lib.TessBaseAPIGetUTF8Text.restype = V
        lib.TessDeleteText.argtypes = [V]
        lib.TessBaseAPIEnd.argtypes = [V]; lib.TessBaseAPIDelete.argtypes = [V]
        self.lib, self.tessdata, self.apis = lib, tessdata, {}

    def get(self, lang, psm, whitelist):
        k = (lang, psm, whitelist)
        if k not in self.apis:
            a = self.lib.TessBaseAPICreate()
            if self.lib.TessBaseAPIInit2(a, self.tessdata.encode(), lang.encode(), 1) != 0:  # oem 1 = LSTM only
                raise OSError(f'tesseract init failed for {lang} in {self.tessdata}')
            self.lib.TessBaseAPISetVariable(a, b'debug_file', b'/dev/null')  # silence "Estimating resolution" chatter
            self.lib.TessBaseAPISetPageSegMode(a, psm)
            if whitelist: self.lib.TessBaseAPISetVariable(a, b'tessedit_char_whitelist', whitelist.encode())
            self.apis[k] = a
        return self.apis[k]

    def read(self, api, im):
        im = np.ascontiguousarray(im, dtype=np.uint8)
        self.lib.TessBaseAPISetImage(api, im.ctypes.data, im.shape[1], im.shape[0], 1, im.shape[1])
        p = self.lib.TessBaseAPIGetUTF8Text(api)
        if not p: return ''
        s = self.c.string_at(p).decode('utf-8', 'replace'); self.lib.TessDeleteText(p)
        return s.strip()


class Engines:
    def __init__(self, crnn_dirs, tessdata=STD_TESSDATA, ort_threads=1, crnn_sort=False, tess2=False):
        self.crnn = []; self.xs = {}
        if crnn_dirs: import onnxruntime as ort
        for d in crnn_dirs:
            so = ort.SessionOptions(); so.intra_op_num_threads = ort_threads; so.inter_op_num_threads = 1
            # no memory arena (2026-11-23): the arena grows with every new crop width and never shrinks; a GE 2024 part took 2.8-4.4 GB
            # per worker and runner VMs (16 GB, 2 processes x 2 workers) ran out of memory. Off: 0.9 GB per worker, output identical.
            so.enable_cpu_mem_arena = False
            s = ort.InferenceSession(os.path.join(d, 'crnn.onnx'), so, providers=['CPUExecutionProvider'])
            self.crnn.append(('crnn_' + os.path.basename(os.path.normpath(d)), s, s.get_inputs()[0].name,
                              json.load(open(os.path.join(d, 'charset.json')))))
            # job3 round 3: horizontal stretch of the input (arch.json 'xs', trainer j3r3.py; default 1.0 = every older model)
            try: self.xs[self.crnn[-1][0]] = float(json.load(open(os.path.join(d, 'arch.json'))).get('xs', 1.0))
            except (OSError, ValueError): pass
        self.tessdata = tessdata
        self.crnn_sort = crnn_sort
        # job2 (2026-11-25): ECI_DIGIT_CRNN=DIR = a CRNN trained only on line2 (house number) and line3 (age, gender) lines; its
        # readings of those lines take the place of Tesseract's (the 5th voter there). Off when unset.
        self.digit = None
        dd = os.environ.get('ECI_DIGIT_CRNN')
        if dd:
            so = ort.SessionOptions(); so.intra_op_num_threads = ort_threads; so.inter_op_num_threads = 1; so.enable_cpu_mem_arena = False
            s = ort.InferenceSession(os.path.join(dd, 'crnn.onnx'), so, providers=['CPUExecutionProvider'])
            self.digit = ('digit', s, s.get_inputs()[0].name, json.load(open(os.path.join(dd, 'charset.json'))))
        # job2 r3 (ECI_NOHINT_VALSET=DIR,DIR..., default off; only without 2027 hints): extra CRNNs that read ONLY the lines of
        # flagged text fields; the value of such a field = strict majority of these extras + the main engines named in
        # ECI_NOHINT_VALKEEP (comma list of engine names, e.g. crnn_job2_r2_indx0). Flags do not change (the field stays flagged).
        self.vcrnn = []
        for d in filter(None, os.environ.get('ECI_NOHINT_VALSET', '').split(',')):
            so = ort.SessionOptions(); so.intra_op_num_threads = ort_threads; so.inter_op_num_threads = 1; so.enable_cpu_mem_arena = False
            s = ort.InferenceSession(os.path.join(d, 'crnn.onnx'), so, providers=['CPUExecutionProvider'])
            self.vcrnn.append(('crnn_' + os.path.basename(os.path.normpath(d)), s, s.get_inputs()[0].name,
                               json.load(open(os.path.join(d, 'charset.json')))))
        self.valkeep = [x for x in os.environ.get('ECI_NOHINT_VALKEEP', '').split(',') if x]
        try:
            self.tapi = TessAPI(tessdata) if not os.environ.get('ECI_TESS_CLI') else None
        except OSError:
            self.tapi = None
        # tess2 (2026-11-25): a second Tesseract reading of the same crops (x2 Lanczos, Otsu binarised) as one more voter,
        # for runs without CRNNs (English baseline): with a single engine decide() would flag every unconfirmed field
        self.tnames = ['tess', 'tess2'] if tess2 else ['tess']
        self.names = [c[0] for c in self.crnn] + self.tnames
        # reading cache (2026-11-22): the SIR final roll repeats ~94 % of the SIR draft's crops pixel for pixel; readings
        # of a crop are reused by (engine, crop hash). cache_in: read-only (loaded before the fork); cache_out: new readings.
        self.cache_in, self.cache_out, self.collect = {}, {}, False

    # engines.py _crnn_read: pad 5 white, resize to height 48 (cubic, width <= 800), invert, /255, greedy CTC (blank 0)
    @staticmethod
    def _crnn_batch(sess, inp, CH, ims, xst=1.0):
        H = sess.get_inputs()[0].shape[2]   # job2: the model's input height (48 for every production model; 32 px students)
        H = H if isinstance(H, int) and H > 0 else 48; xs = []
        for im in ims:
            im = cv2.copyMakeBorder(im, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
            w = min(int(800 * xst), max(8, int(im.shape[1] * H / im.shape[0] * xst)))
            xs.append((255 - cv2.resize(im, (w, H), interpolation=cv2.INTER_CUBIC)).astype(np.float32) / 255.0)
        W = (max(x.shape[1] for x in xs) + 3) // 4 * 4
        X = np.zeros((len(xs), 1, H, W), np.float32)
        for i, x in enumerate(xs): X[i, 0, :, :x.shape[1]] = x
        Y = sess.run(None, {inp: X})[0]
        Y = np.exp(Y - Y.max(-1, keepdims=True)); Y /= Y.sum(-1, keepdims=True)
        out = []
        for P, x in zip(Y, xs):
            P = P[: x.shape[1] // 4 + 1]; seq = P.argmax(-1); conf = 1.0
            s, prev = [], 0
            for t, k in enumerate(seq):
                if k != prev and k != 0: s.append(CH[k - 1]); conf = min(conf, float(P[t, k]))
                prev = k
            out.append((''.join(s), conf))
        return out

    def crnn_read(self, eng, ims):
        _, sess, inp, CH = eng
        xs = self.xs.get(eng[0], 1.0)
        if not ims: return []
        if not self.crnn_sort or len(ims) == 1:
            return self._crnn_batch(sess, inp, CH, ims, xs)
        order = sorted(range(len(ims)), key=lambda i: ims[i].shape[1] * 48 / max(1, ims[i].shape[0]))  # engines.py SORTW=1
        out = [None] * len(ims)
        for j in range(0, len(order), 64):
            idx = order[j:j + 64]
            for i, r in zip(idx, self._crnn_batch(sess, inp, CH, [ims[i] for i in idx], xs)): out[i] = r
        return out

    @staticmethod
    def tess_prep(im, scale=TESS_SCALE):  # engines.py prep(im, 3): Lanczos x3, pad 8*3 white
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        p = int(8 * scale)
        return cv2.copyMakeBorder(im, p, p, p, p, cv2.BORDER_CONSTANT, value=255)

    @staticmethod
    def tess_prep2(im):  # tess2: Lanczos x2, Otsu threshold, pad 16 white
        im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        _, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.copyMakeBorder(im, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)

    def tess_many(self, ims, lang, psm, whitelist=None, tmp=None):
        """Tesseract over many crops. Default: libtesseract C API, one model instance per (lang, psm, whitelist)
        and worker, kept for the whole run. Output equals one `tesseract IMG stdout --psm P --oem 1 -l L` call per crop
        (checked: 465/465 crops identical) and is 4-50x faster. Fallback: the tesseract CLI, one call per crop."""
        if not ims: return []
        if self.tapi is not None:
            try:
                api = self.tapi.get(lang, psm, whitelist)
                return [self.tapi.read(api, im) for im in ims]
            except OSError:
                self.tapi = None
        cmd_tail = ['--psm', str(psm), '--oem', '1', '--tessdata-dir', self.tessdata, '-l', lang]
        if whitelist: cmd_tail += ['-c', f'tessedit_char_whitelist={whitelist}']
        d = tempfile.mkdtemp(dir=tmp)
        try:
            out = []
            for i, im in enumerate(ims):
                p = os.path.join(d, f'{i:04d}.png'); cv2.imwrite(p, im)
                r = subprocess.run(['tesseract', p, 'stdout'] + cmd_tail, capture_output=True)
                out.append(r.stdout.decode('utf-8', 'replace').strip())
            return out
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def read_boxes(self, segs_list, tmp=None, timing=None, crnn_idx=None, tess_groups=None, lang='HIN'):
        """segs_list: per box {field: crop}. Returns per box {engine_name: (raw_fields, conf_fields)}.
        crnn_idx: which CRNN models to run (default all); tess_groups: which Tesseract groups (line/num/epic/code)."""
        timing = timing if timing is not None else {}
        n = len(segs_list)
        res = [{} for _ in range(n)]
        t = time.time()
        keys, ims = [], []
        for b, segs in enumerate(segs_list):
            for k in sorted(segs):  # same order as the bench (lines.jsonl 'fields' = sorted keys)
                keys.append((b, k)); ims.append(segs[k])
        use_cache = bool(self.cache_in) or self.collect
        hs = [hashlib.md5(im.tobytes() + str(im.shape).encode()).hexdigest() for im in ims] if use_cache else None
        for ci, eng in enumerate(self.crnn):
            if crnn_idx is not None and ci not in crnn_idx: continue
            outs = [self.cache_in.get((eng[0], hs[i])) for i in range(len(ims))] if use_cache else [None] * len(ims)
            for j in range(0, n, 32):  # bench: 32 boxes per batch (BOXB)
                sel = [i for i, (b, _) in enumerate(keys) if j <= b < j + 32 and outs[i] is None]
                for i, r in zip(sel, self.crnn_read(eng, [ims[i] for i in sel])):
                    outs[i] = r
                    if self.collect: self.cache_out[(eng[0], hs[i])] = r
            for (b, k), (txt, c) in zip(keys, outs):
                f, cf = res[b].setdefault(eng[0], ({}, {}))
                f[k] = txt; cf[k] = round(c, 4)
        timing['crnn'] = round(timing.get('crnn', 0) + time.time() - t, 3)
        t = time.time()
        groups = {'line': (TESS_LANG.get(lang, 'hin'), 7, None), 'num': ('eng', 7, '0123456789'),
                  'epic': ('eng', 7, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/'), 'code': ('eng', 7, '#SERMQ0123456789')}
        gk = {g: [] for g in groups}
        for b, k in keys:
            g = 'line' if k.startswith('line') else 'num' if k in ('serial', 'part') else k
            gk[g].append((b, k))
        for tn in self.tnames:
            for b in range(n): res[b][tn] = ({}, {})
        for g, (tl, psm, wl) in groups.items():
            if not gk[g] or (tess_groups is not None and g not in tess_groups): continue
            for tn in self.tnames:
                prep = self.tess_prep if tn == 'tess' else self.tess_prep2
                if use_cache:
                    kidx = {bk: i for i, bk in enumerate(keys)}
                    ck = [(tn + '_' + g + ('_' + tl if g == 'line' and tl != 'hin' else ''), hs[kidx[bk]]) for bk in gk[g]]
                    txts = [self.cache_in.get(c) for c in ck]
                    todo = [i for i, t_ in enumerate(txts) if t_ is None]
                    for i, t_ in zip(todo, self.tess_many([prep(segs_list[gk[g][i][0]][gk[g][i][1]]) for i in todo], tl, psm, wl, tmp)):
                        txts[i] = t_
                        if self.collect: self.cache_out[ck[i]] = t_
                else:
                    txts = self.tess_many([prep(segs_list[b][k]) for b, k in gk[g]], tl, psm, wl, tmp)
                for (b, k), txt in zip(gk[g], txts): res[b][tn][0][k] = txt
        if self.digit is not None and (tess_groups is None or 'line' in tess_groups):
            dk = [(b, k) for b, k in gk['line'] if k in ('line2', 'line3')]
            for (b, k), (txt, c) in zip(dk, self.crnn_read(self.digit, [segs_list[b][k] for b, k in dk])):
                res[b]['tess'][0][k] = txt
        timing['tess'] = round(timing.get('tess', 0) + time.time() - t, 3)
        if _segmod.DASH_IDS:   # lang-tam: a band with only a flat short mark (segment_eng dash=True) is the wrapped ' -': text '-'
            for b, segs in enumerate(segs_list):
                for k, im in segs.items():
                    if id(im) in _segmod.DASH_IDS:
                        for nm in res[b]:
                            res[b][nm][0][k] = '-'
                            if k in res[b][nm][1]: res[b][nm][1][k] = 1.0
        return res


def code_pixel(box_gray, lang='HIN'):
    """Printed status code present: inner left 40% of the serial cell has > 15 ink pixels (bench code_pixel.json).
    ENG (2026-11-25): English SIR pages have ~2x the resolution, and the cell's left rule is wider than the 3 px margin
    (English gold: every first-column box read as '#'): only glyph-like ink counts (components >= 3 px wide,
    >= 25 % of the cell height, not touching the crop edges: thin rule remains are dropped)."""
    _, _, small = cells(box_gray)
    if not small: return False
    x, y, cw, ch = small[0]
    m = _mask(box_gray[y + 3: y + ch - 3, x + 3: x + cw - 3])
    if lang in LATIN_TPL and m.size:   # drop ink that touches the crop edges (remains of the cell's rules)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        H, W = m.shape
        for j in range(1, n):
            x0, y0, w0, h0, _a = st[j]
            if x0 == 0 or y0 == 0 or x0 + w0 >= W or y0 + h0 >= H or h0 > 0.8 * H or w0 < 3 or h0 < 0.25 * H: m[lab == j] = 0
    m = m > 0
    return bool(m[:, : int(0.4 * m.shape[1])].sum() > 15)


# ---------------------------------------------------------------- section headings

LISTS = ['परिवर्धन की सूची', 'विलोपन की सूची', 'संशोधन की सूची']  # supplement lists (additions, deletions, changes)


def heading_strips(gray, boxes):
    """Bands above each row of boxes that hold text (page header / section heading). Returns [(y_top_of_row, band)]."""
    H, W = gray.shape
    rows = []
    for b in boxes:
        if rows and abs(rows[-1][0][1] - b[1]) < 0.02 * H: rows[-1].append(b)
        else: rows.append([b])
    out, prev = [], 0
    for r in rows:
        top = min(b[1] for b in r); x0 = max(0, min(b[0] for b in r) - 10); x1 = min(W, max(b[0] + b[2] for b in r) + 10)
        if top - prev > 0.015 * H:
            band = gray[prev: max(prev, top - 2), x0:x1]
            ink = (band < 128).sum(1)
            rows_ink = np.where(ink > 2)[0]
            if len(rows_ink) > 0.006 * H:
                y0, y1 = max(0, rows_ink[0] - 4), min(band.shape[0], rows_ink[-1] + 5)
                out.append((top, band[y0:y1]))
        prev = max(b[1] + b[3] for b in r)
    return out


def text_lines(band):
    """Cut a heading band into single text lines (ink-row runs >= 6 px), trimmed to their ink."""
    ink = (band < 128).sum(1) > 1
    out, y, n = [], 0, len(ink)
    while y < n:
        if ink[y]:
            y0 = y
            while y < n and ink[y]: y += 1
            if y - y0 >= 6:
                sub = band[max(0, y0 - 2): min(n, y + 2)]
                cols = np.where((sub < 128).sum(0) > 0)[0]
                out.append(sub[:, max(0, cols[0] - 3): cols[-1] + 4])
        y += 1
    return out


# section line: 'अनुभाग संख्या और नाम :' (UP, Haryana), 'अनुभाग संख्या एवं नाम :' (Bihar, CG), 'अनुभाग की संख्या व नाम :'
# (MP, Rajasthan, Jharkhand, Delhi), 'अनुभाग संख्या व नाम' without colon (Uttarakhand, Chandigarh), 'खण्डको संख्या अनि नाँउ'
# without colon (West Bengal, Nepali; c11q reads 'ख्डको ... नाँठ 1-...').
# 2026-11-25 (ocrv5 review): at the scaled 1679 px pages the v4a / j2Ps CRNN reads 'अनुभाग' as 'अनुभान' (S04, S07, S26, S28,
# U02: section empty in every box), the hyphen of '1-' as 'न' / 'न्' ('1न्ग्राम-कालुखेडा' for '1-ग्राम-कालुखेडा', S04, S12),
# and sometimes drops the number ('-जैनपुर खुर्द', 'न्ग्राम मलोया' for '1-...'; S25, S28, U02); a colon read after the word
# ('अनुभाग: संख्या व नाम 3-...') left the label in the value. Checked on 68 PDFs of 10 states (page crops, ocrv5 README).
# The word must start the line (the AC line 'विधान सभा क्षेत्र की संख्या व नाम : 175-खंडवा' is no section). The value is the
# first 'N-name' after the label; without a number it is '?-name' and the per-PDF pass gives it a number (section_no_inferred).
SECTION_WORD = re.compile(r'(?:अनु|प्र)भा[^\s:]*|ख[ण्ं]*ड[^\s:]*')
SECTION_LABEL_END = re.compile(r'\s(?:की\s+)?सं[^\s:]*\s+(?:\S+\s+)?ना[^\s:0-9-]*\s*:?')   # ' (की) संख्या व नाम :'
SECTION_VALUE = re.compile(r'([0-9]+)(?:\s*-|न्?(?=[\u0915-\u0939\u0958-\u095f]))?\s*(.*)')
LOST_HYPHEN = re.compile(r'^(?:-|न्(?=[\u0915-\u0937\u0939\u0958-\u095f])(?<!य)\s*)')    # '-name' / 'न्name' with no number
LEAD_MARKS = '\u093c\u094d\u0901\u0902\u0903'   # a heading read as '1-्गोटेगांव': marks cannot start a name


LISTS_ENG = ['List of Additions', 'List of Deletions', 'List of Modifications']


def parse_heading(lines, lang='HIN'):
    """Heading lines (CRNN text) -> section: 'N-name' for SIR sections ('?-name' when the number was not read), or one of
    LISTS. Last match in the band wins."""
    from rapidfuzz import process, fuzz
    for t in reversed(lines):
        t = re.sub(r'\s+', ' ', t).strip()
        if lang == 'TAM':   # 'பிரிவு எண் மற்றும் பெயர் : 1-<name>' (Tesseract tam)
            if ':' in t and re.search(r'பிரிவு|எண்', t):
                v = t.split(':', 1)[1].strip(' :.-')
                m = re.match(r'([0-9]+)\s*-?\s*(.*)', v)
                if m: return f'{m.group(1)}-{m.group(2).strip()}'
            continue
        if lang == 'ENG':   # 'Section No and Name : 1-Lozar ,Aldona'
            m = re.search(r'Sect\w*\s+No\.?\s+and\s+Name\s*[:;.]?\s*([0-9]+)\s*-?\s*(.*)', t)
            if m: return f'{m.group(1)}-{m.group(2).strip()}'
            m = process.extractOne(' '.join(t.split()[:3]), LISTS_ENG, scorer=fuzz.ratio)
            if m and m[1] >= 75: return m[0]
            continue
        w = SECTION_WORD.search(t)
        if w and w.start() <= 3:
            rest = t[w.end():]
            m = SECTION_VALUE.search(rest)
            if m and m.group(2).lstrip(LEAD_MARKS).strip(' :.-'):
                return f'{m.group(1)}-{m.group(2).lstrip(LEAD_MARKS).strip()}'
            e = SECTION_LABEL_END.match(rest)
            v = (rest[e.end():] if e else rest.split(':', 1)[1] if ':' in rest else '').strip(' :.')
            v = LOST_HYPHEN.sub('', v).lstrip(LEAD_MARKS).strip(' :.-')
            if v: return f'?-{v}'
            continue
        m = process.extractOne(' '.join(t.split()[:3]), LISTS, scorer=fuzz.ratio)
        if m and m[1] >= 75: return m[0]
        lw = list_word(t)
        if lw: return lw
    return None


# 2026-11-25 (ocrv5 review 3): the final rolls of Bihar, MP, Rajasthan and Chhattisgarh head the addition list
# 'घटक I - परिवर्धन सूची 1 (dates)'; the CRNN reads 'पटका : परिवर्धन सूची 1 /23-12-2025 से 21-02-2026 तक' (S12, S20, S26) or
# 'पटक ा : परिवर्धन सूची 1 /01-08-2025 ...' (S04). The 3-word match above takes only 'परिवर्धन की सूची' at the line start (UP,
# Uttarakhand, Chandigarh), so every addition record kept the last main-list section (review: 71 of 71 addition records).
# A list word in the first 6 words, followed within 2 words by 'सूची', is the list heading.
LIST_WORDS = {'परिवर्धन': LISTS[0], 'विलोपन': LISTS[1], 'संशोधन': LISTS[2]}


def list_word(t):
    from rapidfuzz import process, fuzz
    w = t.split()[:6]
    for i, x in enumerate(w):
        m = process.extractOne(x, list(LIST_WORDS), scorer=fuzz.ratio)
        if m and m[1] >= 80 and any(fuzz.ratio(y, 'सूची') >= 75 for y in w[i + 1: i + 3]):
            return LIST_WORDS[m[0]]
    return None


def number_sections(recs):
    """'?-name' sections of one PDF (records in reading order) get a number: the previous section's when the name is the same
    (fuzzy, the CRNN reads the same heading a little differently on each page), else the previous number + 1 (1 at the
    start). Marked section_no_inferred. Returns the number of records changed."""
    from rapidfuzz import fuzz
    prev_no, prev_name, raw, val, n = None, None, None, None, 0
    for r in recs:
        s = r.get('section')
        if s != raw:
            raw, val = s, None
            m = re.match(r'([0-9]+)-(.*)', s or '')
            if m:
                prev_no, prev_name = int(m.group(1)), m.group(2)
            elif s and s.startswith('?-'):
                name = s[2:]
                no = prev_no if prev_no is not None and prev_name and fuzz.ratio(name, prev_name) >= 85 else (prev_no or 0) + 1
                val, prev_no, prev_name = f'{no}-{name}', no, name
        if val:
            r['section'], r['section_no_inferred'] = val, True
            n += 1
    return n


# ---------------------------------------------------------------- one page

_E = None
_OPT = {}


RESCALE_H = int(os.environ.get('ECI_RESCALE_H', '1679'))   # 0 = off; see process_page
# 2026-11-25 (ocrv5 review 3), both default off (UP output unchanged), on in the ocr-any model sets; applied by post_rules() after
# the not_voter_box rule, so they cannot turn a voter box into page noise and do not change the cascade:
# ECI_CODE_VALID=1: decide() takes Tesseract's code without a check; a code that is not '#' or S E R Q M (with an optional digit,
# as '#2', 'R3' in the UP gold) is flagged 'code' (Bihar final rolls: '#' read '7', 'S' read 'SS', unflagged).
# ECI_PARTNO_LIST=1: in an SIR roll (file name '-SIR-') a box has a part_no cell only in the supplement lists (section = one of
# LISTS); outside them part_no is junk read from the serial cell ('91', '14118', '1119'): the value moves to
# 'part_no_outside_list', part_no = '', flagged. Not for other rolls: the GE 2024 rolls print supplement boxes with a part_no
# under headings that parse_heading does not know (GE gold run: 548 such boxes with section None).
CODE_VALID = os.environ.get('ECI_CODE_VALID', '0') == '1'
PARTNO_LIST = os.environ.get('ECI_PARTNO_LIST', '0') == '1'
CODE_RE = re.compile(r'(?:#|[SERQM])[0-9]?')
_CACHE_IN, _COLLECT = {}, False   # reading cache: loaded before the fork / collect new readings (--cache-in/--cache-out)
_EPIC_IDX = None
ENG_AGREE = int(os.environ.get('ECI_ENG_AGREE', '0') or 0)   # see ensemble2.agree_eng and sequence_confirm
ENG_S1_TESS = set(filter(None, os.environ.get('ECI_ENG_S1_TESS', 'num,epic,code,line').split(',')))
# ENG_SEQ (job3 round 3, 2026-11-25): English stage 1 reads the serial with the first CRNN only when the CRNN serials of the
# page run in sequence around the box (prev + 1 == s == next - 1; page edges: the two next / two previous boxes); that reading
# also stands for the Tesseract vote. Other boxes get the Tesseract serial as before. Default off.
ENG_SEQ = os.environ.get('ECI_ENG_SEQ', '0') == '1'
# lang-tam (2026-11-25): Tamil SIR rolls (S22, U07) use the English SIR box template: segment_eng, the ENG code-cell test, Tesseract
# 'tam' for the text lines, parse_tam (Tamil labels / genders); everything else as ENG.
LATIN_TPL = ('ENG', 'TAM')
TESS_LANG = {'ENG': 'eng', 'TAM': 'tam'}
PARSERS = {'ENG': (parse_eng, parse_eng.from_lines_eng), 'TAM': (parse_tam, parse_tam.from_lines_tam)}
_HINTS = None   # loaded once in the parent before the fork: the workers share it (a whole AC is ~1.5 GB as dicts)


def _init(crnn_dirs, tessdata, crnn_sort, hints_path, tmp, ort_threads=1, stage1=0, confvote=0.6, tess2=False):
    global _E, _OPT
    cv2.setNumThreads(1)
    _E = Engines(crnn_dirs, tessdata, ort_threads, crnn_sort, tess2)
    _E.cache_in, _E.collect = _CACHE_IN, _COLLECT
    _OPT = dict(tmp=tmp, hints=_HINTS if _HINTS is not None else (load_hints(hints_path) if hints_path else {}), stage1=stage1)
    ensemble2.CONFW = confvote if confvote >= 0 else None


def load_hints(path):
    H = {}
    for l in open(path, encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            if r.get('epic'): H[r['epic']] = r
    return H


NOHINT_CASCADE = int(os.environ.get('ECI_NOHINT_CASCADE', '0') or 0)
_CASC_TXT = ('name', 'relation_label', 'relation_name', 'house_no', 'age', 'gender')


def _stage1_agree(cs):
    c1 = cs[0]
    if not all(visual(c[f]) == visual(c1[f]) for c in cs[1:] for f in _CASC_TXT): return False
    if not all(ensemble2.valid(f, c1[f]) for f in _CASC_TXT): return False
    h = c1['house_no']
    return any(ch.isdigit() for ch in h) and not any(ch in h for ch in '.,;:')


DUMP = os.environ.get('ECI_DUMP', '0') == '1'
# job4: English rolls without 2027 hints -> ensemble2.decide_eng_nohint + 2-CRNN stage 1 (needs --stage1 2) + per-PDF EPIC rule
ENG_NOHINT = os.environ.get('ECI_ENG_NOHINT', '0') == '1'
ENG_LEX = os.environ.get('ECI_ENG_LEX', '0') == '1'   # job4 r2: name lexicon guard (eng_lex_guard)
ENG_UNAN = int(os.environ.get('ECI_ENG_UNAN', '0') or 0)   # job4 r3: accept CRNN-unanimous fields without Tesseract support (eng_unan)
_ENG_LEX = None


def lang_of_pdf_is_eng(recs):
    return any(r.get('lang') == 'ENG' for r in recs)
CASCADE_NOHINT = os.environ.get('ECI_CASCADE_NOHINT', '0') == '1'


def decide_box(per_engine, names, pix, hints, stamp=None, lang='HIN', stage1=False):
    """per_engine: {name: (raw_fields, crnn_conf)} -> record fields, flags, readings for flagged fields."""
    ensemble2.LANG = lang
    if lang in LATIN_TPL:
        # English / Tamil: line roles voted over all engines, then every engine parsed with the same roles (parse_eng / parse_tam)
        P, from_l = PARSERS[lang]
        nl = max([int(k[4:]) + 1 for n in names for k in per_engine[n][0] if k.startswith('line')], default=0)
        roles = P.box_roles([per_engine[n][0] for n in names], nl)
        per_engine = {n: (P.canonical(per_engine[n][0], roles), P.canonical_conf(per_engine[n][1], roles)) for n in names}
        cands = [from_l(per_engine[n][0]) for n in names]
    else:
        cands = [from_lines(per_engine[n][0]) for n in names]
    epics = [e for e, _ in collections.Counter(c['epic'] for c in cands).most_common() if e]
    hint = next((hints[e] for e in epics if e in hints), None)
    if lang in LATIN_TPL and ENG_NOHINT and hint is None:   # lang-tam: Tamil too
        out, flags = ensemble2.decide_eng_nohint(cands, names, pix, stamp, stage1=stage1)
    else:
        out, flags = decide(cands, hint, pix, names, [per_engine[n][1] for n in names], stamp=stamp)
    if DUMP: decide_box.confs = {n: per_engine[n][1] for n in names if per_engine[n][1]}   # canonical line confidences
    readings = {}
    for f in (flags if not os.environ.get('ECI_READINGS_ALL') else list(flags) + [f for f in OUT_FIELDS if f not in flags and f in SRC]):   # debug: readings of every field
        readings[f] = {n: c[f] for n, c in zip(names, cands)}
        if hint and f in ensemble2.HINT_FIELDS + ['age'] and f in hint: readings[f]['hint_2027'] = hint[f]
        conf = {n: per_engine[n][1].get(SRC[f]) for n in names if n.startswith('crnn') and SRC[f] in per_engine[n][1]}
        if conf: readings[f]['crnn_conf'] = conf
    return out, flags, readings, hint is not None, cands


TESS_SKIP = os.environ.get('ECI_NOHINT_TESS_SKIP', '0') != '0'


def _crnn_same(cs, f):
    v = cs[0][f]
    return bool(v) and ensemble2.valid(f, v) and all(visual(c[f]) == visual(v) for c in cs[1:])


_VALF = ('name', 'relation_label', 'relation_name', 'house_no', 'age', 'gender')
# job5 r3 (ECI_NOHINT_VALF=fields, default = all of _VALF): fields whose flagged value the VALSET extras may re-choose. The job2
# VALSET CRNNs have no '(' ')' in their charsets; on Himachal bracket house numbers (13(1)) they would out-vote the bracket
# reading of the j5r2 CRNNs (the field stays flagged; the record would be wrong).
_VALF = tuple(x for x in os.environ.get('ECI_NOHINT_VALF', ','.join(_VALF)).split(',') if x)
_VALGUARD = os.environ.get('ECI_NOHINT_VALGUARD', '0') == '1'


def _valset(recs, segs_list):
    """job2 r3: value of flagged text fields from the ECI_NOHINT_VALSET extras + ECI_NOHINT_VALKEEP engines (see Engines)."""
    need = []  # (record index, line key)
    for i, r in enumerate(recs):
        for ln in sorted({SRC[f] for f in r['flags'] if f in _VALF}):
            if ln in segs_list[i]: need.append((i, ln))
    if not need: return
    ext = {}
    for eng in _E.vcrnn:
        for (i, ln), (txt, _) in zip(need, _E.crnn_read(eng, [segs_list[i][ln] for i, ln in need])):
            ext.setdefault(i, {}).setdefault(eng[0], {})[ln] = txt
    for i, per in ext.items():
        r = recs[i]
        for f in r['flags']:
            if f not in _VALF: continue
            rd = r['readings'].get(f, {})
            vs = [rd[n] for n in _E.valkeep if isinstance(rd.get(n), str) and rd[n]]
            for n, lines in per.items():
                if SRC[f] in lines:
                    v = from_lines(lines)[f]; rd[n] = v
                    if v: vs.append(v)
            if not vs: continue
            c = collections.Counter(visual(v) for v in vs); top, k = c.most_common(1)[0]
            if f == 'house_no' and ensemble2.NOHINT_AMP and '&' in (r[f] or ''): continue   # the '&' choice of decide() stays
            if _VALGUARD and r[f]:
                # job5 r4 (ECI_NOHINT_VALGUARD=1, default off): the extras may not replace a value that a strict majority of
                # the main CRNN voters read (gold: 3 of 4 voters + Tesseract read घांधु, the 3 job2 extras घांथु).
                mv = [rd.get(c[0]) for c in _E.crnn]
                if sum(isinstance(v, str) and visual(v) == visual(r[f]) for v in mv) * 2 > len(mv): continue
            if k * 2 > len(vs): r[f] = next(v for v in vs if visual(v) == top)
        if ensemble2.NOHINT_NUNF:  # job2 r4: clear a name flag when every CRNN (voters + extras) agrees on a common value (ensemble2)
            for f in [f for f in r['flags'] if f in ('name', 'relation_name')]:
                rd = r['readings'].get(f, {})
                cr = [v for n, v in rd.items() if n.startswith('crnn') and isinstance(v, str)]
                if not cr or not cr[0] or len({visual(v) for v in cr}) != 1: continue
                if ensemble2.marks_strip(rd.get('tess', '')) == ensemble2.marks_strip(cr[0]): continue
                if all(ensemble2.lex_count_ge(w, ensemble2.NOHINT_NUNF) for w in visual(cr[0]).split()) and not ensemble2.markvar_suspect(cr[0]):
                    r[f] = cr[0]; r['flags'].remove(f); r['readings'].pop(f, None)


# job2 2026-11-25 (ECI_TOFU=1, default off): missing-glyph boxes after names (Haryana SIR: the roll font has no glyph for a
# character stored after many names, the print shows an empty rectangle; 26 of 90 S07 ACs, 8.5 % of the boxes) are found on the
# page image as stored (before the rescale), painted white there before the boxes are cut, and named in the record:
# 'missing_glyph' = {field: count} (eciocr/ocr/tofu.py). Only boxes inside a voter box's text column; without such boxes the
# output is unchanged.
TOFU = os.environ.get('ECI_TOFU', '0') == '1'


def process_page(task):
    pdf, pno, kind, lang = task
    if ensemble2.EPIC_STATE_ON:   # job5: the state's code of old-format EPICs (ensemble2._epic_state)
        ensemble2.EPIC_STATE = ensemble2.STATE_EPIC_CODE.get(pdf_meta(pdf).get('state'))
    t0 = time.time(); tm = {}
    gray = page_gray(pdf, pno)
    tm['extract'] = round(time.time() - t0, 3)
    if gray is None: return dict(pdf=pdf, page=pno, boxes=0, skip='no image', timing=tm)
    tofus = _tofu.find_tofu(gray) if TOFU else []
    gray0 = gray
    scale = 1.0
    if RESCALE_H and gray.shape[0] > RESCALE_H * 1.25:
        # 2026-11-25: Uttarakhand, Haryana, Jharkhand, Delhi and Chandigarh SIR pages are 2806 px high (240 dpi; UP: 1679 px,
        # 144 dpi). At 240 dpi the EPIC and line crops hold the cell borders and much white space, the CRNNs read e.g.
        # 'UPI127760' for URG1127760 and the code pixel test fires on every left-column box. Pages above 1.25 x 1679 px are
        # scaled to 1679 px (INTER_AREA) before layout; bbox in the output is in the scaled page (record 'page_scale').
        scale = RESCALE_H / gray.shape[0]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    t = time.time()
    boxes = find_boxes(gray)
    tm['layout'] = round(time.time() - t, 3)
    if not boxes: return dict(pdf=pdf, page=pno, boxes=0, skip='no voter boxes', timing=tm, size=gray.shape[::-1])
    tofu_at = {}
    if tofus:
        keep, tofu_at = _tofu.select(tofus, boxes, scale)
        if keep:
            gray = _tofu.erase(gray0, keep)
            if scale != 1.0: gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    t = time.time()
    box_grays = [crop(gray, b, 2) for b in boxes]
    segs_list, stamps = [], []
    _segmod.DASH_IDS.clear()
    for g in box_grays:  # segment.LAST_STAMP: DELETED stamp found (only checked for rolls before 2026, see segment.STAMP_CHECK)
        if lang in LATIN_TPL:   # ENG, TAM: the English SIR box template
            segs_list.append({k: im for k, im in segment_eng(g, dash=lang == 'TAM').items() if im.size}); stamps.append(False)
        else:
            segs_list.append({k: im for k, im in segment(g).items() if im.size}); stamps.append(bool(_segmod.LAST_STAMP))
    pix = [code_pixel(g, lang) for g in box_grays]
    tm['segment'] = round(time.time() - t, 3)
    t = time.time()
    heads = []
    for y, band in heading_strips(gray, boxes):  # section headings: CRNN reads them (Tesseract hin misreads the Latin numbers)
        L = [l for l in text_lines(band) if l.shape[1] < 0.5 * gray.shape[1]]  # headings are short; skip the title line
        if not L: continue
        if lang in LATIN_TPL or not _E.crnn:   # English / Tamil headings: Tesseract reads the print well (and runs without CRNNs)
            heads.append((y, parse_heading(_E.tess_many([_E.tess_prep(l, 2) for l in L], TESS_LANG.get(lang, 'hin'), 7), lang)))
        else:
            heads.append((y, parse_heading([t for t, _ in _E.crnn_read(_E.crnn[0], L)])))
    tm['heading'] = round(time.time() - t, 3)
    K = _OPT.get('stage1', 0)
    if lang in LATIN_TPL and ENG_NOHINT and not _OPT['hints'] and K >= 2:
        # job4 cascade: stage 1 = the first 2 CRNNs (+ Tesseract code only); stage 2 = other CRNNs + all Tesseract groups for
        # boxes where the 2 CRNNs do not agree on every field
        reads = _E.read_boxes(segs_list, _OPT['tmp'], tm, crnn_idx={0, 1}, tess_groups={'code'}, lang=lang)
        names1 = _E.names[:2] + _E.tnames
        need = [i for i, r in enumerate(reads) if decide_box(r, names1, pix[i], {}, stamps[i], lang, stage1=True)[1]]
        if need:
            more = _E.read_boxes([segs_list[i] for i in need], _OPT['tmp'], tm, crnn_idx=set(range(2, len(_E.crnn))),
                                 tess_groups={'line', 'num', 'epic'}, lang=lang)
            for i, r2 in zip(need, more):
                for tn in _E.tnames:
                    t1 = reads[i][tn]; t2 = r2.pop(tn)
                    t1[0].update(t2[0])
                reads[i].update(r2)
        tm['stage2_boxes'] = len(need)
        stage1_only = set(range(len(reads))) - set(need)
        K = -2   # marker: names of stage-1 boxes = the first 2 CRNNs, decided with stage1=True
    # job4 (2026-11-25): ECI_CASCADE_NOHINT=1 runs the cascade without 2027 hints too: stage 2 only for boxes stage 1 flags
    elif K and (_OPT['hints'] or CASCADE_NOHINT):
        # cascade: stage 1 = first K CRNNs + Tesseract Latin fields; the rest only for boxes stage 1 cannot settle
        # ENG: the Tesseract text lines are read in stage 1 too (the English CRNNs are not yet trusted alone; the roles of
        # the wrap lines come from all engines)
        # ENG_S1_TESS (job3, 2026-11-25): Tesseract groups of stage 1 for English rolls (env ECI_ENG_S1_TESS, comma list of
        # num/epic/code/line; default all four = engpipe behaviour). With trusted English CRNNs, 'num,code' keeps Tesseract
        # text lines (95 % of the Tesseract time) for the stage-2 boxes only.
        s1t = ENG_S1_TESS if lang in LATIN_TPL else {'num', 'epic', 'code'}
        seq_ok = set()
        if lang == 'ENG' and ENG_SEQ and 'num' in s1t and _E.crnn:
            reads = _E.read_boxes(segs_list, _OPT['tmp'], tm, crnn_idx=set(range(K)), tess_groups=s1t - {'num'}, lang=lang)
            c0 = _E.names[0]
            sv = [reads[i][c0][0].get('serial', '') if 'part' not in segs_list[i] else '' for i in range(len(reads))]
            sv = [int(v) if v.isdigit() and len(v) <= 5 else None for v in sv]
            n_ = len(sv)
            for i, v in enumerate(sv):
                if v is None: continue
                if 0 < i < n_ - 1: ok = sv[i - 1] == v - 1 and sv[i + 1] == v + 1
                elif i == 0: ok = n_ > 2 and sv[1] == v + 1 and sv[2] == v + 2
                else: ok = n_ > 2 and sv[i - 1] == v - 1 and sv[i - 2] == v - 2
                if ok: seq_ok.add(i)
            for i in seq_ok:
                for tn in _E.tnames: reads[i][tn][0]['serial'] = str(sv[i])
            rest = [i for i in range(n_) if i not in seq_ok and any(k in ('serial', 'part') for k in segs_list[i])]
            if rest:
                more = _E.read_boxes([segs_list[i] for i in rest], _OPT['tmp'], tm, crnn_idx=set(), tess_groups={'num'}, lang=lang)
                for i, r2 in zip(rest, more):
                    for tn in _E.tnames: reads[i][tn][0].update(r2[tn][0])
            tm['seq_serial'] = len(seq_ok)
        else:
            reads = _E.read_boxes(segs_list, _OPT['tmp'], tm, crnn_idx=set(range(K)), tess_groups=s1t, lang=lang)
        names1 = _E.names[:K] + _E.tnames
        need = []
        for i, r in enumerate(reads):
            _, fl, _, hinted, _ = decide_box(r, names1, pix[i], _OPT['hints'], stamps[i], lang)
            if fl or (not hinted and not CASCADE_NOHINT): need.append(i)
        if need:
            more = _E.read_boxes([segs_list[i] for i in need], _OPT['tmp'], tm, crnn_idx=set(range(K, len(_E.crnn))),
                                 tess_groups=({'num', 'epic', 'code', 'line'} - s1t) if lang in LATIN_TPL else {'line'}, lang=lang)
            for i, r2 in zip(need, more):
                for tn in _E.tnames:
                    t1 = reads[i][tn]; t2 = r2.pop(tn)
                    t1[0].update(t2[0])
                reads[i].update(r2)
        tm['stage2_boxes'] = len(need)
        stage1_only = set(range(len(reads))) - set(need)
    elif NOHINT_CASCADE and not _OPT['hints'] and len(_E.crnn) > NOHINT_CASCADE:
        # job2 (no 2027 roll, ECI_NOHINT_CASCADE=N, default 0 = off): stage 1 = first N CRNNs + Tesseract Latin fields; a box whose
        # N stage-1 CRNNs read the same valid text fields (house_no with a digit, without . , ; :) takes them unflagged; the
        # other engines (rest of the CRNNs + Tesseract lines) read only the remaining boxes
        reads = _E.read_boxes(segs_list, _OPT['tmp'], tm, crnn_idx=set(range(NOHINT_CASCADE)), tess_groups={'num', 'epic', 'code'})
        need = [i for i, r in enumerate(reads) if not _stage1_agree([from_lines(r[n][0]) for n in _E.names[:NOHINT_CASCADE]])]
        if need:
            more = _E.read_boxes([segs_list[i] for i in need], _OPT['tmp'], tm, crnn_idx=set(range(NOHINT_CASCADE, len(_E.crnn))), tess_groups={'line'})
            for i, r2 in zip(need, more):
                t1 = reads[i]['tess']; t2 = r2.pop('tess')
                t1[0].update(t2[0]); reads[i].update(r2)
        tm['stage2_boxes'] = len(need)
        stage1_only = set(range(len(reads))) - set(need)
        K = NOHINT_CASCADE
    elif TESS_SKIP and not _OPT['hints']:
        # job2 r3 (ECI_NOHINT_TESS_SKIP=1, no 2027 roll, needs ECI_NOHINT_DIGITS=1 + ECI_NOHINT_CLOSED=1): Tesseract does not read
        # line2 (house) when all CRNNs read the same house number with a digit and without . , ; :, nor line3 (age, gender) when
        # all CRNNs read the same valid age and gender: decide() takes the CRNN value there whatever Tesseract reads, so the
        # output is the same and the Tesseract time of those lines is saved.
        reads = _E.read_boxes(segs_list, _OPT['tmp'], tm, tess_groups={'num', 'epic', 'code'})
        crn = [n for n in _E.names if n.startswith('crnn')]
        sub = []
        for i, r in enumerate(reads):
            cs = [from_lines(r[n][0]) for n in crn]
            skip = set()
            if _crnn_same(cs, 'house_no'):
                h = cs[0]['house_no']
                if any(ch.isdigit() for ch in h) and not any(ch in h for ch in '.,;:'): skip.add('line2')
            if _crnn_same(cs, 'age') and _crnn_same(cs, 'gender'): skip.add('line3')
            sub.append({k: im for k, im in segs_list[i].items() if k.startswith('line') and k not in skip})
        tm['tess_lines_skipped'] = sum(sum(k.startswith('line') for k in segs_list[i]) - len(sub[i]) for i in range(len(sub)))
        more = _E.read_boxes(sub, _OPT['tmp'], tm, crnn_idx=set(), tess_groups={'line'})
        for i, r2 in enumerate(more):
            reads[i]['tess'][0].update(r2['tess'][0])
        stage1_only = set()
    else:
        reads = _E.read_boxes(segs_list, _OPT['tmp'], tm, lang=lang)
        stage1_only = set()
    t = time.time()
    if kind is None: kind = 'SUP' if sum('part' in s for s in segs_list) * 2 > len(segs_list) else 'SIR'
    recs = []
    section = None
    for i, (b, r) in enumerate(zip(boxes, reads)):
        for y, h in heads:
            if y <= b[1] + 5 and h: section = h
        if K == -2:
            nm = (_E.names[:2] + _E.tnames) if i in stage1_only else _E.names
            out, flags, readings, hinted, cands = decide_box(r, nm, pix[i], _OPT['hints'], stamps[i], lang, stage1=i in stage1_only)
        else:
            nm = (_E.names[:K] + _E.tnames) if i in stage1_only else _E.names
            out, flags, readings, hinted, cands = decide_box(r, nm, pix[i], _OPT['hints'], stamps[i], lang)
        if K == -2 and ensemble2.NOHINT_SUPPORT and i not in stage1_only:
            readings['_nohint_crnn_unanimous'] = sorted(ensemble2.NOHINT_SUPPORT)
        if NOHINT_CASCADE and not _OPT['hints'] and i in stage1_only:   # the stage-1 CRNN pair settled the text fields
            for f in _CASC_TXT:
                out[f] = cands[0][f]
                if f in flags: flags.remove(f); readings.pop(f, None)
        if kind == 'SUP' and 'part_no' not in flags:
            pv = {visual(c['part_no']) for c in cands}
            if len(pv) > 1 or not out['part_no'].isdigit(): flags.append('part_no'); readings['part_no'] = {n: c['part_no'] for n, c in zip(nm, cands)}
        # engocr (2026-11-25): 'lang' only on English / Tamil records (engpipe wrote it on every record; Hindi records stay as on main)
        rec = dict(page=pno + 1, box=i, kind=kind, **({'lang': lang} if lang != 'HIN' else {}), section=section, **{f: out[f] for f in OUT_FIELDS})
        if CODE_VALID and out['code'] and not CODE_RE.fullmatch(out['code']) and 'code' not in flags:
            rec['code_invalid'] = True      # flag added by post_rules (after the not_voter_box rule)
            readings['code'] = {n: c['code'] for n, c in zip(nm, cands)}
        rec.update(flags=flags, readings=readings, hint_2027=hinted, code_ink=pix[i], bbox=list(map(int, b)))
        if tofu_at.get(i): rec['missing_glyph'] = tofu_at[i]
        if scale != 1.0: rec['page_scale'] = round(scale, 4)
        if K and _OPT['hints'] and i in seq_ok: rec['serial_s1_seq'] = True
        if DUMP:   # job4: every engine's parsed fields + CRNN line confidences for every box (offline rule studies)
            rec['cands'] = {n: c for n, c in zip(nm, cands)}
            rec['confs'] = decide_box.confs
        recs.append(rec)
    tm['decide'] = round(time.time() - t, 3)
    if _E.vcrnn and not _OPT['hints']:
        t = time.time(); _valset(recs, segs_list); tm['valset'] = round(time.time() - t, 3)
    tm['total'] = round(time.time() - t0, 3)
    for rec in recs: rec['t_box_s'] = round(tm['total'] / len(recs), 3)
    out = dict(pdf=pdf, page=pno, boxes=len(boxes), kind=kind, records=recs, timing=tm, size=gray.shape[::-1])
    if _E.collect:
        out['cache'] = _E.cache_out; _E.cache_out = {}
    return out


# ---------------------------------------------------------------- per-PDF checks

def post_rules(recs, sir_roll):
    """ECI_CODE_VALID / ECI_PARTNO_LIST (see the switches above) on the records of one PDF, after sections are final and after
    the not_voter_box rule. sir_roll: the PDF is an SIR roll ('-SIR-' in its file name). Idempotent. Returns {rule: records
    changed}."""
    n = collections.Counter()
    for r in recs:
        if r.get('code_invalid') and 'code' not in r['flags']:
            r['flags'].append('code'); n['code_invalid'] += 1
        if PARTNO_LIST and sir_roll and r.get('kind') == 'SIR' and r.get('part_no') and r.get('section') not in LISTS:
            r['part_no_outside_list'] = r['part_no']; r['part_no'] = ''
            r.setdefault('readings', {})['part_no'] = {'outside_list': r['part_no_outside_list']}
            if 'part_no' not in r['flags']: r['flags'].append('part_no')
            n['part_no_outside_list'] += 1
    return dict(n)


def sequence_confirm(recs):
    """SIR: a serial_no flagged only by engine disagreement is confirmed when BOTH neighbours in reading order
    agree (prev + 1 == serial == next - 1; first record: serial 1 and next 2). Returns the number confirmed."""
    s = [int(r['serial_no']) if r['serial_no'].isdigit() else None for r in recs]
    n = 0
    for i, r in enumerate(recs):
        if 'serial_no' not in r['flags'] or s[i] is None: continue
        prev_ok = (s[i - 1] is not None and s[i - 1] + 1 == s[i]) if i > 0 else s[i] == 1
        next_ok = (s[i + 1] is not None and s[i + 1] - 1 == s[i]) if i + 1 < len(s) else prev_ok and i > 0
        if (ENG_AGREE and r.get('lang') == 'ENG' and i + 2 < len(s) and (i == 0 or recs[i - 1]['page'] < r['page'] - 1)
                and s[i + 1] is not None and s[i + 2] is not None and s[i + 1] - 1 == s[i] == s[i + 2] - 2
                and recs[i + 2]['page'] == r['page']):
            # job3 (2026-11-25, ECI_ENG_AGREE): first record after pages that were not read (--pages): the next two records
            # of the same page confirm it (no previous neighbour exists)
            prev_ok = next_ok = True
        if prev_ok and next_ok:
            r['flags'].remove('serial_no'); r['readings'].pop('serial_no', None)
            r['serial_by_sequence'] = True; n += 1
    return n


def eng_epic_prefix(recs, min_other=3):
    """job4 (English, no hints): an EPIC flagged only because no Tesseract reading confirmed it (all CRNNs read the same
    valid EPIC) is accepted when its 3-letter prefix is the prefix of >= 3 OTHER unflagged EPICs of the same PDF."""
    pf = collections.Counter(r['epic'][:3] for r in recs if 'epic' not in r['flags'] and ensemble2.valid('epic', r['epic'] or ''))
    n = 0
    for r in recs:
        un = r.get('readings', {}).get('_nohint_crnn_unanimous', [])
        if 'epic' in r['flags'] and 'epic' in un and pf[r['epic'][:3]] >= min_other:
            r['flags'].remove('epic'); r['readings'].pop('epic', None); r['epic_by_prefix'] = True; n += 1
    if not ENG_UNAN:
        for r in recs:
            r.get('readings', {}).pop('_nohint_crnn_unanimous', None)
    return n


def _lex():
    global _ENG_LEX
    if _ENG_LEX is None:
        import gzip
        _ENG_LEX = json.load(gzip.open(os.path.join(os.path.dirname(os.path.abspath(ensemble2.__file__)), 'eng_name_lex.json.gz'), 'rt'))['counts']
    return _ENG_LEX


def eng_unan(recs, lexmin=None, pmin=2):
    """job4 r3 (English, no hints; ECI_ENG_UNAN=N): a field flagged ONLY because no Tesseract reading confirmed it (stage 2: every
    CRNN, N or more of them, read the same valid value; '_nohint_crnn_unanimous') is accepted when an independent check holds:
      name / relation_name: every token of >= 2 letters is common in the non-gold name lexicon (count >= ECI_ENG_UNAN_LEX,
        default 5) or is read in >= pmin OTHER unflagged name fields of the same PDF;
      house_no: the same value is the unflagged house number of the record before or after it (families share a house);
      age: 18..120 and every CRNN agrees (closed numeric range, plausibility already checked);
      gender / relation_label: closed vocabularies (every CRNN agrees).
    Cause: West Bengal SIR final rolls print a grey 'UNDER ADJUDICATION' stamp over the text; Tesseract reads nothing useful
    there, stamp-trained CRNNs do. Returns the number of fields accepted."""
    lexmin = lexmin or int(os.environ.get('ECI_ENG_UNAN_LEX', '5'))
    C = _lex(); tok = re.compile(r'[A-Za-z]+'); n = 0
    pc = collections.Counter(t.lower() for r in recs for k in ('name', 'relation_name') if k not in r['flags']
                             for t in set(tok.findall(r.get(k) or '')))
    for i, r in enumerate(recs):
        un = r.get('readings', {}).pop('_nohint_crnn_unanimous', None)
        if not un: continue
        for f in un:
            if f not in r['flags']: continue
            if sum(k.startswith('crnn') and k != 'crnn_conf' for k in r['readings'].get(f, {})) < ENG_UNAN: continue
            ok = False
            v = r.get(f) or ''
            if f in ('name', 'relation_name'):
                ts = [t.lower() for t in tok.findall(v) if len(t) >= 2]
                own = collections.Counter(ts)
                ok = bool(ts) and all(C.get(t, 0) >= lexmin or pc[t] - own[t] >= pmin for t in ts)
            elif f == 'house_no':
                nb = [recs[j] for j in (i - 1, i + 1) if 0 <= j < len(recs)]
                ok = bool(v) and any(x.get('house_no') == v and 'house_no' not in x['flags'] for x in nb)
            elif f == 'age':
                ok = v.isdigit() and 18 <= int(v) <= 120
            elif f in ('gender', 'relation_label'):
                ok = bool(v)
            if ok:
                r['flags'].remove(f); r['readings'].pop(f, None); r.setdefault('unan', []).append(f); n += 1
    return n


def eng_lex_guard(recs, cmax=2, nmin=30, ratio=30.0, pmin=2):
    """job4 r2 (English, no hints; ECI_ENG_LEX=1): silent-error guard for names. An UNFLAGGED name / relation name is flagged
    when one of its tokens (>= 3 letters) is rare in the non-gold name lexicon (count <= cmax, eciocr/ocr/eng_name_lex.json.gz)
    AND is read in fewer than pmin name fields of the same PDF AND a confusable 1-edit neighbour (y/v, j/i, m/rn, d/cl: lost
    descenders and merged letters) is common (count >= nmin and >= ratio x (count + 1)). Held-out check (22 non-gold PDFs of the
    11 no-2027 states, 13,953 records, config c2): 8 new flags, 7 of them wrong names ('Chaudharv', 'Jvoti', 'Mava', 'Manoi')."""
    global _ENG_LEX
    if _ENG_LEX is None:
        import gzip
        _ENG_LEX = json.load(gzip.open(os.path.join(os.path.dirname(os.path.abspath(ensemble2.__file__)), 'eng_name_lex.json.gz'), 'rt'))['counts']
    C = _ENG_LEX
    sub = {}
    for a, b in (('v', 'y'), ('rn', 'm'), ('cl', 'd'), ('j', 'i')):
        sub.setdefault(a, []).append(b); sub.setdefault(b, []).append(a)
    tok = re.compile(r'[A-Za-z]+')
    pc = collections.Counter(t.lower() for r in recs for k in ('name', 'relation_name') for t in tok.findall(r.get(k) or ''))

    def neigh(t):
        for a, bs in sub.items():
            i = t.find(a)
            while i >= 0:
                for b in bs: yield t[:i] + b + t[i + len(a):]
                i = t.find(a, i + 1)
    n = 0
    for r in recs:
        for f in ('name', 'relation_name'):
            if f in r['flags']: continue
            for t in tok.findall(r.get(f) or ''):
                t = t.lower()
                if len(t) < 3 or C.get(t, 0) > cmax or pc[t] >= pmin: continue
                best = max((C.get(x, 0) for x in neigh(t)), default=0)
                if best >= nmin and best >= ratio * (C.get(t, 0) + 1):
                    r['flags'].append(f); r.setdefault('lex_guard', []).append(t); n += 1
                    break
    return n


def draft_fill(recs, path):
    """job4 (2026-11-25): SIR FINAL roll + the OCR records of the SIR DRAFT roll of the same part. The final roll prints the
    same voters (minus deletions, plus additions); in West Bengal it stamps 'UNDER ADJUDICATION' over the text of some boxes,
    which no engine reads. Only UNFLAGGED draft records are used (their values are as trustworthy as any unflagged record).
    For a final record with flags whose EPIC (output or any engine reading) is an unflagged draft EPIC, each flagged field
    except serial / code / part:
      - an engine of the final read the draft value -> value = draft value, field unflagged ('confirmed_by_draft')
      - else value = draft value, field STAYS flagged ('filled_from_draft'): no value is accepted that no engine read.
    Serial numbers are not taken from the draft (the final renumbers)."""
    D = {}
    for l in open(path, encoding='utf-8'):
        d = json.loads(l)
        if not d.get('flags') and not d.get('not_voter_box') and ensemble2.valid('epic', d.get('epic') or ''): D[d['epic']] = d
    st = collections.Counter()
    for r in recs:
        if not r['flags']: continue
        cand = [r['epic']] + list((r.get('readings', {}).get('epic') or {}).values())
        d = next((D[e] for e in cand if isinstance(e, str) and e in D), None)
        if d is None: continue
        st['matched'] += 1
        for f in list(r['flags']):
            if f in ('serial_no', 'code', 'part_no') or f not in d: continue
            rd = [v for k, v in (r.get('readings', {}).get(f) or {}).items() if isinstance(v, str) and not k.startswith(('hint', 'crnn_conf'))]
            if f == 'epic' or any(visual(v) == visual(d[f]) for v in rd):
                r[f] = d[f]; r['flags'].remove(f); r.setdefault('confirmed_by_draft', []).append(f); st['confirmed'] += 1
            else:
                r[f] = d[f]; r.setdefault('filled_from_draft', []).append(f); st['filled'] += 1
    return dict(st)


def serial_check(recs):
    """SIR: serials must run 1..N in reading order. Flags breaks; returns a summary."""
    s = [int(r['serial_no']) if r['serial_no'].isdigit() else None for r in recs]
    n = len(s)
    bad = []
    for i, v in enumerate(s):
        prev_ok = i > 0 and s[i - 1] is not None and v is not None and v == s[i - 1] + 1
        next_ok = i + 1 < n and s[i + 1] is not None and v is not None and v == s[i + 1] - 1
        if i == 0: prev_ok = v == 1
        if i == n - 1 and n > 1: next_ok = prev_ok
        if not (prev_ok or next_ok):
            bad.append(i)
            r = recs[i]
            if 'serial_no' not in r['flags']: r['flags'].append('serial_no')
            exp = (s[i - 1] + 1) if i > 0 and s[i - 1] is not None else (1 if i == 0 else None)
            nxt = (s[i + 1] - 1) if i + 1 < n and s[i + 1] is not None else None
            r['serial_expected'] = exp if exp is not None and (nxt is None or nxt == exp) else None
            r.setdefault('readings', {}).setdefault('serial_no', {})['sequence'] = 'break'
    good = sorted(v for v in s if v is not None)
    seen = collections.Counter(good)
    last = max(good) if good else None
    missing = [k for k in range(1, (last or 0) + 1) if k not in seen]
    return dict(records=n, first=s[0] if s else None, last=last, records_equal_last=(n == last),
                missing=missing[:200], n_missing=len(missing), duplicates=[k for k, c in seen.items() if c > 1][:200],
                sequence_breaks=len(bad), unreadable=sum(v is None for v in s))


def pdf_kind(path):
    b = os.path.basename(path).upper()
    if 'SUPPLEMENT' in b: return 'SUP'
    if '-SIR-' in b or 'EROLLGEN' in b: return 'SIR'
    return None


def pdf_meta(path):
    b = os.path.basename(path)
    m = re.search(r'-([SU]\d+)-(\d+)-.*-(?:[A-Z]{3})-(\d+)-', b)   # S = state, U = union territory (Delhi U05, Chandigarh U02)
    return dict(state=m.group(1), ac=int(m.group(2)), roll_part=int(m.group(3))) if m else {}


def pdf_lang(path, opt='auto'):
    if opt and opt.upper() in ('HIN', 'ENG', 'TAM'): return opt.upper()
    b = os.path.basename(path).upper()
    return 'ENG' if '-ENG-' in b else 'TAM' if '-TAM-' in b else 'HIN'


def parse_pages(spec, n):
    if not spec: return list(range(n))
    out = []
    for p in spec.split(','):
        a, _, b = p.partition('-')
        out += list(range(int(a) - 1, min(n, int(b or a))))
    return [p for p in out if 0 <= p < n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='roll PDF(s)')
    ap.add_argument('--out', required=True, help='output JSONL (one record per voter box)')
    ap.add_argument('--crnn', action='append', help='CRNN model dir with crnn.onnx + charset.json; repeat for more voters '
                    '(default $ECI_CRNN or /data/models/crnn/c1)')
    ap.add_argument('--hint-2027', help='optional JSONL of 2027 text-layer records keyed by "epic" (confirmation only)')
    ap.add_argument('--workers', type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument('--ort-threads', type=int, default=1, help='ONNX Runtime threads per worker (default 1; workers x threads = CPUs)')
    ap.add_argument('--tessdata', default=STD_TESSDATA)
    ap.add_argument('--pages', help='1-based page list/ranges, e.g. 1-5,9 (testing)')
    ap.add_argument('--summary', help='summary JSON (default OUT.summary.json)')
    ap.add_argument('--crnn-bench-batching', action='store_true',
                    help='CRNN batches exactly as the bench (32 boxes, unsorted): raw text equal to bench 99.95%% but ~30%% slower. '
                         'Default: crops sorted by width, 64 per batch (engines.py SORTW=1)')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--roll-year', type=int, default=2026, help='year of the roll read (the 2027 hints confirm ages with a gap of 2027 - year)')
    ap.add_argument('--cache-in', help='pickle of readings by (engine, crop hash) from an earlier run (e.g. the SIR draft of the same part)')
    ap.add_argument('--cache-out', help='write the readings of this run as a pickle (for --cache-in of a later run)')
    ap.add_argument('--stage1', type=int, default=0, help='cascade: the first K --crnn models (+ Tesseract Latin fields) read every box; '
                    'the other engines run only where stage 1 is flagged or has no 2027 record (needs --hint-2027). 0 = off')
    ap.add_argument('--confvote', type=float, default=0.6, help='confidence-weighted choice for flagged fields (Tesseract weight); <0 = off')
    ap.add_argument('--lang', default='auto', help='auto (default: ENG for *-ENG-*, TAM for *-TAM-* file names, else HIN), HIN, ENG or TAM')
    ap.add_argument('--draft-records', help='job4: eci_ocr output (jsonl) of the SIR DRAFT roll of the same part, for a FINAL roll: '
                    'flagged fields of a final record are filled from the unflagged draft record with the same EPIC (draft_fill)')
    ap.add_argument('--tess2', action='store_true', help='second Tesseract reading (x2, Otsu) as one more voter (default on with --crnn none)')
    a = ap.parse_args()
    crnn = a.crnn or [os.environ.get('ECI_CRNN', '/data/models/crnn/c1')]
    if [c.lower() for c in crnn] == ['none']:   # Tesseract only (English baseline)
        crnn = []; a.tess2 = True
        if a.stage1: ap.error('--stage1 needs CRNNs')
    for d in crnn:
        if not os.path.exists(os.path.join(d, 'crnn.onnx')): ap.error(f'no crnn.onnx in {d}')
    if not shutil.which('tesseract'): ap.error('tesseract not found')
    tmp = '/dev/shm' if os.access('/dev/shm', os.W_OK) else None
    T0 = time.time()
    tasks, meta = [], {}
    for pdf in a.inputs:
        n = page_count(pdf); kind = pdf_kind(pdf)
        meta[pdf] = dict(pages=n, kind=kind, **pdf_meta(pdf))
        lang = pdf_lang(pdf, a.lang); meta[pdf]['lang'] = lang
        tasks += [(pdf, p, kind, lang) for p in parse_pages(a.pages, n)]
    global _HINTS, _CACHE_IN, _COLLECT
    ensemble2.AGE_GAP = 2027 - a.roll_year
    if a.roll_year < 2026 and os.environ.get('ECI_STAMP_TEMPLATE', '1') != '0':  # GE 2024 print: DELETED stamp -> fixed line positions
        import segment as _seg
        _seg.STAMP_CHECK = True; _seg.TEMPLATE_OVERRIDE = _seg.TEMPLATE_GER
    if a.cache_in:
        _CACHE_IN = pickle.load(open(a.cache_in, 'rb'))
    _COLLECT = bool(a.cache_out)
    cache_new = {}
    if a.hint_2027:
        _HINTS = load_hints(a.hint_2027)
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    import gc
    gc.collect(); gc.freeze()   # objects loaded before the fork (hints, cache) stay shared: the workers' GC does not touch them
    by_pdf = collections.defaultdict(list)
    with ctx.Pool(a.workers, _init, (crnn, a.tessdata, not a.crnn_bench_batching, a.hint_2027, tmp, a.ort_threads, a.stage1, a.confvote, a.tess2)) as pool:
        for i, r in enumerate(pool.imap(process_page, tasks, chunksize=1)):
            cache_new.update(r.pop('cache', {}))
            by_pdf[r['pdf']].append(r)
            if not a.quiet:
                print(f"[{time.time() - T0:7.1f}s] {os.path.basename(r['pdf'])} p{r['page'] + 1}: "
                      f"{r['boxes']} boxes {r.get('skip', '')} {r['timing'].get('total', r['timing'].get('extract'))}s", file=sys.stderr, flush=True)
    if a.cache_out:
        pickle.dump(cache_new, open(a.cache_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    wall = time.time() - T0
    import resource
    peak_mb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)
    summ = dict(peak_worker_rss_mb=peak_mb, inputs=a.inputs, crnn=crnn, engines=['crnn_' + os.path.basename(os.path.normpath(d)) for d in crnn] + (['tess', 'tess2'] if a.tess2 else ['tess']),
                hint_2027=a.hint_2027, cache_in=len(_CACHE_IN), cache_out=len(cache_new), workers=a.workers, ort_threads=a.ort_threads, crnn_sorted=not a.crnn_bench_batching, wall_s=round(wall, 1), pdfs=[])
    nrec = nflag = 0; fc = collections.Counter()
    with open(a.out, 'w', encoding='utf-8') as out:
        for pdf in a.inputs:
            pages = sorted(by_pdf[pdf], key=lambda r: r['page'])
            recs = [rec for p in pages for rec in p.get('records', [])]
            sec = None
            for r in recs:  # a page without a heading continues the previous section
                if r['section']: sec = r['section']
                else: r['section'] = sec
            number_sections(recs)
            # A "box" found on a page that is not a voter page (cover, map, photo page) reads as garbage: no valid EPIC and most
            # fields flagged. It is kept in the output, marked, and left out of the counts and the serial check (2026-11-22;
            # 0 of 2,578 records of 3 checked voter PDFs marked, 6 page-3 boxes marked).
            for r in recs:
                if not ensemble2.valid('epic', r['epic'] or '') and len(r['flags']) >= 6:
                    r['not_voter_box'] = True
            post = post_rules(recs, '-SIR-' in os.path.basename(pdf))
            allrecs, recs = recs, [r for r in recs if not r.get('not_voter_box')]
            if _HINTS:   # EPIC prefixes the CRNNs never saw: repaired from the 2027 roll (epicfix.py; hand check 2026-11-22)
                import epicfix
                global _EPIC_IDX
                if _EPIC_IDX is None:
                    _EPIC_IDX = epicfix.index(_HINTS.values())
                epic_fixed = epicfix.fix(recs, _EPIC_IDX, ensemble2.AGE_GAP) + epicfix.fix2(recs, _EPIC_IDX, ensemble2.AGE_GAP)
            else:
                epic_fixed = 0
                if os.environ.get('ECI_EPIC_PREFIX', '0') != '0':   # job2: no 2027 roll -> rare-prefix check (epicfix.prefix_check)
                    import epicfix
                    epic_fixed = epicfix.prefix_check(recs)[0]
            m = meta[pdf]
            kind = m['kind'] or (collections.Counter(p['kind'] for p in pages if p.get('kind')).most_common(1) or [(None,)])[0][0]
            info = dict(pdf=os.path.basename(pdf), kind=kind, **{k: v for k, v in m.items() if k != 'kind'},
                        voter_pages=sum(1 for p in pages if p['boxes']), skipped_pages=[p['page'] + 1 for p in pages if not p['boxes']],
                        records=len(recs), not_voter_boxes=len(allrecs) - len(recs), epic_from_2027=epic_fixed, post_rules=post)
            if ENG_NOHINT and not _HINTS and recs:
                info['epic_by_prefix'] = eng_epic_prefix(recs)
            if ENG_UNAN and ENG_NOHINT and not _HINTS and recs:
                info['unan'] = eng_unan(recs)
            if ENG_LEX and lang_of_pdf_is_eng(recs) and recs:
                info['lex_guard'] = eng_lex_guard(recs)
            if a.draft_records and recs:
                info['draft_fill'] = draft_fill(recs, a.draft_records)
            if kind == 'SIR' and recs:
                info['serial_check'] = serial_check(recs)
                info['serial_check']['confirmed_by_sequence'] = sequence_confirm(recs)
            vp = [p for p in pages if p['boxes']]
            cpu = sum(p['timing'].get('total', 0) for p in pages)
            info['worker_s_per_voter_page'] = round(sum(p['timing']['total'] for p in vp) / max(1, len(vp)), 2)
            info['worker_s_total'] = round(cpu, 1)
            info['stage_s'] = {k: round(sum(p['timing'].get(k, 0) for p in pages), 1) for k in ('extract', 'layout', 'segment', 'heading', 'crnn', 'tess', 'decide')}
            ff = collections.Counter(f for r in recs for f in r['flags'])
            info['flagged_records'] = sum(bool(r['flags']) for r in recs)
            info['flag_rate'] = round(info['flagged_records'] / max(1, len(recs)), 4)
            info['flags_by_field'] = dict(ff.most_common())
            info['hint_matched'] = sum(r['hint_2027'] for r in recs)
            info['sections'] = list(dict.fromkeys(r['section'] for r in recs if r['section']))
            for r in allrecs:
                r = dict(source=os.path.basename(pdf), **{k: v for k, v in m.items() if k not in ('pages', 'kind', 'lang')}, **r)
                out.write(json.dumps(r, ensure_ascii=False) + '\n')
            nrec += len(recs); nflag += info['flagged_records']; fc.update(ff)
            summ['pdfs'].append(info)
    vpages = sum(p['voter_pages'] for p in summ['pdfs'])
    summ.update(records=nrec, voter_pages=vpages, s_per_voter_page_wall=round(wall / max(1, vpages), 2),
                flag_rate=round(nflag / max(1, nrec), 4), flags_by_field=dict(fc.most_common()))
    json.dump(summ, open(a.summary or a.out + '.summary.json', 'w'), ensure_ascii=False, indent=1)
    if not a.quiet:
        for p in summ['pdfs']:
            sc = p.get('serial_check', {})
            print(f"{p['pdf']}: {p['records']} records, {p['voter_pages']} voter pages, flag rate {100 * p['flag_rate']:.1f}%"
                  + (f", last serial {sc.get('last')}, missing {sc.get('n_missing')}, breaks {sc.get('sequence_breaks')}" if sc else ''), file=sys.stderr)
        print(f"total {nrec} records in {wall:.1f}s ({summ['s_per_voter_page_wall']} s/voter page, {a.workers} workers)", file=sys.stderr)


if __name__ == '__main__':
    main()
