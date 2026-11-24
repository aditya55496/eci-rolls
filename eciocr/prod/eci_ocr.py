#!/usr/bin/env python3
"""OCR of image-only electoral roll PDFs (UP, Hindi): SIR rolls and supplements -> one JSONL record per voter box.

Usage:
  eci_ocr.py INPUT.pdf [INPUT2.pdf ...] --out OUT.jsonl [--crnn DIR ...] [--hint-2027 hints.jsonl] [--workers 4]

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
from segment import segment, _mask, cells           # noqa: E402
import segment as _segmod                           # noqa: E402
from parse import from_lines                        # noqa: E402
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
    def __init__(self, crnn_dirs, tessdata=STD_TESSDATA, ort_threads=1, crnn_sort=False):
        import onnxruntime as ort
        self.crnn = []
        for d in crnn_dirs:
            so = ort.SessionOptions(); so.intra_op_num_threads = ort_threads; so.inter_op_num_threads = 1
            # no memory arena (2026-11-23): the arena grows with every new crop width and never shrinks; a GE 2024 part took 2.8-4.4 GB
            # per worker and runner VMs (16 GB, 2 processes x 2 workers) ran out of memory. Off: 0.9 GB per worker, output identical.
            so.enable_cpu_mem_arena = False
            s = ort.InferenceSession(os.path.join(d, 'crnn.onnx'), so, providers=['CPUExecutionProvider'])
            h = s.get_inputs()[0].shape[2]   # line height the model was trained at (48; smaller students: 32)
            self.crnn.append(('crnn_' + os.path.basename(os.path.normpath(d)), s, s.get_inputs()[0].name,
                              json.load(open(os.path.join(d, 'charset.json'))), h if isinstance(h, int) else 48))
        self.tessdata = tessdata
        self.crnn_sort = crnn_sort
        try:
            self.tapi = TessAPI(tessdata) if not os.environ.get('ECI_TESS_CLI') else None
        except OSError:
            self.tapi = None
        self.names = [c[0] for c in self.crnn] + ['tess']
        # reading cache (2026-11-22): the SIR final roll repeats ~94 % of the SIR draft's crops pixel for pixel; readings
        # of a crop are reused by (engine, crop hash). cache_in: read-only (loaded before the fork); cache_out: new readings.
        self.cache_in, self.cache_out, self.collect = {}, {}, False

    # engines.py _crnn_read: pad 5 white, resize to height 48 (cubic, width <= 800), invert, /255, greedy CTC (blank 0)
    @staticmethod
    def _crnn_batch(sess, inp, CH, ims, H=48):
        xs = []
        for im in ims:
            im = cv2.copyMakeBorder(im, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
            w = min(800, max(8, int(im.shape[1] * H / im.shape[0])))
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
        _, sess, inp, CH, H = eng
        if not ims: return []
        if not self.crnn_sort or len(ims) == 1:
            return self._crnn_batch(sess, inp, CH, ims, H)
        order = sorted(range(len(ims)), key=lambda i: ims[i].shape[1] * 48 / max(1, ims[i].shape[0]))  # engines.py SORTW=1
        out = [None] * len(ims)
        for j in range(0, len(order), 64):
            idx = order[j:j + 64]
            for i, r in zip(idx, self._crnn_batch(sess, inp, CH, [ims[i] for i in idx], H)): out[i] = r
        return out

    @staticmethod
    def tess_prep(im, scale=TESS_SCALE):  # engines.py prep(im, 3): Lanczos x3, pad 8*3 white
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        p = int(8 * scale)
        return cv2.copyMakeBorder(im, p, p, p, p, cv2.BORDER_CONSTANT, value=255)

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

    def read_boxes(self, segs_list, tmp=None, timing=None, crnn_idx=None, tess_groups=None):
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
        groups = {'line': ('hin', 7, None), 'num': ('eng', 7, '0123456789'),
                  'epic': ('eng', 7, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/'), 'code': ('eng', 7, '#SERMQ0123456789')}
        gk = {g: [] for g in groups}
        for b, k in keys:
            g = 'line' if k.startswith('line') else 'num' if k in ('serial', 'part') else k
            gk[g].append((b, k))
        for b in range(n): res[b]['tess'] = ({}, {})
        for g, (lang, psm, wl) in groups.items():
            if not gk[g] or (tess_groups is not None and g not in tess_groups): continue
            if use_cache:
                kidx = {bk: i for i, bk in enumerate(keys)}
                ck = [('tess_' + g, hs[kidx[bk]]) for bk in gk[g]]
                txts = [self.cache_in.get(c) for c in ck]
                todo = [i for i, t_ in enumerate(txts) if t_ is None]
                for i, t_ in zip(todo, self.tess_many([self.tess_prep(segs_list[gk[g][i][0]][gk[g][i][1]]) for i in todo], lang, psm, wl, tmp)):
                    txts[i] = t_
                    if self.collect: self.cache_out[ck[i]] = t_
            else:
                txts = self.tess_many([self.tess_prep(segs_list[b][k]) for b, k in gk[g]], lang, psm, wl, tmp)
            for (b, k), txt in zip(gk[g], txts): res[b]['tess'][0][k] = txt
        timing['tess'] = round(timing.get('tess', 0) + time.time() - t, 3)
        return res


def code_pixel(box_gray):
    """Printed status code present: inner left 40% of the serial cell has > 15 ink pixels (bench code_pixel.json)."""
    _, _, small = cells(box_gray)
    if not small: return False
    x, y, cw, ch = small[0]
    m = _mask(box_gray[y + 3: y + ch - 3, x + 3: x + cw - 3]) > 0
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


def parse_heading(lines):
    """Heading lines (CRNN text) -> section: 'N-name' for SIR sections, or one of LISTS. Last match in the band wins."""
    from rapidfuzz import process, fuzz
    for t in reversed(lines):
        t = re.sub(r'\s+', ' ', t).strip()
        if 'अनुभाग' in t and ':' in t:
            v = t.split(':', 1)[1].strip(' :.-')
            m = re.match(r'([0-9]+)\s*-?\s*(.*)', v)
            return f'{m.group(1)}-{m.group(2)}' if m else v
        m = process.extractOne(' '.join(t.split()[:3]), LISTS, scorer=fuzz.ratio)
        if m and m[1] >= 75: return m[0]
    return None


# ---------------------------------------------------------------- one page

_E = None
_OPT = {}
ALLREAD = os.environ.get('ECI_ALLREAD', '0') != '0'   # job1 2026-11-25: dump all engine readings per record (eval only)


_CACHE_IN, _COLLECT = {}, False   # reading cache: loaded before the fork / collect new readings (--cache-in/--cache-out)
_EPIC_IDX = None
_HINTS = None   # loaded once in the parent before the fork: the workers share it (a whole AC is ~1.5 GB as dicts)


def _init(crnn_dirs, tessdata, crnn_sort, hints_path, tmp, ort_threads=1, stage1=0, confvote=0.6):
    global _E, _OPT
    cv2.setNumThreads(1)
    _E = Engines(crnn_dirs, tessdata, ort_threads, crnn_sort)
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


def decide_box(per_engine, names, pix, hints, stamp=None):
    """per_engine: {name: (raw_fields, crnn_conf)} -> record fields, flags, readings for flagged fields."""
    cands = [from_lines(per_engine[n][0]) for n in names]
    epics = [e for e, _ in collections.Counter(c['epic'] for c in cands).most_common() if e]
    hint = next((hints[e] for e in epics if e in hints), None)
    out, flags = decide(cands, hint, pix, names, [per_engine[n][1] for n in names], stamp=stamp)
    readings = {}
    for f in flags:
        readings[f] = {n: c[f] for n, c in zip(names, cands)}
        if hint and f in ensemble2.HINT_FIELDS + ['age'] and f in hint: readings[f]['hint_2027'] = hint[f]
        conf = {n: per_engine[n][1].get(SRC[f]) for n in names if n.startswith('crnn') and SRC[f] in per_engine[n][1]}
        if conf: readings[f]['crnn_conf'] = conf
    return out, flags, readings, hint is not None, cands


def process_page(task):
    pdf, pno, kind = task
    t0 = time.time(); tm = {}
    gray = page_gray(pdf, pno)
    tm['extract'] = round(time.time() - t0, 3)
    if gray is None: return dict(pdf=pdf, page=pno, boxes=0, skip='no image', timing=tm)
    t = time.time()
    boxes = find_boxes(gray)
    tm['layout'] = round(time.time() - t, 3)
    if not boxes: return dict(pdf=pdf, page=pno, boxes=0, skip='no voter boxes', timing=tm, size=gray.shape[::-1])
    t = time.time()
    box_grays = [crop(gray, b, 2) for b in boxes]
    segs_list, stamps = [], []
    for g in box_grays:  # segment.LAST_STAMP: DELETED stamp found (only checked for rolls before 2026, see segment.STAMP_CHECK)
        segs_list.append({k: im for k, im in segment(g).items() if im.size}); stamps.append(bool(_segmod.LAST_STAMP))
    pix = [code_pixel(g) for g in box_grays]
    tm['segment'] = round(time.time() - t, 3)
    t = time.time()
    heads = []
    for y, band in heading_strips(gray, boxes):  # section headings: CRNN reads them (Tesseract hin misreads the Latin numbers)
        L = [l for l in text_lines(band) if l.shape[1] < 0.5 * gray.shape[1]]  # headings are short; skip the title line
        if L: heads.append((y, parse_heading([t for t, _ in _E.crnn_read(_E.crnn[0], L)])))
    tm['heading'] = round(time.time() - t, 3)
    K = _OPT.get('stage1', 0)
    if K and _OPT['hints']:
        # cascade: stage 1 = first K CRNNs + Tesseract Latin fields; the rest only for boxes stage 1 cannot settle
        reads = _E.read_boxes(segs_list, _OPT['tmp'], tm, crnn_idx=set(range(K)), tess_groups={'num', 'epic', 'code'})
        names1 = _E.names[:K] + ['tess']
        need = []
        for i, r in enumerate(reads):
            _, fl, _, hinted, _ = decide_box(r, names1, pix[i], _OPT['hints'], stamps[i])
            if fl or not hinted: need.append(i)
        if need:
            more = _E.read_boxes([segs_list[i] for i in need], _OPT['tmp'], tm, crnn_idx=set(range(K, len(_E.crnn))), tess_groups={'line'})
            for i, r2 in zip(need, more):
                t1 = reads[i]['tess']; t2 = r2.pop('tess')
                t1[0].update(t2[0]); reads[i].update(r2)
        tm['stage2_boxes'] = len(need)
        stage1_only = set(range(len(reads))) - set(need)
    else:
        reads = _E.read_boxes(segs_list, _OPT['tmp'], tm)
        stage1_only = set()
    t = time.time()
    if kind is None: kind = 'SUP' if sum('part' in s for s in segs_list) * 2 > len(segs_list) else 'SIR'
    recs = []
    section = None
    for i, (b, r) in enumerate(zip(boxes, reads)):
        for y, h in heads:
            if y <= b[1] + 5 and h: section = h
        nm = (_E.names[:K] + ['tess']) if i in stage1_only else _E.names
        out, flags, readings, hinted, cands = decide_box(r, nm, pix[i], _OPT['hints'], stamps[i])
        if kind == 'SUP' and 'part_no' not in flags:
            pv = {visual(c['part_no']) for c in cands}
            if len(pv) > 1 or not out['part_no'].isdigit(): flags.append('part_no'); readings['part_no'] = {n: c['part_no'] for n, c in zip(nm, cands)}
        rec = dict(page=pno + 1, box=i, kind=kind, section=section, **{f: out[f] for f in OUT_FIELDS})
        rec.update(flags=flags, readings=readings, hint_2027=hinted, code_ink=pix[i], bbox=list(map(int, b)))
        if ALLREAD:  # job1 (evaluation only): every engine's fields + CRNN line confidences, for offline decide() studies
            rec['allreads'] = {n: [r[n][0], {k: float(v) for k, v in r[n][1].items()}] for n in nm}
            rec['stage1_only'] = i in stage1_only
        recs.append(rec)
    tm['decide'] = round(time.time() - t, 3)
    tm['total'] = round(time.time() - t0, 3)
    for rec in recs: rec['t_box_s'] = round(tm['total'] / len(recs), 3)
    out = dict(pdf=pdf, page=pno, boxes=len(boxes), kind=kind, records=recs, timing=tm, size=gray.shape[::-1])
    if _E.collect:
        out['cache'] = _E.cache_out; _E.cache_out = {}
    return out


# ---------------------------------------------------------------- per-PDF checks

def sequence_confirm(recs):
    """SIR: a serial_no flagged only by engine disagreement is confirmed when BOTH neighbours in reading order
    agree (prev + 1 == serial == next - 1; first record: serial 1 and next 2). Returns the number confirmed."""
    s = [int(r['serial_no']) if r['serial_no'].isdigit() else None for r in recs]
    n = 0
    for i, r in enumerate(recs):
        if 'serial_no' not in r['flags'] or s[i] is None: continue
        prev_ok = (s[i - 1] is not None and s[i - 1] + 1 == s[i]) if i > 0 else s[i] == 1
        next_ok = (s[i + 1] is not None and s[i + 1] - 1 == s[i]) if i + 1 < len(s) else prev_ok and i > 0
        if prev_ok and next_ok:
            r['flags'].remove('serial_no'); r['readings'].pop('serial_no', None)
            r['serial_by_sequence'] = True; n += 1
    return n


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
    m = re.search(r'-S(\d+)-(\d+)-.*-HIN-(\d+)-', b)
    return dict(state=f'S{m.group(1)}', ac=int(m.group(2)), roll_part=int(m.group(3))) if m else {}


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
    a = ap.parse_args()
    crnn = a.crnn or [os.environ.get('ECI_CRNN', '/data/models/crnn/c1')]
    for d in crnn:
        if not os.path.exists(os.path.join(d, 'crnn.onnx')): ap.error(f'no crnn.onnx in {d}')
    if not shutil.which('tesseract'): ap.error('tesseract not found')
    tmp = '/dev/shm' if os.access('/dev/shm', os.W_OK) else None
    T0 = time.time()
    tasks, meta = [], {}
    for pdf in a.inputs:
        n = page_count(pdf); kind = pdf_kind(pdf)
        meta[pdf] = dict(pages=n, kind=kind, **pdf_meta(pdf))
        tasks += [(pdf, p, kind) for p in parse_pages(a.pages, n)]
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
    if os.environ.get('ECI_STAGE1'):   # job1 2026-11-25 (evaluation switch): overrides --stage1 (number of stage-1 CRNNs)
        a.stage1 = int(os.environ['ECI_STAGE1'])
    with ctx.Pool(a.workers, _init, (crnn, a.tessdata, not a.crnn_bench_batching, a.hint_2027, tmp, a.ort_threads, a.stage1, a.confvote)) as pool:
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
    summ = dict(peak_worker_rss_mb=peak_mb, inputs=a.inputs, crnn=crnn, engines=['crnn_' + os.path.basename(os.path.normpath(d)) for d in crnn] + ['tess'],
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
            # A "box" found on a page that is not a voter page (cover, map, photo page) reads as garbage: no valid EPIC and most
            # fields flagged. It is kept in the output, marked, and left out of the counts and the serial check (2026-11-22;
            # 0 of 2,578 records of 3 checked voter PDFs marked, 6 page-3 boxes marked).
            for r in recs:
                if not ensemble2.valid('epic', r['epic'] or '') and len(r['flags']) >= 6:
                    r['not_voter_box'] = True
            allrecs, recs = recs, [r for r in recs if not r.get('not_voter_box')]
            if _HINTS:   # EPIC prefixes the CRNNs never saw: repaired from the 2027 roll (epicfix.py; hand check 2026-11-22)
                import epicfix
                global _EPIC_IDX
                if _EPIC_IDX is None:
                    _EPIC_IDX = epicfix.index(_HINTS.values())
                epic_fixed = epicfix.fix(recs, _EPIC_IDX) + epicfix.fix2(recs, _EPIC_IDX, ensemble2.AGE_GAP)
            else:
                epic_fixed = 0
            m = meta[pdf]
            kind = m['kind'] or (collections.Counter(p['kind'] for p in pages if p.get('kind')).most_common(1) or [(None,)])[0][0]
            info = dict(pdf=os.path.basename(pdf), kind=kind, **{k: v for k, v in m.items() if k != 'kind'},
                        voter_pages=sum(1 for p in pages if p['boxes']), skipped_pages=[p['page'] + 1 for p in pages if not p['boxes']],
                        records=len(recs), not_voter_boxes=len(allrecs) - len(recs), epic_from_2027=epic_fixed)
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
                r = dict(source=os.path.basename(pdf), **{k: v for k, v in m.items() if k not in ('pages', 'kind')}, **r)
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
