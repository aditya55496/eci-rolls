"""Printed totals of one roll part PDF (image-only EROLLGEN PDFs; no text layer), for the completeness check of task fetch-hi.

Read with Tesseract (digits only matter; they are Latin on every roll seen, Bengali rolls print the page footer in Bengali
digits, read with the 'ben' model when installed):
  part_printed   the part number in the header cell at the top right of the cover ("भाग संख्या : 5", "অংশ নং : 5")
  pages_printed  the page total in the footer of the cover and the last page ("कुल पृष्ठ 25 - का पृष्ठ 1")
  cover          the cover's electors table (male, female, third gender, total; last serial number)
  summary_net    the net electors row of the summary on the last page
check(): ok when the PDF has the printed number of pages, the printed part number is the file's part number, and the cover
total equals the summary's net total. A value that cannot be read is "unread", not a fault. Tested 2026-11-24 on MP, Bihar
(form route), Tamil Nadu (SIR, SSR 2027), Gujarat, West Bengal, Andaman, Manipur, Lakshadweep rolls: 1.6-2.5 s CPU per
PDF with Tesseract, up to 6 s when the first resolution does not read the totals.
"""
import collections
import os
import re
import shutil
import subprocess
import unicodedata

import pymupdf

DPI = 200
_LANGS = None


def _have(lang):
    global _LANGS
    if _LANGS is None:
        try:
            out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, timeout=30).stdout
            _LANGS = set(out.split()[1:]) if out else set()
        except (OSError, subprocess.SubprocessError):
            _LANGS = set()
    return lang in _LANGS


def available():
    return shutil.which("tesseract") is not None


def _ocr(png, psm, lang="eng"):
    out = subprocess.run(["tesseract", "stdin", "stdout", "-l", lang, "--psm", str(psm), "tsv"], input=png, capture_output=True,
                         env={**os.environ, "OMP_THREAD_LIMIT": "1"}, timeout=120).stdout.decode("utf-8", "replace")
    words = []
    for line in out.splitlines()[1:]:
        f = line.split("\t")
        if len(f) == 12 and f[11].strip():
            words.append((int(f[6]), int(f[7]), int(f[8]), int(f[9]), f[11].strip()))
    return words


def _num(t):
    t = "".join(str(unicodedata.digit(c)) if unicodedata.digit(c, None) is not None else c for c in t)
    m = re.fullmatch(r"[\(\[:]*(\d{1,6})[\)\],.:]*", t)
    return int(m.group(1)) if m else None


def _ints(words):
    return [(x, y, w, h, v) for x, y, w, h, t in words for v in [_num(t)] if v is not None]


def _png(page, clip=None, dpi=None):
    return page.get_pixmap(dpi=dpi or DPI, clip=clip, colorspace=pymupdf.csGRAY).tobytes("png")


def _crop(page, x0, y0, x1, y1):
    r = page.rect
    return pymupdf.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height, r.x0 + x1 * r.width, r.y0 + y1 * r.height)


def _rows(ints, tol=12):
    R = collections.defaultdict(list)
    for x, y, w, h, v in ints:
        R[round((y + h / 2) / tol)].append((x, v))
    return [[v for x, v in sorted(R[k])] for k in sorted(R)]


def _totals(r):
    """(male, female, third, total, rest) from the right end of a row; Tesseract drops lone '0' cells, so a missing third
    gender cell (male + female = total) counts as 0."""
    if len(r) >= 4 and r[-4] + r[-3] + r[-2] == r[-1] > 0:
        return r[-4], r[-3], r[-2], r[-1], r[:-4]
    if len(r) >= 3 and r[-3] + r[-2] == r[-1] > 0:
        return r[-3], r[-2], 0, r[-1], r[:-3]
    return None


def _table_totals(page, y0, y1):
    """Rows of (male, female, third, total, rest) between y0 and y1 (page fractions). Tesseract drops or splits digits at one
    resolution and not at another (Manipur 588 -> 88 at 200 dpi, Bihar cover only at 250): 200 dpi first, then 150 and 250."""
    for dpi in (DPI, 150, 250):
        H = page.rect.height * dpi / 72
        ints = [i for i in _ints(_ocr(_png(page, dpi=dpi), 11)) if y0 * H < i[1] < y1 * H]
        found = [t for t in map(_totals, _rows(ints, 12 * dpi / 200)) if t]
        if found:
            return found
    return []


def _footer_pages(page, lang):
    clip = _crop(page, 0.55, 0.93, 1.0, 1.0)
    for lg in ["eng"] + ([lang] if lang and _have(lang) else []):
        v = [x[4] for x in _ints(_ocr(_png(page, clip), 6, lg)) if 0 < x[4] < 1000]
        if v:
            return max(v)
    return None


TESS = {"BEN": "ben", "ASM": "asm", "ORI": "ori", "TAM": "tam", "GUJ": "guj", "MAR": "mar", "HIN": "hin", "TEL": "tel", "KAN": "kan",
        "MAL": "mal", "PAN": "pan", "URD": "urd"}


def printed(pdf_bytes, roll_lang=None):
    lang = TESS.get(roll_lang or "")
    d = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    n = len(d)
    out = {"pdf_pages": n}
    p0, pl = d[0], d[n - 1]
    hdr = set()      # header cell: "भाग संख्या : : 5", "Part No.:5"; the number can stick to the colon or the label
    for w in _ocr(_png(p0, _crop(p0, 0.72, 0.02, 1.0, 0.13)), 11):
        m = re.search(r"(\d{1,4})\W*$", "".join(str(unicodedata.digit(c)) if unicodedata.digit(c, None) is not None else c for c in w[4]))
        if m:
            hdr.add(int(m.group(1)))
    if hdr:
        out["part_read"] = sorted(hdr)
    cov = _table_totals(p0, 0.12, 0.92)
    if cov:
        t = cov[0]
        out["cover"] = {"male": t[0], "female": t[1], "third_gender": t[2], "total": t[3], "serial_to": t[4][-1] if t[4] else None}
    out["pages_printed"] = _footer_pages(p0, lang)
    if n > 1:
        net = _table_totals(pl, 0.0, 0.9)
        if net:
            out["summary_net"] = {"male": net[-1][0], "female": net[-1][1], "third_gender": net[-1][2], "total": net[-1][3]}
        out["pages_printed_last"] = _footer_pages(pl, lang)
    return out


def check(pdf_bytes, part, roll_lang=None):
    """printed() plus verdicts: {"pages": ok|bad|unread, "part": ok|bad|unread, "totals": ok|bad|unread, "electors": int|None, "ok": bool}."""
    p = printed(pdf_bytes, roll_lang)
    pg = [x for x in (p.get("pages_printed"), p.get("pages_printed_last")) if x]
    p["pages"] = "unread" if not pg else "ok" if p["pdf_pages"] in pg else "bad"
    # the colon before the number is sometimes read as a digit (": : 10" -> "710"): one extra leading digit still counts
    read = [v for v in p.get("part_read") or [] if v > 0]
    p["part"] = "unread" if not read else "ok" if any(v == part or (str(v).endswith(str(part)) and len(str(v)) == len(str(part)) + 1)
                                                     for v in read) else "bad"
    c, s = p.get("cover"), p.get("summary_net")
    p["totals"] = "unread" if not (c and s) else "ok" if c["total"] == s["total"] else "bad"
    p["electors"] = (s or c or {}).get("total")
    p["ok"] = "bad" not in (p["pages"], p["part"], p["totals"])
    return p


if __name__ == "__main__":      # printcheck.py PART LANG FILE.pdf
    import json
    import sys
    print(json.dumps(check(open(sys.argv[3], "rb").read(), int(sys.argv[1]), sys.argv[2]), ensure_ascii=False))
