"""Fixes of the 2027 SSR text-layer parsers for the non-Hindi rolls (2026-11-25, release faults). Offline: no hub, no network.
Strings and geometry are as found in the PDFs named in each test (bucket eci-rolls, eci-rolls-all-cdn/pdf/...)."""
import collections
import os
import sys

import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "eciocr", "textlayer")]
os.environ.setdefault("SCRAPE_POOL_PROJECT", "eci-rolls-test")
import parse_textlayer_any as A  # noqa: E402
import parse_textlayer_eng as E  # noqa: E402


def _chars(rows, sid0=0, step=1.0):
    """rows: [(x0, y, text)] -> rawdict-like chars (ch, x, y, span id, span text, bbox), one span per row."""
    out = []
    for k, (x0, y, t) in enumerate(rows):
        for i, ch in enumerate(t):
            out.append((ch, x0 + i * step, y, sid0 + k, t, (x0 + i * step, y - 8, x0 + i * step + step, y)))
    return out


def _trace(text, x0=0.0, y=0.0, gids=None):
    return [(ch, x0 + i, y, (gids or {}).get(i, 100 + i)) for i, ch in enumerate(text)]


# --------------------------------------------------------------------------- S03 Assam: khanda ta and Bengali ra per glyph

def test_khanda_ta_only_for_the_khanda_glyph():
    # 'ৰনজিত্ ৰায' prints ত + hasanta (gid 468): kept; 'মনজিৎ' prints the khanda ta (gid 714): ৎ
    tl = [("ন", 1, 0, 358), ("জ", 2, 0, 350), ("ি", 3, 0, 373), ("ত", 4, 0, 468), ("্", 4, 0, -1)]
    assert A.bengali_trace_fixes("ৰনজিত্", tl) == ("ৰনজিত্", [])
    tl = [("ম", 1, 0, 358), ("ন", 2, 0, 358), ("জ", 3, 0, 350), ("ি", 4, 0, 373), ("ত", 5, 0, 714), ("্", 5, 0, -1)]
    assert A.bengali_trace_fixes("মনজিত্", tl) == ("মনজিৎ", ["khanda_ta"])
    # two candidates, first drawn with hasanta, second khanda: only the second becomes ৎ
    tl = [("ত", 1, 0, 468), ("্", 1, 0, -1), (" ", 2, 0, 3), ("ত", 3, 0, 714), ("্", 3, 0, -1)]
    assert A.bengali_trace_fixes("ৰণজিত্ সত্যজিত্", tl)[0] == "ৰণজিত্ সত্যজিৎ"


def test_bengali_ra_glyph_is_bengali_ra_in_assamese_rolls_too():
    c = collections.Counter()
    assert A.repair("চৌধুব়ী", "ASM", c) == "চৌধুরী" and c["bengali_ra"] == 1     # glyph 365 = Bengali র
    assert A.repair("বেশ্ব়", "ASM") == "বেশ্বর"                                 # ব-phala + র
    assert A.repair("পিতাৰ নাম", "ASM") == "পিতাৰ নাম"                          # ৰ (glyph 402) kept
    assert A.repair("পিতাব় নাম", "BEN") == "পিতার নাম"


# --------------------------------------------------------------------------- stretched page (wide box)

def _page_with_columns(cols, rows=2, epic_x=None):
    doc = pymupdf.open()
    pg = doc.new_page(width=595, height=842)
    n = 0
    for r in range(rows):
        y0 = 28 + r * 79
        for x0, x1 in cols:
            pg.draw_line((x0, y0), (x1, y0)); pg.draw_line((x0, y0 + 75), (x1, y0 + 75))
            n += 1
            pg.insert_text((x0 + 40, y0 + 11), str(n), fontsize=9)
            pg.insert_text((x1 - 44, y0 + 11), "ABC%07d" % n, fontsize=9)
            pg.insert_text((x0 + 4, y0 + 24), "Name : P%d" % n, fontsize=8)
    return doc, pg


def test_wide_page_keeps_every_column():
    # S03 AC 42 part 243 page 25: columns 200.5 / 185 / 165 pt; find_boxes lost the 165 pt column (5 voters)
    import parse_textlayer as H
    doc, pg = _page_with_columns([(14, 214.5), (227, 411.8), (415.8, 581)])
    old = H.find_boxes(pg, pg.get_drawings())
    st = collections.Counter()
    new = A.find_boxes_wide(pg, pg.get_drawings(), st)
    assert len(new) == 6 and st["wide_pages"] == 1
    assert [round(b.x0) for b in new[:3]] == [14, 227, 416] and [round(b.width) for b in new[:3]] == [200, 185, 165]
    assert sorted(round(b.x0) for b in old[:3]) != [14, 227, 416]       # before: EPIC-anchored boxes for 2 of the 3 columns


def test_normal_page_unchanged():
    import parse_textlayer as H
    doc, pg = _page_with_columns([(14, 200), (204, 390), (394, 580)])
    st = collections.Counter()
    assert A.find_boxes_wide(pg, pg.get_drawings(), st) == H.find_boxes(pg, pg.get_drawings()) and not st


# --------------------------------------------------------------------------- Kerala

def test_chillu_space_removed_only_where_the_trace_has_no_space():
    # S11 AC 26 part 208 page 12 serial 282: text 'മേക്കോാണ്‍ പൊായില്', trace '...ണ്‍െപായില്' (no space glyph)
    raw = "മേക്കോാണ്‍ പൊായില്‍"
    tl = _trace("േമേക്കാണ്‍െപായില്‍")
    assert A.chillu_space_fix(raw, tl) == "മേക്കോാണ്‍പൊായില്‍"
    assert A.repair(A.chillu_space_fix(raw, tl), "MAL") == "മേക്കോൺപൊയിൽ"
    # S11 AC 71: 'സുബിന്‍ െക എസ്' has a space glyph in the trace: kept
    raw = "സുബിന്‍ െക എസ്"
    assert A.chillu_space_fix(raw, _trace("സുബിന്‍ െക എസ്")) == raw


def test_malayalam_trace_fixes():
    def fix(v, trace):
        a, b = A.letters(v, "MAL"), A.letters(trace, "MAL", trace=True)
        for kind, nv in A.mal_trace_fixes(v, b - a, a - b):
            if A.letters(nv, "MAL") == b:
                return kind, nv
        return None, v
    # 'റഹ്‍മാമാൻ' (the syllable after virama + ZWJ repeated)
    assert fix("അബ്ദുൽ റഹ്‍മാമാൻ", "അബ്ദുൽ റഹ്‍മാൻ") == ("zwj_repeat", "അബ്ദുൽ റഹ്‍മാൻ")
    # S11 AC 82 part 142 page 13 serial 327: 'സെയിൻ്റ്റ്' (text layer ൻ്റ്്റ്), trace 'െസയിന്റ്'
    assert fix("സെയിൻ്റ്റ്", "െസയിന്റ്") == ("nta_doubled", "സെയിന്റ്")
    assert fix("ആൻ്റു്റു", "ആന്റു") == ("nta_doubled", "ആന്റു")
    # S11 AC 133 part 205 page 12 serial 282: 'അൽാൺസ്' (ഫ + േ lost), trace 'അല്‍േഫാണ്‍സ്'
    assert fix("അൽാൺസ്", "അല്‍േഫാണ്‍സ്") == ("chillu_lost", "അൽഫോൺസ്")
    # nothing to do when the trace agrees, or when the difference is of another kind
    assert fix("ബാലൻ", "ബാലന്‍") == (None, "ബാലൻ")
    assert fix("മാമാങ്കം", "മാമാങ്കം") == (None, "മാമാങ്കം")


def test_colon_in_house_line_is_not_the_age_row():
    box = pymupdf.Rect(14, 28, 200, 103)
    # S19 AC 101 part 160 page 14 serial 334
    rows = [(56, 38.6, "334"), (142, 38.6, "YJD3589637"), (5, 50, "ਨਾਮ : ਸੋਨਿਕਾ ਰਾਣੀ "), (5, 59, "ਪਤੀ ਦਾ ਨਾਮ: ਜੋਨੀ ਕੁਮਾਰ "),
            (5, 68, "ਮਕਾਨ ਨੰ. : 162 ਵਾਰਡ ਨੰ: 13"), (5, 77, "ਉਮਰ : 40 ਲਿੰੰਗ : ਇਸਤਰੀ")]
    rec, rep, warn = A.parse_box(box, _chars(rows), [], "PAN", set())
    assert rec["house_no"] == "162 ਵਾਰਡ ਨੰ: 13" and rec["age"] == 40 and rec["gender_std"] == "F"
    assert not any(w.startswith("layout:") for w in warn) and "extra_text" not in warn
    # S11 AC 135 part 45 page 24 serial 644: the house line with ':' wraps to a row without ':'
    rows = [(56, 38.6, "644"), (142, 38.6, "UYX1676238"), (5, 50, "പേര് : സജികുമാര്‍ "),
            (5, 59, "അച്ഛന്റെ പേര്: എപിഎസ് നായര്‍ "), (5, 68, "വീട്ടു നമ്പർ : 10 ഏ വര്‍മ്മ ഡോ: പൈ "),
            (5, 77, "ഹെറിറ്റേജ"), (5, 86, "പ്രായം : 62 ലിംഗം : പുരുഷൻ")]
    rec, rep, warn = A.parse_box(pymupdf.Rect(14, 28, 200, 110), _chars(rows), [], "MAL", set())
    assert rec["house_no"] == "10 ഏ വർമ്മ ഡോ: പൈ ഹെറിറ്റേജ" and rec["age"] == 62 and rec["gender_std"] == "M"


def test_wrap_without_space_after_a_hyphen():
    box = pymupdf.Rect(14, 28, 200, 103)
    rows = [(56, 38.6, "5"), (142, 38.6, "ABC1234567"), (5, 50, "ਨਾਮ : ਰਾਜੂ "), (5, 59, "ਪਿਤਾ ਦਾ ਨਾਮ: ਸਰਵਣ "),
            (5, 68, "ਮਕਾਨ ਨੰ. : ਮਕਾਨ ਨੰਬਰ 155 , ਗਲੀ ਨੰਬਰ-"), (5, 74, "01 ,"), (5, 82, "ਉਮਰ : 51 ਲਿੰੰਗ : ਪੁਰਸ਼")]
    rec, rep, warn = A.parse_box(pymupdf.Rect(14, 28, 200, 106), _chars(rows), [], "PAN", set())
    assert rec["house_no"] == "ਮਕਾਨ ਨੰਬਰ 155 , ਗਲੀ ਨੰਬਰ-01 ,"
    # the line breaks at a space (S19 AC 19 part 171 page 27 serial 701: the trace draws the space): joined with it
    rows[4] = (5, 68, "ਮਕਾਨ ਨੰ. : ਥੀ ਫ਼੍ਰੀਡਮ-189/ਏਮਸੀਏ-ਐਮ-2 "); rows[5] = (5, 74, "835")
    rec, rep, warn = A.parse_box(pymupdf.Rect(14, 28, 200, 106), _chars(rows), [], "PAN", set())
    assert rec["house_no"].endswith("ਐਮ-2 835")


def test_photo_frame_when_no_photo_text_is_known():
    # S19 AC 86 part 165 page 3: the first voter page has one box, no text repeats: the photo text inside the frame is dropped
    box = pymupdf.Rect(14, 28.2, 200, 103)
    rows = [(68, 39, "1"), (154, 38.8, "ARC0002618"), (5 + 14, 52.3, "ਨਾਮ : ਮਨਜਿੰੰਦਰ ਸਿੰੰਘ"),
            (19, 61.3, "ਪਿਤਾ ਦਾ ਨਾਮ: ਕੁਲਵੰਤ ਸਿੰੰਘ"), (157, 66.4, "ਫੋਟੋ ਉਪਲਬਧ "), (19, 70.3, "ਮਕਾਨ ਨੰ. : 1266"),
            (19, 79.3, "ਉਮਰ : 37 ਲਿੰੰਗ : ਪੁਰਸ਼"), (172.3, 79.0, "ਹੈ")]
    chars = _chars(rows, step=0.3)
    frame = pymupdf.Rect(153.5, 44.2, 195.5, 98.8)
    rec, rep, warn = A.parse_box(box, chars, [], "PAN", set(), frame)
    assert rec["gender"] == "ਪੁਰਸ਼" and rec["gender_std"] == "M" and rec["house_no"] == "1266" and "right_span" not in warn
    rec, rep, warn = A.parse_box(box, chars, [], "PAN", set())          # without the frame: as before
    assert rec["gender"] != "ਪੁਰਸ਼"


# --------------------------------------------------------------------------- English (S22)

def _top(spans):
    out = []
    for sid, (x0, t) in enumerate(spans):
        out += [(ch, x0 + i * 4.0, 37.7, sid, t, (x0 + i * 4.0, 29, x0 + i * 4.0 + 4, 38)) for i, ch in enumerate(t)]
    return out


def test_section_label_over_the_serial_cell_english():
    toks, lab = E.top_tokens_eng(_top([(16.0, "Door No ( 1 To 81 ) & ( 2 To 82 )"), (155.7, "UKV1922103"), (65.3, "31")]))
    assert toks == ["31", "UKV1922103"] and lab == "Door No ( 1 To 81 ) & ( 2 To 82 )"
    toks, lab = E.top_tokens_eng(_top([(16.0, "999.OVERSEAS ELECTORS"), (156.5, "IQB1035344"), (62.7, "901")]))
    assert toks == ["901", "IQB1035344"] and lab == "999.OVERSEAS ELECTORS"
    toks, lab = E.top_tokens_eng(_top([(16.0, "Door No { 1 To 73 & 2 To 54 }"), (150, "ABC1234567"), (70, "721")]))
    assert toks == ["721", "ABC1234567"]
    toks, lab = E.top_tokens_eng(_top([(16.0, "00 )"), (150, "ABC1234567"), (70, "12")]))   # tail of a label from the left box
    assert toks == ["12", "ABC1234567"] and lab == "00 )"
    toks, lab = E.top_tokens_eng(_top([(16.0, "( MAIBANG PT-I)"), (150, "MSY7535037"), (68, "1")]))   # S03: as before
    assert toks == ["1", "MSY7535037"] and lab == "( MAIBANG PT-I)"
    toks, lab = E.top_tokens_eng(_top([(150, "ABC1234567"), (70, "12")]))
    assert toks == ["12", "ABC1234567"] and lab is None
