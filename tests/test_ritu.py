"""Hindi text-layer parser fixes of 2026-11-25 (owner order "fix bihar parser" / "fix all the faults"). Offline: no PDF, no hub.
Glyph ids and texts are as in the 2027 SSR draft rolls (Bihar AC 175 part 178, AC 8 part 284, AC 11 part 212; MP AC 69 part 33).

- glyph uni0943 (vowel sign ृ, gid 3811 in the NotoSans-Regular subsets) given as 'र्' by a wrong ToUnicode CMap
- a box row in visual order (rows with Arabic-script characters): 'िनवार्चक का नाम', 'पित का नाम'
- an EPIC printed with non-ASCII characters or without letters: in epic, not in status_code
"""
import collections
import os
import sys

import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "eciocr", "textlayer")]
os.environ.setdefault("SCRAPE_POOL_PROJECT", "eci-rolls-test")
import parse_textlayer as H  # noqa: E402

BB = (0, 0, 1, 1)
# the embedded font's own cmap, reversed (font_reverse_cmaps): gid -> code points
REV = {"NotoSans-Regular": [{3765: {0x915}, 3792: {0x930}, 3811: {0x943}, 3755: {0x90B}, 3951: set(), 3779: {0x923}}]}


def span(chars, font="NotoSans-Regular"):
    return {"font": font, "chars": chars}


# --------------------------------------------------------------------------- glyph uni0943 given as 'र्'

def faulty_trace():
    # 'कृष्ण' as the faulty PDF has it: क (3765), glyph 3811 with text 'र' + '्' (second char gid -1, same origin), ष् (3951), ण
    return [span([(0x915, 3765, (66.1, 300.0), BB), (0x930, 3811, (69.9, 300.0), BB), (0x94D, -1, (69.9, 300.0), BB),
                  (0x937, 3951, (71.4, 300.0), BB), (0x94D, -1, (71.4, 300.0), BB), (0x923, 3779, (73.7, 300.0), BB)])]


def test_glyph_fix_vocalic_r_from_trace():
    fixes = H.glyph_fixes(faulty_trace(), REV)
    assert fixes == {(69.9, 300.0): ("र्", "ृ")}


def test_no_fix_when_the_cmap_is_right():
    good = [span([(0x915, 3765, (66.1, 300.0), BB), (0x943, 3811, (68.5, 300.0), BB), (0x937, 3951, (71.4, 300.0), BB)])]
    assert H.glyph_fixes(good, REV) == {}
    # a real reph (gid 3918, no code in the font's own cmap) is never changed
    reph = [span([(0x930, 3918, (30.2, 300.0), BB), (0x94D, -1, (30.2, 300.0), BB)])]
    assert H.glyph_fixes(reph, REV) == {}
    # 'र्' text on glyph 3811 but the virama has its own glyph (not the same glyph): no fix
    other = [span([(0x930, 3811, (69.9, 300.0), BB), (0x94D, 4000, (70.9, 300.0), BB)])]
    assert H.glyph_fixes(other, REV) == {}
    # the font is unknown: no fix
    assert H.glyph_fixes([dict(faulty_trace()[0], font="GJOT-Avantika")], REV) == {}


def test_old_fix_of_vocalic_r_letter_unchanged():
    tr = [span([(0x930, 3755, (10.0, 300.0), BB), (0x937, 3951, (14.0, 300.0), BB)])]
    assert H.glyph_fixes(tr, REV) == {(10.0, 300.0): ("र", "ऋ")}


def raw_span(text_xs):
    return [{"c": c, "origin": (x, 300.0), "bbox": (x, 290, x + 2, 302)} for c, x in text_xs]


def test_apply_multi_fixes_on_rawdict_chars():
    fixes = H.glyph_fixes(faulty_trace(), REV)
    chs = raw_span([("क", 66.1), ("र", 69.9), ("्", 69.9), ("ष", 71.4), ("्", 74.1), ("ण", 73.7)])
    got = H.apply_multi_fixes(chs, fixes)
    assert "".join(c["c"] for c in got) == "कृष्ण" and got[1]["origin"] == (69.9, 300.0)
    # a real 'र्' at another origin stays ('शर्मा')
    chs = raw_span([("श", 10.0), ("र", 69.9), ("्", 70.4), ("म", 71.0), ("ा", 75.0)])
    assert "".join(c["c"] for c in H.apply_multi_fixes(chs, fixes)) == "शर्मा"


def test_texttrace_chars_apply_the_fix():
    fixes = H.glyph_fixes(faulty_trace(), REV)
    t = H.texttrace_chars(faulty_trace(), fixes)
    assert "".join(c[0] for c in t) == "कृष्ण"
    assert H.texttrace_chars(faulty_trace(), {}) and "".join(c[0] for c in H.texttrace_chars(faulty_trace(), {})) == "कर्ष्ण"


def box_chars(rows, x0=5):
    """rows: [(text, y)] -> rawdict-like chars (one span per row; an EPIC row at x 120)."""
    out = []
    for sid, (t, y) in enumerate(rows):
        x = 120 if t[:1].isalpha() and t.isascii() else x0
        for i, ch in enumerate(t):
            out.append((ch, x + i * 1.0, y, sid, t, (x + i, y - 8, x + i + 1, y)))
    return out


def tr(chars):
    """glyph-trace chars for parse_box (same text; spaces drawn)"""
    return [(c[0], c[1], c[2], 1) for c in chars]


def test_parse_box_counts_the_fix_and_reads_the_label():
    # MP AC 69 part 33: the house label 'गृह संख्या' came as 'गर्ह संख्या' (the house number was lost into the relation name)
    box = pymupdf.Rect(0, 0, 186, 75)
    rows = [("12", 10), ("निर्वाचक का नाम : कृष्णा देवी", 28), ("पति का नाम: रामकृष्ण", 38), ("गृह संख्या : 01", 48),
            ("उम्र : 45 लिंग : महिला", 58)]
    chars = box_chars(rows)
    # the 'ृ' chars as page_chars gives them after the fix; fixes holds the glyph origins
    fixes = {(round(c[1], 1), round(c[2], 1)): ("र्", "ृ") for c in chars if c[0] == "ृ"}
    rec, rep, warn = H.parse_box(box, chars, tr(chars), fixes)
    assert rec["name"] == "कृष्णा देवी" and rec["relation_name"] == "रामकृष्ण" and rec["house_no"] == "01"
    assert rep["glyph_fix_vocalic_r"] == 3


# --------------------------------------------------------------------------- rows in visual order

def test_visual_row_to_logical():
    v = H.visual_to_logical
    assert v("िनवार्चक का नाम : मो۰सािबर हुसैन") == "निर्वाचक का नाम : मो۰साबिर हुसैन"
    assert v("पित का नाम: मो ٠वजैर") == "पति का नाम: मो ٠वजैर"
    assert v("िनवार्चक का नाम : कलاमुन िनशा") == "निर्वाचक का नाम : कलاमुन निशा"
    assert v("िनवार्चक का नाम : िंसह") == "निर्वाचक का नाम : सिंह"
    for t in ("िनवार्चक का नाम : x", "पित का नाम: x", " िपता का नाम:: x"):
        assert H.VISUAL_LABEL.match(t), t
    for t in ("निर्वाचक का नाम : किशोर", "पिता का नाम: x", "पति का नाम: x", "पितांबर का नाम: x", "मकान संख्या : 3"):
        assert not H.VISUAL_LABEL.match(t), t


def test_parse_box_visual_label_row():
    # Bihar AC 13 part 291 serial 842: the name row came in visual order; the name was null and the relative's name lost
    box = pymupdf.Rect(0, 0, 186, 75)
    rows = [("842", 10), ("AZO0135699", 10), ("िनवार्चक का नाम : मो۰सािबर हुसैन", 28), ("पिता का नाम:: शेख इसा मोहम्मद", 38),
            ("मकान संख्या : 123", 48), ("उम्र : 43 लिंग: : पुरुष", 58)]
    rec, rep, warn = H.parse_box(box, box_chars(rows), tr(box_chars(rows)), {})
    assert rec["name"] == "मो۰साबिर हुसैन" and rec["relation_type"] == "पिता" and rec["relation_name"] == "शेख इसा मोहम्मद"
    assert rep["visual_row"] == 1 and "visual_row" in warn and "extra_text" not in warn
    rows[3] = ("पित का नाम: मो ٠वजैर", 38)
    rows[2] = ("निर्वाचक का नाम : नूर जहाँ", 28)
    rec, rep, warn = H.parse_box(box, box_chars(rows), tr(box_chars(rows)), {})
    assert rec["relation_type"] == "पति" and rec["relation_name"] == "मो ٠वजैर" and rec["name"] == "नूर जहाँ"


def test_normal_rows_unchanged():
    box = pymupdf.Rect(0, 0, 186, 75)
    rows = [("5", 10), ("ABC1234567", 10), ("निर्वाचक का नाम : किशोर शर्मा", 28), ("पिता का नाम:: रामकिशोर", 38),
            ("मकान संख्या : 7", 48), ("उम्र : 33 लिंग: : पुरुष", 58)]
    rec, rep, warn = H.parse_box(box, box_chars(rows), tr(box_chars(rows)), {})
    assert rec["name"] == "किशोर शर्मा" and rec["relation_name"] == "रामकिशोर" and rec["epic"] == "ABC1234567"
    assert "visual_row" not in rep and not [w for w in warn if not w.startswith("glyph_mismatch")]


# --------------------------------------------------------------------------- EPIC printed with odd characters

def test_odd_epic_print_goes_to_epic():
    box = pymupdf.Rect(0, 0, 186, 75)
    for epic in ("MGच2486231", "BR/12/015Ü201135", "8607269969", "H«H4413258"):
        chars = box_chars([("208", 10), ("निर्वाचक का नाम : शिव महतो", 28), ("पिता का नाम:: छठु महतो", 38),
                           ("मकान संख्या : 31", 48), ("उम्र : 43 लिंग: : पुरुष", 58)])
        chars += [(ch, 120 + i, 10, 99, epic, (120 + i, 2, 121 + i, 10)) for i, ch in enumerate(epic)]
        rec, rep, warn = H.parse_box(box, chars, tr(chars), {})
        assert rec["epic"] == epic and "status_code" not in rec and "epic_nonstandard" in warn, epic
        assert "missing:epic" not in warn
    # a short stamp stays in status_code
    chars = box_chars([("208", 10), ("ABC1234567", 10), ("निर्वाचक का नाम : शिव महतो", 28)])
    chars += [("D", 100, 10, 98, "D", (100, 2, 101, 10))]
    rec, rep, warn = H.parse_box(box, chars, tr(chars), {})
    assert rec["epic"] == "ABC1234567" and rec["status_code"] == "D"
