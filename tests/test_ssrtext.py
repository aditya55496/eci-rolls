"""Task ssrtext and the any-language text-layer parser (2026-11-25). Offline: no hub, no network.
Strings are as printed in the 2027 SSR draft rolls (samples in models2/ssrtext/samples)."""
import gzip
import io
import json
import os
import sys

import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "eciocr", "textlayer")]
os.environ.setdefault("SCRAPE_POOL_PROJECT", "eci-rolls-test")
import parse_textlayer_any as A  # noqa: E402
import ssrtext_task as S  # noqa: E402


# --------------------------------------------------------------------------- repairs

def test_repair_doubled_marks():
    assert A.repair("ਹਰਮਨ ਸਿੰੰਘ", "PAN") == "ਹਰਮਨ ਸਿੰਘ"
    assert A.repair("পুরুুষ", "BEN") == "পুরুষ"
    assert A.repair("সিংংহ", "BEN") == "সিংহ"
    assert A.repair("ശോാഭന", "MAL") == "ശോഭന"                 # NFD: ോ = െ + ാ, then ാ again
    assert A.repair("സ്ത്രീീ", "MAL") == "സ്ത്രീ"
    assert A.repair("ফটোা উপলব্ধ", "ASM") == "ফটো উপলব্ধ"


def test_repair_repeated_mark_blocks():
    assert A.repair("বৰগোঁাঁহাই", "ASM") == "বৰগোঁহাই"           # 'াঁ' twice
    assert A.repair("মোঃাঃ", "BEN") == "মোঃ"
    assert A.repair("সুভ্ৰাংাংশু", "ASM") == "সুভ্ৰাংশু"
    assert A.repair("এক্কা", "ASM") == "এক্কা"                    # letters repeat: kept
    assert A.repair("ಮೊನಪ್ಪ  ಶೆಟ್ಟಿಗರ್ ", "KAN") == "ಮೊನಪ್ಪ ಶೆಟ್ಟಿಗರ್"


def test_repair_bengali_ra_and_chillu():
    c = A.collections.Counter()
    assert A.repair("পিতাব় নাম", "BEN", c) == "পিতার নাম" and c["bengali_ra"] == 1
    assert A.repair("দনেশ্ব়", "BEN") == "দনেশ্বর"                 # after a virama: ব-phala + র
    assert A.repair("পিতাৰ নাম", "ASM") == "পিতাৰ নাম"            # Assamese ra untouched
    assert A.repair("ബാലന്‍", "MAL") == "ബാലൻ"
    assert A.repair("ബാലന്‍", "KAN") == "ബാലന്‍"        # chillu only for Malayalam


def test_letters_fold_for_glyph_trace():
    # the glyph trace of Bengali-script fonts gives ৰ for র and ব় for র; Kannada conjuncts carry no virama
    assert A.letters("কর্মকার", "BEN") == A.letters("কৰ্মকাব়", "BEN", trace=True)
    assert A.letters("ಶೆಟ್ಟಿಗರ್", "KAN") == A.letters("ಶೆಟಿ್ಟಗರ್", "KAN", trace=True)


def test_toint_native_digits():
    assert A.toint("১২") == 12 and A.toint("12") == 12 and A.toint("൧൦") == 10
    assert A.toint("12a") is None and A.toint("") is None


# --------------------------------------------------------------------------- rows -> fields

def test_split_label_and_age_rows():
    assert A.split_label("ਮਕਾਨ ਨੰ. : 00") == ("ਮਕਾਨ ਨੰ.", "00")
    assert A.split_label("ಭಾಗ ಸಂಖ್ಯೆ : : 3") == ("ಭಾಗ ಸಂಖ್ಯೆ", "3")
    assert A.split_label("ਮਕਾਨ ਨੰ. : ") == ("ਮਕਾਨ ਨੰ.", "")
    for t, age, g in (("ਉਮਰ : 36 ਲਿੰਗ : ਇਸਤਰੀ", "36", "ਇਸਤਰੀ"), ("പ്രായം : 63 ലിംഗം : സ്ത്രീ", "63", "സ്ത്രീ"),
                      ("বয়স : 60 লিঙ্গ : পুৰুষ", "60", "পুৰুষ"), ("ವಯಸ್ಸು : 75 ಲಿಂಗ : ಹೆಣ್ಣು", "75", "ಹೆಣ್ಣು")):
        assert A.is_age_row(t)
        m = A.AGE_ROW.match(t)
        assert (m.group(2), m.group(4)) == (age, g)
    assert not A.is_age_row("ਮਕਾਨ ਨੰ. : 12")
    assert not A.is_age_row("ਨਾਮ : ਸੀਤਾ")


def _box_chars(box, spans):
    """Synthetic rawdict chars (ch, ox, oy, span_id, span_text, bbox) for spans [(dx, dy, text)] in a box."""
    out = []
    for sid, (dx, dy, text) in enumerate(spans, 1):
        x = box.x0 + dx
        for ch in text:
            out.append((ch, x, box.y0 + dy, sid, text, (x, box.y0 + dy - 7, x + 4, box.y0 + dy + 1)))
            x += 4
    return out


def test_parse_box_by_position_wrap_and_photo():
    box = pymupdf.Rect(14, 28, 200, 103)
    chars = _box_chars(box, [
        (56, 10.6, "17"), (142, 10.6, "TQL8576613"),
        (5, 21.9, "ਨਾਮ : ਰਮਨਦੀਪ ਕੌਰ "),
        (5, 30.9, "ਪਤੀ ਦਾ ਨਾਮ: ਸ਼ਮਸ਼ੇਰ "),
        (150, 35.4, "ਫੋਟੋ ਉਪਲਬਧ "),
        (5, 39.9, "ਸਿੰੰਘ"),                                   # wrapped relation name
        (5, 48.9, "ਮਕਾਨ ਨੰ. : 1"),
        (5, 57.9, "ਉਮਰ : 32 ਲਿੰੰਗ : ਇਸਤਰੀ"), (158, 57.9, "ਹੈ"),
    ])
    rec, rep, warn = A.parse_box(box, chars, [], "PAN", {"ਫੋਟੋ ਉਪਲਬਧ", "ਹੈ"})
    assert rec["serial_no"] == 17 and rec["epic"] == "TQL8576613"
    assert rec["name"] == "ਰਮਨਦੀਪ ਕੌਰ" and rec["relation_label"] == "ਪਤੀ ਦਾ ਨਾਮ" and rec["relation_type"] == "Husband"
    assert rec["relation_name"] == "ਸ਼ਮਸ਼ੇਰ ਸਿੰਘ" and rec["house_no"] == "1"
    assert rec["age"] == 32 and rec["gender"] == "ਇਸਤਰੀ" and rec["gender_std"] == "F"
    assert "wrapped_line:1" in warn and rep["doubled_mark"] >= 1
    assert rec["raw_lines"][0] == "17TQL8576613" or rec["raw_lines"][0].endswith("TQL8576613")
    assert any("ਸਿੰੰਘ" in x for x in rec["raw_lines"])               # raw text kept before the repair


def test_parse_box_unknown_label_and_empty_house():
    box = pymupdf.Rect(14, 28, 200, 103)
    chars = _box_chars(box, [(56, 10.6, "3"), (142, 10.6, "INR4240305"), (5, 22, "ਨਾਮ : ਰਾਜੂ "), (5, 31, "ਭਰਾ ਦਾ ਨਾਮ: ਸਰਵਣ "),
                             (5, 40, "ਮਕਾਨ ਨੰ. : "), (5, 49, "ਉਮਰ : 51 ਲਿੰੰਗ : ਪੁਰਸ਼")])
    rec, rep, warn = A.parse_box(box, chars, [], "PAN", set())
    assert rec["relation_type"] is None and "relation_label_unknown" in warn
    assert rec["house_no"] == "" and "missing:house_no" in warn and rec["gender_std"] == "M"


# --------------------------------------------------------------------------- a whole PDF through the task

LAB = ("Name", "Fathers Name", "House Number", "Age", "Gender")


def make_pdf(path, voters, cover=True, summary=True, ac=7, part=2):
    """A roll-like PDF: cover with the totals row, one page of 186 x 75 pt boxes, a summary page. Latin text: the
    any-language parser needs no label words, so the layout code is tested without Indic fonts."""
    doc = pymupdf.open()
    m = sum(1 for v in voters if v[3] == "Male"); f = len(voters) - m
    pg = doc.new_page(width=595, height=842)
    pg.insert_text((16, 30), f"ELECTORAL ROLL 2027 S99 Test : {ac} - Test (GEN)")
    if cover:
        for x, v in zip((40, 120, 200, 280, 360, 440), (1, len(voters), m, f, 0, len(voters))):
            pg.insert_text((x, 500), str(v))
    pg.insert_text((480, 830), "Pages 3 - Page 1")
    pg = doc.new_page(width=595, height=842)
    pg.insert_text((16, 10), f"Assembly : {ac}-Test")
    pg.insert_text((520, 10), f"Part : {part}")
    pg.insert_text((16, 22), "Section : 1-Main Road")
    for i, (name, rel, house, gender, age) in enumerate(voters):
        x0, y0 = 14 + (i % 3) * 190, 28 + (i // 3) * 80
        pg.draw_line((x0, y0), (x0 + 186, y0)); pg.draw_line((x0, y0 + 75), (x0 + 186, y0 + 75))
        pg.insert_text((x0 + 56, y0 + 10.6), str(i + 1)); pg.insert_text((x0 + 120, y0 + 10.6), "ABC%07d" % (i + 1))
        pg.insert_text((x0 + 5, y0 + 22), f"Name : {name}", fontsize=8)
        pg.insert_text((x0 + 5, y0 + 31), f"Fathers Name: {rel}", fontsize=8)
        pg.insert_text((x0 + 150, y0 + 36), "Photo", fontsize=7)
        pg.insert_text((x0 + 5, y0 + 40), f"House Number : {house}", fontsize=8)
        pg.insert_text((x0 + 5, y0 + 49), f"Age : {age} Gender : {gender}", fontsize=8)
    pg.insert_text((480, 830), "Pages 3 - Page 2")
    pg = doc.new_page(width=595, height=842)
    if summary:
        for x, v in zip((400, 450, 500, 550), (m, f, 0, len(voters))):
            pg.insert_text((x, 300), str(v))
    pg.insert_text((480, 830), "Pages 3 - Page 3")
    doc.save(path)


VOTERS = [("Asha Devi", "Ram Lal", "1", "Female", 40), ("Ram Lal", "Shyam Lal", "1", "Male", 66),
          ("Gopal", "Ram Lal", "2A", "Male", 31), ("Sita", "Gopal", "2A", "Female", 29)]


def run_item(tmp_path, pdfs, lang="TST", extra_parts=()):
    A.GENDER["TST"] = {"Male": "M", "Female": "F"}
    A.REL["TST"] = {"Fathers Name": "Father"}
    keys = [f"src/pdf/S99-2027-DR/7/{os.path.basename(p)}" for p in pdfs]
    src = dict(zip(keys, pdfs))
    item = {"key": "S99-2027-DR/7", "payload": {"state": "S99", "ac": 7, "lang": lang, "roll": {"id": "S99-2027-DR", "year": 2027},
                                               "src_project": "src", "keys": keys + ["src/pdf/S99-2027-DR/7/absent.pdf"],
                                               "extra_parts": list(extra_parts)}}
    files = {}

    def fetch(ks, dest):
        return {k: src.get(k) for k in ks}

    def upload(fs):
        files.update({p: d for p, d, _ in fs})

    rec = S.do_item(None, item, fetch=fetch, upload=upload, workers=1)
    return rec, files


def test_task_end_to_end(tmp_path):
    p1 = str(tmp_path / "2027-EROLLGEN-S99-7-SSR-DraftRoll-Revision1-TST-2-WI.pdf")
    p2 = str(tmp_path / "2027-EROLLGEN-S99-7-SSR-DraftRoll-Revision1-TST-3-WI.pdf")
    make_pdf(p1, VOTERS, part=2)
    make_pdf(p2, VOTERS[:3], part=3, summary=False)
    rec, files = run_item(tmp_path, [p1, p2], extra_parts=[3])
    assert rec["records"] == 7 and rec["pdfs"] == 3 and rec["pdfs_parsed"] == 2 and rec["n_missing_pdfs"] == 1
    assert rec["match_pdfs"] == 2 and rec["gender_match_pdfs"] == 2 and rec["printed_total"] == 7 and rec["parser"] == "any"
    recs = [json.loads(x) for x in gzip.decompress(files["text/S99-2027-DR/7.jsonl.gz"]).decode().splitlines()]
    r = recs[0]
    assert (r["name"], r["relation_name"], r["house_no"], r["age"], r["gender_std"], r["relation_type"]) == \
        ("Asha Devi", "Ram Lal", "1", 40, "F", "Father")
    assert r["state"] == "S99" and r["ac"] == 7 and r["part"] == 2 and r["roll"] == "S99-2027-DR" and r["section_no"] == 1
    assert r["raw_lines"] and all(x.get("extra_part") for x in recs if x["part"] == 3)
    hints = [json.loads(x) for x in gzip.decompress(files["hints/S99-2027-DR/7.jsonl.gz"]).decode().splitlines()]
    assert len(hints) == 7 and hints[0]["epic"] == "ABC0000001" and hints[0]["age"] == "40" and hints[0]["serial_no"] == "1"
    summ = json.loads(files["text/S99-2027-DR/7.summary.json"])
    assert [s["printed"]["cover_total"] for s in summ] == [4, 3] and summ[0]["printed"]["summary_total"] == 4
    assert summ[0]["match"] and summ[1]["match"] and summ[0]["printed"]["printed_pages"] == 3
    from parquet_task import pq
    t = pq.read_table(io.BytesIO(files["parquet/S99-2027-DR/ac_7/data.parquet"]))
    assert t.num_rows == 7 and t.column("roll")[0].as_py() == "S99-2027-DR" and t.column("age")[0].as_py() == 40
    extra = json.loads(t.column("extra")[0].as_py())
    assert extra["state"] == "S99" and extra["gender_std"] == "F" and extra["raw_lines"]


def test_task_count_mismatch(tmp_path):
    p = str(tmp_path / "2027-EROLLGEN-S99-7-SSR-DraftRoll-Revision1-TST-4-WI.pdf")
    make_pdf(p, VOTERS, part=4)
    doc = pymupdf.open(p)                         # cover says 4; drop one voter box's text: 3 records
    pg = doc[1]
    pg.add_redact_annot(pymupdf.Rect(204, 28, 390, 103)); pg.apply_redactions()
    q = p.replace("-4-WI", "-5-WI"); doc.save(q)
    rec, files = run_item(tmp_path, [q])
    assert rec["records"] == 3 and rec["match_pdfs"] == 0 and rec["mismatch"][0]["cover_total"] == 4


def test_normalise_hindi_and_english_records():
    item = {"payload": {"state": "S04", "ac": 12, "lang": "HIN", "roll": {"id": "S04-2027-DR"}}}
    r = S.normalise({"ac": 64, "part": 5, "relation_type": "पति", "gender": "महिला", "epic": "ABC1234567", "serial_no": 3},
                    item, 5, {"parser": "hin"}, False)
    assert r["ac"] == 12 and r["ac_in_pdf"] == 64 and r["relation_label"] == "पति का नाम" and r["gender_std"] == "F"
    assert r["relation_type"] == "Husband" and r["relation_type_printed"] == "पति" and r["state"] == "S04" and "extra_part" not in r
    h = S.hint(r, "hin")
    assert h["relation_label"] == "पति का नाम" and h["ac"] == "12" and h["state"] == "S04"
    item["payload"].update(state="S22", lang="ENG")
    r = S.normalise({"ac": 12, "part": 9, "relation_type": "Father", "relation_label": "Fathers Name", "gender": "Male",
                     "epic": "abc1234567", "serial_no": 1, "state": "S22"}, item, 9, {"parser": "eng"}, True)
    assert r["gender_std"] == "M" and r["relation_type"] == "Father" and r["extra_part"] is True
    assert S.hint(r, "eng")["epic"] == "ABC1234567"
    r = S.normalise({"ac": 12, "part": 9, "relation_type": "Other", "relation_label": "Other", "gender": "Male",
                     "epic": "ABC1234567", "serial_no": 2}, item, 9, {"parser": "eng"}, False)
    assert r["relation_type"] == "Others" and r["relation_type_printed"] == "Other"   # one vocabulary for all languages


def test_hint_missing_age_is_null():
    r = {"epic": "PUV0815399", "name": "X", "relation_label": "Fathers Name", "relation_name": "Y", "gender": None,
         "serial_no": 554, "state": "U06", "ac": 1, "part": 3}
    assert S.hint(r, "eng")["age"] is None and S.hint(r, "any")["age"] is None and S.hint(r, "eng")["serial_no"] == "554"
    r.update(relation_type="Father", relation_label="पिता का नाम")
    assert S.hint(r, "hin")["age"] is None and S.hint(r, "hin")["relation_label"] == "पिता का नाम"


# --------------------------------------------------------------------------- 2026-11-25 verifier findings

def test_hindi_space_before_mark():
    import parse_textlayer as H
    c = A.collections.Counter()
    assert H.repair("गिरीश ंिसंह", c) == "गिरीश सिंह" and c["orphan_mark_dropped"] and c["visual_order"]
    assert H.repair("संगीता ंकंवर चौहान", c) == "संगीता कंवर चौहान"
    assert H.repair("अस्मिता ुदेवी", c) == "अस्मिता देवी"
    assert H.repair("सुनीता ेदेवी", c) == "सुनीता देवी"                 # the sign of the next syllable
    assert H.repair("ललन ुसाह", c) == "ललन साह"                       # a frequent word on the right
    assert H.repair("उमा शंकर ॉसिंह", c) == "उमा शंकर सिंह"
    assert H.repair(" ुदुरपती देबी", c) == "दुरपती देबी"               # start of the value
    assert H.repair("बृजकिशोर िंसह", c) == "बृजकिशोर सिंह"             # visual order (Bihar)
    assert H.repair("मंजु िंसंह", c) == "मंजु सिंह"
    c = A.collections.Counter()
    assert H.repair("कृष्णा क ुशवाहा", c) == "कृष्णा कुशवाहा" and c["space_before_mark"] and not c["orphan_mark_dropped"]
    assert H.repair("भगत प्रस ाद", c) == "भगत प्रसाद"                 # fragments: joined
    assert H.repair("विक्र ांत कुमार", c) == "विक्रांत कुमार"
    assert H.repair("जानक ी देवी", c) == "जानकी देवी"
    assert H.repair("परसराम े", c) == "परसराम"                       # a stray sign at the end of a word
    assert H.repair("राम सिंह", c) == "राम सिंह" and H.repair("सिंह", c) == "सिंह"


def test_bengali_wa_khanda_ba_phala():
    assert A.repair("পোরোমেশ্ৱর", "BEN") == "পোরোমেশ্বর"
    assert A.repair("পোরোমেশ্ৱর", "ASM") == "পোরোমেশ্ৱর"           # Assamese wa kept
    assert A.repair("ৰাজিয়া", "BEN") == "ৰাজিয়া"                   # ৰ has its own glyph: kept
    # glyph trace of 'নীলোৎপল' (khanda ta glyph = ত + virama) and of 'উদ্বব শিল' (ব-phala conjunct, then ব)
    tl = [(c, x, 0, g) for c, x, g in [("ন", 1, 358), ("ী", 2, 374), ("ে", 3, 379), ("ল", 4, 366), ("া", 5, 372),
                                         ("ত", 6, 714), ("্", 6, -1), ("প", 7, 359), ("ল", 8, 366)]]
    assert A.bengali_trace_fixes("নীলোত্পল", tl) == ("নীলোৎপল", ["khanda_ta"])
    assert A.letters("নীলোৎপল", "BEN") == A.letters("নীলোত্পল", "BEN", trace=True)
    tl = [(c, x, 0, g) for c, x, g in [("উ", 1, 331), ("দ", 2, 570), ("ব", 2, -1), ("্", 2, -1), ("ব", 3, 361),
                                         (" ", 4, 3), ("ি", 5, 373), ("শ", 6, 367), ("ল", 7, 366)]]
    assert A.bengali_trace_fixes("উদ্ব শিল", tl) == ("উদ্বব শিল", ["ba_after_ba_phala"])
    tl = [(c, x, 0, g) for c, x, g in [("শ", 1, 634), ("ব", 1, -1), ("্", 1, -1), ("ব", 2, 365), ("়", 2, -1)]]
    assert A.bengali_trace_fixes("শ্বর", tl) == ("শ্বর", [])              # ব-phala + র: nothing to add


def test_lam_alef_order():
    # rawdict: ' ' 'ا' 'ل' 'ل' (alef before lam); glyph trace: ligature lam (gid) + alef (gid -1) at x 255.9, then lam
    chars = [(" ", 260.0, 100.0, 7, "", None), ("ا", 260.0, 100.0, 7, "", None), ("ل", 255.9, 100.0, 7, "", None),
             ("ل", 251.6, 100.0, 7, "", None)]
    tch = [("ل", 251.6, 100.0, 149), ("ل", 255.9, 100.0, 227), ("ا", 255.9, 100.0, -1)]
    out, n = A.lam_alef_fix(chars, tch)
    assert n == 1 and "".join(c[0] for c in out) == " لال"
    tch = [("ا", 260.0, 100.0, 12), ("ل", 255.9, 100.0, 40)]                # 'ال' as two glyphs ('الطاف'): kept
    assert A.lam_alef_fix(chars, tch)[1] == 0


def test_box_edge_widths_and_epic_anchor():
    import parse_textlayer as H
    doc = pymupdf.open()
    pg = doc.new_page(width=595, height=842)
    for x0, x1 in ((14, 197.1), (201.1, 394.5), (398.5, 581)):            # 183 / 193 / 183 pt columns (S11 AC 81 part 277)
        for y in (30, 105):
            pg.draw_line((x0, y), (x1, y))
        pg.insert_text((x1 - 70, 42), "ABC%07d" % int(x0), fontsize=9)
    boxes = H.find_boxes(pg, pg.get_drawings())
    assert len(boxes) == 3 and [round(b.width) for b in boxes] == [183, 193, 183]
    H.BOX_W, old = (180, 192), H.BOX_W                                   # the 193 pt column out of range: anchored on its EPIC
    try:
        st = A.collections.Counter()
        boxes = H.find_boxes(pg, pg.get_drawings(), st)
        assert len(boxes) == 3 and st["epic_anchored"] == 1 and 185 < boxes[1].x0 < 205 and boxes[0].x0 < 20
    finally:
        H.BOX_W = old


def test_english_section_without_colon():
    import parse_textlayer_eng as E
    assert E.SECTION_RE.search("Section No and Name  1-Pudhur (R.V)").groups() == ("1", "Pudhur (R.V)")
    assert E.SECTION_RE.search("Section No and Name : 2-Block 2").groups() == ("2", "Block 2")


def test_plan_items():
    import plan_ssrtext as PL
    k = "eci-rolls-all-cdn/pdf/S19-2027-DR/21/2027-EROLLGEN-S19-21-SSR-DraftRoll-Revision1-PAN-%d-WI.pdf"
    lines = ["eci-rolls-all-cdn\tS19-2027-DR/21\t" + json.dumps({"lang": "PAN", "lang_used": "PAN", "extra_parts": [12]}),
             "eci-rolls-all-cdn\tS19-2027-DR/21/10\t" + json.dumps({"key": k % 10}),
             "eci-rolls-all-cdn\tS19-2027-DR/21/2\t" + json.dumps({"key": k % 2}),
             "eci-rolls-all-cdn\tS19-2027-DR/21/12\t" + json.dumps({"key": k % 12}),
             "eci-rolls-hi-cdn\tS24-2027-DR/1/1\t" + json.dumps({"key": "x/S24-1-HIN-1-WI.pdf"})]
    items, summary, notes = PL.plan(lines)
    assert len(items) == 1 and summary == {"S19 PAN": 1, "pdfs": 3} and not notes
    p = items[0]["payload"]
    assert items[0]["key"] == "S19-2027-DR/21" and p["lang"] == "PAN" and p["src_project"] == "eci-rolls-all-cdn"
    assert [S.part_of(x) for x in p["keys"]] == [2, 10, 12] and p["extra_parts"] == [12] and p["roll"] == {"id": "S19-2027-DR", "year": 2027}


def test_ba_after_ba_phala_kept_only_when_trace_agrees():
    # 'দ্বিব' is drawn ি, দ্ব, ব: the ব after the ব-phala glyph is the next syllable, nothing is missing
    tl = [(c, x, 0, g) for c, x, g in [("ি", 1, 373), ("দ", 2, 570), ("ব", 2, -1), ("্", 2, -1), ("ব", 3, 361)]]
    v, kinds = A.bengali_trace_fixes("দ্বিব", tl)
    assert kinds == ["ba_after_ba_phala"] and A.letters(v, "BEN") != A.letters("িদব্ব", "BEN", trace=True)
    assert A.bengali_trace_fixes("দ্বিব", tl, ba=False) == ("দ্বিব", [])
