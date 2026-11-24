"""Printed label variants of the Hindi-script SIR states (2026-11-25): the OCR line parser and the 2027 text-layer row regexes
give the same fields for every variant. Lines as printed (checked on page images and 2027 text layers, models2/hinvar/NOTES.md)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "eciocr", "ocr"), os.path.join(ROOT, "eciocr", "textlayer"), os.path.join(ROOT, "eciocr", "prod")]
from parse import from_lines  # noqa: E402

BOX = {  # state: (line0, line1, line2, line3) -> expected (name, relation_label, relation_name, house_no, age, gender)
    "UP": (("नाम : राम कुमार", "पिता का नाम: श्याम लाल", "मकान संख्या : 12", "आयु : 40 लिंग : पुरुष"),
           ("राम कुमार", "पिता का नाम", "श्याम लाल", "12", "40", "पुरुष")),
    "Bihar": (("निर्वाचक का नाम : निखिल कुमार", "पिता का नाम:: संजीव कुमार सिंह", "मकान संख्या : 00", "उम्र : 26 लिंग: : पुरुष"),
              ("निखिल कुमार", "पिता का नाम", "संजीव कुमार सिंह", "00", "26", "पुरुष")),
    "MP": (("निर्वाचक का नाम : सुषमा जैन", "पति का नाम: अजित कुमार", "गृह संख्या : 1", "उम्र : 67 लिंग : महिला"),
           ("सुषमा जैन", "पति का नाम", "अजित कुमार", "1", "67", "महिला")),
    "Chandigarh": (("नाम : सतीश कुमार", "पिता: पिरथी राम", "मकान नं. : 4141", "आयु : 40 लिंग : पुरुष"),
                   ("सतीश कुमार", "पिता का नाम", "पिरथी राम", "4141", "40", "पुरुष")),
    "WB-Nepali": (("नाँउ : कल्पना सुब्बा", "पिताको नाँउ: आशबहादुर सुब्बा", "घर संख्या : n0065", "उमेर : 35 लिंग : स्त्री"),
                  ("कल्पना सुब्बा", "पिता का नाम", "आशबहादुर सुब्बा", "n0065", "35", "महिला")),
    "no-colon": (("निर्वाचक का नाम सुषमा जैन", "पति का नाम: अजित कुमार", "गृह संख्या 1", "उम्र 67 लिंग : महिला"),
                 ("सुषमा जैन", "पति का नाम", "अजित कुमार", "1", "67", "महिला")),
}


def test_ocr_lines_all_variants():
    for st, (lines, want) in BOX.items():
        r = from_lines(dict(serial="5", epic="ABC1234567", **{f"line{i}": t for i, t in enumerate(lines)}))
        got = tuple(r[k] for k in ("name", "relation_label", "relation_name", "house_no", "age", "gender"))
        assert got == want, (st, got)


def test_textlayer_rows_all_variants():
    import parse_textlayer as T
    m = T.NAME_RE.match("निर्वाचक का नाम : गीता देवी"); assert m and m.group(2) == "गीता देवी"
    assert not T.REL_RE.match("निर्वाचक का नाम : गीता देवी")           # the voter's own name, not a relative
    m = T.REL_RE.match("पिता का नाम:: महंत शर्मा"); assert m.group(1) == "पिता" and m.group(3) == "महंत शर्मा"
    m = T.HOUSE_RE.match("गृह संख्या : 244"); assert m.group(2) == "244"
    m = T.AGE_RE.match("उम्र : 53 लिंग: : महिला"); assert (m.group(2), m.group(3)) == ("53", "महिला")
    m = T.AGE_RE.match("आयु : 36 लिंग : महिला"); assert (m.group(2), m.group(3)) == ("36", "महिला")
    for t, no in (("अनुभाग संख्या और नाम : 1-मालवीय नगर", 1), ("प्रभाग संख्या एवं नाम : 2-फरसाहनी", 2),
                  ("अनुभाग की संख्या व नाम : 3-सुभाष वार्ड", 3), ("अनुभाग संख्या व नाम 4-कक्ष", 4)):
        assert int(T.SECTION_RE.search(t).group(1)) == no


def test_ocr_headings_and_meta():
    import eci_ocr as E
    assert E.parse_heading(["अनुभाग संख्या और नाम : 1-मालवीय नगर ऐशबाग"]) == "1-मालवीय नगर ऐशबाग"
    assert E.parse_heading(["अनुभाग संख्या व नाम 1-हरचन्दपुर"]) == "1-हरचन्दपुर"
    assert E.parse_heading(["खण्डको संख्या अनि नाँउ 1-रंगपु फरेष्ट बस्ती"]) == "1-रंगपु फरेष्ट बस्ती"
    assert E.pdf_meta("2026-EROLLGEN-U05-10-SIR-DraftRoll-Revision1-HIN-10-WI.pdf") == dict(state="U05", ac=10, roll_part=10)
    assert E.pdf_meta("2026-EROLLGEN-S24-174-SIR-DraftRoll-Revision1-HIN-1-WI.pdf") == dict(state="S24", ac=174, roll_part=1)
