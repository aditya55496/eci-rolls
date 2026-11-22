"""File names and paths must match what the portal serves (checked against PDFs downloaded by hand, 2026-11-22)."""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eci_api as E  # noqa: E402
from common import gz_jsonl, hint_of, pdf_key  # noqa: E402


def test_names():
    assert E.pdf_name("2026-sir-draft", 174, 1) == "2026-EROLLGEN-S24-174-SIR-DraftRoll-Revision1-HIN-1-WI.pdf"
    assert E.pdf_name("2026-sir-final", 86, 3) == "2026-EROLLGEN-S24-86-SIR-FinalRoll-Revision1-HIN-3-WI.pdf"
    assert E.pdf_name("2027-ssr-draft", 8, 2) == "2027-EROLLGEN-S24-8-SSR-DraftRoll-Revision1-HIN-2-WI.pdf"
    assert E.pdf_url("2026-sir-final", 174, 1) == ("https://voters.eci.gov.in/eroll/2026/s24/sir-finalroll/174/"
                                                   "2026-EROLLGEN-S24-174-SIR-FinalRoll-Revision1-HIN-1-WI.pdf")
    assert pdf_key("2027-ssr-draft", "174", "x.pdf") == "eci-rolls-up/pdf/2027-ssr-draft/174/x.pdf"


def test_body_is_encrypted():
    url, body = E.part_list_request("2026-sir-draft", 174)
    assert url.endswith("get-publish-part-list") and set(body) == {"encryptedPayload", "encryptedKey", "iv"}
    assert "174" not in json.dumps(body)


def test_pdf_check():
    assert E.is_pdf(b"%PDF-1.7\n...\n%%EOF\n") and not E.is_pdf(b"<html>error</html>")


def test_hint():
    r = {"ac": 174, "part": 1, "serial_no": 1, "epic": "ABC1234567", "name": "क", "relation_type": "पिता", "relation_name": "ख",
         "house_no": "1/ 1", "age": 59, "gender": "पुरुष"}
    h = hint_of(r)
    assert h["relation_label"] == "पिता का नाम" and h["age"] == "59" and h["serial_no"] == "1" and h["part"] == "1"
    assert hint_of(dict(r, relation_type="अन्य"))["relation_label"] == "अन्य"
    assert json.loads(gzip.decompress(gz_jsonl([h])))["epic"] == "ABC1234567"
