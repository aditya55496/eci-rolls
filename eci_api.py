"""The parts of the ECI portal (voters.eci.gov.in "Download Electoral Roll") this family uses, as the portal's own page calls them.

- Part list: POST gateway-voters.eci.gov.in/api/v1/printing-publish/get-publish-part-list. The page encrypts the JSON body:
  a random AES-256-GCM key and 12-byte iv, the key wrapped with RSA-OAEP-SHA256 under the site's public key (eci_rsa_pub_b64.txt,
  from the page's JS); body {"encryptedPayload", "encryptedKey", "iv"}. No login, token or captcha.
- Roll PDFs of the EROLLGEN rolls (SIR draft/final 2026, SSR draft 2027): after the captcha form, the page itself GETs a fixed
  public path https://voters.eci.gov.in/eroll/<year>/s24/<rollTypeRefId lowercase>/<ac>/<file name>; the same GET works without
  the form (checked 2026-11-22: byte-identical to PDFs downloaded by hand through the form).
- Supplements are only served through the captcha form (not used here yet).
"""
import base64
import json
import os
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

GW = "https://gateway-voters.eci.gov.in/"
SITE = "https://voters.eci.gov.in"
MISKEY = "EROLLA32DVI09AJH"
_PUB = serialization.load_der_public_key(base64.b64decode(
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eci_rsa_pub_b64.txt")).read().strip()))
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*", "Origin": SITE, "Referer": SITE + "/",
    "applicationname": "VSP", "channelidobo": "VSP", "platform-type": "ECIWEB",
}

# roll id -> how the portal names it (from get-publish-eroll-type for S24, 2026-11-21)
EROLL = {
    "2026-sir-draft": {"year": 2026, "ref": "SIR-DraftRoll", "gen": "EROLLGEN", "rev": 1},
    "2026-sir-final": {"year": 2026, "ref": "SIR-FinalRoll", "gen": "EROLLGEN", "rev": 1},
    "2027-ssr-draft": {"year": 2027, "ref": "SSR-DraftRoll", "gen": "EROLLGEN", "rev": 1},
}


def enc_body(obj):
    k, iv = os.urandom(32), os.urandom(12)
    ct = AESGCM(k).encrypt(iv, json.dumps(obj, separators=(",", ":")).encode(), None)
    wrapped = _PUB.encrypt(k, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    b = lambda x: base64.b64encode(x).decode()  # noqa: E731
    return {"encryptedPayload": b(ct), "encryptedKey": b(wrapped), "iv": b(iv)}


def part_list_request(roll, ac, state="S24"):
    e = EROLL[roll]
    body = {"stateCd": state, "acNumber": str(ac), "rollTypeRefId": e["ref"], "pdfGenType": e["gen"], "revisionNo": e["rev"],
            "year": str(e["year"]), "misKey": MISKEY}
    return GW + "api/v1/printing-publish/get-publish-part-list", enc_body(body)


def pdf_name(roll, ac, part, state="S24"):
    e = EROLL[roll]
    return f"{e['year']}-{e['gen']}-{state}-{ac}-{e['ref']}-Revision{e['rev']}-HIN-{part}-WI.pdf"


def pdf_url(roll, ac, part, state="S24"):
    e = EROLL[roll]
    return f"{SITE}/eroll/{e['year']}/{state.lower()}/{e['ref'].lower()}/{ac}/{pdf_name(roll, ac, part, state)}"


def is_pdf(b):
    return b[:5] == b"%PDF-" and b.rstrip()[-5:] == b"%%EOF"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ---- captcha form (used for the General Election 2024 roll, which has no public file path) ----
# getCaptcha/EROLL returns {"data": base64(iv12 || AES-GCM(json))} with a static key from the page's JS; the json has the captcha
# image (base64 JPEG) and its id. generate-published-pdfs (encrypted body, like the part list) takes the answer and a list of part
# numbers (a whole AC in one call, measured 327 parts) and returns one single-use file id per part; the ids expire within minutes
# ("File Fetch Time Expire"). get-published-file on gateway-vpd returns {"payload": base64 PDF, "refId": file name}.
VPD = "https://gateway-vpd.eci.gov.in/"
_CAPTCHA_KEY = base64.b64decode("e855n97lc4tcPkj7WWsi38yNWpalLBLZzQdkqHWYbZ0=")
GER = {"2024-ge-final": {"id": "S24-2024-GEN-FIR", "year": 2024, "ref": "FinalRoll", "gen": "FC-EROLLGEN", "rev": 2}}


def captcha_url():
    return GW + "api/v1/captcha-service/getCaptcha/EROLL"


def captcha_decode(data_b64):
    raw = base64.b64decode(data_b64)
    d = json.loads(AESGCM(_CAPTCHA_KEY).decrypt(raw[:12], raw[12:], None))
    return d["id"], base64.b64decode(d["captcha"])


def ger_part_list_request(roll, ac, state="S24"):
    e = GER[roll]
    body = {"stateCd": state, "acNumber": str(ac), "rollTypeRefId": e["ref"], "pdfGenType": e["gen"], "revisionNo": e["rev"],
            "year": str(e["year"]), "misKey": MISKEY}
    return GW + "api/v1/printing-publish/get-publish-part-list", enc_body(body)


def generate_request(roll, ac, district, parts, answer, captcha_id, state="S24"):
    body = {"stateCd": state, "acNumber": str(ac), "partNumberList": list(parts), "districtCd": district, "captcha": answer,
            "captchaId": captcha_id, "langCd": "HIN", "publishedRollId": GER[roll]["id"], "misKey": MISKEY}
    return GW + "api/v1/printing-publish/generate-published-pdfs", enc_body(body)


def file_url(file_id):
    return VPD + f"api/v1/ext-printing-publish/get-published-file?fileId={file_id}"
