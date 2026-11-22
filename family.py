"""Projects of the eci-rolls family: the 5 UP electoral rolls in scope (owner 2026-11-22), Hindi, all 403 assembly constituencies.

Scale (owner 2026-11-22: "use 300 github runners"): fetch 48 workers x 4 threads (no pushback from ECI at 32), textlayer 12, ocr 256.

eci-rolls-up       task fetch      one item per roll and AC (3 rolls x 403 ACs): every part PDF once into the bucket
eci-rolls-up-2027  task textlayer  one item per AC, seeded by plan.py: 2027 SSR draft records from the PDF text layer (no OCR)
eci-rolls-up-ocr   task ocr        one item per AC and part, seeded by plan.py: OCR of the SIR draft and final 2026 rolls
Supplement-2 and -3 (2026) are skipped (owner 2026-11-22); the portal serves them only through its captcha form.
"""
import json
import os
from pathlib import Path

FAMILY = json.loads((Path(__file__).parent / "scrape.json").read_text())["family"]
UP_ACS = range(1, 404)
FETCH_ROLLS = ["2027-ssr-draft", "2026-sir-final", "2026-sir-draft"]     # 2027 first: its text confirms the OCR
# FETCH_ACS="174,86" (Setup workflow input) seeds only these ACs: the first pool run is small, then everything
ONLY = [int(a) for a in os.environ.get("FETCH_ACS", "").split(",") if a.strip()]

PROJECTS = [
    {
        "name": f"{FAMILY}-up", "title": "UP electoral roll PDFs (SIR draft + final 2026, SSR draft 2027)", "task": "fetch",
        "target": "voters.eci.gov.in", "env": {}, "probe": f"{FAMILY}:voters.eci.gov.in",
        "unit_label": "roll x AC", "record_label": "PDF", "threads": 4, "max_workers": 48,
        # partition_id = AC x 10 + roll: the hub hands items out in partition order, so all 3 rolls of an AC finish together
        # and the planner can start that AC's OCR early, instead of every AC waiting for the end of the fetch
        "items": [{"key": f"{roll}/{ac}", "partition_id": ac * 10 + i, "payload": {"roll": roll, "ac": ac, "expected": 350}}
                  for i, roll in enumerate(FETCH_ROLLS) for ac in (ONLY or UP_ACS)],
    },
    {
        "name": f"{FAMILY}-up-2027", "title": "UP SSR draft roll 2027 from the PDF text layer", "task": "textlayer",
        "target": "eci-rolls-bucket", "env": {}, "probe": None,
        "unit_label": "AC", "record_label": "AC", "threads": 1, "max_workers": 12, "items": [],
    },
    {
        "name": f"{FAMILY}-up-ocr", "title": "UP SIR rolls 2026: OCR (no LLM, 4 CPUs per runner)", "task": "ocr",
        "target": "eci-rolls-ocr", "env": {}, "probe": None,
        "unit_label": "part", "record_label": "part", "threads": 1, "max_workers": 256, "items": [],
    },
]

PROBES = [{"name": f"{FAMILY}:voters.eci.gov.in", "url": "https://voters.eci.gov.in/eroll/2027/s24/ssr-draftroll/174/"
           "2027-EROLLGEN-S24-174-SSR-DraftRoll-Revision1-HIN-1-WI.pdf", "expect": "%PDF-"}]
