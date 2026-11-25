"""Projects of the eci-rolls family: the 5 UP electoral rolls in scope (owner 2026-11-22), Hindi, all 403 assembly constituencies.

Scale (owner 2026-11-22: "use 300 github runners"): fetch 80 workers x 4 threads (no pushback from ECI at 32; owner: more fetch slots), textlayer 12, ocr 256.

eci-rolls-up       task fetch      one item per roll and AC (3 rolls x 403 ACs): every part PDF once into the bucket
eci-rolls-up-ger   task fetch-ger  one item per AC (403): General Election roll 2024 PDFs through the captcha form
eci-rolls-up-2027  task textlayer  one item per AC, seeded by plan.py: 2027 SSR draft records from the PDF text layer (no OCR)
eci-rolls-up-ocr   task ocr        one item per AC and part, seeded by plan.py: OCR of the SIR draft and final 2026 rolls
eci-rolls-ssr-text task ssrtext    one item per AC, seeded by plan_ssrtext.py: 2027 SSR draft rolls of all other states from the text layer
Worker limits (2026-11-25, ocrv5 review): the values here equal the hub's where the hub accepts them (max_workers <= 256 through
set_pool). Where the hub holds more (set by SQL: eci-rolls-up, -hi-cdn, -all-cdn 320, -up-ocr 760) the value here is 256: a Setup
run with projects 'all' would LOWER those projects. Run Setup with the names of the projects to set up (setup_family.py).
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
        "unit_label": "roll x AC", "record_label": "PDF", "threads": 4, "max_workers": 256,   # hub: 320 (SQL)
        # partition_id = AC x 10 + roll: the hub hands items out in partition order, so all 3 rolls of an AC finish together
        # and the planner can start that AC's OCR early, instead of every AC waiting for the end of the fetch
        "items": [{"key": f"{roll}/{ac}", "partition_id": ac * 10 + i, "payload": {"roll": roll, "ac": ac, "expected": 350}}
                  for i, roll in enumerate(FETCH_ROLLS) for ac in (ONLY or UP_ACS)],
    },
    {
        # General Election 2024 roll (owner 2026-11-22): only through the portal's captcha form; offline captcha model in captcha/
        "name": f"{FAMILY}-up-ger", "title": "UP General Election roll 2024 PDFs (captcha form)", "task": "fetch-ger",
        # the file server gateway-vpd.eci.gov.in answers only from India (checked 2026-11-22: timeouts from US/SG, no DNS from DE):
        # the hub's Mumbai proxies, one per worker
        # and the pool server dev-vps-3 (Mumbai, owner 2026-11-22: "use this server"; geo "IN"), whose workers set larger
        # chunks and more threads in their own environment (ECI_GER_CHUNK, ECI_GER_THREADS): one captcha per AC there
        "target": "gateway-voters.eci.gov.in", "env": {}, "probe": None, "proxy": "IN", "geo": "IN",
        "unit_label": "AC", "record_label": "PDF", "threads": 8, "max_workers": 12,
        "items": [{"key": f"2024-ge-final/{ac}", "partition_id": ac, "payload": {"roll": "2024-ge-final", "ac": ac, "expected": 420}}
                  for ac in (ONLY or UP_ACS)],
    },
    {
        # EPIC repair of SIR OCR output made before the repairs existed (2026-11-23); seeded per AC from the server
        "name": f"{FAMILY}-up-repair", "title": "UP SIR OCR: EPIC repair from the 2027 roll (no new OCR)", "task": "repair",
        "target": "eci-rolls-repair", "env": {}, "probe": None,
        "unit_label": "AC", "record_label": "AC", "threads": 1, "max_workers": 60, "items": [],
    },
    {
        # Every roll of the states that publish Hindi rolls, in Hindi (owner 2026-11-24). Items from plan_hi.py, seeded from the
        # server. CDN route (SIR draft/final 2026, SSR draft 2027): any IP, all GitHub runners.
        "name": f"{FAMILY}-hi-cdn", "title": "Hindi-state rolls: PDFs from the public path (SIR 2026, SSR 2027)", "task": "fetch-hi",
        "target": "voters.eci.gov.in", "env": {}, "probe": "eci-rolls:voters.eci.gov.in",
        "unit_label": "AC roll", "record_label": "PDF", "threads": 4, "max_workers": 256, "items": [],   # hub: 320 (SQL)
    },
    {
        # CDN rolls of ALL other states (owner 2026-11-24: "complete the CDN downloads for all states"; English where offered,
        # else the state's language). Items from plan_cdn_all.py.
        "name": f"{FAMILY}-all-cdn", "title": "All-state rolls: PDFs from the public path (SIR 2026, SSR 2027), English or state language",
        "task": "fetch-hi", "target": "voters.eci.gov.in", "env": {}, "probe": "eci-rolls:voters.eci.gov.in",
        "unit_label": "AC roll", "record_label": "PDF", "threads": 4, "max_workers": 256, "items": [],   # hub: 320 (SQL)
    },
    {
        # Form route (every other roll: GE 2024, draft/final 2024-2025, supplements, bye-elections): the file server answers only
        # from India -> the hub's Mumbai proxies and dev-vps-3 (181-eci-hi-form-local.sh)
        "name": f"{FAMILY}-hi-form", "title": "Hindi-state rolls: PDFs through the captcha form (GE 2024, 2024-25 rolls, supplements)",
        "task": "fetch-hi", "target": "gateway-voters.eci.gov.in", "env": {"ECI_FORM_CHUNK": "100"}, "probe": None, "proxy": "IN", "geo": "IN",
        "unit_label": "AC roll", "record_label": "PDF", "threads": 6, "max_workers": 55, "items": [],    # = hub 2026-11-25
    },
    {
        # Parquet export of the finished rolls (2026-11-24), one item per AC, seeded from the server; a small pool share
        "name": f"{FAMILY}-up-parquet", "title": "UP rolls as Parquet (SIR draft + final 2026, SSR draft 2027; later GE 2024)", "task": "parquet",
        "target": "eci-rolls-parquet", "env": {}, "probe": None,
        "unit_label": "AC", "record_label": "AC", "threads": 1, "max_workers": 16, "items": [],
    },
    {
        "name": f"{FAMILY}-up-2027", "title": "UP SSR draft roll 2027 from the PDF text layer", "task": "textlayer",
        "target": "eci-rolls-bucket", "env": {}, "probe": None,
        "unit_label": "AC", "record_label": "AC", "threads": 1, "max_workers": 12, "items": [],
    },
    {
        # 2027 SSR draft rolls of every state the CDN projects hold (2026-11-25; UP has its own, eci-rolls-up-2027), from the PDF
        # text layer, any language: Hindi, English and the generic parser (ASM, BEN, MAL, KAN, PAN). Items from plan_ssrtext.py,
        # seeded from the server. CPU only (4 parser processes per runner), no request to ECI.
        "name": f"{FAMILY}-ssr-text", "title": "SSR draft rolls 2027, all states: records from the PDF text layer (no OCR)",
        "task": "ssrtext", "target": "eci-rolls-bucket", "env": {}, "probe": None,
        "unit_label": "AC", "record_label": "AC", "threads": 1, "max_workers": 200, "items": [],   # = hub 2026-11-25
    },
    {
        # OCR of the Hindi-state SIR rolls (owner plan 2026-11-25 03:49): one item per part of one AC of one state, both SIR rolls;
        # model set per item (model_sets.json: v4a with 2027 hints, j2Ps without); hub setting ocr_model_set_map swaps sets for
        # all pending items. Items from plan_hi_ocr.py, seeded from the server. Same target as the UP OCR (its worker limit).
        "name": f"{FAMILY}-hi-ocr", "title": "Hindi-state SIR rolls (2025 Bihar, 2026 others): OCR, model set per item (no LLM)",
        # max_workers 128 (2026-11-25): a fair share while the fetch projects still run (hi-form 68 jobs, all-form 11,506 items
        # pending); the main session raises it in the hub when the pool is free.
        "task": "ocr-any", "target": "eci-rolls-ocr", "env": {}, "probe": None,
        "unit_label": "part", "record_label": "part", "threads": 1, "max_workers": 128, "items": [],
    },
    {
        # the part repairs again on stored ocr-any output (ocr-any runs them itself; this is for outputs older than a fix)
        "name": f"{FAMILY}-hi-repair", "title": "Hindi-state SIR OCR: repairs on stored output (no new OCR)", "task": "repair-any",
        "target": "eci-rolls-repair", "env": {}, "probe": None,
        "unit_label": "AC", "record_label": "AC", "threads": 1, "max_workers": 16, "items": [],
    },
    {
        # Parquet per state / roll / AC from the ocr-any output (SIR schema + provenance columns); plan_hi_ocr.py --parquet
        "name": f"{FAMILY}-hi-parquet", "title": "Hindi-state SIR rolls as Parquet (per state, roll and AC)", "task": "parquet-any",
        "target": "eci-rolls-parquet", "env": {}, "probe": None,
        "unit_label": "AC", "record_label": "AC", "threads": 1, "max_workers": 16, "items": [],
    },
    {
        "name": f"{FAMILY}-up-ocr", "title": "UP SIR rolls 2026: OCR (no LLM, 4 CPUs per runner)", "task": "ocr",
        "target": "eci-rolls-ocr", "env": {}, "probe": None,
        "unit_label": "part", "record_label": "part", "threads": 1, "max_workers": 256, "items": [],   # hub: 760 (SQL)
    },
]

PROBES = [{"name": f"{FAMILY}:voters.eci.gov.in", "url": "https://voters.eci.gov.in/eroll/2027/s24/ssr-draftroll/174/"
           "2027-EROLLGEN-S24-174-SSR-DraftRoll-Revision1-HIN-1-WI.pdf", "expect": "%PDF-"}]
