# eci-rolls

Electoral rolls of Uttar Pradesh (ECI, voters.eci.gov.in), Hindi, all 403 assembly constituencies (ACs). Public records.

Rolls in scope: SIR draft roll 2026, SIR final roll 2026, SSR draft roll 2027. Supplement-2 and -3 (2026) come later. The portal
serves them only through its captcha form.

## Projects and tasks

| Project | Task | Queue item | Output (bucket `eci-rolls`) |
|---|---|---|---|
| `eci-rolls-up` | `fetch` | `<roll>/<ac>` (3 x 403) | `pdf/<roll>/<ac>/<file name>`, `parts/<roll>/<ac>.json`; one hub record per PDF (bytes, sha256, Last-Modified, polling station name) |
| `eci-rolls-up-2027` | `textlayer` | `<ac>` (seeded by `plan.py`) | `text/<ac>.jsonl.gz` (every voter record of the 2027 draft, from the PDF text layer), `hints/<ac>.jsonl.gz`, `text/<ac>.summary.json` |
| `eci-rolls-up-ocr` | `ocr` | `<ac>-<part>` (seeded by `plan.py`) | `ocr/<ac>/<part>.jsonl.gz` (one record per voter box of the SIR draft and final 2026), `ocr/<ac>/<part>.summary.json` |

- **fetch:** one call to the portal's part-list API (`eci_api.py`; the page encrypts the request body, the same is done here).
  Then each part PDF comes from its fixed public path `https://voters.eci.gov.in/eroll/<year>/s24/<roll type>/<ac>/<file>`, the
  same GET that the portal's page makes after its form. Each PDF is fetched once.
- **textlayer:** `eciocr/textlayer/parse_textlayer.py`. There is no OCR. The script repairs the text-layer artefacts of the
  2027 PDFs and checks the record counts against each PDF's cover totals.
- **ocr:** `eciocr/prod/eci_ocr.py` with the int8 CRNN models in `models/` and Tesseract 5 (hin, eng), on the runner's 4 CPUs. There is
  no LLM and no network use apart from the hub. The 2027 records of the same AC are used only to confirm readings, never as values.
  - Measured on a 4,266-box gold set (180 pages from 3 ACs): 97.84 % of voter records have every field right (95 % CI
    97.13–98.43). 4.5 % of records are flagged, and 99.63 % of unflagged records are right.
  - Speed: about 1.2–1.4 s per page.
  - Every record keeps its `flags` (fields where the engines disagree).
- **plan.py** (workflow `plan`, hourly) seeds `textlayer` once an AC's 2027 roll is fetched. It seeds `ocr` once the AC's SIR
  draft, SIR final and 2027 text are done.

## Running

- `setup` workflow: registers the projects and seeds `eci-rolls-up`. The input `fetch_acs` (for example `174,86`) seeds only
  those ACs. Run it again with an empty input for all 403.
- `check` workflow (every push): tests, then `scrape-check` of `fetch` on 2 items with 2 PDFs each (`ECI_MAX_PARTS=2`).
- Locally: `python -m pytest tests`; `scrape-check --project eci-rolls-up --task fetch --env ECI_MAX_PARTS=2 --items 2`.

The code in `eciocr/` is copied unchanged from the benchmark (methods and measurements: paper "Reading Hindi electoral rolls
without an LLM", 2026-11-22).
