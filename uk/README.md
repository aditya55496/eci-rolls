# uk/ — Uttarakhand (S28) ingestion, hub-free

Stage-1 driver for the Uttarakhand voter-roll audit. Added 2026-11-26 (Claude Code). It **reuses this
repo's own** `eci_api.py` (part-list API + public PDF paths, replicating ECI's own request crypto with
ECI's own public key) and `eciocr/textlayer/parse_textlayer.py` (2027 SSR records from the text layer,
no OCR). It does **not** use `scrape_hub_client` — it loops locally and writes to an output dir.

## Scope (confirmed live for S28, 2026-11-26)
Three EROLLGEN rolls with public paths, no captcha:
`2026 SIR Draft`, `2026 SIR Final`, `2027 SSR Draft`. (2024 GE needs the captcha form — out of scope here.)

## Run (Colab or GitHub Actions — never the Mac)
```
pip install requests cryptography pymupdf pyarrow
python uk/uk_driver.py check                               # self-test: 1 AC, 1 PDF, parse -> record count
python uk/uk_driver.py parts     --out <DRIVE>/data        # exact download size (part-list sweep)
python uk/uk_driver.py textlayer --out <DRIVE>/data --acs 1-70   # 2027 records -> Parquet(zstd) per AC
```
Easiest: open `uk/uk_rolls_runner.ipynb` in Colab (signed in as aditya55496), Run all.

## Outputs (all private)
- `data/parts/<roll>/<ac>.json` — part lists; `data/manifest.json` — total PDFs + GB estimate.
- `data/release/2027-ssr/ac=<n>/data.parquet` — 2027 voter records (zstd).

## Invariants
- **PII stays private.** 2027 records contain real names → written only to the private out dir, never to
  the public site. `check` prints counts only, never names.
- **0 lasting footprint.** PDFs are streamed to a temp file, parsed, and `os.unlink`ed immediately.
- **2026 rolls are scanned** → their records come from the OCR stage (`eciocr/`, CC#2), not this driver.

## Next (CC#2, OCR stage)
Run `eciocr/prod/eci_ocr.py` (CRNN + Tesseract) on the 2026 SIR draft/final PDFs on a Colab GPU →
Parquet; then the DuckDB discrepancy engine diffs draft vs final (mass deletion), and scans 2027 for
duplicate EPICs / ghost voters.
