"""Planner (GitHub Actions, workflow `plan`, hourly): seeds the textlayer and ocr queues from what the fetch task has finished.

- An AC whose 2027 SSR draft is fetched -> textlayer item "<ac>" (keys from parts/2027-ssr-draft/<ac>.json).
- An AC whose SIR draft and SIR final 2026 are fetched and whose 2027 text is done -> one ocr item per part number
  "<ac>-<part>" with the keys of the PDFs that exist for that part (draft, final, or both).
seed_queue keeps keys that exist, so running it again only adds what is new.
"""
import json
import os

import requests
from scrape_hub_client import HubClient, list_queue, seed_queue

import eci_api as E
from common import FETCH_PROJECT, OCR_PROJECT, TEXT_PROJECT, pdf_key

GER_PROJECT = "eci-rolls-up-ger"
GER_OCR = False      # General Election 2024 OCR: off until the model is checked on that print (owner 2026-11-22: OCR it too)


def parts_of(hub, s, roll, ac):
    k = f"{FETCH_PROJECT}/parts/{roll}/{ac}.json"
    url = hub.read_urls(FETCH_PROJECT, [k])[k]
    r = s.get(url, timeout=60)
    return [int(p["partNumber"]) for p in r.json()["parts"]] if r.status_code == 200 else None


def main():
    done = {i["key"] for i in list_queue(FETCH_PROJECT, "done")}
    text_done = {i["key"] for i in list_queue(TEXT_PROJECT, "done")}
    acs = sorted({int(k.split("/")[1]) for k in done})
    hub = HubClient(TEXT_PROJECT, note="planner")
    s = requests.Session()
    text_items, ocr_items = [], []
    try:
        for ac in acs:
            if f"2027-ssr-draft/{ac}" in done:
                p27 = parts_of(hub, s, "2027-ssr-draft", ac)
                if p27:
                    text_items.append({"key": str(ac), "payload": {"ac": ac, "keys": [
                        pdf_key("2027-ssr-draft", ac, E.pdf_name("2027-ssr-draft", ac, p)) for p in p27]}})
            if str(ac) in text_done and f"2026-sir-draft/{ac}" in done and f"2026-sir-final/{ac}" in done:
                pd, pf = parts_of(hub, s, "2026-sir-draft", ac) or [], parts_of(hub, s, "2026-sir-final", ac) or []
                for p in sorted(set(pd) | set(pf)):
                    keys = {r: pdf_key(r, ac, E.pdf_name(r, ac, p)) for r, have in (("2026-sir-draft", pd), ("2026-sir-final", pf)) if p in have}
                    ocr_items.append({"key": f"{ac}-{p}", "payload": {"ac": ac, "part": p, "keys": keys, "expected": 0}})
        hub.close()
    except BaseException as e:
        hub.close("failed", f"{type(e).__name__}: {e}"[:200])
        raise
    print("fetch done:", len(done), "ACs seen:", len(acs))
    print("textlayer:", seed_queue(TEXT_PROJECT, text_items) if text_items else "nothing")
    print("ocr:", seed_queue(OCR_PROJECT, ocr_items) if ocr_items else "nothing")


if __name__ == "__main__":
    main()
