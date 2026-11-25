"""Model-set choice, roll order, provenance and repairs of task ocr-any (2026-11-25). Offline: no network, no OCR."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SCRAPE_POOL_PROJECT", "eci-rolls-test")
import hi_common as H  # noqa: E402
import pytest  # noqa: E402

SETS = {"v4a": {"hints": "required", "models": ["a", "b"]}, "j2Ps": {"hints": "none", "models": ["a", "c"]},
        "any": {"hints": "optional", "models": ["d"]}, "v5h": {"hints": "required", "models": ["e"]},
        "j3": {"hints": "none", "models": ["f"]}}
HINT = {"model_set": "v4a", "hints": "eci-rolls-ssr-text/hints/S12-2027-DR/1.jsonl.gz"}
NOHINT = {"model_set": "j2Ps", "hints": None}


def test_model_sets_file():
    for name, ms in H.MODEL_SETS.items():
        assert ms["hints"] in ("required", "none", "optional"), name
        assert set(ms["sha256_16"]) == set(ms["models"]), name
        for m in ms["models"]:
            assert os.path.exists(os.path.join(H.HERE, "models", m, "charset.json")), (name, m)
        assert all(k.startswith("ECI_") and isinstance(v, str) for k, v in ms["env"].items()), name
    assert H.MODEL_SETS["v4a"]["hints"] == "required" and H.MODEL_SETS["j2Ps"]["hints"] == "none"


def test_choose_payload():
    assert H.choose_set(HINT, {}, SETS) == ("v4a", "payload")
    assert H.choose_set(NOHINT, {"threads": 1}, SETS) == ("j2Ps", "payload")


def test_choose_map_swaps_only_the_named_set():
    s = {"ocr_model_set_map": {"j2Ps": "j3"}}
    assert H.choose_set(NOHINT, s, SETS) == ("j3", "hub ocr_model_set_map")
    assert H.choose_set(HINT, s, SETS) == ("v4a", "payload")


def test_choose_incompatible_hub_choice_is_ignored():
    # a set that needs hints must not go onto an item without hints, a no-hint set not onto an item with hints
    assert H.choose_set(NOHINT, {"ocr_model_set": "v5h"}, SETS) == ("j2Ps", "payload")
    assert H.choose_set(HINT, {"ocr_model_set": "j3"}, SETS) == ("v4a", "payload")
    assert H.choose_set(HINT, {"ocr_model_set": "any"}, SETS) == ("any", "hub ocr_model_set")
    assert H.choose_set(HINT, {"ocr_model_set_map": {"v4a": "v5h"}, "ocr_model_set": "any"}, SETS) == ("v5h", "hub ocr_model_set_map")


def test_unknown_set():
    with pytest.raises(H.UnknownModelSet):
        H.choose_set(NOHINT, {"ocr_model_set_map": {"j2Ps": "newer"}}, SETS)
    with pytest.raises(H.UnknownModelSet):
        H.choose_set({"model_set": "newer", "hints": None}, {}, SETS)
    assert H.hub_override_unknown({"ocr_model_set_map": {"j2Ps": "newer"}, "ocr_model_set": "j3"}, SETS) == ["newer"]
    assert H.hub_override_unknown({"paused": False}, SETS) == []


def test_rolls():
    assert H.is_draft("S12-2026-DR") and H.is_draft("S04-2025-SIR-DR")
    assert not H.is_draft("S12-2026-FIR") and not H.is_draft("S28-2026-SIR-FIR-1") and not H.is_draft("S04-2025-SIR-FIR")
    assert H.roll_order(["S28-2026-SIR-FIR-1", "S28-2026-DR"]) == ["S28-2026-DR", "S28-2026-SIR-FIR-1"]
    assert H.roll_year("S04-2025-SIR-DR") == 2025 and H.roll_year("U05-2026-DR") == 2026


def test_rules_and_model_check():
    ms = H.MODEL_SETS["v4a"]
    r = H.rules_of(ms, 2026, True)
    assert r["stage1"] == 1 and r["roll_year"] == 2026 and r["ECI_EPIC_PAIR"] == "1"
    assert H.rules_of(ms, 2026, False)["stage1"] == 0          # eci_ocr has no cascade without hints
    dirs = H.model_dirs("j2Ps")
    assert [os.path.basename(d) for d in dirs] == H.MODEL_SETS["j2Ps"]["models"]


def _rec(src, serial, epic, nvb=False, **kw):
    r = {"source": src, "serial_no": str(serial), "epic": epic, "name": "n", "relation_label": "पिता का नाम", "relation_name": "r",
         "house_no": "1", "age": "30", "gender": "पुरुष", "flags": [], "code": "#"}
    if nvb:
        r.update(not_voter_box=True, flags=["epic", "name", "relation_name", "house_no", "age", "gender"])
    r.update(kw)
    return r


def test_repairs_with_set_engines():
    d, f = "2026-EROLLGEN-S12-1-SIR-DraftRoll-Revision1-HIN-1-WI.pdf", "2026-EROLLGEN-S12-1-SIR-FinalRoll-Revision1-HIN-1-WI.pdf"
    recs = [_rec(d, 1, "ABC1234567"), _rec(d, 2, "ABC2345678"), _rec(d, 3, "ABC3456789"),
            _rec(f, 1, "ABC1234567"), _rec(f, 2, "ABCZ345678", nvb=True, readings={"code": {"crnn_c11g9s": "Q", "crnn_c9g9s": "Q"}}),
            _rec(f, 3, "", nvb=True), _rec(f, 4, "ABC4567890")]
    c = H.repair_part(recs, ["crnn_c11g9s", "crnn_c9g9s"])
    assert c == {"final_deleted": 1, "serial_gaps": 1}
    assert recs[4]["deleted_in_final"] and recs[4]["epic"] == "ABC2345678" and recs[4]["code"] == "Q"
    assert recs[5].get("voter_by_serial") and "not_voter_box" not in recs[5]


def test_epicfix_relative_name_needs_same_person():
    """ocrv5 review 2026-11-25 (S12 AC 222 part 23 p24 b7): a relative-name-only match with another gender / age is no repair."""
    import epicfix
    hints = [dict(epic="LWP1463645", name="ओमप्रकाश", relation_name="कन्हैयालाल", gender="पुरुष", age="63"),
             dict(epic="BRA1234567", name="सीता देवी", relation_name="राम लाल", gender="महिला", age="41")]
    idx = epicfix.index(hints)
    recs = [dict(epic="ZAX1463645", name="केसरबाई", relation_name="कन्हैयालाल", gender="महिला", age="64", flags=[]),
            dict(epic="8RA1234567", name="सीता देवी", relation_name="x", gender="महिला", age="40", flags=["epic"]),
            dict(epic="BHA1234567", name="y", relation_name="राम लाल", gender="महिला", age="40", flags=["epic"])]
    assert epicfix.fix(recs[:2], idx, 1) == 1
    assert recs[0]["epic"] == "ZAX1463645" and not recs[0].get("epic_from_2027")
    assert recs[1]["epic"] == "BRA1234567" and recs[1]["flags"] == []
    recs[1].update(epic="8RA1234567", epic_from_2027=False)
    assert epicfix.fix([recs[2]], idx, 1) == 1 and recs[2]["epic"] == "BRA1234567"     # relative + same gender + age in window
    r = dict(epic="BHA1234567", name="y", relation_name="राम लाल", gender="महिला", age="30", flags=[])
    assert epicfix.fix([r], idx, 1) == 0                                                  # age gap 11: no repair


# ---- do_item / main with the OCR engine and the hub replaced (2026-11-25, ocrv5 review: these paths had no offline test)

def _fake_item(model_set="v4a", hints=True):
    return {"key": "S12/1/10", "payload": {
        "state": "S12", "ac": 1, "part": 10, "lang": "HIN", "model_set": model_set,
        "keys": {"S12-2026-FIR": "eci-rolls-hi-cdn/pdf/S12-2026-FIR/1/x-FinalRoll-10.pdf",
                 "S12-2026-DR": "eci-rolls-hi-cdn/pdf/S12-2026-DR/1/x-DraftRoll-10.pdf"},
        "hints": "eci-rolls-ssr-text/hints/S12-2027-DR/1.jsonl.gz" if hints else None}}


def _fake_fetch(keys, dest):
    import gzip
    import json
    got = {}
    for k in keys:
        p = os.path.join(dest, k.replace("/", "__"))
        if k.endswith(".gz"):
            with gzip.open(p, "wt", encoding="utf-8") as f:
                f.write(json.dumps({"epic": "ABC1234567", "name": "x"}) + "\n")
        else:
            open(p, "wb").write(b"%PDF-")
        got[k] = p
    return got


def _fake_run_eci(calls):
    def run(pdfs, hints_path, ms, dirs, year, tmp):
        calls.append({"rolls": [r for r, _ in pdfs], "hints": hints_path, "env": dict(ms["env"]), "dirs": dirs, "year": year})
        recs = [{"roll": r, "page": 3, "box": 0, "serial_no": "1", "epic": "ABC1234567", "flags": []} for r, _ in pdfs]
        summs = [{"roll": r, "voter_pages": 1, "pdfs": [{"pdf": os.path.basename(p), "records": 1}]} for r, p in pdfs]
        return recs, summs, 12.5
    return run


def test_do_item_output_and_provenance(monkeypatch):
    import gzip
    import json
    import ocr_any_task as O
    calls, up = [], []
    monkeypatch.setattr(O, "run_eci", _fake_run_eci(calls))
    monkeypatch.setattr(H, "repair_part", lambda recs, engines: {"final_deleted": 0, "serial_gaps": 0})
    monkeypatch.setattr(H, "CODE", "0123456789ab")
    monkeypatch.setattr(H, "code_commit", lambda: "0123456789ab")
    rec = O.do_item(None, _fake_item(), settings={}, fetch=_fake_fetch, upload=up.extend)
    assert calls[0]["rolls"] == ["S12-2026-DR", "S12-2026-FIR"] and calls[0]["hints"] and calls[0]["year"] == 2026
    assert calls[0]["env"] == H.MODEL_SETS["v4a"]["env"]
    assert [u[0] for u in up] == ["ocr/S12/1/10.jsonl.gz", "ocr/S12/1/10.summary.json"]
    recs = [json.loads(l) for l in gzip.decompress(up[0][1]).decode().splitlines()]
    for r in recs:
        assert r["ocr_model_set"] == "v4a" and r["ocr_models"] == H.MODEL_SETS["v4a"]["models"]
        assert r["ocr_code"] == "0123456789ab" and r["hints_used"] is True
        assert r["ocr_rules"]["stage1"] == 1 and r["ocr_rules"]["roll_year"] == 2026
    for k in H.PROVENANCE_KEYS + ("model_set_source", "cpu_s", "by_roll"):
        assert k in rec, k
    assert rec["records"] == 2 and rec["model_set_source"] == "payload"


def test_do_item_hub_swap_and_no_hint_set(monkeypatch):
    import ocr_any_task as O
    calls = []
    monkeypatch.setattr(O, "run_eci", _fake_run_eci(calls))
    monkeypatch.setattr(H, "repair_part", lambda recs, engines: {})
    monkeypatch.setattr(H, "code_commit", lambda: H.CODE)
    rec = O.do_item(None, _fake_item("j2Ps", hints=False), settings={"ocr_model_set_map": {"j2Ps": "j2Psf"}},
                    fetch=_fake_fetch, upload=lambda files: None)
    assert rec["ocr_model_set"] == "j2Psf" and rec["model_set_source"] == "hub ocr_model_set_map"
    assert rec["hints_used"] is False and calls[0]["hints"] is None and rec["ocr_rules"]["stage1"] == 0
    with pytest.raises(RuntimeError, match="needs hints"):
        O.do_item(None, _fake_item("v4a", hints=False), settings={}, fetch=_fake_fetch, upload=lambda files: None)


def test_do_item_code_guards(monkeypatch):
    import ocr_any_task as O
    monkeypatch.setattr(O, "run_eci", _fake_run_eci([]))
    monkeypatch.setattr(H, "repair_part", lambda recs, engines: {})
    monkeypatch.setattr(H, "CODE", "0123456789ab")
    monkeypatch.setattr(H, "code_commit", lambda: "ba9876543210")      # the checkout moved during the item
    with pytest.raises(H.CodeChanged):
        O.do_item(None, _fake_item(), settings={}, fetch=_fake_fetch, upload=lambda files: None)
    monkeypatch.setattr(H, "CODE", "")                                  # git could not tell the commit
    with pytest.raises(RuntimeError, match="commit is unknown"):
        O.do_item(None, _fake_item(), settings={}, fetch=_fake_fetch, upload=lambda files: None)


class _FakeHub:
    instances = []

    def __init__(self, *a, **k):
        self.settings, self.completed, self.closed, self.metrics = {}, [], None, []
        self.items = [_fake_item("nosuchset")]
        _FakeHub.instances.append(self)

    def claim(self, n, **k):
        return [self.items.pop()] if self.items else []

    def complete(self, keys, status="done", error="", fetched=None):
        self.completed.append((list(keys), status, error))

    def metric(self, name, value=1):
        self.metrics.append(name)

    def close(self, status="done", note=""):
        self.closed = (status, note)

    def record(self, *a, **k):
        raise AssertionError("no record for an item that was not run")

    def error(self):
        pass

    def failure(self, *a, **k):
        pass


def test_main_unknown_set_gives_item_back_untried(monkeypatch):
    import ocr_any_task as O
    _FakeHub.instances.clear()
    monkeypatch.setattr(O, "HubClient", _FakeHub)
    monkeypatch.setattr(H, "CODE", "0123456789ab")
    with pytest.raises(SystemExit) as e:
        O.main()
    hub = _FakeHub.instances[0]
    assert e.value.code == 3
    assert hub.completed == [(["S12/1/10"], "pending", "model set nosuchset not in this checkout (0123456789ab)")]


def test_main_refuses_without_code_commit(monkeypatch):
    import ocr_any_task as O
    _FakeHub.instances.clear()
    monkeypatch.setattr(O, "HubClient", _FakeHub)
    monkeypatch.setattr(H, "CODE", "")
    with pytest.raises(SystemExit) as e:
        O.main()
    hub = _FakeHub.instances[0]
    assert e.value.code == 2 and hub.completed == [] and hub.closed[0] == "failed" and len(hub.items) == 1


def test_planner_holds_partial_lang_rank(monkeypatch):
    """plan_hi_ocr with the hub database replaced: held states, partial parts, AC lang, fixed partition ids, resets."""
    import plan_hi_ocr as P
    rows = {
        "U02-2026-DR": [{"project": "eci-rolls-hi-cdn", "record_id": "U02-2026-DR/1", "ac": 1, "part": None, "key": None,
                         "part_numbers": [1, 2, 3], "extra_parts": [], "missing": [], "lang": "HIN"}]
        + [{"project": "eci-rolls-hi-cdn", "record_id": f"U02-2026-DR/1/{x}", "ac": 1, "part": x, "key": f"k/DR/{x}"} for x in (1, 2, 3)],
        "U02-2026-FIR": [{"project": "eci-rolls-hi-cdn", "record_id": "U02-2026-FIR/1", "ac": 1, "part": None, "key": None,
                          "part_numbers": [1, 2, 3], "extra_parts": [], "missing": [], "lang": "HIN"}]
        + [{"project": "eci-rolls-hi-cdn", "record_id": f"U02-2026-FIR/1/{x}", "ac": 1, "part": x, "key": f"k/FIR/{x}"} for x in (1, 2)],
        "S07-2026-DR": [{"project": "eci-rolls-hi-cdn", "record_id": "S07-2026-DR/5", "ac": 5, "part": None, "key": None,
                         "part_numbers": [1], "extra_parts": [], "missing": [], "lang": "ENG"},
                        {"project": "eci-rolls-hi-cdn", "record_id": "S07-2026-DR/5/1", "ac": 5, "part": 1, "key": "k/S07/1"}],
    }
    monkeypatch.setattr(P, "roll_rows", lambda roll: rows.get(roll, []))
    monkeypatch.setattr(P, "fetch_status", lambda roll: {})
    items, summ = P.plan_ocr({"U02", "S07"}, None, None, "v4a", "j2Ps", "x", True)
    assert [i["key"] for i in items] == ["U02/1/1", "U02/1/2"]                # part 3 has no final PDF yet: held
    assert summ["U02"]["items_partial_held"] == 1 and items[0]["partition_id"] == 3 * 1_000_000 + 1000 + 1
    assert summ["S07"]["held_state"] and not any(i["key"].startswith("S07") for i in items)
    items, summ = P.plan_ocr({"U02", "S07"}, None, None, "v4a", "j2Ps", "x", True, state_set={"S07": "j2Psf"}, include_partial=True)
    assert [i["key"] for i in items] == ["U02/1/1", "U02/1/2", "U02/1/3"]    # S07 AC 5 is ENG: not planned
    assert summ["S07"]["items_other_lang"] == 1 and summ["S07"]["model_set"] == "j2Psf"
    monkeypatch.setattr(P, "psql_json", lambda sql: [{"key": "U02/1/3", "status": "done", "payload": {"keys": {"U02-2026-DR": "k/DR/3"}}}])
    rows["U02-2026-FIR"].append({"project": "eci-rolls-hi-cdn", "record_id": "U02-2026-FIR/1/3", "ac": 1, "part": 3, "key": "k/FIR/3"})
    items, _ = P.plan_ocr({"U02"}, None, None, "v4a", "j2Ps", "x", True)
    sql, n = P.resets_sql("eci-rolls-hi-ocr", items)
    assert n == {"done": 1} and "U02-2026-FIR" in sql and "key = 'U02/1/3'" in sql
