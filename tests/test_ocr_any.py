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
    assert [u[0] for u in up] == ["ocr/S12/1/10.summary.json", "ocr/S12/1/10.jsonl.gz"]      # summary first, records last
    recs = [json.loads(l) for l in gzip.decompress(up[1][1]).decode().splitlines()]
    summ = json.loads(up[0][1])
    assert len(summ["hints_sha256"]) == 64 and recs[0]["hints_sha256_16"] == summ["hints_sha256"][:16]
    assert "tesseract" in summ["ocr_tools"] or summ["ocr_tools"] == {} or "pymupdf" in summ["ocr_tools"]
    for r in recs:
        assert r["ocr_model_set"] == "v4a" and r["ocr_models"] == H.MODEL_SETS["v4a"]["models"]
        assert r["ocr_code"] == "0123456789ab" and r["hints_used"] is True
        assert r["ocr_rules"]["stage1"] == 1 and r["ocr_rules"]["roll_year"] == 2026
    for k in H.PROVENANCE_KEYS + ("model_set_source", "cpu_s", "by_roll", "hints_sha256", "ocr_tools", "epic_fix2_refused"):
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
    assert e.value.code == H.EXIT_NEEDS_NEWER_CODE == 75
    assert hub.completed == [(["S12/1/10"], "pending", "model set nosuchset not in this checkout (0123456789ab)")]


def test_main_incomplete_model_dirs_give_item_back(monkeypatch, tmp_path):
    """ocrv5 review 2026-11-25: a set pushed without its model files (or a hub swap to it) must not use up the attempts of the
    pending items: the item goes back as 'pending' and the worker exits 75, as for an unknown set."""
    import ocr_any_task as O
    _FakeHub.instances.clear()
    sets = dict(H.MODEL_SETS, nofiles=dict(H.MODEL_SETS["j2Ps"], models=["no_such_model"], sha256_16={"no_such_model": "0" * 16}))
    monkeypatch.setattr(H, "MODEL_SETS", sets)
    monkeypatch.setattr(H, "CODE", "0123456789ab")

    class Hub(_FakeHub):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.items = [_fake_item("nofiles", hints=False)]
    monkeypatch.setattr(O, "HubClient", Hub)
    with pytest.raises(SystemExit) as e:
        O.main()
    hub = _FakeHub.instances[0]
    assert e.value.code == 75 and hub.completed[0][1] == "pending" and "incomplete" in hub.completed[0][2]
    assert "model_set_incomplete" in hub.metrics
    # a wrong sha256 is incomplete too
    sets["badsha"] = dict(H.MODEL_SETS["j2Ps"], sha256_16=dict(H.MODEL_SETS["j2Ps"]["sha256_16"], c11g9s="0" * 16))
    with pytest.raises(H.ModelSetIncomplete):
        H.model_dirs("badsha")


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


def test_epicfix_fix2_guard(monkeypatch):
    """ocrv5 review 2026-11-25: fix2 changed printed EPICs to another voter's (S20/14/70 serial 937 UIO1338565 -> HSD1338565;
    S20/72/10 serial 622 RJ/09/070/198129 -> RJ/09/071/198129); with the guard it needs a close name or relative's name."""
    import epicfix
    monkeypatch.setattr(epicfix, "FIX2_GUARD", True)
    monkeypatch.setattr(epicfix, "FIX2_MIN", 70.0)
    hints = [dict(epic="HSD1338565", name="मुकेशकुमार", relation_name="बजरंगलाल", relation_label="पिता का नाम", gender="पुरुष", age="34"),
             dict(epic="RJ/09/071/198129", name="कृष्णकुमार", relation_name="गोर्धनसिंह", relation_label="पिता का नाम", gender="पुरुष", age="41"),
             dict(epic="BR/25/142/216172", name="मन्नु ऋृषि", relation_name="गंगा प्रसाद ऋृषि", relation_label="पिता का नाम", gender="पुरुष", age="52")]
    idx = epicfix.index(hints)
    wrong1 = dict(epic="UIO1338565", name="पिंटू राम", relation_name="जय गोपाल", relation_label="पिता का नाम", gender="पुरुष", age="34", flags=[])
    wrong2 = dict(epic="RJ/09/070/198129", name="राजकुमार", relation_name="बुद्धीलाल", relation_label="पिता का नाम", gender="पुरुष", age="40",
                  flags=["house_no"])
    right = dict(epic="8R/25/142/216172", name="मन्नु ऋषि", relation_name="गंगा प्रसाद ऋषि", relation_label="पिता का नाम", gender="पुरुष",
                 age="51", flags=["name"])
    assert epicfix.fix([wrong1, wrong2, right], idx, 1) == 0          # no exact name: rule 1 does not fire
    assert epicfix.fix2([wrong1, wrong2, right], idx, 1) == 1
    assert wrong1["epic"] == "UIO1338565" and wrong1["flags"] == ["epic"] and wrong1["readings"]["epic"]["fix2_candidate"] == "HSD1338565"
    assert wrong2["epic"] == "RJ/09/070/198129" and wrong2["flags"] == ["house_no", "epic"]
    assert right["epic"] == "BR/25/142/216172" and right["epic_from_2027"] and right["flags"] == ["name"]
    monkeypatch.setattr(epicfix, "FIX2_GUARD", False)                 # guard off (UP production): the old behaviour
    w = dict(wrong1, epic="UIO1338565", flags=[]); w.pop("readings")
    assert epicfix.fix2([w], idx, 1) == 1 and w["epic"] == "HSD1338565"


def test_age_hint_majority(monkeypatch):
    """ocrv5 review 2026-11-25 (S20/14/70 final roll serial 135): 4 CRNNs read 21, Tesseract 24 = the 2027 age -> 24 without a
    flag. With ECI_AGE_HINT_MAJ the outside majority wins and the field is flagged; a Tesseract-only window value is flagged."""
    sys.path.insert(0, os.path.join(H.HERE, "eciocr", "ocr"))
    import ensemble2 as E
    monkeypatch.setattr(E, "AGE_GAP", 1)
    names = ["crnn_a", "crnn_b", "crnn_c", "crnn_d", "tess"]
    base = dict(code="", serial_no="135", part_no="", epic="ABC1234567", name="x", relation_label="पिता का नाम", relation_name="y",
                house_no="1", gender="पुरुष")
    hint = dict(base, age="24")

    def run(ages, nm=names):
        return E.decide([dict(base, age=a) for a in ages], hint, False, nm, [{}] * len(ages))

    monkeypatch.setattr(E, "AGE_HINT_MAJ", False)
    out, fl = run(["21", "21", "21", "21", "24"])
    assert out["age"] == "24" and "age" not in fl                        # the old rule: the reviewed error
    monkeypatch.setattr(E, "AGE_HINT_MAJ", True)
    out, fl = run(["21", "21", "21", "21", "24"])
    assert out["age"] == "21" and "age" in fl
    out, fl = run(["21", "24"], ["crnn_a", "tess"])                      # stage 1: only Tesseract fits the window
    assert out["age"] == "24" and "age" in fl
    out, fl = run(["23", "2"], ["crnn_a", "tess"])                       # stage 1: the CRNN fits the window: no flag
    assert out["age"] == "23" and "age" not in fl
    out, fl = run(["24", "24", "23", "21", "24"])
    assert out["age"] == "24" and "age" not in fl


def test_setup_family_selection(monkeypatch):
    import setup_family as S
    projs = [{"name": "a", "max_workers": 10, "threads": 1}, {"name": "b", "max_workers": 640, "threads": 4}]
    assert [p["name"] for p in S.chosen(projs, "a")[0]] == ["a"] and S.chosen(projs, "a")[1] is False
    assert S.chosen(projs, None)[1] is True and S.chosen(projs, "all")[1] is True
    with pytest.raises(SystemExit):
        S.chosen(projs, "")
    with pytest.raises(SystemExit):
        S.chosen(projs, "a,zz")
    S.check(projs[:1])
    with pytest.raises(SystemExit, match="max_workers 640"):
        S.check(projs)
    import family
    S.check(family.PROJECTS)                                             # every family.py value is one the hub accepts


def test_planner_holds_incomplete_hints(monkeypatch):
    import plan_hi_ocr as P
    recs = [{"record_id": "S12-2027-DR/1", "pdfs": 10, "parsed": 10, "missing": 0, "match": 10},
            {"record_id": "S12-2027-DR/2", "pdfs": 10, "parsed": 10, "missing": 2, "match": 8},
            {"record_id": "S12-2027-DR/3", "pdfs": 10, "parsed": 9, "missing": 0, "match": 9},
            {"record_id": "S12-2027-DR/4", "pdfs": 10, "parsed": 10, "missing": 0, "match": 9}]
    monkeypatch.setattr(P, "psql_json", lambda sql: recs)
    ready, bad = P.ssr_done("eci-rolls-ssr-text", "S12-2027-DR")
    assert ready == {1, 4} and set(bad) == {2, 3, 4} and "not held" in bad[4]
    ready, bad = P.ssr_done("eci-rolls-ssr-text", "S12-2027-DR", strict=True)
    assert ready == {1}


def test_repair_any_epic_keeps_the_runs_fix2_guard():
    """ocrv5 review 2: repair-any 'epic' ran fix2 without the guard of the OCR run and put back the refused swap."""
    import gzip
    import json
    import repair_any_task as RA
    hints = [dict(epic="HSD1338565", name="मुकेशकुमार", relation_name="बजरंगलाल", relation_label="पिता का नाम", gender="पुरुष", age="34")]
    rec = dict(roll="S20-2026-DR", source="x.pdf", page=35, box=10, epic="UIO1338565", name="पिंटू राम", relation_name="जय गोपाल",
               relation_label="पिता का नाम", gender="पुरुष", age="34", flags=["epic"], readings={"epic": {"fix2_candidate": "HSD1338565"}},
               ocr_rules={"ECI_EPICFIX2_GUARD": "1", "ECI_EPICFIX2_MIN": "70"})
    files = {"eci-rolls-ssr-text/hints/S20-2027-DR/14.jsonl.gz": gzip.compress("".join(json.dumps(h) + "\n" for h in hints).encode()),
             "eci-rolls-hi-ocr/ocr/S20/14/70.jsonl.gz": gzip.compress((json.dumps(rec, ensure_ascii=False) + "\n").encode())}

    def fetch(keys, dest):
        out = {}
        for k in keys:
            p = os.path.join(dest, k.replace("/", "__"))
            open(p, "wb").write(files[k])
            out[k] = p
        return out
    up = []
    item = {"key": "epic/S20/14", "payload": {"state": "S20", "ac": 14, "p": [70], "fix": "epic",
                                              "hints": "eci-rolls-ssr-text/hints/S20-2027-DR/14.jsonl.gz"}}
    out = RA.do_item(None, item, None, fetch=fetch, upload_ocr=up.extend, upload_own=up.extend)
    assert out["changed"] == {"epic_fix1": 0, "epic_fix2": 0} and out["files_changed"] == 0 and up == []
    rec["ocr_rules"] = {"ECI_EPICFIX2_GUARD": "0"}                  # a run without the guard: the old fix2 (UP behaviour)
    files["eci-rolls-hi-ocr/ocr/S20/14/70.jsonl.gz"] = gzip.compress((json.dumps(rec, ensure_ascii=False) + "\n").encode())
    out = RA.do_item(None, item, None, fetch=fetch, upload_ocr=up.extend, upload_own=up.extend)
    assert out["changed"]["epic_fix2"] == 1
