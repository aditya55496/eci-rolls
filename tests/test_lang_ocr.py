"""engocr (2026-11-25): English and Tamil SIR items of task ocr-any, the sets engD / engJ / tamH2 / tamN2, the planner
plan_lang_ocr.py and parquet-any lang items. Offline: no network, no OCR (eci_ocr replaced)."""
import gzip
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SCRAPE_POOL_PROJECT", "eci-rolls-test")
import hi_common as H  # noqa: E402
import pytest  # noqa: E402

NEW = ("engD", "engJ", "engK11", "tamH2", "tamN2")
# eci_ocr switches that are no decision (call path, debug default, model path when no --crnn is given)
PASS = {"ECI_OCR_WORKERS", "ECI_TESS_CLI", "ECI_CRNN"}
# job1 speed switches (2026-11-25): added after the engocr sets were deployed (a deployed entry is frozen); their defaults
# keep every older set's behaviour (batch 64, no lazy Tesseract / hints, no stamp set, no screening, headings by CRNN 0)
PASS |= {"ECI_CRNN_BATCH", "ECI_NOTRIM_PAD", "ECI_TESS_LAZY", "ECI_HINTS_LAZY", "ECI_STAMP_SET", "ECI_S1_SCREEN", "ECI_HEAD_CRNN",
         "ECI_NOHINT_EARLY", "ECI_NOHINT_EARLY_CONF", "ECI_EARLY_DUMP"}


def _code_switches():
    names = set()
    for d in ("eciocr/ocr", "eciocr/prod"):
        for f in os.listdir(os.path.join(H.HERE, d)):
            if f.endswith(".py"):
                names |= set(re.findall(r"ECI_[A-Z0-9_]+", open(os.path.join(H.HERE, d, f), encoding="utf-8").read()))
    return names - PASS


def test_new_sets_files_sha_and_every_switch_named():
    sw = _code_switches()
    for name in NEW:
        ms = H.MODEL_SETS[name]
        assert len(H.model_dirs(name)) == len(ms["models"])
        assert H.set_lang(ms) in ("ENG", "TAM")
        missing = sw - set(ms["env"])
        assert not missing, (name, sorted(missing))
        # main's Hindi rules: off (the job evaluations ran without them); main's code defaults that are ON must be named 0
        assert ms["env"]["ECI_RESCALE_H"] == "0" and ms["env"]["ECI_EPIC_TESS_NOHINT"] == "0"
        assert ms["env"]["ECI_ENG_DESC"] == "0.35"
        for m in ms["models"]:
            if os.path.exists(os.path.join(H.HERE, "models", m, "arch.json")):
                assert m in ms["arch_sha256_16"], (name, m)
    assert H.MODEL_SETS["engD"]["env"]["ECI_ENG_DARK"] == "0"           # job3 code has no dark-ink cut
    assert H.MODEL_SETS["engJ"]["env"]["ECI_ENG_DARK"] == "170"
    assert H.MODEL_SETS["engJ"]["args"] == ["--tess2"] and H.MODEL_SETS["engJ"]["draft_records"] is True
    for name in ("engD", "engJ"):
        assert H.MODEL_SETS[name]["env"]["ECI_EPIC_PAIR"] == "0" and H.MODEL_SETS[name]["env"]["ECI_CODE_CRNN"] == "0"
    for name in ("tamH2", "tamN2"):
        e = H.MODEL_SETS[name]["env"]
        assert e["ECI_EPICFIX2_GUARD"] == "1" and e["ECI_EPICFIX2_MIN"] == "60" and e["ECI_TAM_EPIC_L2D8"] == "1"
    # a wrong arch.json sha makes the set incomplete (the stretch 'xs' changes the readings)
    sets = dict(H.MODEL_SETS, badarch=dict(H.MODEL_SETS["engD"], arch_sha256_16={"job3_r3_ekd15x0": "0" * 16, "job3_r4_tqx15sx0": "0" * 16}))
    with pytest.raises(H.ModelSetIncomplete, match="arch.json"):
        H.model_dirs("badarch", sets)
    sets["noarch"] = dict(H.MODEL_SETS["engD"], arch_sha256_16={})
    with pytest.raises(H.ModelSetIncomplete, match="arch.json"):
        H.model_dirs("noarch", sets)


def test_hindi_sets_unchanged_by_engocr():
    """The Hindi sets keep their entries (frozen): no lang key, rules without the engocr keys."""
    for name in ("v4a", "j2Ps", "j2Psf", "j5r4"):
        ms = H.MODEL_SETS[name]
        assert H.set_lang(ms) == "HIN" and "lang" not in ms and "args" not in ms and "stage1_nohint" not in ms
        r = H.rules_of(ms, 2026, False)
        assert "lang" not in r and "args" not in r and r["stage1"] == 0
    assert H.rules_of(H.MODEL_SETS["v4a"], 2026, True)["stage1"] == 1


def test_lang_set_choice_and_payload_versions():
    eng = {"model_set": "engJ", "hints": None, "lang": "ENG"}
    assert H.choose_set(eng, {}) == ("engJ", "payload")
    # a hub swap to a Hindi set is ignored for an English item (and the other way round)
    assert H.choose_set(eng, {"ocr_model_set": "j2Ps"}) == ("engJ", "payload")
    assert H.choose_set({"model_set": "j2Ps", "hints": None, "lang": "HIN"}, {"ocr_model_set": "engJ"}) == ("j2Ps", "payload")
    assert H.choose_set({"model_set": "tamN2", "hints": None, "lang": "TAM"}, {"ocr_model_set_map": {"tamN2": "engJ"}}) == ("tamN2", "payload")
    with pytest.raises(RuntimeError, match="planner error"):
        H.choose_set({"model_set": "engJ", "hints": None, "lang": "TAM"}, {})
    assert H.compatible(H.MODEL_SETS["engD"], True, "ENG") and not H.compatible(H.MODEL_SETS["engD"], False, "ENG")
    assert not H.compatible(H.MODEL_SETS["engJ"], False, "HIN") and not H.compatible(H.MODEL_SETS["v4a"], True, "ENG")
    assert H.ocr_base("S10", 60, None, "ENG") == "ocr/S10/ENG/60" and H.ocr_base("S10", 60) == "ocr/S10/60"
    assert H.ocr_base("S12", 1, None, "HIN") == "ocr/S12/1"
    for bad in (("S10", 60, None, "XYZ"), ("S10", 60, "S10-2024-GEN-FIR", "ENG")):
        with pytest.raises(RuntimeError):
            H.ocr_base(*bad)
    assert H.payload_version({"v": 3, "lang": "ENG"}) == 3
    with pytest.raises(RuntimeError, match="planner error"):
        H.payload_version({"v": 2, "lang": "ENG"})
    assert H.rules_of(H.MODEL_SETS["engJ"], 2026, False)["stage1"] == 2        # no-hint cascade keeps --stage1
    r = H.rules_of(H.MODEL_SETS["engJ"], 2026, False)
    assert r["lang"] == "ENG" and r["args"] == ["--tess2"] and r["draft_records"] is True
    with pytest.raises(RuntimeError, match="not allowed"):
        H.set_args({"args": ["--hint-2027", "/etc/passwd"]})


def _item(model_set="engJ", hints=None, lang="ENG", state="S10", ac=60, part=12, rolls=("S10-2026-DR", "S10-2026-SIR-FIR-1")):
    names = {"DR": "SIR-DraftRoll", "FIR": "SIR-FinalRoll"}
    keys = {r: f"eci-rolls-all-cdn/pdf/{r}/{ac}/2026-EROLLGEN-{state}-{ac}-{names['DR' if H.is_draft(r) else 'FIR']}-Revision1-{lang}-{part}-WI.pdf"
            for r in rolls}
    return {"key": f"{state}/{lang}/{ac}/{part}", "payload": {"state": state, "ac": ac, "part": part, "lang": lang, "keys": keys,
                                                              "hints": hints, "model_set": model_set, "v": 3}}


def _fetch(keys, dest):
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


def _fake_eci(calls, lang="ENG"):
    """O._run replaced: records the eci_ocr command, writes one record + the summary eci_ocr would write."""
    def run(cmd, env, timeout=3600):
        calls.append((list(cmd), dict(env)))
        out = cmd[cmd.index("--out") + 1]
        src = os.path.basename(cmd[2])
        rec = {"source": src, "page": 3, "box": 0, "kind": "SIR", "lang": lang, "serial_no": "1", "epic": "ABC1234567", "flags": []}
        open(out, "w").write(json.dumps(rec) + "\n")
        json.dump({"pdfs": [{"pdf": src, "records": 1, "draft_fill": {"matched": 0}}], "voter_pages": 1}, open(out + ".summary.json", "w"))

        class RU:
            ru_utime, ru_stime = 1.0, 0.5
        return 0, "", RU()
    return run


def test_do_item_english_pair_draft_records_and_paths(monkeypatch):
    import ocr_any_task as O
    calls, up = [], []
    monkeypatch.setattr(O, "_run", _fake_eci(calls))
    monkeypatch.setattr(O, "ensure_tessdata", lambda lang: None)
    monkeypatch.setattr(H, "CODE", "0123456789ab")
    monkeypatch.setattr(H, "code_commit", lambda: "0123456789ab")
    rec = O.do_item(None, _item(), settings={}, fetch=_fetch, upload=up.extend)
    (c1, e1), (c2, e2) = calls
    assert "--lang" in c1 and c1[c1.index("--lang") + 1] == "ENG" and "--tess2" in c1 and c1[c1.index("--stage1") + 1] == "2"
    assert "--hint-2027" not in c1 and "--draft-records" not in c1 and "--cache-out" in c1
    assert c2[c2.index("--draft-records") + 1] == c1[c1.index("--out") + 1]      # final reads the draft's output
    assert "--cache-in" in c2
    assert all(e1[k] == v for k, v in H.MODEL_SETS["engJ"]["env"].items())
    assert not [k for k in e1 if k.startswith("ECI_") and k not in H.MODEL_SETS["engJ"]["env"] and k not in ("ECI_OCR_WORKERS", "ECI_TESS_CLI")]
    assert [u[0] for u in up] == ["ocr/S10/ENG/60/12.summary.json", "ocr/S10/ENG/60/12.jsonl.gz"]
    recs = [json.loads(l) for l in gzip.decompress(up[1][1]).decode().splitlines()]
    assert {r["roll"] for r in recs} == {"S10-2026-DR", "S10-2026-SIR-FIR-1"}
    assert all(r["ocr_model_set"] == "engJ" and r["ocr_rules"]["lang"] == "ENG" and r["hints_used"] is False for r in recs)
    assert rec["lang"] == "ENG" and rec["pdfs"][0]["draft_fill"] == {"matched": 0}
    summ = json.loads(up[0][1])
    assert summ["lang"] == "ENG" and summ["item"] == "S10/ENG/60/12"


def test_do_item_hint_set_and_tamil(monkeypatch):
    import ocr_any_task as O
    calls, up = [], []
    monkeypatch.setattr(O, "_run", _fake_eci(calls))
    monkeypatch.setattr(O, "ensure_tessdata", lambda lang: None)
    monkeypatch.setattr(H, "CODE", "0123456789ab")
    monkeypatch.setattr(H, "code_commit", lambda: "0123456789ab")
    it = _item("engD", "eci-rolls-ssr-text/hints/S05-2027-DR/12.jsonl.gz", state="S05", ac=12, part=5,
               rolls=("S05-2026-DR", "S05-2026-FIR"))
    O.do_item(None, it, settings={}, fetch=_fetch, upload=up.extend)
    c1 = calls[0][0]
    assert "--hint-2027" in c1 and c1[c1.index("--stage1") + 1] == "1" and "--tess2" not in c1
    assert not any("--draft-records" in c for c, _ in calls)          # engD: no draft_records
    calls.clear()
    up.clear()
    monkeypatch.setattr(O, "_run", _fake_eci(calls, "TAM"))
    O.do_item(None, _item("tamN2", lang="TAM", state="S22", ac=80, part=40, rolls=("S22-2026-DR", "S22-2026-FIR")),
              settings={}, fetch=_fetch, upload=up.extend)
    c1 = calls[0][0]
    assert c1[c1.index("--lang") + 1] == "TAM" and c1[c1.index("--stage1") + 1] == "2" and "--tess2" not in c1
    assert up[1][0] == "ocr/S22/TAM/80/40.jsonl.gz"
    with pytest.raises(RuntimeError, match="needs hints"):
        O.do_item(None, _item("tamH2", lang="TAM", state="S22", rolls=("S22-2026-DR",)), settings={}, fetch=_fetch, upload=up.extend)


def test_main_missing_traineddata_gives_item_back(monkeypatch):
    import ocr_any_task as O
    from test_ocr_any import _FakeHub
    _FakeHub.instances.clear()

    class Hub(_FakeHub):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.items = [_item("tamN2", lang="TAM", state="S22", rolls=("S22-2026-DR",))]
    monkeypatch.setattr(O, "HubClient", Hub)
    monkeypatch.setattr(H, "CODE", "0123456789ab")
    monkeypatch.setattr(O, "TESSDATA", "/nonexistent")
    monkeypatch.setattr(O.subprocess, "run", lambda *a, **k: None)      # no apt-get here
    with pytest.raises(SystemExit) as e:
        O.main()
    hub = _FakeHub.instances[0]
    assert e.value.code == 2 and hub.completed[0][1] == "pending" and "traineddata" in hub.completed[0][2]
    assert "tool_missing" in hub.metrics


def test_old_payload_version_code_gives_lang_items_back():
    """Workers of main db823e0 know payload v 1 and 2: a v 3 item raises UnknownPayloadVersion (given back, not failed)."""
    old = {1, 2}
    assert H.PAYLOAD_V == 3 and H.PAYLOAD_V not in old


# ---- planner (hub database replaced)

def _rows():
    def pdf(roll, ac, part, lang="ENG"):
        kind = "DraftRoll" if H.is_draft(roll) else "FinalRoll"
        st = roll[:3]
        f = f"2026-EROLLGEN-{st}-{ac}-SIR-{kind}-Revision1-{lang}-{part}-WI.pdf"
        return {"record_id": f"{roll}/{ac}/{part}", "key": f"eci-rolls-all-cdn/pdf/{roll}/{ac}/{f}", "file": f, "ac": ac, "part": part}

    def acr(roll, ac, parts, lang_used="ENG", missing=()):
        return {"record_id": f"{roll}/{ac}", "key": None, "file": None, "ac": ac, "part": None, "part_numbers": list(parts),
                "extra_parts": [], "missing": list(missing), "lang_used": lang_used}
    return {
        "S05-2026-DR": [acr("S05-2026-DR", 12, [1, 2]), pdf("S05-2026-DR", 12, 1), pdf("S05-2026-DR", 12, 2),
                        acr("S05-2026-DR", 13, [1]), pdf("S05-2026-DR", 13, 1)],
        "S05-2026-FIR": [acr("S05-2026-FIR", 12, [1, 2]), pdf("S05-2026-FIR", 12, 1), pdf("S05-2026-FIR", 12, 2),
                         acr("S05-2026-FIR", 13, [1]), pdf("S05-2026-FIR", 13, 1)],
        "S10-2026-DR": [acr("S10-2026-DR", 60, [1, 2, 3]), pdf("S10-2026-DR", 60, 1), pdf("S10-2026-DR", 60, 2), pdf("S10-2026-DR", 60, 3)],
        # FIR: part 3 listed but no PDF (fetch gap) -> held; AC 61 interim record without lang_used: language from the file names
        "S10-2026-SIR-FIR-1": [acr("S10-2026-SIR-FIR-1", 60, [1, 2, 3]), pdf("S10-2026-SIR-FIR-1", 60, 1), pdf("S10-2026-SIR-FIR-1", 60, 2),
                               dict(acr("S10-2026-SIR-FIR-1", 61, [7], None)), pdf("S10-2026-SIR-FIR-1", 61, 7)],
        "S22-2026-DR": [acr("S22-2026-DR", 80, [40], "TAM"), pdf("S22-2026-DR", 80, 40, "TAM")],
        "S22-2026-FIR": [acr("S22-2026-FIR", 80, [40], "TAM"), pdf("S22-2026-FIR", 80, 40, "TAM")],
        "S25-2026-DR": [acr("S25-2026-DR", 5, [1], "ENG"), pdf("S25-2026-DR", 5, 1), acr("S25-2026-DR", 6, [1], "BEN"),
                        pdf("S25-2026-DR", 6, 1, "BEN")],
        "S25-2026-FIR": [acr("S25-2026-FIR", 5, [1], "ENG"), pdf("S25-2026-FIR", 5, 1)],
    }


def test_plan_lang_ocr(monkeypatch):
    import plan_lang_ocr as L
    rows = _rows()
    monkeypatch.setattr(L, "roll_rows", lambda roll: rows.get(roll, []))
    monkeypatch.setattr(L, "fetch_status", lambda roll: {})
    monkeypatch.setattr(L.P, "ssr_done", lambda project, roll, strict=False: ({12}, {13: "missing_pdfs 1"}) if roll == "S05-2027-DR" else (set(), {}))
    items, summ = L.plan({"S05", "S10", "S22", "S25"})
    by = {i["key"]: i for i in items}
    assert sorted(by) == ["S05/ENG/12/1", "S05/ENG/12/2", "S05/ENG/13/1", "S10/ENG/60/1", "S10/ENG/60/2", "S10/ENG/61/7", "S22/TAM/80/40"]
    assert by["S05/ENG/12/1"]["payload"]["model_set"] == "engD"
    assert by["S05/ENG/12/1"]["payload"]["hints"] == "eci-rolls-ssr-text/hints/S05-2027-DR/12.jsonl.gz"
    assert by["S05/ENG/13/1"]["payload"]["model_set"] == "engK11" and by["S05/ENG/13/1"]["payload"]["hints"] is None   # hints incomplete
    assert summ["S05"]["acs_hints_incomplete"] == {"13": "missing_pdfs 1"}
    assert list(by["S10/ENG/60/1"]["payload"]["keys"]) == ["S10-2026-DR", "S10-2026-SIR-FIR-1"]
    assert summ["S10"]["parts_partial_held"] == {"S10-2026-SIR-FIR-1": 1}                              # part 3: FIR listed, no PDF
    assert list(by["S10/ENG/61/7"]["payload"]["keys"]) == ["S10-2026-SIR-FIR-1"] and summ["S10"]["langs"]["ENG"] == 2
    assert by["S22/TAM/80/40"]["payload"]["model_set"] == "tamN2" and by["S22/TAM/80/40"]["payload"]["lang"] == "TAM"
    assert all(i["payload"]["v"] == 3 for i in items)
    assert by["S10/ENG/60/1"]["partition_id"] == L.RANK["S10"] * 1_000_000 + 60 * 1000 + 1
    assert summ["S25"]["held_state"] and summ["S25"]["items"] == 1 and summ["S25"]["acs_other_lang"] == {"6": "BEN"}
    items, summ = L.plan({"S25"}, release={"S25"})
    assert [i["key"] for i in items] == ["S25/ENG/5/1"]
    items, _ = L.plan({"S05"}, wait_hints=True)
    assert sorted(i["key"] for i in items) == ["S05/ENG/12/1", "S05/ENG/12/2"]
    items, _ = L.plan({"S10"}, include_partial=True)
    assert "S10/ENG/60/3" in {i["key"] for i in items}
    items, _ = L.plan({"S05"}, state_set={"S05": "engJ"})
    assert {i["payload"]["model_set"] for i in items} == {"engJ"} and all(i["payload"]["hints"] is None for i in items)
    with pytest.raises(SystemExit):
        L.plan({"S05"}, state_set={"S05": "j2Ps"})          # a Hindi set does not fit English ACs
    monkeypatch.setattr(L, "queued_acs", lambda project: {("S05", "ENG", 12)})
    items, summ = L.plan({"S05"}, new_only=True)
    assert [i["key"] for i in items] == ["S05/ENG/13/1"] and summ["S05"]["acs_new_only_skipped"] == 1
    # keys never collide with Hindi SIR keys (3 segments) or form keys (group starts with the state code)
    assert all(len(i["key"].split("/")) == 4 and i["key"].split("/")[1] in L.PLAN_LANGS for i in items)


def test_plan_parquet_lang_and_form_planner_skip(monkeypatch):
    import plan_form_ocr as F
    import plan_lang_ocr as L
    q = [{"key": "S10/ENG/60/1", "status": "done", "keys": {"S10-2026-SIR-FIR-1": "x", "S10-2026-DR": "y"}, "done_at": 5},
         {"key": "S10/ENG/60/2", "status": "done", "keys": {"S10-2026-DR": "y"}, "done_at": 6},
         {"key": "S22/TAM/80/40", "status": "pending", "keys": {"S22-2026-DR": "z"}, "done_at": None}]
    monkeypatch.setattr(L.P, "psql_json", lambda sql: q if "status, payload" in sql else [{"key": "S10/ENG/60", "done_at": 5.5}])
    items, summ, stale = L.plan_parquet()
    assert [i["key"] for i in items] == ["S10/ENG/60"] and stale == ["S10/ENG/60"]
    p = items[0]["payload"]
    assert p["lang"] == "ENG" and p["ocr_project"] == H.OCR_ALL_PROJECT and p["parts"] == [1, 2] and p["v"] == 3
    assert p["rolls"] == ["S10-2026-DR", "S10-2026-SIR-FIR-1"] and summ["S22 TAM not ready"] == 1
    monkeypatch.setattr(F.P, "psql_json", lambda sql: [{"key": "S10/ENG/60/1", "status": "done", "keys": {"S10-2026-DR": "y"}}])
    assert F.plan_parquet()[0] == []                       # lang keys are no form-route groups


def test_parquet_any_lang_item():
    import pyarrow.parquet as pq
    import parquet_any_task as Q
    recs = [{"roll": "S10-2026-DR", "source": "a.pdf", "page": 3, "box": i, "kind": "SIR", "lang": "ENG", "serial_no": str(i + 1),
             "epic": "ABC1234567", "name": "Ravi", "flags": [], "confirmed_by_draft": ["name"], "ocr_model_set": "engJ",
             "ocr_models": ["a"], "ocr_code": "0123456789ab", "ocr_rules": {"lang": "ENG"}, "hints_used": False} for i in range(3)]
    body = gzip.compress("".join(json.dumps(r) + "\n" for r in recs).encode())
    seen, up = [], []

    def fetch(keys, dest):
        seen.extend(keys)
        out = {}
        for k in keys:
            p = os.path.join(dest, k.replace("/", "__"))
            open(p, "wb").write(body)
            out[k] = p
        return out
    it = {"key": "S10/ENG/60", "payload": {"state": "S10", "ac": 60, "lang": "ENG", "parts": [1], "rolls": ["S10-2026-DR"],
                                           "ocr_project": H.OCR_ALL_PROJECT, "v": 3}}
    rec = Q.do_item(None, it, fetch=fetch, upload=up.extend)
    assert seen == ["eci-rolls-all-ocr/ocr/S10/ENG/60/1.jsonl.gz"]
    assert up[0][0] == "parquet/S10/S10-2026-DR/ENG/ac_60/data.parquet" and rec["lang"] == "ENG"
    t = pq.read_table(io.BytesIO(up[0][1]))
    assert t.schema.names[-6:] == ["ocr_model_set", "ocr_models", "ocr_code", "ocr_rules", "hints_used", "lang"]
    assert t.num_rows == 3 and set(t.column("lang").to_pylist()) == {"ENG"}
    assert json.loads(t.column("extra").to_pylist()[0])["confirmed_by_draft"] == ["name"]
    bad = json.loads(json.dumps(it))
    bad["payload"]["ocr_project"] = H.OCR_HI_PROJECT
    with pytest.raises(RuntimeError, match="expected"):
        Q.do_item(None, bad, fetch=fetch, upload=up.extend)
