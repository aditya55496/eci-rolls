"""Form-route OCR (formocr, 2026-11-25): set j5r4, payload v 2 roll_group, output paths, planner plan_form_ocr.py, parquet-any.
Offline: no network, no OCR (engine and hub database replaced)."""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SCRAPE_POOL_PROJECT", "eci-rolls-test")
import hi_common as H  # noqa: E402
import pytest  # noqa: E402
from test_ocr_any import _fake_fetch, _fake_run_eci  # noqa: E402


def _form_item(group="S08-2025-SSR", rolls=("S08-2025-DR", "S08-2025-FIR"), v=2, model_set="j5r4"):
    return {"key": f"S08/{group}/55/52", "payload": {
        "state": "S08", "ac": 55, "part": 52, "lang": "HIN", "model_set": model_set, "roll_group": group, "v": v, "hints": None,
        "keys": {r: f"eci-rolls-hi-form/pdf/{r}/55/2025-EROLLGEN-S08-55-{'DraftRoll' if r.endswith('-DR') else 'FinalRoll'}-52.pdf"
                 for r in rolls}}}


def test_j5r4_set_files_env_and_sha():
    ms = H.MODEL_SETS["j5r4"]
    assert ms["hints"] == "none" and ms["models"] == ["j5r2_s11x0", "j5r2_s9x0", "j5r2_s4x0", "j5r2_s3x0"]
    dirs = H.model_dirs("j5r4")      # crnn.onnx + charset.json of the models and every aux file, sha256 checked
    assert [os.path.basename(d) for d in dirs] == ms["models"]
    env = H.set_env(ms)
    models = os.path.join(H.HERE, "models")
    assert env["ECI_NOHINT_VALSET"] == ",".join(os.path.join(models, a) for a in ms["aux"][:3])
    assert env["ECI_NOHINT_LEX"] == os.path.join(models, "job2_lex4.npz") and os.path.isfile(env["ECI_NOHINT_MARKVAR"])
    assert "<models>" in ms["env"]["ECI_NOHINT_VALSET"]          # provenance keeps the path-free form
    for k in ("ECI_EPIC_STATE", "ECI_PAREN_GUARD", "ECI_NOHINT_VALGUARD", "ECI_CODE_VALID", "ECI_NOHINT_CLOSED"):
        assert ms["env"][k] == "1", k
    assert ms["env"]["ECI_NOHINT_VALF"] == "name,relation_label,relation_name,age,gender"     # no house_no (brackets)
    # the proposal of job5 round 4 (models2/job5/models/model_sets_proposal.json 'j5r4'): same models, sha256 and switches
    for k, v in {"ECI_NOHINT_TESSCLEAN": "3", "ECI_NOHINT_NUNF": "1000", "ECI_NOHINT_VALKEEP": "crnn_j5r2_s3x0"}.items():
        assert ms["env"][k] == v
    for name in ("v4a", "j2Ps", "j2Psf"):     # the older sets have no '<models>' paths: set_env = env
        assert H.set_env(H.MODEL_SETS[name]) == H.MODEL_SETS[name]["env"]


def test_j5r4_aux_sha_checked():
    ms = dict(H.MODEL_SETS["j5r4"], sha256_16=dict(H.MODEL_SETS["j5r4"]["sha256_16"], **{"job2_lex4.npz": "0" * 16}))
    with pytest.raises(H.ModelSetIncomplete, match="job2_lex4.npz"):
        H.model_dirs("bad_aux", {"bad_aux": ms})
    ms = dict(H.MODEL_SETS["j5r4"], aux=H.MODEL_SETS["j5r4"]["aux"] + ["missing.npz"])
    with pytest.raises(H.ModelSetIncomplete, match="missing.npz"):
        H.model_dirs("bad_aux2", {"bad_aux2": ms})


def test_ocr_base_and_payload_versions():
    assert H.ocr_base("S12", 1) == "ocr/S12/1"
    assert H.ocr_base("S08", 55, "S08-2025-SSR") == "ocr/S08/S08-2025-SSR/55"
    for bad in ("S12-2025-SSR", "../x", "S08/2025", ""):
        with pytest.raises(RuntimeError):
            H.ocr_base("S08", 55, bad)
    assert H.payload_version({"v": 2, "roll_group": "S08-2025-SSR"}) == 2 and H.payload_version({}) == 1
    with pytest.raises(RuntimeError, match="planner error"):
        H.payload_version({"v": 1, "roll_group": "S08-2025-SSR"})
    with pytest.raises(H.UnknownPayloadVersion):
        H.payload_version({"v": 3})


def test_do_item_form_route_paths_env_and_order(monkeypatch):
    import ocr_any_task as O
    calls, up = [], []
    monkeypatch.setattr(O, "run_eci", _fake_run_eci(calls))
    monkeypatch.setattr(H, "repair_part", lambda recs, engines, rules=None: {"final_deleted": 0, "serial_gaps": 0})
    monkeypatch.setattr(H, "code_commit", lambda: H.CODE)
    rec = O.do_item(None, _form_item(), settings={}, fetch=_fake_fetch, upload=up.extend)
    assert calls[0]["rolls"] == ["S08-2025-DR", "S08-2025-FIR"] and calls[0]["year"] == 2025 and calls[0]["hints"] is None
    assert [os.path.basename(d) for d in calls[0]["dirs"]] == H.MODEL_SETS["j5r4"]["models"]
    assert [u[0] for u in up] == ["ocr/S08/S08-2025-SSR/55/52.summary.json", "ocr/S08/S08-2025-SSR/55/52.jsonl.gz"]
    recs = [json.loads(x) for x in gzip.decompress(up[1][1]).decode().splitlines()]
    assert {r["roll"] for r in recs} == {"S08-2025-DR", "S08-2025-FIR"}
    for r in recs:
        assert r["ocr_model_set"] == "j5r4" and r["hints_used"] is False and r["ocr_rules"]["roll_year"] == 2025
        assert r["ocr_rules"]["stage1"] == 0 and r["ocr_rules"]["ECI_NOHINT_LEX"] == "<models>/job2_lex4.npz"
    assert rec["roll_group"] == "S08-2025-SSR" and json.loads(up[0][1])["roll_group"] == "S08-2025-SSR"
    # GE: one roll, year 2024
    calls.clear()
    up.clear()
    O.do_item(None, _form_item("S08-2024-GEN-FIR", ("S08-2024-GEN-FIR",)), settings={}, fetch=_fake_fetch, upload=up.extend)
    assert calls[0]["year"] == 2024 and up[1][0] == "ocr/S08/S08-2024-GEN-FIR/55/52.jsonl.gz"
    # a roll of another state in a form item, a v 1 payload with a group: refused
    with pytest.raises(RuntimeError, match="not of state"):
        O.do_item(None, _form_item("S08-2025-SSR", ("S12-2025-DR",)), settings={}, fetch=_fake_fetch, upload=up.extend)
    with pytest.raises(RuntimeError, match="planner error"):
        O.do_item(None, _form_item(v=1), settings={}, fetch=_fake_fetch, upload=up.extend)


def test_run_eci_env_expands_models(monkeypatch, tmp_path):
    """run_eci passes the set's env with '<models>' replaced, and removes other ECI_* variables (as before)."""
    import ocr_any_task as O
    seen = {}

    def fake_run(cmd, env, timeout=3600):
        seen.update(env)
        out = cmd[cmd.index("--out") + 1]
        open(out, "w").write(json.dumps({"page": 1, "box": 0}) + "\n")
        json.dump({"voter_pages": 1, "pdfs": []}, open(out + ".summary.json", "w"))

        class RU:
            ru_utime = ru_stime = 0.0
        return 0, "", RU()
    monkeypatch.setattr(O, "_run", fake_run)
    monkeypatch.setenv("ECI_NOHINT_NAMES", "3")
    ms = H.MODEL_SETS["j5r4"]
    O.run_eci([("S08-2025-DR", str(tmp_path / "a.pdf"))], None, ms, H.model_dirs("j5r4"), 2025, str(tmp_path))
    assert seen["ECI_NOHINT_VALSET"].startswith(os.path.join(H.HERE, "models")) and "<models>" not in seen["ECI_NOHINT_LEX"]
    assert seen["ECI_NOHINT_NAMES"] == "0"            # the set's own value, not the worker env's
    seen.clear()
    O.run_eci([("S12-2026-DR", str(tmp_path / "b.pdf"))], None, H.MODEL_SETS["j2Ps"], [], 2026, str(tmp_path))
    # an older set: exactly its env (the new switches stay unset = code default off; the worker's ECI_NOHINT_NAMES is removed)
    eci = {k: v for k, v in seen.items() if k.startswith("ECI_") and k not in ("ECI_OCR_WORKERS", "ECI_TESS_CLI")}
    assert eci == H.MODEL_SETS["j2Ps"]["env"]


# ---- planner (hub database replaced)

def _rows(roll, ac, parts, listed=None, lang="HIN"):
    st = roll.split("-")[0]
    r = [{"project": "eci-rolls-hi-form", "record_id": f"{roll}/{ac}", "ac": ac, "part": None, "key": None,
          "part_numbers": listed or parts, "extra_parts": [], "missing": [], "lang": lang}]
    return r + [{"project": "eci-rolls-hi-form", "record_id": f"{roll}/{ac}/{p}", "ac": ac, "part": p,
                 "key": f"eci-rolls-hi-form/pdf/{roll}/{ac}/{st}-{p}.pdf"} for p in parts]


def test_classify_and_groups():
    import plan_form_ocr as F
    assert F.group_of("S12-2024-GEN-FIR") == ("S12-2024-GEN-FIR", "GE", 2024)
    assert F.group_of("U05-2025-GEN-FIR") == ("U05-2025-GEN-FIR", "GE", 2025)
    assert F.group_of("S08-2025-DR") == ("S08-2025-SSR", "SSR", 2025) == F.group_of("S08-2025-FIR")
    for r, why in (("S04-2025-SIR-DR", "SIR"), ("S12-2026-SUPP-2", "supplement"), ("S25-2026-SUPP-JO-A-12A", "adjudication"),
                   ("S20-2025-BY-DR", "bye-election"), ("S04-2025-ST-FIR", "other"), ("S25-2026-FIR-2", "other")):
        assert F.group_of(r) is None and why in F.classify(r)[0], r
    assert F.slot("GE", 2024) == 0 and F.slot("SSR", 2024) == 1 and F.slot("GE", 2025) == 2 and F.slot("SSR", 2025) == 3


def test_plan_form(monkeypatch):
    import plan_form_ocr as F
    import plan_hi_ocr as P
    rows = {"S08-2025-DR": _rows("S08-2025-DR", 55, [1, 2, 3]) + _rows("S08-2025-DR", 56, [1]),
            "S08-2025-FIR": _rows("S08-2025-FIR", 55, [1, 2], listed=[1, 2, 3]) + _rows("S08-2025-FIR", 56, [1]),
            "S08-2024-GEN-FIR": _rows("S08-2024-GEN-FIR", 55, [1, 2], listed=[1, 2, 3]) + _rows("S08-2024-GEN-FIR", 57, [1], lang="ENG"),
            "S12-2024-DR": _rows("S12-2024-DR", 1, [1]), "S12-2025-FIR": _rows("S12-2025-FIR", 1, [1]),
            "S07-2024-GEN-FIR": _rows("S07-2024-GEN-FIR", 1, [1])}
    monkeypatch.setattr(P, "roll_rows", lambda roll: rows.get(roll, []))
    rolls = {"S08-2025-DR": {55: "done", 56: "done"}, "S08-2025-FIR": {55: "done", 56: "failed"},
             "S08-2024-GEN-FIR": {55: "done", 57: "done"}, "S08-2025-SUPP-4": {55: "pending"},
             "S12-2024-DR": {1: "done"}, "S12-2025-FIR": {1: "done"}, "S07-2024-GEN-FIR": {1: "done"}, "S20-2025-BY-DR": {2: "pending"}}
    items, summ = F.plan(rolls=rolls, planned=set())
    keys = [i["key"] for i in items]
    # GE first (slot 0), then SSR; AC 56 waits for its final roll; part 3 has only the draft (final done: planned with one roll)
    assert keys == ["S08/S08-2024-GEN-FIR/55/1", "S08/S08-2024-GEN-FIR/55/2", "S08/S08-2025-SSR/55/1", "S08/S08-2025-SSR/55/2",
                    "S08/S08-2025-SSR/55/3", "S12/S12-2024-SSR/1/1", "S12/S12-2025-SSR/1/1"]
    it = {i["key"]: i for i in items}
    p = it["S08/S08-2025-SSR/55/1"]["payload"]
    assert p["v"] == 2 and p["roll_group"] == "S08-2025-SSR" and p["model_set"] == "j5r4" and p["hints"] is None
    assert list(p["keys"]) == ["S08-2025-DR", "S08-2025-FIR"]
    assert list(it["S08/S08-2025-SSR/55/3"]["payload"]["keys"]) == ["S08-2025-DR"]
    assert it["S08/S08-2024-GEN-FIR/55/1"]["partition_id"] == 100_000_000 + (4 * 4 + 0) * 1_000_000 + 55_000 + 1
    assert it["S08/S08-2025-SSR/55/1"]["partition_id"] == 100_000_000 + (4 * 4 + 3) * 1_000_000 + 55_000 + 1
    s = summ["S08/S08-2025-SSR"]
    assert s["acs_done"] == 1 and s["acs_waiting_fetch"] == 1 and s["parts_one_roll"] == 1
    assert summ["S08/S08-2024-GEN-FIR"]["acs_other_lang"] == 1
    assert summ["S08/S08-2024-GEN-FIR"]["listed_parts_without_pdf"] == {"S08-2024-GEN-FIR": {"55": [3]}}
    assert summ["S07/S07-2024-GEN-FIR"]["held_state"] and summ["S07/S07-2024-GEN-FIR"]["items"] == 1
    assert summ["excluded: supplement"] == {"S08": 1} and summ["excluded: bye-election"] == {"S20": 1}
    assert not any(k.startswith("S07/") for k in keys)
    # --release S07, --include-partial (AC 56 with its draft only), --new-only (AC 55 groups already planned)
    items, summ = F.plan(rolls=rolls, planned={("S08", "S08-2025-SSR", 55), ("S08", "S08-2024-GEN-FIR", 55)}, release={"S07"},
                         include_partial=True)
    keys = [i["key"] for i in items]
    assert keys == ["S08/S08-2025-SSR/56/1", "S12/S12-2024-SSR/1/1", "S12/S12-2025-SSR/1/1", "S07/S07-2024-GEN-FIR/1/1"]
    assert list(items[0]["payload"]["keys"]) == ["S08-2025-DR"] and summ["S08/S08-2025-SSR"]["acs_planned_before"] == 1
    # keys never collide with the SIR planner's '<state>/<ac>/<part>'
    assert all(k.count("/") == 3 for k in keys)
    # a set that needs hints waits for them; S12 hints from ssrtext, others none
    sets = dict(H.MODEL_SETS, hreq=dict(H.MODEL_SETS["j5r4"], hints="required"))
    monkeypatch.setattr(H, "MODEL_SETS", sets)
    items, summ = F.plan(rolls=rolls, planned=set(), type_set={"SSR": "hreq"}, hints_ready={"S12": {1}})
    ssr = [i for i in items if "-SSR/" in i["key"]]
    assert [i["key"] for i in ssr] == ["S12/S12-2024-SSR/1/1", "S12/S12-2025-SSR/1/1"]
    assert ssr[0]["payload"]["hints"] == "eci-rolls-ssr-text/hints/S12-2027-DR/1.jsonl.gz"
    assert summ["S08/S08-2025-SSR"]["acs_waiting_hints"] == 1
    assert F.hints_for("S24", 5, {"S24": {5}}) == ("eci-rolls-up-2027/hints/5.jsonl.gz", True)
    assert F.hints_for("S08", 5, {}) == (None, None)


def test_plan_parquet_form_and_sir_skip(monkeypatch):
    import plan_form_ocr as F
    import plan_hi_ocr as P
    q = [{"key": "S08/S08-2025-SSR/55/1", "status": "done", "keys": {"S08-2025-DR": "a", "S08-2025-FIR": "b"}},
         {"key": "S08/S08-2025-SSR/55/2", "status": "done", "keys": {"S08-2025-DR": "c"}},
         {"key": "S08/S08-2024-GEN-FIR/55/1", "status": "pending", "keys": {"S08-2024-GEN-FIR": "d"}}]
    monkeypatch.setattr(P, "psql_json", lambda sql: q)
    items, summ = F.plan_parquet()
    assert [i["key"] for i in items] == ["S08/S08-2025-SSR/55"]
    assert items[0]["payload"] == {"state": "S08", "ac": 55, "parts": [1, 2], "roll_group": "S08-2025-SSR",
                                   "rolls": ["S08-2025-DR", "S08-2025-FIR"]}
    assert summ == {"S08/S08-2024-GEN-FIR not ready": 1, "S08/S08-2025-SSR ready": 1}
    # the SIR planner's --parquet must skip the 4-segment form keys (before: ValueError on split)
    q2 = q + [{"key": "U02/1/1", "status": "done", "keys": {"U02-2026-DR": "x"}, "done_at": 1}]
    monkeypatch.setattr(P, "psql_json", lambda sql: [dict(r, done_at=1) for r in q2] if "hi-ocr" in sql else [])
    items, _, _ = P.plan_parquet(None)
    assert [i["key"] for i in items] == ["U02/1"]


def test_parquet_any_form_item(monkeypatch, tmp_path):
    import parquet_any_task as Q
    recs = [{"roll": "S08-2025-DR", "page": 3, "box": 0, "serial_no": "1", "name": "क", "ocr_model_set": "j5r4", "ocr_models": ["m"],
             "ocr_code": "0123456789ab", "ocr_rules": {"roll_year": 2025}, "hints_used": False, "flags": []},
            {"roll": "S08-2025-FIR", "page": 3, "box": 0, "serial_no": "1", "name": "क", "ocr_model_set": "j5r4", "ocr_models": ["m"],
             "ocr_code": "0123456789ab", "ocr_rules": {"roll_year": 2025}, "hints_used": False, "flags": ["name"]}]
    seen, up = [], []

    def fetch(keys, dest):
        seen.extend(keys)
        out = {}
        for k in keys:
            p = os.path.join(dest, k.replace("/", "__"))
            with gzip.open(p, "wt", encoding="utf-8") as f:
                f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
            out[k] = p
        return out
    item = {"key": "S08/S08-2025-SSR/55", "payload": {"state": "S08", "ac": 55, "parts": [52], "roll_group": "S08-2025-SSR",
                                                       "rolls": ["S08-2025-DR", "S08-2025-FIR"]}}
    rec = Q.do_item(None, item, fetch=fetch, upload=up.extend)
    assert seen == ["eci-rolls-hi-ocr/ocr/S08/S08-2025-SSR/55/52.jsonl.gz"]
    assert [u[0] for u in up] == ["parquet/S08/S08-2025-DR/ac_55/data.parquet", "parquet/S08/S08-2025-FIR/ac_55/data.parquet"]
    assert rec["roll_group"] == "S08-2025-SSR" and rec["rolls"]["S08-2025-FIR"]["rows"] == 1
    import io
    t = Q.pq.read_table(io.BytesIO(up[0][1]))
    assert t.column("roll").to_pylist() == ["S08-2025-DR"] and t.column("ocr_model_set").to_pylist() == ["j5r4"]
    assert t.schema.names[-5:] == ["ocr_model_set", "ocr_models", "ocr_code", "ocr_rules", "hints_used"]
