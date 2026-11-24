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
