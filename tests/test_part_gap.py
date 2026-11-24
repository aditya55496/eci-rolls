"""Part-list gap and completeness logic of task fetch-hi (2026-11-24). Offline: no network, no Tesseract."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SCRAPE_POOL_PROJECT", "eci-rolls-test")
import fetch_hi_task as T  # noqa: E402
import printcheck as P  # noqa: E402


def hub(checks):
    return types.SimpleNamespace(checks=checks)


def ok(e=100):
    return {"ok": True, "pages": "ok", "part": "ok", "totals": "ok", "electors": e}


def test_complete_with_extras_and_missing():
    parts = [{"partNumber": n} for n in range(1, 6)]                   # current list 1..5
    info = {"missing": [5], "extra_parts": [6, 7]}                     # 5 has no file in this roll, 6-7 above the list
    c = T.completeness(hub({n: ok() for n in (1, 2, 3, 4, 6, 7)}), parts, info, "fetch")
    assert c["parts_with_file"] == 6 and c["highest_part"] == 7
    assert c["holes"] == [5] and c["complete"] is False                 # a hole inside 1..highest is not complete


def test_complete_when_roll_is_shorter_than_list():
    parts = [{"partNumber": n} for n in range(1, 55)]
    info = {"missing": [53, 54], "extra_parts": []}
    c = T.completeness(hub({n: ok(10) for n in range(1, 53)}), parts, info, "check_only")
    assert c["holes"] == [] and c["complete"] is True and c["electors_printed"] == 520


def test_printed_fault_and_unchecked():
    parts = [{"partNumber": n} for n in range(1, 4)]
    info = {"missing": [], "extra_parts": []}
    bad = {1: ok(), 2: {**ok(), "ok": False, "pages": "bad"}, 3: ok()}
    assert T.completeness(hub(bad), parts, info, "fetch")["complete"] is False
    assert T.completeness(hub({1: ok()}), parts, info, "fetch")["complete"] is None   # not every PDF checked


def test_totals_row_rules():
    assert P._totals([1, 615, 328, 287, 0, 615])[:4] == (328, 287, 0, 615)
    assert P._totals([627, 330, 297, 627])[:4] == (330, 297, 0, 627)                   # lone '0' dropped by Tesseract
    assert P._totals([2026, 1, 5]) is None
    assert P._num(":5") == 5 and P._num("২৬") == 26 and P._num("No.") is None


def test_parquet_extra_part_marker():
    try:
        import parquet_task as Q
    except Exception:       # pyarrow not installed here: the task installs it on the runner
        return
    rows = [{"ac": 5, "roll_part": 1, "name": "a"}, {"ac": 5, "roll_part": 2, "name": "b", "extra_part": True}]
    t = Q.to_table(rows, Q.OCR_SCHEMA, "2026-sir-final", 5).to_pylist()
    assert t[0]["extra"] is None and t[1]["extra"] == '{"extra_part": true}'
    assert [c.name for c in Q.OCR_SCHEMA] == [c for c in t[0]]           # schema unchanged
