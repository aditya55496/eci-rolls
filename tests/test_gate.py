"""Back-off gate and file retries of task fetch-hi (2026-11-25). Offline: no network."""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SCRAPE_POOL_PROJECT", "eci-rolls-test")
import fetch_hi_task as T  # noqa: E402


class Hub:
    def __init__(self):
        self.m = {}

    def metric(self, name, v=1):
        self.m[name] = self.m.get(name, 0) + v


def resp(code, payload=None):
    return types.SimpleNamespace(status_code=code, json=lambda: {"payload": payload, "refId": "X-HIN-7-WI.pdf"})


def test_gate_doubles_once_per_overload_and_opens():
    g, h = T.Gate(), Hub()
    t0 = time.time()
    g.bad(h, t0)
    assert g.pause == 2.0
    g.bad(h, t0)                      # started before the gate closed: same overload, no new step
    assert g.pause == 2.0
    g.bad(h, time.time() + 0.001)
    assert g.pause == 4.0 and h.m["gate_close"] == 2
    g.good(); g.good(); g.good()
    assert g.pause == 0.0


def test_file_retry_same_id(monkeypatch):
    seq = [None, resp(504), resp(200, "JVBERi0xLjQK")]  # base64 of "%PDF-1.4\n"
    calls = []

    def fake(hub, sess, url, **kw):
        calls.append(url)
        return seq.pop(0)
    monkeypatch.setattr(T, "timed_get", fake)
    monkeypatch.setattr(T, "GATE", types.SimpleNamespace(wait=lambda hub: time.time(), bad=lambda *a: None, good=lambda: None))
    monkeypatch.setattr(T.E, "is_pdf", lambda b: b.startswith(b"%PDF"))
    h = Hub()
    b, name, err = T.get_file(h, "fid1")
    assert b.startswith(b"%PDF") and err == "" and len(set(calls)) == 1 and len(calls) == 3
    assert h.m["file_retry"] == 2 and h.m["ep_file_timeout"] == 1 and h.m["ep_file_504"] == 1


def test_file_404_then_ok(monkeypatch):
    seq = [resp(404), resp(200, "JVBERi0xLjQK")]
    monkeypatch.setattr(T, "timed_get", lambda *a, **k: seq.pop(0))
    monkeypatch.setattr(T, "GATE", types.SimpleNamespace(wait=lambda hub: time.time(), bad=lambda *a: None, good=lambda: None))
    monkeypatch.setattr(T.E, "is_pdf", lambda b: b.startswith(b"%PDF"))
    monkeypatch.setattr(T, "FILE_404_WAIT_S", 0)
    h = Hub()
    b, name, err = T.get_file(h, "fid")
    assert err == "" and h.m["file_404_retry"] == 1 and h.m["file_404_then_ok"] == 1


def test_file_404_gives_up(monkeypatch):
    calls = []
    monkeypatch.setattr(T, "timed_get", lambda *a, **k: calls.append(1) or resp(404))
    monkeypatch.setattr(T, "GATE", types.SimpleNamespace(wait=lambda hub: time.time(), bad=lambda *a: None, good=lambda: None))
    monkeypatch.setattr(T, "FILE_404_WAIT_S", 0)
    h = Hub()
    assert T.get_file(h, "fid") == (None, None, 404) and len(calls) == 1 + T.FILE_404_TRIES


def test_file_no_retry_on_4xx(monkeypatch):
    monkeypatch.setattr(T, "timed_get", lambda *a, **k: resp(403))
    monkeypatch.setattr(T, "GATE", types.SimpleNamespace(wait=lambda hub: time.time(), bad=lambda *a: None, good=lambda: None))
    h = Hub()
    assert T.get_file(h, "fid") == (None, None, 403) and "file_retry" not in h.m
