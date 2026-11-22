"""Offline solver for the ECI voter-portal captcha (getCaptcha/EROLL).

    from captcha_solver import solve
    guesses = solve(jpg_bytes, k=5)   # top-k strings, best first

Needs numpy, opencv-python-headless, onnxruntime. Files next to this module: model.onnx, charset.txt.
The model takes the decoded RGB image (1x3x80x200 float32, 0..255); the fixed background is inside the model.
"""
import os
import numpy as np

# Characters seen in real captchas: a-z and 2-9 without i, l, o (0, 1, i, l, o never occur).
ALLOWED = set("abcdefghjkmnpqrstuvwxyz23456789")
_HERE = os.path.dirname(os.path.abspath(__file__))
_sess = None
_charset = None


def _load():
    global _sess, _charset
    if _sess is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("CAPTCHA_THREADS", "1"))
        so.inter_op_num_threads = 1
        _sess = ort.InferenceSession(os.path.join(_HERE, "model.onnx"), so, providers=["CPUExecutionProvider"])
        _charset = open(os.path.join(_HERE, "charset.txt")).read().strip()
    return _sess, _charset


def _decode(jpg_bytes):
    import cv2
    im = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("not an image")
    if im.shape[:2] != (80, 200):
        im = cv2.resize(im, (200, 80), interpolation=cv2.INTER_AREA)
    return im[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)  # 1x3x80x200 RGB


def probs(jpg_bytes):
    """Per-position probabilities, shape (6, len(charset))."""
    sess, _ = _load()
    return sess.run(None, {sess.get_inputs()[0].name: _decode(jpg_bytes)})[0][0]


def solve(jpg_bytes, k=5):
    """Return the k most likely 6-character answers, best first (exact k-best over independent positions)."""
    sess, cs = _load()
    p = probs(jpg_bytes)
    allowed = np.array([c in ALLOWED for c in cs])
    lp = np.where(allowed[None, :], np.log(np.maximum(p, 1e-12)), -1e9)
    beams = [(0.0, "")]
    for pos in range(lp.shape[0]):
        top = np.argsort(-lp[pos])[:k]
        beams = sorted(((s + lp[pos, c], t + cs[c]) for s, t in beams for c in top), key=lambda b: -b[0])[:k]
    return [t for _, t in beams]


if __name__ == "__main__":
    import sys, time
    for p in sys.argv[1:]:
        b = open(p, "rb").read()
        t = time.time(); g = solve(b); ms = (time.time() - t) * 1000
        print(p, " ".join(g), f"{ms:.1f} ms")
