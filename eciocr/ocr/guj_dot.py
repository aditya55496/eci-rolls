"""Relation-label anusvara check for Gujarati SIR rolls (lang-guj round 5).

The rolls print the relation label as 'પિતાનું નામ' / 'પતિનું નામ' (with the anusvara dot above ન) in most boxes, and without
the dot ('પિતાનુ' / 'પતિનુ') in ~8 % of boxes (data entry). The CRNNs were trained on pseudo labels that always carry the dot,
and Tesseract drops it in ~18 % of boxes where it is printed. This module decides dot / no dot from the pixels of the relation
line crop (segment_guj line crop): a small numpy MLP (1968 -> 64 -> 1) on the left part of the crop, height 24 px, width 82 px.
Trained on 17,475 real non-gold relation lines with Opus line labels (reads A == B, lang-guj round 4 api1 + api2).
prob(dot) >= HI -> dot, < LO -> no dot, between -> unchanged + field flagged."""
import os
import numpy as np
import cv2

H, W = 24, 82
_P = None
LO = float(os.environ.get('ECI_GUJ_DOT_LO', '0.1'))
HI = float(os.environ.get('ECI_GUJ_DOT_HI', '0.9'))
FORMS = {'પિતાનું': 'પિતાનુ', 'પતિનું': 'પતિનુ'}


def feat(a):
    a = 1.0 - np.asarray(a, dtype=np.float32) / 255.0
    cols = np.where(a.max(0) > 0.35)[0]
    if len(cols) == 0: return np.zeros(H * W, np.float32)
    a = a[:, cols[0]:]
    h, w = a.shape; nw = max(1, int(round(w * H / h)))
    x = cv2.resize(a, (nw, H), interpolation=cv2.INTER_AREA)[:, :W]
    b = np.zeros((H, W), np.float32); b[:, :x.shape[1]] = x
    return b.ravel()


def prob(crop):
    global _P
    if _P is None:
        _P = dict(np.load(os.environ.get('ECI_GUJ_DOT_FILE', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'guj_dot.npz'))))
    h = np.maximum(0, feat(crop) @ _P['W1'] + _P['b1'])
    return float(1 / (1 + np.exp(-(h @ _P['W2'] + _P['b2'])[0])))


def decide(label, crop):
    """label: current relation_label value; crop: the relation line crop. -> (label, decided: bool, p)"""
    parts = label.split(' ', 1)
    w = parts[0]
    base = w if w in FORMS else next((k for k, v in FORMS.items() if v == w), None)
    if base is None or crop is None or crop.size == 0: return label, False, None
    p = prob(crop)
    rest = (' ' + parts[1]) if len(parts) > 1 else ''
    if p >= HI: return base + rest, True, p
    if p < LO: return FORMS[base] + rest, True, p
    return label, False, p
