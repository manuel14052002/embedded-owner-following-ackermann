"""Offline replay of four identity rules on FollowMe cue logs (per-candidate rows).
strict : ReID distance <= threshold and not more like a known stranger (original FollowMe rule)
cascade: the decision FollowMe actually made (logged 'chosen')
fixed  : weighted mean of cue similarities, every cue counted (c_i = 1, unavailable cue votes 0)
gated  : S = sum w c s / sum w c, unavailable cues abstain (c_i = 0)
Cue models are a-priori (not fitted to these data)."""
import csv, collections, math, sys
W = {"reid": 0.40, "neg": 0.15, "cont": 0.15, "kf": 0.15, "hint": 0.15}
TAU = 0.60          # acceptance threshold for fused rules (a priori)
NEG_MARGIN = 0.03   # FollowMe identity.negative_margin
def F(x):
    try:
        v = float(x); return v if math.isfinite(v) else None
    except: return None
def cues(r):
    d, thr = F(r["d_reid"]), F(r["thr"])
    out = {}
    # ReID: d = thr maps to 0.5; reliability from mask size (crop resolution)
    if d is not None and thr:
        s = min(1, max(0, 1 - d / (2 * thr)))
        m = F(r["mask_px"]) or 0
        out["reid"] = (s, min(1.0, m / 20000.0))
    else: out["reid"] = (0.0, 0.0)
    neg = F(r["neg_sim"])
    if neg is not None and d is not None:
        margin = (1 - d) - neg
        out["neg"] = (min(1, max(0, 0.5 + margin / 0.2)), 1.0)
    else: out["neg"] = (0.0, 0.0)
    iou, age = F(r["iou_prev"]), F(r["prev_age_s"])
    out["cont"] = (iou, 1.0) if (iou is not None and age is not None and age <= 0.5 and r["crossing"] == "0") else (0.0, 0.0)
    kd, kg, ka = F(r["kf_dist"]), F(r["kf_gate"]), F(r["kf_age_s"])
    out["kf"] = (min(1, max(0, 1 - kd / kg)), max(0.0, 1 - ka / 2.0)) if (kd is not None and kg) else (0.0, 0.0)
    hd, hr = F(r["hint_dist"]), F(r["hint_r"])
    out["hint"] = (min(1, max(0, 1 - hd / (2 * hr))), 1.0) if (hd is not None and hr) else (0.0, 0.0)
    return out
def fuse(c, gated):
    n = dd = 0.0
    for k, w in W.items():
        s, conf = c[k]
        cc = conf if gated else 1.0
        n += w * cc * s; dd += w * cc
    return n / dd if dd > 1e-9 else 0.0
def strict_ok(r):
    d, thr, neg = F(r["d_reid"]), F(r["thr"]), F(r["neg_sim"])
    if d is None or thr is None or d > thr: return False
    return neg is None or (1 - d) - neg >= NEG_MARGIN
def frames(rows):
    by = collections.OrderedDict()
    for r in rows:
        if r["mode"] != "IDENTIFYING" or r["cand"] == "-1": continue
        by.setdefault(r["frame"], []).append(r)
    return by
def decide(cands):
    """returns dict rule -> index of accepted candidate in cands or None"""
    res = {}
    ch = [i for i, r in enumerate(cands) if r["chosen"] == "1"]
    res["cascade"] = ch[0] if ch else None
    ok = [(F(r["d_reid"]), i) for i, r in enumerate(cands) if strict_ok(r)]
    res["strict"] = min(ok)[1] if ok else None
    C = [cues(r) for r in cands]
    for name, g in (("fixed", False), ("gated", True)):
        S = [fuse(c, g) for c in C]
        k = max(range(len(S)), key=lambda i: S[i])
        res[name] = k if S[k] >= TAU else None
    return res, C
