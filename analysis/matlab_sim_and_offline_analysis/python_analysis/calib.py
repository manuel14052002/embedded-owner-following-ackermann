"""Calibrate per-cue reliability by range on block A only; select every rule's threshold on block A;
evaluate on held-out blocks (B1, B2 owner-only; volunteer-only block)."""
import csv, collections, math, json
import replay as R
from replay import frames, F, cues
rows = list(csv.DictReader(open('../followme_logs/cues_20261010_145514.csv')))
B = [float(x.split(',')[0]) for x in open('../followme_logs/blocks.csv') if x.strip()]
E = B + [1e12]
def blk(t):
    for i in range(len(B)):
        if E[i] <= t < E[i + 1]: return i
    return -1
FR = [(blk(float(c[0]["t_wall"])), c) for c in frames(rows).values()]
def oidx(c): return max(range(len(c)), key=lambda i: int(c[i]["mask_px"] or 0))
CAL = [c for b, c in FR if b == 0]                      # block A: owner (largest) + background people
TEST_OWN = [c for b, c in FR if b in (3, 4)]
TEST_STR = [c for b, c in FR if b == 1]
BINS = [0, 1.5, 2.5, 3.5, 99]
def rbin(r):
    x = F(r["range_m"])
    if x is None: return None
    return next(i for i in range(len(BINS) - 1) if BINS[i] <= x < BINS[i + 1])
# --- calibration: Youden J at s>=0.5, per cue, per owner-range bin; impostor rate pooled over block A
gen = collections.defaultdict(lambda: [0, 0]); imp = collections.defaultdict(lambda: [0, 0])
for c in CAL:
    o = oidx(c)
    for i, r in enumerate(c):
        C = cues(r)
        for k, (s, conf) in C.items():
            if conf <= 0: continue
            if i == o:
                g = gen[(k, rbin(r))]; g[0] += s >= 0.5; g[1] += 1
            else:
                m = imp[k]; m[0] += s >= 0.5; m[1] += 1
REL = {}
for k in R.W:
    fpr = imp[k][0] / imp[k][1] if imp[k][1] else 0.5
    for b in list(range(len(BINS) - 1)) + [None]:
        g = gen[(k, b)]
        tpr = g[0] / g[1] if g[1] >= 20 else None
        REL[(k, b)] = max(0.0, tpr - fpr) if tpr is not None else None
    REL[(k, "pooled")] = max(0.0, sum(gen[(k, b)][0] for b in range(4)) / max(1, sum(gen[(k, b)][1] for b in range(4))) - fpr)
print("calibrated reliability (Youden J) per cue x range bin  [<1.5, 1.5-2.5, 2.5-3.5, >3.5 m]; impostor FPR pooled:")
for k in R.W:
    print(f"  {k:5s} " + "  ".join(f"{REL[(k,b)]:.2f}" if REL[(k,b)] is not None else " -- " for b in range(4)),
          f"  FPR={imp[k][0]/max(1,imp[k][1]):.2f} (n_imp={imp[k][1]})")
def cues_cal(r):
    C = cues(r); b = rbin(r); out = {}
    for k, (s, conf) in C.items():
        if conf <= 0: out[k] = (s, 0.0); continue
        j = REL.get((k, b)); j = REL[(k, "pooled")] if j is None else j
        out[k] = (s, j)
    return out
def fuse(C, mode):
    n = d = 0.0
    for k, w in R.W.items():
        s, c = C[k]
        cc = 1.0 if mode == "fixed" else c
        n += w * cc * s; d += w * cc
    return n / d if d > 1e-9 else 0.0
def pick(c, rule, p):
    if rule == "cascade":
        return next((i for i, r in enumerate(c) if r["chosen"] == "1"), None)
    if rule == "strict":
        ok = []
        for i, r in enumerate(c):
            d, thr, neg = F(r["d_reid"]), F(r["thr"]), F(r["neg_sim"])
            if d is None or thr is None or d > p * thr: continue
            if neg is not None and (1 - d) - neg < R.NEG_MARGIN: continue
            ok.append((d, i))
        return min(ok)[1] if ok else None
    Cs = [cues_cal(r) if rule == "gated_cal" else cues(r) for r in c]
    S = [fuse(x, "fixed" if rule == "fixed" else "gated") for x in Cs]
    k = max(range(len(S)), key=lambda i: S[i])
    return k if S[k] >= p else None
def evaluate(own, strg, rule, p):
    ta = wr = 0
    for c in own:
        a = pick(c, rule, p); o = oidx(c); ta += a == o; wr += a is not None and a != o
    fa = sum(pick(c, rule, p) is not None for c in strg)
    return ta / max(1, len(own)), wr / max(1, len(own)), (fa / len(strg) if strg else 0.0)
GRID = {"strict": [x / 20 for x in range(10, 61)], "fixed": [x / 100 for x in range(30, 96)],
        "gated": [x / 100 for x in range(30, 96)], "gated_cal": [x / 100 for x in range(30, 96)], "cascade": [None]}
print(f"\ncalibration frames {len(CAL)} | test: owner {len(TEST_OWN)}, volunteer-only {len(TEST_STR)}")
out = {}
for rule, grid in GRID.items():
    best = None
    for p in grid:                         # select on block A: no wrong person, maximise owner acceptance
        ta, wr, _ = evaluate(CAL, [], rule, p)
        if wr == 0 and (best is None or ta > best[1]): best = (p, ta)
    p = best[0]
    ta, wr, fa = evaluate(TEST_OWN, TEST_STR, rule, p)
    out[rule] = dict(param=p, cal_ta=best[1], test_ta=ta, test_wrong=wr, test_fa=fa)
    print(f"  {rule:9s} param={p}  [A] owner {best[1]:.3f} | TEST owner accepted {ta:.3f}  wrong {wr:.3f}  volunteer accepted {fa:.3f}")
json.dump({"rel": {f"{k}|{b}": v for (k, b), v in REL.items()}, "out": out}, open("calib.json", "w"), indent=1)
