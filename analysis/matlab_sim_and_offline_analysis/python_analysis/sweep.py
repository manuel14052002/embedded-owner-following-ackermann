import csv, collections, math
import replay as R
from replay import frames, F, cues, fuse
rows = list(csv.DictReader(open('../followme_logs/cues_20261010_145514.csv')))
B = [(float(t), l) for t, l, _ in (x.strip().split(',', 2) for x in open('../followme_logs/blocks.csv') if x.strip())]
E = [b[0] for b in B] + [1e12]
def blk(t):
    for i in range(len(B)):
        if E[i] <= t < E[i + 1]: return i
    return -1
# confirmed by Manuel: 0=A owner(+calibration), 1=volunteer only, 2=owner+volunteer, 3,4=owner only
OWNER, STRANGER = {0, 3, 4}, {1}
FR = [(blk(float(c[0]["t_wall"])), c) for c in frames(rows).values()]
own_fr = [c for b, c in FR if b in OWNER]; str_fr = [c for b, c in FR if b in STRANGER]
def owner_idx(c): return max(range(len(c)), key=lambda i: int(c[i]["mask_px"] or 0))
pre_own = [(c, owner_idx(c), [cues(r) for r in c]) for c in own_fr]
pre_str = [(c, [cues(r) for r in c]) for c in str_fr]
def strict_pick(c, k):
    ok = []
    for i, r in enumerate(c):
        d, thr, neg = F(r["d_reid"]), F(r["thr"]), F(r["neg_sim"])
        if d is None or thr is None or d > k * thr: continue
        if neg is not None and (1 - d) - neg < R.NEG_MARGIN: continue
        ok.append((d, i))
    return min(ok)[1] if ok else None
def fused_pick(C, g, tau):
    S = [fuse(x, g) for x in C]; k = max(range(len(S)), key=lambda i: S[i])
    return k if S[k] >= tau else None
def point(pick_own, pick_str):
    ta = wr = 0
    for (c, o, C) in pre_own:
        a = pick_own(c, C); ta += a == o; wr += a is not None and a != o
    fa = sum(pick_str(c, C) is not None for (c, C) in pre_str)
    n = len(pre_own); m = len(pre_str)
    return ta / n, wr / n, fa / m
res = {}
res["strict"] = [(k,) + point(lambda c, C, k=k: strict_pick(c, k), lambda c, C, k=k: strict_pick(c, k)) for k in [x / 20 for x in range(10, 61)]]
for name, g in (("fixed", False), ("gated", True)):
    res[name] = [(t,) + point(lambda c, C, t=t: fused_pick(C, g, t), lambda c, C, t=t: fused_pick(C, g, t)) for t in [x / 100 for x in range(30, 96)]]
casc = point(lambda c, C: next((i for i, r in enumerate(c) if r["chosen"] == "1"), None),
             lambda c, C: next((i for i, r in enumerate(c) if r["chosen"] == "1"), None))
print(f"owner frames {len(pre_own)}  volunteer-only frames {len(pre_str)}")
print(f"cascade (deployed): owner accepted {casc[0]:.3f} wrong {casc[1]:.3f} FA {casc[2]:.3f}")
for name, pts in res.items():
    z = [p for p in pts if p[3] == 0 and p[2] == 0]
    best = max(z, key=lambda p: p[1]) if z else None
    dep = [p for p in pts if abs(p[0] - (1.0 if name == "strict" else R.TAU)) < 1e-9][0]
    print(f"{name:7s} at default ({dep[0]}): TA {dep[1]:.3f} wrong {dep[2]:.3f} FA {dep[3]:.3f} | best with zero FA & zero wrong: param {best[0] if best else None} TA {best[1] if best else float('nan'):.3f}")
import json; json.dump({"res": res, "casc": casc}, open("sweep.json", "w"))
# owner acceptance by range bin (owner-only frames), strict@1.0 vs cascade vs gated@best-zero-FA
bins = [(0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 8)]
gz = max([p for p in res["gated"] if p[3] == 0 and p[2] == 0], key=lambda p: p[1])[0]
tab = collections.defaultdict(lambda: collections.Counter())
for (c, o, C) in pre_own:
    rg = F(c[o]["range_m"])
    if rg is None: continue
    b = next((bb for bb in bins if bb[0] <= rg < bb[1]), None)
    if b is None: continue
    t = tab[b]; t["n"] += 1
    t["strict"] += strict_pick(c, 1.0) == o
    t["cascade"] += next((i for i, r in enumerate(c) if r["chosen"] == "1"), None) == o
    t["gated"] += fused_pick(C, True, gz) == o
    t["reid_avail"] += C[o]["reid"][1] >= 0.5
    t["kf"] += C[o]["kf"][1] > 0; t["hint"] += C[o]["hint"][1] > 0
print(f"\nowner accepted by range (gated at tau={gz}):")
for b in bins:
    t = tab[b]; n = t["n"] or 1
    print(f"  {b[0]}-{b[1]} m n={t['n']:4d} strict {t['strict']/n:.3f} cascade {t['cascade']/n:.3f} gated {t['gated']/n:.3f} | ReID c>=0.5 {t['reid_avail']/n:.2f} KF {t['kf']/n:.2f} lidar {t['hint']/n:.2f}")
