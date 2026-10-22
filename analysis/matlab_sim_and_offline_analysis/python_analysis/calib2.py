exec(open('calib.py').read().split('GRID = ')[0])
h = len(TEST_STR) // 2
STR_CAL, STR_TEST = TEST_STR[:h], TEST_STR[h:]
# recalibrate reliabilities with the first half of the volunteer block added as impostors
imp2 = collections.defaultdict(lambda: [0, 0])
for k in R.W: imp2[k] = list(imp[k])
for c in STR_CAL:
    for r in c:
        for k, (s, conf) in cues(r).items():
            if conf > 0: imp2[k][0] += s >= 0.5; imp2[k][1] += 1
for k in R.W:
    fpr = imp2[k][0] / imp2[k][1]
    for b in range(4):
        g = gen[(k, b)]
        REL[(k, b)] = max(0.0, g[0] / g[1] - fpr) if g[1] >= 20 else None
    REL[(k, "pooled")] = max(0.0, sum(gen[(k, b)][0] for b in range(4)) / max(1, sum(gen[(k, b)][1] for b in range(4))) - fpr)
print("reliability with volunteer impostors (first half) added:")
for k in R.W:
    print(f"  {k:5s} " + "  ".join(f"{REL[(k,b)]:.2f}" if REL[(k,b)] is not None else " -- " for b in range(4)), f" FPR={imp2[k][0]/imp2[k][1]:.2f} n={imp2[k][1]}")
GRID = {"strict": [x / 20 for x in range(10, 61)], "fixed": [x / 100 for x in range(30, 96)],
        "gated": [x / 100 for x in range(30, 96)], "gated_cal": [x / 100 for x in range(30, 96)], "cascade": [None]}
print(f"\nselect on: A owner {len(CAL)} + volunteer 1st half {len(STR_CAL)} | test: B owner {len(TEST_OWN)} + volunteer 2nd half {len(STR_TEST)}")
res = {}
for rule, grid in GRID.items():
    best = None
    for p in grid:
        ta, wr, fa = evaluate(CAL, STR_CAL, rule, p)
        if wr == 0 and fa == 0 and (best is None or ta > best[1]): best = (p, ta)
    if best is None: print(rule, "no zero-FA setting"); continue
    ta, wr, fa = evaluate(TEST_OWN, STR_TEST, rule, best[0])
    res[rule] = (best[0], best[1], ta, wr, fa, int(round(fa * len(STR_TEST))))
    print(f"  {rule:9s} param={best[0]}  [select] owner {best[1]:.3f} | TEST owner {ta:.3f} wrong {wr:.3f} volunteer accepted {fa:.3f} ({res[rule][5]}/{len(STR_TEST)})")
# by range on test, for the table
bins = [(0,1.5),(1.5,2.5),(2.5,3.5),(3.5,99)]
print("\nTEST owner acceptance by range:")
for lo, hi in bins:
    sel = [c for c in TEST_OWN if (F(c[oidx(c)]["range_m"]) or -1) >= lo and (F(c[oidx(c)]["range_m"]) or -1) < hi]
    line = f"  {lo}-{hi} m n={len(sel):4d} "
    for rule in res:
        line += f"{rule} {sum(pick(c, rule, res[rule][0]) == oidx(c) for c in sel)/max(1,len(sel)):.2f}  "
    print(line)
json.dump(res, open("calib2.json", "w"), indent=1)
