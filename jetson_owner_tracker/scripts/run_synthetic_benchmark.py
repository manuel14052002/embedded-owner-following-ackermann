#!/usr/bin/env python3
"""Run the synthetic identity-preservation suite: OwnerTracker vs a proximity baseline.

    python scripts/run_synthetic_benchmark.py --seeds 10 --jitter 3 --miss-rate 0.05
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from owner_tracking.baselines import NearestNeighborBaseline  # noqa: E402
from owner_tracking.config import TrackerConfig  # noqa: E402
from owner_tracking.evaluation import evaluate_scenario_results  # noqa: E402
from owner_tracking.reid import HistogramReID  # noqa: E402
from owner_tracking.simulation import build_scenarios, run_scenario  # noqa: E402
from owner_tracking.tracker import OwnerTracker  # noqa: E402

KEYS = ["incorrect_owner_switches", "false_owner_frames", "recall", "precision", "idf1", "hota",
        "reid_success_rate"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--jitter", type=float, default=1.5)
    p.add_argument("--miss-rate", type=float, default=0.0)
    args = p.parse_args(argv)
    agg: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    names: list[str] = []
    for seed in range(args.seeds):
        for method in ("owner_tracker", "nearest_nb"):
            for name, sc in build_scenarios(seed, args.jitter, args.miss_rate).items():
                if name not in names:
                    names.append(name)
                tracker = (OwnerTracker(TrackerConfig(), HistogramReID()) if method == "owner_tracker"
                           else NearestNeighborBaseline())
                m = evaluate_scenario_results(run_scenario(sc, tracker), sc.owner_id).as_dict()
                for k in KEYS:
                    agg[(name, method)][k].append(m[k])
    head = f"{'scenario':30s} {'method':14s} " + " ".join(f"{k[:12]:>12s}" for k in KEYS)
    print(head)
    print("-" * len(head))
    for name in names:
        for method in ("owner_tracker", "nearest_nb"):
            row = agg[(name, method)]
            cells = []
            for k in KEYS:
                vals = [v for v in row[k] if not (isinstance(v, float) and math.isnan(v))]
                total = k in ("incorrect_owner_switches", "false_owner_frames")
                v = (sum(vals) if total else sum(vals) / len(vals)) if vals else float("nan")
                cells.append(f"{v:12.0f}" if total else f"{v:12.3f}")
            print(f"{name:30s} {method:14s} " + " ".join(cells))
    print("\nincorrect_owner_switches / false_owner_frames are totals over seeds; others are means.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
