#!/usr/bin/env python3
"""Evaluate a tracker CSV log against owner ground truth.

Ground truth CSV: ``frame,person_id,x1,y1,x2,y2`` (0-based frames, all
annotated people, so that wrong-person reports can be identified).

    python evaluate.py --gt gt.csv --log logs/owner_track.csv --owner-id 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from owner_tracking.evaluation import evaluate_owner_track, load_ground_truth, load_tracker_log


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--owner-id", type=int, required=True, help="GT person_id the operator clicked")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--min-reid-gap", type=int, default=5)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    preds, lat = load_tracker_log(args.log)
    gt = load_ground_truth(args.gt, num_frames=len(preds))
    m = evaluate_owner_track(gt, preds, args.owner_id, args.iou, args.min_reid_gap, lat)
    d = m.as_dict()
    if args.json:
        print(json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()},
                         indent=2))
    else:
        for k, v in d.items():
            print(f"{k:28s} {v:.4f}" if isinstance(v, float) else f"{k:28s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
