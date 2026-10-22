#!/usr/bin/env python3
"""Train and evaluate the gesture SVM (paper Sect. III-D, VI-A, Fig. 6a, Table I left).

    python scripts/followme_train_gestures.py data/gestures/*.npz --out models/gesture_svm.joblib
    python scripts/followme_train_gestures.py data/gestures/*.npz --protocol loso

Protocols (the confusion matrix is averaged over subjects, as in Fig. 6a):
    split  per subject, a stratified ``--test-fraction`` of its samples is held out;
           the SVM is trained on all remaining samples of all subjects.
    loso   leave-one-subject-out: the tested subject is never seen in training
           (stricter: measures generalisation to a new user).
The final model saved to ``--out`` is trained on all samples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from followme.evaluation import confusion_matrix, format_report  # noqa: E402
from followme.gestures import CLASSES, GestureClassifier, landmark_features  # noqa: E402


def load(paths: list[str], mirror: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ss = [], [], []
    for p in paths:
        d = np.load(p)
        for lm, h in zip(d["landmarks"], d["handedness"]):
            xs.append(landmark_features(lm, str(h), mirror))
        ys.extend(d["labels"].tolist())
        ss.extend(d["subjects"].tolist())
    return np.array(xs), np.array(ys), np.array(ss)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", default="models/gesture_svm.joblib")
    ap.add_argument("--protocol", choices=["split", "loso"], default="split")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--c", type=float, default=10.0)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    x, y, s = load(args.data, not args.no_mirror)
    print(f"{len(y)} samples, subjects={sorted(set(s))}, " +
          ", ".join(f"{c}={int((y == c).sum())}" for c in CLASSES))
    rng = np.random.default_rng(args.seed)
    cms = []
    for subj in sorted(set(s)):
        if args.protocol == "loso":
            test = s == subj
        else:
            test = np.zeros(len(y), bool)
            for c in CLASSES:
                idx = np.flatnonzero((s == subj) & (y == c))
                rng.shuffle(idx)
                test[idx[:int(round(len(idx) * args.test_fraction))]] = True
        if test.sum() == 0 or (~test).sum() == 0:
            continue
        clf = GestureClassifier(c=args.c).fit(x[~test], y[~test])
        cm = confusion_matrix(y[test], clf.predict(x[test]), CLASSES)
        cms.append(cm)
        print(format_report(cm, CLASSES, f"\n== subject {subj} ({args.protocol})"))
    if cms:
        print(format_report(np.mean(cms, axis=0), CLASSES,
                            f"\n== MEAN over {len(cms)} subjects (Fig. 6a / Table I)"))
    GestureClassifier(c=args.c).fit(x, y).save(args.out)
    print(f"\nfinal model trained on all {len(y)} samples -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
