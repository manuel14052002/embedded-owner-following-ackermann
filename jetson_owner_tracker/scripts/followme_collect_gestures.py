#!/usr/bin/env python3
"""Record hand-landmark samples for the gesture SVM (paper Sect. III-D, VI-A).

Live::

    python scripts/followme_collect_gestures.py --subject manuel --source realsense
      1 = record WAIT (open hand), 2 = FOLLOW (closed fist), 3 = OTHER (anything else),
      0 / space = pause, Q = save and quit.  Move the hand around, change distance
      and rotation while recording (paper: 500 samples per class per subject).

From images laid out as ``root/<subject>/<WAIT|FOLLOW|OTHER>/*.jpg``::

    python scripts/followme_collect_gestures.py --images data/gesture_images --out data/gestures

Output: ``<out>/<subject>.npz`` with ``landmarks (N,21,3)``, ``handedness``,
``labels`` and ``subjects`` (features are recomputed at training time).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from followme.config import GestureConfig  # noqa: E402
from followme.gestures import CLASSES, FOLLOW, OTHER, WAIT, HandLandmarkDetector  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def save(path: Path, lms, hands, labels, subject: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():                                   # append to an existing session
        old = np.load(path)
        lms = list(old["landmarks"]) + lms
        hands = list(old["handedness"]) + hands
        labels = list(old["labels"]) + labels
    np.savez(path, landmarks=np.array(lms, np.float32).reshape(-1, 21, 3), handedness=np.array(hands),
             labels=np.array(labels), subjects=np.array([subject] * len(labels)))
    counts = {c: int(np.sum(np.array(labels) == c)) for c in CLASSES}
    print(f"saved {path}: {counts}")


def from_images(root: Path, out: Path, cfg: GestureConfig) -> None:
    det = HandLandmarkDetector(cfg, static_images=True)
    for subj in sorted(p for p in root.iterdir() if p.is_dir()):
        lms, hands, labels = [], [], []
        for cls in CLASSES:
            for f in sorted((subj / cls).glob("*")):
                if f.suffix.lower() not in EXTS:
                    continue
                obs = det.detect(cv2.imread(str(f)))
                if obs is not None:
                    lms.append(obs.landmarks)
                    hands.append(obs.handedness)
                    labels.append(cls)
        save(out / f"{subj.name}.npz", lms, hands, labels, subj.name)


def live(args, cfg: GestureConfig) -> None:
    if args.source.startswith("realsense"):
        from owner_tracking.camera import RealSenseCapture
        cap = RealSenseCapture()
    else:
        cap = cv2.VideoCapture(int(args.source)) if args.source.isdigit() else cv2.VideoCapture(args.source)
    det = HandLandmarkDetector(cfg, static_images=False)
    keys = {ord("1"): WAIT, ord("2"): FOLLOW, ord("3"): OTHER}
    rec, lms, hands, labels = None, [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        obs = det.detect(frame)
        disp = frame.copy()
        if obs is not None:
            for x, y, _ in obs.landmarks.astype(int):
                cv2.circle(disp, (x, y), 3, (0, 0, 255), -1)
            if rec is not None:
                lms.append(obs.landmarks)
                hands.append(obs.handedness)
                labels.append(rec)
        counts = " ".join(f"{c}:{labels.count(c)}" for c in CLASSES)
        cv2.putText(disp, f"REC {rec or '-'}   {counts}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("collect gestures", disp)
        k = cv2.waitKey(1) & 0xFF
        if k in keys:
            rec = keys[k]
        elif k in (ord("0"), ord(" ")):
            rec = None
        elif k in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    save(Path(args.out) / f"{args.subject}.npz", lms, hands, labels, args.subject)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="subject0")
    ap.add_argument("--source", default="0")
    ap.add_argument("--images", default=None)
    ap.add_argument("--out", default="data/gestures")
    ap.add_argument("--task-model", default="hand_landmarker.task")
    args = ap.parse_args()
    cfg = GestureConfig(task_model_path=args.task_model)
    if args.images:
        from_images(Path(args.images), Path(args.out), cfg)
    else:
        live(args, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
