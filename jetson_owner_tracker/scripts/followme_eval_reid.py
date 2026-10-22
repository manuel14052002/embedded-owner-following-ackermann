#!/usr/bin/env python3
"""Re-ID experiment of the paper (Sect. VI-B): Fig. 2, Fig. 6b, Table I (right).

Dataset layout (images are .jpg/.png)::

    root/
      <subject>/calib/*.jpg        calibration images, the subject alone (paper: 500)
      <subject>/without/*.jpg      everyone except the subject (paper: 500)
      all_present/*.jpg            everyone present (paper: 500, shared)
      all_present/boxes.csv        image,subject,x1,y1,x2,y2  (GT box of each subject)

Each image is segmented and embedded once (cached in ``--cache``); then each
subject is calibrated in turn (2/3 : 1/3 split) and evaluated against the
others, and the confusion matrices are averaged over subjects.

    python scripts/followme_eval_reid.py --root data/reid --config config/followme.yaml \
        --reid-model mmt_msmt17_ibn50a_256.pth.tar --out results/reid
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from followme.config import FollowMeConfig  # noqa: E402
from followme.evaluation import REID_LABELS, ImageInstances, evaluate_reid_subject, format_report  # noqa: E402
from followme.reid import build_feature_extractor  # noqa: E402
from followme.segmentation import build_segmenter, masked_crop  # noqa: E402
from owner_tracking.runtime import resolve_device  # noqa: E402

LOGGER = logging.getLogger("followme.eval_reid")
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS) if folder.is_dir() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--reid-model", default=None)
    ap.add_argument("--reid-backend", default=None)
    ap.add_argument("--cache", default=None, help="npz feature cache (default <out>/features.npz)")
    ap.add_argument("--out", default="results/reid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = FollowMeConfig.from_yaml(args.config) if args.config else FollowMeConfig()
    if args.reid_model:
        cfg.reid.model_path = args.reid_model
    if args.reid_backend:
        cfg.reid.backend = args.reid_backend
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else out / "features.npz"

    subjects = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "all_present")
    files = [f for s in subjects for f in images(root / s / "calib") + images(root / s / "without")]
    files += images(root / "all_present")
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=True)
        cache = data["cache"].item()
        LOGGER.info("Loaded %d cached images", len(cache))
    todo = [f for f in files if str(f.relative_to(root)) not in cache]
    if todo:
        device = resolve_device(cfg.device)
        seg = build_segmenter(cfg.segmentation, device)
        ext = build_feature_extractor(cfg.reid, device)
        for n, f in enumerate(todo):
            img = cv2.imread(str(f))
            persons = seg.segment(img) if img is not None else []
            crops, boxes = [], []
            for p in persons:
                c = masked_crop(img, p, cfg.segmentation.remove_background)
                if c is not None:
                    crops.append(c)
                    boxes.append(p.bbox)
            feats = ext.extract(crops) if crops else np.zeros((0, ext.feature_dim), np.float32)
            cache[str(f.relative_to(root))] = (feats, np.array(boxes, np.float32).reshape(-1, 4))
            if n % 200 == 0:
                LOGGER.info("embedded %d / %d", n, len(todo))
        np.savez(cache_path, cache=np.array(cache, dtype=object))

    gt: dict[tuple[str, str], np.ndarray] = {}
    gt_file = root / "all_present" / "boxes.csv"
    if gt_file.is_file():
        with open(gt_file, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                gt[(row["image"], row["subject"])] = np.array(
                    [float(row[k]) for k in ("x1", "y1", "x2", "y2")], np.float32)
    else:
        LOGGER.warning("No all_present/boxes.csv: test set 1 cannot be scored (every hit counts as a miss)")

    results = []
    for s in subjects:
        calib = []
        for f in images(root / s / "calib"):
            feats, _ = cache[str(f.relative_to(root))]
            if len(feats) == 1:                  # calibration images must show the subject alone
                calib.append(feats[0])
        if len(calib) < 6:
            LOGGER.warning("subject %s: only %d usable calibration images, skipped", s, len(calib))
            continue
        present = [ImageInstances(*cache[str(f.relative_to(root))], target_box=gt.get((f.name, s)))
                   for f in images(root / "all_present")]
        absent = [ImageInstances(*cache[str(f.relative_to(root))]) for f in images(root / s / "without")]
        r = evaluate_reid_subject(s, np.stack(calib), present, absent, cfg.calibration, seed=args.seed)
        results.append(r)
        print(format_report(r.cm, REID_LABELS, f"\n== subject {s}  lambda_d={r.model.threshold:.3f} "
                                               f"(wrong-person selections: {r.wrong_person})"))
    if not results:
        LOGGER.error("no subject evaluated")
        return 1
    mean_cm = np.mean([r.cm for r in results], axis=0)
    report = format_report(mean_cm, REID_LABELS, f"\n== MEAN over {len(results)} subjects (Fig. 6b / Table I)")
    print(report)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    json.dump({r.subject: {"cm": r.cm.tolist(), "lambda_d": r.model.threshold, "wrong_person": r.wrong_person}
               for r in results}, open(out / "per_subject.json", "w"), indent=2)
    try:  # Fig. 2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r = results[0]
        n = min(200, len(r.target_distances), len(r.distractor_distances)) or len(r.target_distances)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(r.target_distances[:n], "g", lw=1, label="Calibrated person distances")
        ax.plot(r.distractor_distances[:n], "r", lw=1, label="Distractor person distances")
        ax.axhline(r.model.mean_distance, color="c", lw=1, label=r"Calibrated mean $\mu_d$")
        ax.axhline(r.model.threshold, color="m", lw=1, label=r"Threshold $\lambda_d=\mu_d+2\sigma_d$")
        ax.set_xlabel("# Samples")
        ax.set_ylabel(r"Feature distance $d_{\mu,\sigma}$")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"fig2_{r.subject}.png", dpi=150)
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
