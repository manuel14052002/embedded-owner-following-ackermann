"""Target calibration and identification (paper Sect. III-B) + extensions.

Metrics (``calibration.metric``):
    eq1      paper: d = sqrt(1/D sum((x_i - mu_i)/sigma_i)^2) <= lambda_d = mu_d + k sigma_d
    cosine   1 - cos(x, prototype) <= threshold learned the same way (mean + k std)
    both     must pass eq1 AND cosine
    gallery  multi-view: d = 1 - mean(top-k cosine similarities to the stored views of the
             target) <= threshold learned the same way; additionally the candidate must look
             more like the target than like any known stranger (negative bank), and confident
             views are added online (:class:`IdentityMemory`).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from followme.config import CalibrationConfig, IdentityConfig

LOGGER = logging.getLogger(__name__)


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, np.float64))
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _topk_mean_sim(x_unit: np.ndarray, gallery_unit: np.ndarray, k: int) -> np.ndarray:
    """Mean of the k highest cosine similarities of each row of x to the gallery."""
    sims = x_unit @ gallery_unit.T
    k = max(1, min(k, sims.shape[1]))
    return np.sort(sims, axis=1)[:, -k:].mean(axis=1)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class TargetModel:
    mu: np.ndarray
    sigma: np.ndarray
    threshold: float            # lambda_d (Eq. 1)
    mean_distance: float
    std_distance: float
    n_calibration: int
    n_threshold: int
    threshold_k: float = 2.0
    meta: dict = field(default_factory=dict)
    proto: Optional[np.ndarray] = None      # unit mean feature (cosine metric)
    cos_threshold: float = float("nan")
    cos_mean: float = float("nan")
    cos_std: float = float("nan")
    metric: str = "eq1"
    gallery: Optional[np.ndarray] = None    # (M, D) unit calibration views (gallery metric)
    gal_threshold: float = float("nan")
    gal_mean: float = float("nan")
    gal_std: float = float("nan")
    top_k: int = 3

    @property
    def dim(self) -> int:
        return int(self.mu.shape[0])

    @property
    def active_metric(self) -> str:
        if self.metric == "gallery" and self.gallery is not None:
            return "gallery"
        if self.metric in ("cosine", "both") and self.proto is not None:
            return self.metric
        return "eq1"

    @property
    def active_threshold(self) -> float:
        m = self.active_metric
        return {"eq1": self.threshold, "gallery": self.gal_threshold}.get(m, self.cos_threshold)

    def distances(self, features: np.ndarray) -> np.ndarray:
        """Eq. 1 for each row."""
        feats = np.atleast_2d(np.asarray(features, dtype=np.float64))
        if feats.shape[1] != self.dim:
            raise ValueError(f"feature dim {feats.shape[1]} != calibrated dim {self.dim}")
        z = (feats - self.mu) / self.sigma
        return np.sqrt(np.mean(z * z, axis=1))

    def cos_distances(self, features: np.ndarray) -> np.ndarray:
        return 1.0 - _unit(features) @ self.proto

    def gallery_distances(self, features: np.ndarray, extra: Optional[np.ndarray] = None) -> np.ndarray:
        g = self.gallery if extra is None or len(extra) == 0 else np.vstack([self.gallery, extra])
        return 1.0 - _topk_mean_sim(_unit(features), g, self.top_k)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if self.proto is not None:
            extra.update(proto=self.proto, cos=np.array([self.cos_threshold, self.cos_mean, self.cos_std]))
        if self.gallery is not None:
            extra.update(gallery=self.gallery.astype(np.float32),
                         gal=np.array([self.gal_threshold, self.gal_mean, self.gal_std, self.top_k]))
        np.savez(path, mu=self.mu, sigma=self.sigma,
                 scalars=np.array([self.threshold, self.mean_distance, self.std_distance,
                                   self.n_calibration, self.n_threshold, self.threshold_k]),
                 meta=np.array(json.dumps(self.meta)), metric=np.array(self.metric), **extra)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TargetModel":
        data = np.load(path, allow_pickle=False)
        s = data["scalars"]
        m = cls(mu=data["mu"], sigma=data["sigma"], threshold=float(s[0]), mean_distance=float(s[1]),
                std_distance=float(s[2]), n_calibration=int(s[3]), n_threshold=int(s[4]),
                threshold_k=float(s[5]), meta=json.loads(str(data["meta"])))
        if "proto" in data:
            m.proto = data["proto"]
            m.cos_threshold, m.cos_mean, m.cos_std = (float(v) for v in data["cos"])
        if "gallery" in data:
            m.gallery = data["gallery"].astype(np.float64)
            g = data["gal"]
            m.gal_threshold, m.gal_mean, m.gal_std, m.top_k = float(g[0]), float(g[1]), float(g[2]), int(g[3])
        if "metric" in data:
            m.metric = str(data["metric"])
        return m


def fit_target_model(features: np.ndarray, config: CalibrationConfig,
                     rng: Optional[np.random.Generator] = None, top_k: int = 3) -> TargetModel:
    feats = np.asarray(features, dtype=np.float64)
    if feats.ndim != 2 or len(feats) < 6:
        raise ValueError("need at least 6 calibration feature vectors")
    rng = rng if rng is not None else np.random.default_rng(config.shuffle_seed)
    order = rng.permutation(len(feats))
    n_c = int(round(len(feats) * config.calibration_fraction))
    n_c = min(max(n_c, 3), len(feats) - 3)
    calib, thresh = feats[order[:n_c]], feats[order[n_c:]]
    mu = calib.mean(axis=0)
    sigma = calib.std(axis=0)
    floor = max(config.sigma_floor_absolute, config.sigma_floor_relative * float(np.mean(sigma)))
    sigma = np.maximum(sigma, floor)
    model = TargetModel(mu=mu, sigma=sigma, threshold=0.0, mean_distance=0.0, std_distance=0.0,
                        n_calibration=len(calib), n_threshold=len(thresh), threshold_k=config.threshold_k,
                        metric=config.metric, top_k=top_k)
    k = config.threshold_k
    d = model.distances(thresh)
    model.mean_distance, model.std_distance = float(d.mean()), float(d.std())
    model.threshold = model.mean_distance + k * model.std_distance
    model.proto = _unit(_unit(calib).mean(axis=0))[0]
    dc = model.cos_distances(thresh)
    model.cos_mean, model.cos_std = float(dc.mean()), float(dc.std())
    model.cos_threshold = model.cos_mean + k * model.cos_std
    model.gallery = _unit(calib)
    dg = model.gallery_distances(thresh)
    model.gal_mean, model.gal_std = float(dg.mean()), float(dg.std())
    model.gal_threshold = model.gal_mean + k * model.gal_std
    LOGGER.info("Calibrated target: N_c=%d N_lambda=%d | eq1 mu_d=%.3f sigma_d=%.3f lambda=%.3f | "
                "cosine mean=%.3f thr=%.3f | gallery mean=%.3f std=%.3f thr=%.3f | metric=%s",
                len(calib), len(thresh), model.mean_distance, model.std_distance, model.threshold,
                model.cos_mean, model.cos_threshold, model.gal_mean, model.gal_std, model.gal_threshold,
                model.metric)
    return model


class IdentityMemory:
    """Online state for the gallery metric: new target views + known strangers (negatives)."""

    def __init__(self, cfg: IdentityConfig) -> None:
        self.cfg = cfg
        self.online: list[np.ndarray] = []
        self.negatives: list[np.ndarray] = []
        self._last_add = -np.inf
        self.counters = {"negatives_added": 0, "views_added": 0, "rejected_by_negatives": 0}

    def reset(self) -> None:
        self.__init__(self.cfg)

    @property
    def online_array(self) -> Optional[np.ndarray]:
        return np.stack(self.online) if self.online else None

    def negative_sims(self, feats_unit: np.ndarray, k: Optional[int] = None) -> Optional[np.ndarray]:
        """Top-k mean similarity to known strangers (same statistic as the gallery, so the
        comparison is fair: a max over hundreds of negatives would be inflated)."""
        if not self.negatives:
            return None
        return _topk_mean_sim(feats_unit, np.stack(self.negatives), k or self.cfg.top_k)

    def observe(self, model: TargetModel, feats: np.ndarray, boxes: Sequence[np.ndarray],
                valid: np.ndarray, target: int, d_target: float, now: float) -> None:
        """Called after the target was accepted this frame."""
        c = self.cfg
        u = _unit(feats)
        thr = model.gal_threshold
        tbox = boxes[target]
        if d_target <= c.negative_confident_factor * thr:
            for i in range(len(boxes)):
                if i != target and valid[i] and _iou(boxes[i], tbox) < c.negative_max_iou:
                    self.negatives.append(u[i])
                    self.counters["negatives_added"] += 1
            del self.negatives[:-c.max_negatives]
        if not c.online_update or now - self._last_add < c.online_interval_s:
            return
        alone = sum(1 for i in range(len(boxes)) if valid[i]) == 1
        adaptive_ok = (c.online_from_tracking and alone
                       and d_target <= max(1.0, c.tracking_threshold_factor) * thr)
        if d_target > c.online_confident_factor * thr and not adaptive_ok:
            return
        if any(i != target and valid[i] and _iou(boxes[i], tbox) > c.online_max_iou for i in range(len(boxes))):
            return
        x = u[target]
        neg = self.negative_sims(x[None])
        if neg is not None and (1.0 - d_target) - neg[0] < c.online_min_negative_margin:
            return
        existing = model.gallery if not self.online else np.vstack([model.gallery, np.stack(self.online)])
        if float((existing @ x).max()) >= c.online_duplicate_sim:
            return
        self.online.append(x)
        del self.online[:-c.max_online]
        self._last_add = now
        self.counters["views_added"] += 1


@dataclass
class Identification:
    target_index: Optional[int]
    distances: np.ndarray       # distance of the active metric (inf = invalid / rejected)


def identify(model: TargetModel, features: np.ndarray, valid: Optional[np.ndarray] = None,
             memory: Optional[IdentityMemory] = None, negative_margin: float = 0.03) -> Identification:
    feats = np.atleast_2d(features)
    if len(feats) == 0 or feats.shape[1] == 0:
        return Identification(None, np.zeros(0))
    metric = model.active_metric
    if metric == "gallery":
        d = model.gallery_distances(feats, memory.online_array if memory is not None else None)
        ok = d <= model.gal_threshold
        if memory is not None:
            neg = memory.negative_sims(_unit(feats))
            if neg is not None:
                beaten = (1.0 - d) - neg < negative_margin
                memory.counters["rejected_by_negatives"] += int((ok & beaten).sum())
                ok &= ~beaten
    elif metric == "eq1":
        d = model.distances(feats)
        ok = d <= model.threshold
    else:
        d = model.cos_distances(feats)
        ok = d <= model.cos_threshold
        if metric == "both":
            ok &= model.distances(feats) <= model.threshold
    if valid is not None:
        v = np.asarray(valid, bool)
        d = np.where(v, d, np.inf)
        ok &= v
    cand = np.where(ok, d, np.inf)
    best = int(np.argmin(cand))
    return Identification(best if np.isfinite(cand[best]) else None, d)


class CalibrationCollector:
    """Collects calibration features (target alone), spaced over ``duration_s``."""

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config
        self.features: list[np.ndarray] = []
        self.distances_m: list[float] = []
        self._last_t = -np.inf
        self.rejected_multi = 0
        self.interval = config.min_interval_s
        if config.duration_s:
            self.interval = max(config.min_interval_s, config.duration_s / config.num_samples)

    @property
    def done(self) -> bool:
        return len(self.features) >= self.config.num_samples

    @property
    def progress(self) -> float:
        return min(1.0, len(self.features) / self.config.num_samples)

    def offer(self, features: np.ndarray, person_distances_m: Sequence[float],
              now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if self.done or len(features) == 0:
            return False
        if self.config.require_single_person and len(features) != 1:
            self.rejected_multi += 1
            return False
        if now - self._last_t < self.interval:
            return False
        self._last_t = now
        self.features.append(np.asarray(features[0], np.float64))
        self.distances_m.append(float(person_distances_m[0]) if len(person_distances_m) else float("nan"))
        return True

    def bins_covered(self) -> int:
        d = np.asarray([x for x in self.distances_m if np.isfinite(x)])
        if d.size == 0:
            return 0
        edges = [0.0] + list(self.config.distance_bins_m) + [np.inf]
        hist, _ = np.histogram(d, bins=edges)
        return int((hist >= max(3, self.config.num_samples // 20)).sum())

    def finish(self, rng: Optional[np.random.Generator] = None, top_k: int = 3) -> TargetModel:
        if len(self.features) < 6:
            raise RuntimeError("not enough calibration samples")
        covered = self.bins_covered()
        if covered < self.config.min_bins_covered:
            LOGGER.warning("Calibration covered only %d distance bin(s): walk nearer/farther", covered)
        model = fit_target_model(np.stack(self.features), self.config, rng, top_k)
        finite = [x for x in self.distances_m if np.isfinite(x)]
        model.meta.update({"distance_bins_covered": covered,
                           "distance_range_m": [min(finite), max(finite)] if finite else None,
                           "rejected_multi_person_frames": self.rejected_multi})
        return model
