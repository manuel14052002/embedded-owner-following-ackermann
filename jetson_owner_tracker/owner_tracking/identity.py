"""Owner appearance model: fixed anchor + controlled adaptive gallery,
plus a bank of *distractor* embeddings (other people seen near the owner)."""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

from owner_tracking.config import IdentityConfig
from owner_tracking.reid import PersonReID

LOGGER = logging.getLogger(__name__)


class IdentityManager:
    """Maintains the owner's identity representation.

    * The **anchor** is the embedding captured at the operator's click. It is
      never replaced, so the model cannot drift arbitrarily far from the person
      the operator actually selected.
    * The **gallery** holds up to ``gallery_size`` additional embeddings,
      admitted only under strict conditions (see :meth:`maybe_update`), to
      cover viewpoint, pose, lighting and distance changes.
    * The **distractor bank** stores embeddings of other people observed while
      the owner was confidently tracked. A candidate that resembles a known
      distractor more than the owner is rejected.
    """

    def __init__(self, config: IdentityConfig, reid: Optional[PersonReID] = None) -> None:
        self.config = config
        self.reid = reid
        self.reset()

    def reset(self) -> None:
        self.anchor: Optional[np.ndarray] = None
        self.gallery: deque[np.ndarray] = deque(maxlen=self.config.gallery_size)
        self.distractors: deque[np.ndarray] = deque(maxlen=self.config.distractor_bank_size)
        self._last_update_frame = -10 ** 9
        self.updates = 0

    @property
    def initialized(self) -> bool:
        return self.anchor is not None

    # --- spec-level convenience API -------------------------------------
    def initialize_owner(self, frame: Optional[np.ndarray] = None, bbox: Optional[np.ndarray] = None,
                         embedding: Optional[np.ndarray] = None) -> np.ndarray:
        """Create the anchor from an embedding or from ``frame`` + ``bbox``."""
        if embedding is None:
            if self.reid is None or frame is None or bbox is None:
                raise ValueError("Need an embedding, or frame + bbox with a Re-ID model")
            embedding = self.reid.extract_embedding(frame, bbox)
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        if not np.any(embedding):
            raise ValueError("Cannot initialise owner from an invalid (zero) embedding")
        self.reset()
        self.anchor = embedding / np.linalg.norm(embedding)
        return self.anchor

    def compare(self, frame: np.ndarray, bbox: np.ndarray) -> float:
        """Identity similarity of one box to the owner model."""
        if self.reid is None:
            raise RuntimeError("compare() requires a Re-ID model")
        return float(self.similarity(self.reid.extract_embedding(frame, bbox)[None])[0])

    # --- batch API used by the tracker ----------------------------------
    def similarity(self, embeddings: Optional[np.ndarray]) -> np.ndarray:
        """Owner similarity per row: anchor blended with top-k gallery mean."""
        if embeddings is None or len(embeddings) == 0 or self.anchor is None:
            return np.zeros(0 if embeddings is None else len(embeddings), dtype=np.float32)
        anchor_sim = embeddings @ self.anchor
        if not self.gallery:
            return anchor_sim.astype(np.float32)
        gallery = np.stack(self.gallery)
        sims = embeddings @ gallery.T
        k = min(self.config.gallery_top_k, sims.shape[1])
        top_k = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
        w = self.config.anchor_weight
        return (w * anchor_sim + (1.0 - w) * top_k).astype(np.float32)

    def distractor_similarity(self, embeddings: Optional[np.ndarray]) -> np.ndarray:
        """Max similarity of each row to any stored distractor (0 if none)."""
        if embeddings is None or len(embeddings) == 0:
            return np.zeros(0 if embeddings is None else len(embeddings), dtype=np.float32)
        if not self.distractors:
            return np.zeros(len(embeddings), dtype=np.float32)
        return (embeddings @ np.stack(self.distractors).T).max(axis=1).astype(np.float32)

    def maybe_update(self, embedding: np.ndarray, *, confidence: float, similarity: float,
                     margin: float, isolated: bool, stable_frames: int, frame_index: int) -> bool:
        """Admit ``embedding`` to the gallery only if every safeguard passes.

        Safeguards: high overall confidence, a clear margin over competitors,
        no other person overlapping the crop, enough stable frames since the
        last (re)acquisition, a minimum similarity (no drift towards a
        different-looking person), rate limiting, and de-duplication.
        """
        c = self.config
        if self.anchor is None or not np.any(embedding):
            return False
        if (confidence < c.gallery_update_confidence or similarity < c.gallery_update_min_similarity
                or margin < c.gallery_update_min_margin or not isolated
                or stable_frames < c.gallery_min_stable_frames
                or frame_index - self._last_update_frame < c.gallery_update_interval):
            return False
        references = [self.anchor, *self.gallery]
        if max(float(embedding @ r) for r in references) >= c.gallery_dedup_similarity:
            return False  # redundant view; keeps the gallery diverse
        self.gallery.append(np.asarray(embedding, dtype=np.float32).copy())
        self._last_update_frame = frame_index
        self.updates += 1
        return True

    def add_distractors(self, embeddings: np.ndarray) -> int:
        """Store embeddings of other people as hard negatives.

        People whose appearance is nearly indistinguishable from the owner are
        *not* stored: they carry no discriminative information and would make
        the distractor rule reject the owner. Such look-alikes are handled by
        the margin rule and by motion evidence instead. Returns the number of
        embeddings stored.
        """
        added = 0
        for emb in embeddings:
            if not np.any(emb):
                continue
            if self.anchor is not None:
                owner_sim = float(self.similarity(emb[None])[0])
                if owner_sim >= self.config.distractor_max_owner_similarity:
                    continue
            self.distractors.append(np.asarray(emb, dtype=np.float32).copy())
            added += 1
        return added
