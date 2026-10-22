#!/usr/bin/env python3
"""
apply_learning_patch.py
-----------------------
Reads  multicue_owner.py   (never modified)
Writes multicue_owner.py.new

Every edit is an insertion at an anchor string that must occur EXACTLY once.
If any anchor is missing or ambiguous the script aborts and writes nothing.
All original lines and comments are preserved.

Usage:  python3 apply_learning_patch.py [path/to/multicue_owner.py]
"""
import ast
import sys
from pathlib import Path

MARK = "# [A2RBOT-LEARN]"

EDITS = [
    # 1 ── imports + availability flag ───────────────────────────────────────
    ("insert_after", "    HAS_CV2 = False\n", f'''
{MARK} opt-in learning extension. Defaults reproduce legacy behaviour.
import os as _os
try:
    try:
        from .a2rbot_learning_patch import (EnrollmentPhase, ContinuousLearner,
            compute_leg_profile, leg_similarity, LEG_EMA_KEEP, LEARN_GALLERY_MAXLEN)
    except ImportError:
        from a2rbot_learning_patch import (EnrollmentPhase, ContinuousLearner,
            compute_leg_profile, leg_similarity, LEG_EMA_KEEP, LEARN_GALLERY_MAXLEN)
    _LEARN_AVAILABLE = True
except Exception as _e:        # missing file must never stop the robot
    print(f"[LEARN] extension unavailable ({{_e}}) - legacy mode")
    _LEARN_AVAILABLE = False
'''),
    # 2 ── profile fields ───────────────────────────────────────────────────
    ("insert_after", "        self.enrolled = False\n", f'''        self.anchor_legs = None          {MARK} leg occupancy anchor
'''),
    # 3 ── constructor: modes (env vars or kwargs) ───────────────────────────
    ("replace", "    def __init__(self, reid_fn=None, face_detector=None, face_embed_fn=None):\n",
     "    def __init__(self, reid_fn=None, face_detector=None, face_embed_fn=None,\n"
     f"                 learning_mode=None, leg_mode=None):   {MARK}\n"),
    ("insert_after", "        self._body_ratio_ema = None   # smoothed aspect ratio for body3d cue\n", f'''        {MARK} modes: learning legacy|extended, legs off|shadow|on
        _lm = learning_mode or _os.environ.get("A2RBOT_LEARNING", "legacy")
        _gm = leg_mode or _os.environ.get("A2RBOT_LEG_CUE", "off")
        self._extended = _LEARN_AVAILABLE and _lm == "extended"
        self._leg_mode = _gm if (_LEARN_AVAILABLE and _gm in ("shadow", "on")) else "off"
        self._leg_weight = float(_os.environ.get("A2RBOT_LEG_WEIGHT", "0.08"))
        self._enrollment = EnrollmentPhase() if self._extended else None
        self._learner = ContinuousLearner() if self._extended else None
        self._leg_buf = []
        self.last_legs = (0.0, 0.0)      # always available for CSV logging
        if self._extended:
            self.profile.reid_gallery = deque(maxlen=LEARN_GALLERY_MAXLEN)
            self.profile.color_gallery = deque(maxlen=LEARN_GALLERY_MAXLEN)
        if self._extended or self._leg_mode != "off":
            print(f"[LEARN] learning={{_lm}} legs={{self._leg_mode}}")
'''),
    # 4 ── enroll(): optional depth_frame, extended path ─────────────────────
    ("replace", "    def enroll(self, frame, bbox, distance_m: Optional[float]):\n",
     f"    def enroll(self, frame, bbox, distance_m: Optional[float], depth_frame=None):  {MARK}\n"),
    ("insert_after", '        """Capture all cues for the owner. Call over ~20 frames, then finalize."""\n', f'''        {MARK}
        if self._leg_mode != "off" and depth_frame is not None:
            _lp, _ = compute_leg_profile(depth_frame, bbox, distance_m)
            if _lp is not None:
                self._leg_buf.append(_lp)
        if self._extended:
            return self._enroll_extended(frame, bbox, distance_m)
'''),
    # 5 ── finalize_enrollment(): extended path + leg anchor ─────────────────
    ("insert_after", "    def finalize_enrollment(self) -> bool:\n", f'''        {MARK}
        if self._leg_buf:
            self.profile.anchor_legs = np.median(np.stack(self._leg_buf), axis=0)
            print(f"[LEGS] anchor from {{len(self._leg_buf)}} frame(s)")
            self._leg_buf = []
        if self._extended:
            return self._finalize_extended()
'''),
    # 6 ── score(): optional depth_frame; legs computed (shadow) or voting (on)
    ("replace", "              reid_emb=None, depth_roi=None, other_dets=None) -> Dict:\n",
     f"              reid_emb=None, depth_roi=None, other_dets=None,\n"
     f"              depth_frame=None) -> Dict:   {MARK}\n"),
    ("insert_after", "            # (written as 0.0) so existing analysis scripts still parse.\n        }\n", f'''        {MARK} leg cue: computed in shadow/on, votes only in "on"
        self.last_legs = (0.0, 0.0)
        if self._leg_mode != "off" and self.profile.anchor_legs is not None:
            _lp, _n = compute_leg_profile(depth_frame, bbox, distance_m)
            self.last_legs = leg_similarity(_lp, self.profile.anchor_legs, _n)
            if self._leg_mode == "on":
                cues["legs"] = self.last_legs
'''),
    ("replace", "            w = self.BASE_WEIGHTS[name]\n",
     f"            w = self.BASE_WEIGHTS[name] if name in self.BASE_WEIGHTS else self._leg_weight  {MARK}\n"),
    # 7 ── new methods, placed before set_owned ──────────────────────────────
    ("insert_before", "    def set_owned(self, owned: bool):\n", f'''    {MARK} ── extension API (inactive in legacy mode) ─────────────────────
    def start_enrollment(self):
        """Call when the enrollment gesture is confirmed."""
        self._leg_buf = []
        if self._extended:
            self._enrollment.start()

    @property
    def enrollment_ready(self) -> bool:
        """Legacy: always True (caller's frame count decides, as before)."""
        return self._enrollment.ready if self._extended else True

    @property
    def enrollment_progress(self) -> float:
        return self._enrollment.progress if self._extended else 1.0

    def learn_status(self) -> dict:
        if not self._extended:
            return dict(mode="legacy")
        d = self._learner.status()
        d.update(mode="extended", enroll_s=round(self._enrollment.elapsed, 1),
                 enroll_frames=self._enrollment.n_accepted)
        return d

    def _enroll_extended(self, frame, bbox, distance_m):
        emb = self._reid_fn(frame, bbox) if self._reid_fn else None
        ok, q = self._enrollment.feed(emb, color_hist(frame, bbox), bbox,
                                      distance_m, frame.shape[0])
        if self._face_detect and self._face_embed:
            fc = self._face_detect(frame, bbox)
            fe = self._face_embed(fc) if fc is not None else None
            if fe is not None:
                if not hasattr(self, "_face_enroll_buf"):
                    self._face_enroll_buf = []
                self._face_enroll_buf.append(fe.ravel())
        bw = (bbox[2] - bbox[0]) / max(1, frame.shape[1])
        bh = (bbox[3] - bbox[1]) / max(1, frame.shape[0])
        a = bw / max(0.01, bh)
        r = self.profile.body_width_ratio
        self.profile.body_width_ratio = a if r is None else 0.7 * r + 0.3 * a
        return ok, q

    def _finalize_extended(self) -> bool:
        res = self._enrollment.finalize()
        if res is None:
            return False                      # keep collecting; caller retries
        p = self.profile
        p.reid, p.color = res["anchor_emb"], res["anchor_hist"]
        p.reid_gallery.clear(); p.reid_gallery.extend(res["seed_embs"])
        p.color_gallery.clear(); p.color_gallery.extend(res["seed_hists"])
        p.height_ratio = res["height_ratio"]
        if res["depth_m"]:
            p.enroll_distance, p.enroll_bbox_h = res["depth_m"], res["bbox_h_px"]
        buf = getattr(self, "_face_enroll_buf", [])
        if buf:
            avg = np.mean(np.stack(buf), axis=0)
            p.face = avg / (np.linalg.norm(avg) + 1e-6)
            print(f"[FACE] enrolled from {{len(buf)}} face capture(s)")
        else:
            p.face = None
            print("[FACE] no face captured during enrollment — cue will abstain")
        self._face_enroll_buf = []
        p.enrolled = True
        print(f"[ENROLL] done: {{res['n_frames']}} frames in {{res['duration_s']:.0f}} s")
        return True

'''),
    # 8 ── update_galleries(): learner gate + leg anchor adaptation ──────────
    ("replace", "    def update_galleries(self, frame, bbox, result: Dict, reid_emb=None):\n",
     f"    def update_galleries(self, frame, bbox, result: Dict, reid_emb=None,\n"
     f"                         depth_frame=None):   {MARK}\n"),
    ("insert_after", '        if result["score"] < 0.82 or result["occluded"]:\n            return\n', f'''        {MARK} extended: count verified following time, rate-limit learning
        if self._extended:
            self._learner.observe()
            if not self._learner.should_update(result["score"]):
                return
            if self._leg_mode != "off" and self.profile.anchor_legs is not None:
                _lp, _ = compute_leg_profile(depth_frame, bbox)
                if _lp is not None:
                    self.profile.anchor_legs = (LEG_EMA_KEEP * self.profile.anchor_legs
                                                + (1 - LEG_EMA_KEEP) * _lp)
'''),
]


def main():
    src_path = Path(sys.argv[1] if len(sys.argv) > 1 else "multicue_owner.py")
    src = src_path.read_text()
    if MARK in src or "<<PATCH>>" in src:
        sys.exit(f"ABORT: {src_path} is already patched. Restore the original first:\n"
                 f"  cp {src_path}.bak {src_path}")
    out = src
    for i, (kind, anchor, text) in enumerate(EDITS, 1):
        n = out.count(anchor)
        if n != 1:
            sys.exit(f"ABORT: edit {i}: anchor found {n} times (need 1):\n{anchor!r}\n"
                     "Nothing was written. Send this message to Claude.")
        if kind == "insert_after":
            out = out.replace(anchor, anchor + text, 1)
        elif kind == "insert_before":
            out = out.replace(anchor, text + anchor, 1)
        else:
            out = out.replace(anchor, text, 1)
    ast.parse(out)                                   # syntax check
    # every original line must still be present (replaced signature lines excepted)
    replaced = {a for k, a, _ in EDITS if k == "replace"}
    orig_lines = [l for l in src.splitlines(True) if l not in replaced]
    missing = [l for l in orig_lines if l not in out]
    if missing:
        sys.exit(f"ABORT: {len(missing)} original lines lost — nothing written")
    dst = src_path.with_name(src_path.name + ".new")
    dst.write_text(out)
    print(f"OK: wrote {dst}  ({len(EDITS)} edits, {src.count(chr(10))} -> "
          f"{out.count(chr(10))} lines). Original untouched.")


if __name__ == "__main__":
    main()
