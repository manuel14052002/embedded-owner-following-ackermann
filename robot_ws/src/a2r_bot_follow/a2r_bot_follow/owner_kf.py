"""owner_kf.py - constant-velocity Kalman track of the owner in the odom frame (a2r_v12.1)

Pure Python + numpy, no ROS: unit-testable (test_owner_kf.py).

State x = [px, py, vx, vy] in odom.  Process model: constant velocity, white
acceleration noise (q_acc, m/s^2).  Measurement: position only (sigma_meas).

States reported by status(now):
  TRACKING    last accepted measurement younger than fresh_s
  PREDICTING  no measurement, but younger than fresh_s + predict_s AND the
              predicted position sigma is below max_sigma_m
  LOST        otherwise (or never initialised)

Data association: a measurement is accepted only if its Mahalanobis distance
to the predicted position is inside the chi-square gate (2 dof).  The gate
grows by itself because the covariance grows while nothing is measured - that
is the "expanding gate" during an occlusion.

Re-initialisation (a different place than predicted) needs CONFIRMATION:
reinit_hits consistent out-of-gate measurements spread over >= reinit_min_s,
and only when the track has had no accepted measurement for >= reinit_quiet_s.
A single chair-leg pair or a passer-by can therefore never replace the owner.

Out-of-order / too-old measurements are dropped.
"""
import math
import numpy as np

TRACKING, PREDICTING, LOST = "TRACKING", "PREDICTING", "LOST"


class OwnerKF:
    def __init__(self, q_acc=0.6, sigma_meas=0.08, gate_chi2=9.21, fresh_s=0.8,
                 predict_s=2.0, max_sigma_m=1.0, max_speed=2.0,
                 reinit_hits=3, reinit_min_s=0.25, reinit_quiet_s=0.5, reinit_spread_m=0.45,
                 init_vel_sigma=1.0, gate_max_m=0.7, reacq_confirm=2):
        self.q, self.r = q_acc, sigma_meas
        self.gate, self.fresh_s, self.predict_s = gate_chi2, fresh_s, predict_s
        self.max_sigma, self.vmax = max_sigma_m, max_speed
        self.reinit_hits, self.reinit_min_s = reinit_hits, reinit_min_s
        self.reinit_quiet_s, self.reinit_spread = reinit_quiet_s, reinit_spread_m
        self.init_vs = init_vel_sigma
        self.gate_max, self.reacq_confirm = gate_max_m, reacq_confirm
        self.pend = []           # in-gate hits after a gap, waiting for confirmation
        self.x = None            # state at time self.t
        self.P = None
        self.t = None            # time of the state (measurement time)
        self.t_meas = -math.inf  # time of the last ACCEPTED measurement
        self.cand = []           # out-of-gate candidates (t, xy)
        self.stats = {"accepted": 0, "gated": 0, "reinit": 0, "stale": 0}

    # ── model ────────────────────────────────────────────────────────────
    def _FQ(self, dt):
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        q = self.q ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        Q = q * np.array([[dt4 / 4, 0, dt3 / 2, 0], [0, dt4 / 4, 0, dt3 / 2],
                          [dt3 / 2, 0, dt2, 0], [0, dt3 / 2, 0, dt2]])
        return F, Q

    def _predicted(self, t):
        """(x, P) propagated to time t (no side effects)."""
        dt = max(0.0, t - self.t)
        F, Q = self._FQ(dt)
        return F @ self.x, F @ self.P @ F.T + Q

    def init(self, t, xy):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([self.r ** 2, self.r ** 2, self.init_vs ** 2, self.init_vs ** 2])
        self.t = self.t_meas = t
        self.cand = []

    # ── measurement ──────────────────────────────────────────────────────
    def update(self, t, xy):
        """Offer one measurement (odom xy, measured at time t).
        Returns 'init' | 'accepted' | 'pending' | 'gated' | 'reinit' | 'stale'."""
        xy = np.asarray(xy, dtype=float)
        if self.x is None:
            self.init(t, xy)
            self.stats["accepted"] += 1
            return "init"
        if t < self.t - 1e-3:                                  # out of order
            self.stats["stale"] += 1
            return "stale"
        xp, Pp = self._predicted(t)
        H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0
        S = H @ Pp @ H.T + np.eye(2) * self.r ** 2
        y = xy - xp[:2]
        d2 = float(y @ np.linalg.solve(S, y))
        # gate: statistical (chi2 on the innovation) AND physical (never wider than gate_max_m,
        # or a long prediction would swallow any chair/person within a few sigma)
        in_gate = d2 <= self.gate and float(np.hypot(*y)) <= self.gate_max
        if in_gate and t - self.t_meas > self.fresh_s:
            # re-acquisition after a gap: reacq_confirm consistent hits before the track moves
            self.pend = [p for p in self.pend if t - p[0] <= 0.5 and np.hypot(*(xy - p[1])) <= 0.35]
            self.pend.append((t, xy))
            if len(self.pend) < self.reacq_confirm:
                self.stats["gated"] += 1
                return "pending"
        if in_gate:
            self.pend = []
            K = Pp @ H.T @ np.linalg.inv(S)
            self.x = xp + K @ y
            self.P = (np.eye(4) - K @ H) @ Pp
            sp = math.hypot(self.x[2], self.x[3])
            if sp > self.vmax:
                self.x[2:] *= self.vmax / sp
            self.t = self.t_meas = t
            self.cand = []
            self.stats["accepted"] += 1
            return "accepted"
        # outside the gate: candidate for a CONFIRMED re-initialisation only
        self.cand = [c for c in self.cand if t - c[0] <= 1.0 and
                     np.hypot(*(xy - c[1])) <= self.reinit_spread + 1.5 * (t - c[0])]
        self.cand.append((t, xy))
        # only a LOST track may be replaced - while it is TRACKING/PREDICTING a static chair
        # or a passer-by repeating outside the gate must never take over (hijack guard)
        quiet = (t - self.t_meas >= self.reinit_quiet_s) and self.status(t)[0] == LOST
        if (quiet and len(self.cand) >= self.reinit_hits
                and self.cand[-1][0] - self.cand[0][0] >= self.reinit_min_s):
            first, last = self.cand[0], self.cand[-1]
            self.init(t, xy)
            dt = last[0] - first[0]
            if dt > 1e-3:
                v = (last[1] - first[1]) / dt
                sp = float(np.hypot(*v))
                self.x[2:] = v * (min(sp, self.vmax) / sp if sp > 1e-6 else 0.0)
            self.stats["reinit"] += 1
            return "reinit"
        self.stats["gated"] += 1
        return "gated"

    # ── output ───────────────────────────────────────────────────────────
    def status(self, now):
        """(state, xy_now, vel, sigma_m, age_s) - xy predicted to `now`."""
        if self.x is None:
            return LOST, None, None, math.inf, math.inf
        xp, Pp = self._predicted(now)
        sigma = math.sqrt(max(Pp[0, 0], Pp[1, 1]))
        age = now - self.t_meas
        if age <= self.fresh_s:
            st = TRACKING
        elif age <= self.fresh_s + self.predict_s and sigma <= self.max_sigma:
            st = PREDICTING
        else:
            st = LOST
        return st, (float(xp[0]), float(xp[1])), (float(xp[2]), float(xp[3])), sigma, age

    def reset(self):
        self.__init__(self.q, self.r, self.gate, self.fresh_s, self.predict_s, self.max_sigma,
                      self.vmax, self.reinit_hits, self.reinit_min_s, self.reinit_quiet_s,
                      self.reinit_spread, self.init_vs, self.gate_max, self.reacq_confirm)
