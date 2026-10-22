"""Turns state-machine actions into navigation-stack calls (paper Sect. IV).

The paper drives the robot with the ROS navigation stack (goal = projected
target position, obstacle avoidance by the planner, other people are
obstacles in the costmap). :class:`NavigationBackend` abstracts that stack;
``followme_ros`` implements it for ROS 2 Nav2 (NavigateToPose + cmd_vel).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional

from followme.config import NavigationConfig
from followme.localization import Goal2D
from followme.state_machine import Action, ActionKind

LOGGER = logging.getLogger(__name__)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class NavigationBackend(ABC):
    @abstractmethod
    def send_goal(self, goal: Goal2D) -> None: ...

    @abstractmethod
    def cancel_goal(self) -> None: ...

    @abstractmethod
    def set_velocity(self, linear: float, angular: float) -> None: ...


class RecordingNavigation(NavigationBackend):
    """No robot: records calls (offline app, tests)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def send_goal(self, goal: Goal2D) -> None:
        self.calls.append(("goal", goal))

    def cancel_goal(self) -> None:
        self.calls.append(("cancel",))

    def set_velocity(self, linear: float, angular: float) -> None:
        self.calls.append(("vel", linear, angular))


class NavigationExecutor:
    def __init__(self, cfg: NavigationConfig, backend: NavigationBackend) -> None:
        self.cfg = cfg
        self.backend = backend
        self.active_goal: Optional[Goal2D] = None
        self.last_goal_t = -math.inf
        self.last_kind: Optional[ActionKind] = None
        self.search_angle = 0.0
        self._last_t: Optional[float] = None
        self._last_yaw: Optional[float] = None
        self._search_phase_t = 0.0
        self._search_moving = True

    def _stop(self) -> None:
        if self.active_goal is not None:
            self.backend.cancel_goal()
            self.active_goal = None
        self.backend.set_velocity(0.0, 0.0)

    def _search_linear(self) -> float:
        """Search forward speed; a base that only steers while rolling needs min_steer_speed."""
        v = self.cfg.search_linear_speed
        if self.cfg.search_angular_speed > 0:
            v = max(v, getattr(self.cfg, "min_steer_speed_mps", 0.0))
        return v

    def _search_yaw_rate(self) -> float:
        """Real turn rate while searching. Car-like base: v * tan(max_steer) / L, which is far
        below the commanded angular speed at low speed."""
        w = abs(self.cfg.search_angular_speed)
        L = self.cfg.ackermann_wheelbase_m
        if L > 0:
            w = min(w, abs(self._search_linear()) * math.tan(math.radians(self.cfg.ackermann_max_steer_deg)) / L)
        return w

    def execute(self, action: Action, now: float, robot_yaw: Optional[float] = None) -> None:
        kind = action.kind
        if kind != ActionKind.ROTATE:
            self.search_angle = 0.0
            self._last_yaw = None
        if kind == ActionKind.STOP:
            if self.last_kind != ActionKind.STOP or self.active_goal is not None:
                self._stop()
        elif kind == ActionKind.GOAL and action.goal is not None:
            g = action.goal
            moved = (math.inf if self.active_goal is None
                     else math.hypot(g.x - self.active_goal.x, g.y - self.active_goal.y))
            continuous = getattr(self.backend, "continuous", False)   # cmd_vel control: every frame
            if (continuous or moved >= self.cfg.goal_update_distance_m
                    or now - self.last_goal_t >= self.cfg.goal_update_period_s):
                self.backend.send_goal(g)
                self.active_goal, self.last_goal_t = g, now
        elif kind == ActionKind.ROTATE:
            if self.last_kind != ActionKind.ROTATE and self.active_goal is not None:
                self.backend.cancel_goal()
                self.active_goal = None
            # accumulate the rotated angle: odometry if available, else w * dt
            if robot_yaw is not None:
                if self._last_yaw is not None:
                    self.search_angle += abs(wrap_angle(robot_yaw - self._last_yaw))
                self._last_yaw = robot_yaw
            elif self.last_kind == ActionKind.ROTATE and self._last_t is not None and self._search_moving:
                self.search_angle += self._search_yaw_rate() * max(0.0, now - self._last_t)
            if self.last_kind != ActionKind.ROTATE:
                self._search_phase_t, self._search_moving = now, True
            move, pause = self.cfg.search_move_s, self.cfg.search_pause_s
            if move > 0 and pause > 0:          # stop-and-look pattern
                limit = move if self._search_moving else pause
                if now - self._search_phase_t >= limit:
                    self._search_moving, self._search_phase_t = not self._search_moving, now
            if self._search_moving:
                self.backend.set_velocity(self._search_linear(),
                                          action.direction * abs(self.cfg.search_angular_speed))
            else:
                self.backend.set_velocity(0.0, 0.0)
        self.last_kind = kind
        self._last_t = now
