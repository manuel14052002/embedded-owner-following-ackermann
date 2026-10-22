"""Decision making (paper Sect. V, Fig. 5).

States
    STEADY  robot stopped, waiting for the target position (and optionally a
            Follow gesture, ``steady_requires_follow_gesture``).
    FOLLOW  robot follows the target (navigation goals).
    SEARCH  no target position: rotate toward the last detected direction,
            for at most one complete turn.
    WAIT    stopped by the target's Wait gesture until a Follow gesture.

Events
    alpha   the target is re-identified in the camera FOV, i.e. a filtered
            target position is available (the KF keeps it for ``t_exp``).
    beta    the last target position is inside the safety circle.
    gamma   a gesture command (FOLLOW / WAIT) from the target.

Transitions (Fig. 5)
    STEADY --alpha--> FOLLOW
    FOLLOW --beta & !alpha--> STEADY
    FOLLOW --!alpha & !beta--> SEARCH
    SEARCH --alpha--> FOLLOW
    SEARCH --!alpha (full turn done)--> STEADY
    FOLLOW --gamma wait--> WAIT
    WAIT   --gamma follow--> FOLLOW
In FOLLOW with alpha & beta the robot stays in FOLLOW but deletes the last
goal and stops (Sect. IV). With ``wait_from_any_state`` the Wait gesture
also stops the robot from STEADY and SEARCH (a safety extension; Fig. 5
only draws it from FOLLOW).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from followme.config import StateMachineConfig
from followme.gestures import FOLLOW, WAIT
from followme.localization import Goal2D


class State(str, Enum):
    STEADY = "STEADY"
    FOLLOW = "FOLLOW"
    SEARCH = "SEARCH"
    WAIT = "WAIT"


class ActionKind(str, Enum):
    STOP = "STOP"              # cancel the goal and command zero velocity
    GOAL = "GOAL"              # (re)send a navigation goal
    ROTATE = "ROTATE"          # search rotation, direction = +1 left / -1 right


@dataclass
class Action:
    kind: ActionKind
    goal: Optional[Goal2D] = None
    direction: int = 0


@dataclass
class Inputs:
    alpha: bool
    beta: bool
    gamma: Optional[str] = None
    goal: Optional[Goal2D] = None           # current goal when alpha
    search_angle_done: float = 0.0          # |yaw| rotated since SEARCH started (rad)


class FollowMeStateMachine:
    def __init__(self, cfg: StateMachineConfig, search_max_angle: float = 2 * math.pi) -> None:
        self.cfg = cfg
        self.search_max_angle = search_max_angle
        self.state = State(cfg.initial_state)
        self.last_heading_sign = 1
        self.transitions: list[tuple[State, State, str]] = []

    def reset(self) -> None:
        self.state = State(self.cfg.initial_state)
        self.transitions.clear()

    def note_heading(self, theta: float) -> None:
        """Remember on which side the target was last seen (search direction)."""
        if abs(theta) > 1e-3:
            self.last_heading_sign = 1 if theta > 0 else -1

    def _go(self, new: State, why: str) -> None:
        if new != self.state:
            self.transitions.append((self.state, new, why))
            self.state = new

    def step(self, inp: Inputs) -> Action:
        s = self.state
        g = inp.gamma
        if g == WAIT and (s == State.FOLLOW or (self.cfg.wait_from_any_state and s != State.WAIT)):
            self._go(State.WAIT, "gamma wait")
            return Action(ActionKind.STOP)

        if s == State.WAIT:
            if g == FOLLOW:
                self._go(State.FOLLOW if inp.alpha else State.SEARCH, "gamma follow")
                return self.step(Inputs(inp.alpha, inp.beta, None, inp.goal, 0.0))
            return Action(ActionKind.STOP)

        if s == State.STEADY:
            ok = inp.alpha and (g == FOLLOW or not self.cfg.steady_requires_follow_gesture)
            if ok:
                self._go(State.FOLLOW, "alpha")
                return self._follow_action(inp)
            return Action(ActionKind.STOP)

        if s == State.SEARCH:
            if inp.alpha:
                self._go(State.FOLLOW, "alpha")
                return self._follow_action(inp)
            if inp.search_angle_done >= self.search_max_angle:
                self._go(State.STEADY, "!alpha, full turn")
                return Action(ActionKind.STOP)
            return Action(ActionKind.ROTATE, direction=self.last_heading_sign)

        # FOLLOW
        if not inp.alpha:
            if inp.beta:
                self._go(State.STEADY, "beta & !alpha")
                return Action(ActionKind.STOP)
            self._go(State.SEARCH, "!alpha & !beta")
            return Action(ActionKind.ROTATE, direction=self.last_heading_sign)
        return self._follow_action(inp)

    @staticmethod
    def _follow_action(inp: Inputs) -> Action:
        if inp.beta or inp.goal is None:
            return Action(ActionKind.STOP)      # target inside the safety circle
        return Action(ActionKind.GOAL, goal=inp.goal)
