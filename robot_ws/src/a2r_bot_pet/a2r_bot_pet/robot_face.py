#!/usr/bin/env python3
"""
robot_face.py — a2r_bot_pet
===========================
The animated face, imported as a library.

This is the original robot_face.py, unchanged in every part that draws.  The
eye geometry, the blink curves, the gaze wander, the wobble, the drift, the
mouth shapes and the whole EXPRESSIONS table are exactly as they were, because
they were tuned by eye and there is no way to improve them from a terminal.

Three things were added, all additive:

  1. `run_standalone()` wraps the original `main()`, so running this file
     directly still behaves exactly as before — keyboard control, auto
     sequence, ESC to quit.  Nothing about the original workflow is lost.

  2. `FaceRenderer` packages up the state the original `main()` kept in local
     variables (the blinkers, the gaze, the wobble, the drift, the talker,
     the expression) so that a caller can own the loop instead.  That is what
     face_node.py needs: ROS wants to spin its own executor, and pygame wants
     to own the frame timing, and the only clean way to have both is for the
     face to expose `step(dt)` rather than a `while True`.

  3. `list_expressions()` so the state bridge can validate a mapping at
     startup rather than failing on the first unusual state the Jetson sends.

The auto sequence is still here and still works, but face_node disables it by
default: when the Jetson is driving the expression, an auto sequence fighting
it for control is just a bug that looks like a feature.
"""

"""
Cute robot face for an LCD screen (Raspberry Pi friendly).

Black background, two blue eyes with black pupils, a simple mouth.
Eyes blink naturally at randomized intervals and occasionally glance
in a random direction before returning to center.

The default look (breathing/blinking/glancing) is the "idle" expression.
Until you press an expression key for the first time, the face
automatically cycles through AUTO_SEQUENCE (idle/curious/happy/sad/
bored/sleep) on its own. Pressing an expression key morphs the face to a
new mood and turns off the automatic cycling for the rest of the run;
there's no dedicated "back to idle" key -- press I.

Controls (when running windowed on a PC for testing):
  ESC or close window -> quit
  F                    -> toggle fullscreen
  I                    -> idle (default)
  C                    -> curious
  H                    -> happy
  A                    -> angry
  S                    -> sad
  B                    -> bored (auto-falls asleep after BORED_TO_SLEEP_TIME)
  Z                    -> sleep (also auto-entered after being bored)
  T                    -> talking

On Raspberry Pi, set FULLSCREEN = True below (or press F) and hide
the mouse cursor via pygame.mouse.set_visible(False).
"""

import math
import random
import sys

import pygame

# ---------------------------------------------------------------------------
# Configuration -- every knob you're likely to want to tweak lives here.
# ---------------------------------------------------------------------------

# --- Display / window ---
SCREEN_W, SCREEN_H = 800, 480       # match your LCD's resolution
FPS = 90                            # animation runs on real elapsed time (dt),
                                     # so a higher cap just means smoother motion
FULLSCREEN = True                  # flip to True on the Raspberry Pi

# Supersampling factor: the face is drawn onto an internal canvas this many
# times larger than the screen, then smoothly downscaled. That's what removes
# the jagged/staircase edges on the curves and rounded corners. 1 = off
# (cheapest, some aliasing), 2 = clean on a Pi 4/5, 3+ for a beefier GPU.
RENDER_SCALE = 1.0

# --- What's shown on the face ---
SHOW_MOUTH = True                   # set False to hide the mouth entirely

# --- Overall size & color ---
# Scales the whole face (both eyes and the mouth) up or down as one unit.
# 1.0 = the base sizes defined below; e.g. 1.25 makes everything 25% bigger.
FACE_SCALE = 1.55

BG_COLOR = (0, 0, 0)
EYE_COLOR = (30, 140, 255)
PUPIL_COLOR = (0, 0, 0)
MOUTH_COLOR = (30, 140, 255)

# Base (FACE_SCALE == 1.0) sizes, in logical screen pixels.
_BASE_EYE_W, _BASE_EYE_H = 160, 200   # base (open) eye size, rounded rect
_BASE_EYE_RADIUS = 60                 # corner rounding
_BASE_EYE_GAP = 100                   # gap between the two eyes
_BASE_GAZE_OFFSET_X = 46              # how far the pupil wanders, in pixels
_BASE_GAZE_OFFSET_Y = 60
_BASE_MOUTH_W, _BASE_MOUTH_H = 110, 70
_BASE_MOUTH_THICKNESS = 10
_BASE_MOUTH_GAP = 0                   # vertical gap between eyes and mouth

# Everything below is derived from FACE_SCALE, so a single knob resizes the
# whole face consistently.
EYE_W = _BASE_EYE_W * FACE_SCALE
EYE_H = _BASE_EYE_H * FACE_SCALE
EYE_RADIUS = _BASE_EYE_RADIUS * FACE_SCALE
EYE_GAP = _BASE_EYE_GAP * FACE_SCALE
PUPIL_SCALE = 0.85                     # pupil size relative to the eye (unitless)
GAZE_MAX_OFFSET_X = _BASE_GAZE_OFFSET_X * FACE_SCALE
GAZE_MAX_OFFSET_Y = _BASE_GAZE_OFFSET_Y * FACE_SCALE
MOUTH_W = _BASE_MOUTH_W * FACE_SCALE
MOUTH_H = _BASE_MOUTH_H * FACE_SCALE
MOUTH_THICKNESS = _BASE_MOUTH_THICKNESS * FACE_SCALE
MOUTH_GAP = _BASE_MOUTH_GAP * FACE_SCALE

# Blink timing (seconds)
BLINK_MIN_INTERVAL = 2.0
BLINK_MAX_INTERVAL = 5.0
BLINK_CLOSE_TIME = 0.08
BLINK_OPEN_TIME = 0.10
BLINK_HOLD_TIME = 0.03

# Gaze (look-around) timing
GAZE_MIN_INTERVAL = 3.0
GAZE_MAX_INTERVAL = 7.0
GAZE_MOVE_TIME = 0.35
GAZE_HOLD_TIME = 0.9
GAZE_RETURN_TIME = 0.35
GAZE_DIRECTIONS = [
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
]

# Subtle eye size "breathing" so the face doesn't look perfectly static.
# Every so often one or both eyes ease to a slightly different size and
# back. Adjust WOBBLE_AMOUNT to change how strong the effect is.
EYE_WOBBLE_AMOUNT = 0.1          # max size change, e.g. 0.05 = +/-5%
EYE_WOBBLE_MIN_INTERVAL = 4.0
EYE_WOBBLE_MAX_INTERVAL = 9.0
EYE_WOBBLE_MOVE_TIME = 0.5
EYE_WOBBLE_HOLD_TIME = 1.0
EYE_WOBBLE_RETURN_TIME = 0.5
EYE_WOBBLE_EQUAL_CHANCE = 0.5     # both eyes change together most of the time

# Subtle whole-eyes drift: both eyes (together, keeping their spacing) ease
# a few pixels off-center -- up/down, left/right, or diagonally -- and back.
# Keep the *_MAX values small; this should read as a subtle "not a frozen
# picture" effect, not the face visibly moving around.
EYE_DRIFT_MAX_X = 20 #6               # max horizontal drift, in pixels
EYE_DRIFT_MAX_Y = 10 #4               # max vertical drift, in pixels
EYE_DRIFT_MIN_INTERVAL = 3.0
EYE_DRIFT_MAX_INTERVAL = 10.0
EYE_DRIFT_MOVE_TIME = 1.2
EYE_DRIFT_HOLD_TIME = 1.5
EYE_DRIFT_RETURN_TIME = 1.2

# --- Expressions ---
# How long (seconds) it takes to morph from one expression to another when
# you switch. Higher = a slower, more deliberate change of mood.
EXPRESSION_TRANSITION_TIME = 0.3

# "happy" expression extras: the eyes tilt outward (left eye counter-
# clockwise, right eye clockwise, mirrored) and gently bounce up and down.
HAPPY_EYE_ROTATE_DEG = 14     # how far each eye tilts, in degrees
HAPPY_BOUNCE_AMOUNT = 6       # how far the eyes bounce up/down, in pixels
HAPPY_BOUNCE_SPEED = 9.0      # bounce oscillation speed, in radians/sec

# "bored" auto-falls asleep after sitting bored this long (seconds).
BORED_TO_SLEEP_TIME = 7.0

# "talking" mouth animation: the mouth (a flat-line/rectangle "bar" shape)
# randomly opens and closes at a natural, slightly randomized pace to fake
# talking, like a mouth bar growing taller then flattening back out.
TALK_SPEED = 6.0          # average mouth movements per second -- adjust me
TALK_RANDOMNESS = 0.8     # 0..1, timing/size jitter so it doesn't look robotic
TALK_MOUTH_MIN = 0.0     # smallest ("closed"/flat) mouth curve value
TALK_MOUTH_MAX = 0.4      # largest ("wide open") mouth curve value

# "sleep" mouth animation: a small circle mouth slowly grows/shrinks like
# a slow breathing cycle.
SLEEP_BREATHE_SPEED = 1.2  # breathing oscillation speed, in radians/sec -- slow
SLEEP_MOUTH_MIN = 0.2      # smallest (exhaled) mouth circle curve value
SLEEP_MOUTH_MAX = 0.5      # largest (inhaled) mouth circle curve value

# Keyboard keys that trigger each expression while the window has focus.
# "idle" (the default, breathing/blinking/glancing face developed earlier)
# is what the face returns to -- there's no dedicated idle key, press any
# expression key to leave idle, idle is just what's active at startup.
EXPRESSION_KEYS = {
    pygame.K_i: "idle",
    pygame.K_c: "curious",
    pygame.K_h: "happy",
    pygame.K_a: "angry",
    pygame.K_s: "sad",
    pygame.K_b: "bored",
    pygame.K_z: "sleep",
    pygame.K_t: "talking",
}

# Automatic idle-mood cycling: until the user presses an expression key
# for the first time, the face automatically works through this list,
# holding each expression for its listed duration (seconds) before
# advancing to the next -- both the order and each duration are
# adjustable. A manual expression keypress turns this off for the rest of
# the run.
AUTO_SEQUENCE_ENABLED = True
AUTO_SEQUENCE = [
    ("idle", 15.0),
    ("curious", 4.0),
    ("happy", 8.0),
    ("sad", 6.0),
    ("bored", 4.5),   # prevent dozes off into "sleep" earlier, via BORED_TO_SLEEP_TIME
]

# Per-expression look. Each field blends smoothly (over
# EXPRESSION_TRANSITION_TIME) from whatever the current expression is:
#   eye_h        - eye height multiplier (< 1 = squint, > 1 = wide/alert)
#   brow_inner   - how much of the top-inner corner (nose side) is "cut"
#                  away by a dark eyelid/eyebrow wedge (0 = none, 0-0.5ish)
#   brow_outer   - same, but the top-outer corner (temple side)
#   pupil_mult   - pupil size multiplier relative to the normal pupil
#   mouth_curve  - >0 = smile (bigger = wider grin), <0 = frown, magnitude
#                  controls how deep the curve is
#   mouth_width  - mouth width multiplier
#   mouth_shape  - "curve" (smile/frown arc, default), "circle" (a round
#                  filled "o" mouth), "flat" (a straight horizontal line),
#                  or "bar" (a filled rectangle that grows/shrinks
#                  vertically, e.g. talking); not blended, switches
#                  immediately
#   eye_rotate   - degrees each eye tilts (left eye CCW, right eye CW,
#                  mirrored); 0 = no tilt
#   bounce       - 0..1 strength of the happy up/down eye bounce
#   wobble_mult  - 0..1 how much of the idle eye-size "breathing" wobble
#                  still applies (0 = suppressed, e.g. happy's bounce
#                  reads cleaner without the size changing too)
#   mouth_offset_y - extra vertical offset for the mouth, in pixels
#   gaze_interval_mult - multiplies how often the eyes glance sideways
#                  (>1 = more frequent, e.g. curious; 1 = normal)
#   eyes_closed  - 0..1, crossfades the eye from its normal look to a
#                  wide "U" closed-eye curve (1 = fully closed, e.g. sleep)
#   talk_mult    - 0..1 how much the "talking" mouth-cycling animation
#                  (see Talker) drives the mouth instead of mouth_curve
#   breathe_mult - 0..1 how much the slow "sleep" breathing mouth
#                  animation drives the mouth instead of mouth_curve
#   color        - eye/mouth color (the "mood" tint); pupils stay black
EXPRESSIONS = {
    "idle": dict(
        eye_h=1.0, brow_inner=0.0, brow_outer=0.0, pupil_mult=1.0,
        mouth_curve=1.0, mouth_width=1.0, mouth_shape="curve",
        eye_rotate=0.0, bounce=0.0, wobble_mult=1.0, mouth_offset_y=0.0,
        gaze_interval_mult=1.0, eyes_closed=0.0, talk_mult=0.0, breathe_mult=0.0,
        color=EYE_COLOR,
    ),
    "curious": dict(
        eye_h=1.15, brow_inner=0.0, brow_outer=0.0, pupil_mult=1.10,
        mouth_curve=0.4, mouth_width=0.6, mouth_shape="circle",
        eye_rotate=0.0, bounce=0.0, wobble_mult=1.0, mouth_offset_y=60.0,
        gaze_interval_mult=2.5, eyes_closed=0.0, talk_mult=0.0, breathe_mult=0.0,
        color=EYE_COLOR,
    ),
    "happy": dict(
        eye_h=0.75, brow_inner=0.0, brow_outer=0.0, pupil_mult=1.0,
        mouth_curve=1.8, mouth_width=1.3, mouth_shape="curve",
        eye_rotate=HAPPY_EYE_ROTATE_DEG, bounce=1.0, wobble_mult=0.0, mouth_offset_y=0.0,
        gaze_interval_mult=1.0, eyes_closed=0.0, talk_mult=0.0, breathe_mult=0.0,
        color=EYE_COLOR,
    ),
    "angry": dict(
        eye_h=0.55, brow_inner=0.4, brow_outer=0.05, pupil_mult=0.8,
        mouth_curve=-1.0, mouth_width=0.9, mouth_shape="curve",
        eye_rotate=0.0, bounce=0.0, wobble_mult=1.0, mouth_offset_y=0.0,
        gaze_interval_mult=1.0, eyes_closed=0.0, talk_mult=0.0, breathe_mult=0.0,
        color=(255, 70, 60),
    ),
    "sad": dict(
        eye_h=0.85, brow_inner=0.05, brow_outer=0.35, pupil_mult=0.9,
        mouth_curve=-0.7, mouth_width=0.75, mouth_shape="curve",
        eye_rotate=0.0, bounce=0.0, wobble_mult=1.0, mouth_offset_y=60.0,
        gaze_interval_mult=1.0, eyes_closed=0.0, talk_mult=0.0, breathe_mult=0.0,
        color=(130, 175, 225),
    ),
    "bored": dict(
        eye_h=0.5, brow_inner=0.0, brow_outer=0.1, pupil_mult=0.9,
        mouth_curve=0.15, mouth_width=0.7, mouth_shape="flat",
        eye_rotate=0.0, bounce=0.0, wobble_mult=0.5, mouth_offset_y=0.0,
        gaze_interval_mult=0.4, eyes_closed=0.0, talk_mult=0.0, breathe_mult=0.0,
        color=(80, 120, 170),
    ),
    "sleep": dict(
        eye_h=0.5, brow_inner=0.0, brow_outer=0.0, pupil_mult=1.0,
        mouth_curve=0.3, mouth_width=0.5, mouth_shape="circle",
        eye_rotate=0.0, bounce=0.0, wobble_mult=0.0, mouth_offset_y=0.0,
        gaze_interval_mult=1.0, eyes_closed=1.0, talk_mult=0.0, breathe_mult=1.0,
        color=(235, 240, 250),
    ),
    "talking": dict(
        eye_h=1.05, brow_inner=0.0, brow_outer=0.0, pupil_mult=1.0,
        mouth_curve=0.3, mouth_width=0.6, mouth_shape="bar",
        eye_rotate=0.0, bounce=0.0, wobble_mult=0.0, mouth_offset_y=60.0,
        gaze_interval_mult=0.4, eyes_closed=0.0, talk_mult=1.0, breathe_mult=0.0,
        color=EYE_COLOR,
    ),
}


def ease_in_out(t: float) -> float:
    """Smooth 0..1 easing curve for natural-looking motion."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


class Blinker:
    """Drives the eyelid open/closed animation state (0 = open, 1 = closed)."""

    def __init__(self):
        self.state = "open"
        self.timer = random.uniform(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL)
        self.progress = 0.0
        self.amount = 0.0  # 0 = fully open, 1 = fully closed

    def trigger(self):
        if self.state == "open":
            self.state = "closing"
            self.progress = 0.0

    def update(self, dt):
        if self.state == "open":
            self.timer -= dt
            if self.timer <= 0:
                self.trigger()
        elif self.state == "closing":
            self.progress += dt / BLINK_CLOSE_TIME
            self.amount = min(1.0, self.progress)
            if self.progress >= 1.0:
                self.state = "hold"
                self.progress = 0.0
        elif self.state == "hold":
            self.progress += dt
            if self.progress >= BLINK_HOLD_TIME:
                self.state = "opening"
                self.progress = 0.0
        elif self.state == "opening":
            self.progress += dt / BLINK_OPEN_TIME
            self.amount = max(0.0, 1.0 - self.progress)
            if self.progress >= 1.0:
                self.state = "open"
                self.amount = 0.0
                self.timer = random.uniform(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL)


class Gaze:
    """Drives the pupils looking toward a random offset and back to center."""

    def __init__(self):
        self.state = "center"
        self.timer = random.uniform(GAZE_MIN_INTERVAL, GAZE_MAX_INTERVAL)
        self.progress = 0.0
        self.start = (0.0, 0.0)
        self.target = (0.0, 0.0)
        self.offset = (0.0, 0.0)

    def _random_target(self):
        dx, dy = random.choice(GAZE_DIRECTIONS)
        strength = random.uniform(0.7, 1.0)
        return (dx * GAZE_MAX_OFFSET_X * strength, dy * GAZE_MAX_OFFSET_Y * strength)

    def update(self, dt, interval_mult=1.0):
        """`interval_mult` > 1 makes glances happen more often (e.g. a
        curious expression), < 1 less often, without changing how a single
        glance moves/holds/returns.
        """
        if self.state == "center":
            self.timer -= dt * interval_mult
            if self.timer <= 0:
                self.start = self.offset
                self.target = self._random_target()
                self.state = "moving"
                self.progress = 0.0
        elif self.state == "moving":
            self.progress += dt / GAZE_MOVE_TIME
            t = ease_in_out(min(1.0, self.progress))
            self.offset = (
                self.start[0] + (self.target[0] - self.start[0]) * t,
                self.start[1] + (self.target[1] - self.start[1]) * t,
            )
            if self.progress >= 1.0:
                self.state = "hold"
                self.progress = 0.0
        elif self.state == "hold":
            self.progress += dt
            if self.progress >= GAZE_HOLD_TIME:
                self.state = "returning"
                self.progress = 0.0
                self.start = self.offset
        elif self.state == "returning":
            self.progress += dt / GAZE_RETURN_TIME
            t = ease_in_out(min(1.0, self.progress))
            self.offset = (
                self.start[0] + (0.0 - self.start[0]) * t,
                self.start[1] + (0.0 - self.start[1]) * t,
            )
            if self.progress >= 1.0:
                self.offset = (0.0, 0.0)
                self.state = "center"
                self.timer = random.uniform(GAZE_MIN_INTERVAL, GAZE_MAX_INTERVAL)


class EyeWobble:
    """Eases each eye's size toward a slightly different target and back,
    so the face reads as alive instead of a static image. Most of the time
    both eyes change together (equal); occasionally they move opposite each
    other -- one enlarges by exactly as much as the other shrinks.
    """

    def __init__(self):
        self.state = "wait"
        self.timer = random.uniform(EYE_WOBBLE_MIN_INTERVAL, EYE_WOBBLE_MAX_INTERVAL)
        self.progress = 0.0
        self.start = (1.0, 1.0)
        self.target = (1.0, 1.0)
        self.scale = (1.0, 1.0)  # (left, right) size multipliers

    def _random_target(self):
        delta = random.uniform(0.4, 1.0) * EYE_WOBBLE_AMOUNT
        sign = random.choice((-1, 1))
        if random.random() < EYE_WOBBLE_EQUAL_CHANCE:
            # Both eyes change the same way (both bigger or both smaller).
            return (1.0 + sign * delta, 1.0 + sign * delta)
        else:
            # Opposite: one eye enlarges exactly as much as the other shrinks.
            return (1.0 + sign * delta, 1.0 - sign * delta)

    def update(self, dt):
        if self.state == "wait":
            self.timer -= dt
            if self.timer <= 0:
                self.start = self.scale
                self.target = self._random_target()
                self.state = "moving"
                self.progress = 0.0
        elif self.state == "moving":
            self.progress += dt / EYE_WOBBLE_MOVE_TIME
            t = ease_in_out(min(1.0, self.progress))
            self.scale = (
                self.start[0] + (self.target[0] - self.start[0]) * t,
                self.start[1] + (self.target[1] - self.start[1]) * t,
            )
            if self.progress >= 1.0:
                self.state = "hold"
                self.progress = 0.0
        elif self.state == "hold":
            self.progress += dt
            if self.progress >= EYE_WOBBLE_HOLD_TIME:
                self.state = "returning"
                self.progress = 0.0
                self.start = self.scale
        elif self.state == "returning":
            self.progress += dt / EYE_WOBBLE_RETURN_TIME
            t = ease_in_out(min(1.0, self.progress))
            self.scale = (
                self.start[0] + (1.0 - self.start[0]) * t,
                self.start[1] + (1.0 - self.start[1]) * t,
            )
            if self.progress >= 1.0:
                self.scale = (1.0, 1.0)
                self.state = "wait"
                self.timer = random.uniform(EYE_WOBBLE_MIN_INTERVAL, EYE_WOBBLE_MAX_INTERVAL)


class EyeDrift:
    """Slowly eases the whole eye pair a few pixels off-center and back, in a
    random direction (horizontal, vertical, or diagonal), so the face isn't
    perfectly still between blinks/gazes.
    """

    def __init__(self):
        self.state = "wait"
        self.timer = random.uniform(EYE_DRIFT_MIN_INTERVAL, EYE_DRIFT_MAX_INTERVAL)
        self.progress = 0.0
        self.start = (0.0, 0.0)
        self.target = (0.0, 0.0)
        self.offset = (0.0, 0.0)

    def _random_target(self):
        return (
            random.uniform(-EYE_DRIFT_MAX_X, EYE_DRIFT_MAX_X),
            random.uniform(-EYE_DRIFT_MAX_Y, EYE_DRIFT_MAX_Y),
        )

    def update(self, dt):
        if self.state == "wait":
            self.timer -= dt
            if self.timer <= 0:
                self.start = self.offset
                self.target = self._random_target()
                self.state = "moving"
                self.progress = 0.0
        elif self.state == "moving":
            self.progress += dt / EYE_DRIFT_MOVE_TIME
            t = ease_in_out(min(1.0, self.progress))
            self.offset = (
                self.start[0] + (self.target[0] - self.start[0]) * t,
                self.start[1] + (self.target[1] - self.start[1]) * t,
            )
            if self.progress >= 1.0:
                self.state = "hold"
                self.progress = 0.0
        elif self.state == "hold":
            self.progress += dt
            if self.progress >= EYE_DRIFT_HOLD_TIME:
                self.state = "returning"
                self.progress = 0.0
                self.start = self.offset
        elif self.state == "returning":
            self.progress += dt / EYE_DRIFT_RETURN_TIME
            t = ease_in_out(min(1.0, self.progress))
            self.offset = (
                self.start[0] + (0.0 - self.start[0]) * t,
                self.start[1] + (0.0 - self.start[1]) * t,
            )
            if self.progress >= 1.0:
                self.offset = (0.0, 0.0)
                self.state = "wait"
                self.timer = random.uniform(EYE_DRIFT_MIN_INTERVAL, EYE_DRIFT_MAX_INTERVAL)


class Talker:
    """Randomly cycles a 0..1 "mouth openness" value to fake natural
    talking, instead of a rigid metronome-like open/close. TALK_SPEED sets
    the average pace; TALK_RANDOMNESS jitters each movement's timing and
    target size so it doesn't look robotic.
    """

    def __init__(self):
        self.value = 0.0
        self._start = 0.0
        self._target = 0.0
        self._timer = 0.0
        self._duration = 0.2
        self._pick_next()

    def _pick_next(self):
        self._start = self.value
        self._target = random.uniform(0.15, 1.0)
        base = 1.0 / max(0.05, TALK_SPEED)
        jitter = 1.0 + random.uniform(-TALK_RANDOMNESS, TALK_RANDOMNESS)
        self._duration = max(0.03, base * jitter)
        self._timer = 0.0

    def update(self, dt):
        self._timer += dt
        t = min(1.0, self._timer / self._duration)
        self.value = self._start + (self._target - self._start) * ease_in_out(t)
        if t >= 1.0:
            self._pick_next()


class FaceExpression:
    """Holds the current expression and smoothly blends its look (eye
    height, brow cuts, pupil size, mouth curve, color) toward a new
    expression's params whenever `set()` picks a different one.
    """

    def __init__(self, name="idle"):
        self.name = name
        self.params = dict(EXPRESSIONS[name])
        self._from = dict(self.params)
        self._to = dict(self.params)
        self._t = 1.0

    def set(self, name):
        if name == self.name:
            return
        self.name = name
        self._from = dict(self.params)
        self._to = dict(EXPRESSIONS[name])
        self._t = 0.0
        # Not a blendable number -- switches immediately.
        self.params["mouth_shape"] = self._to["mouth_shape"]

    def update(self, dt):
        if self._t >= 1.0:
            return
        self._t = min(1.0, self._t + dt / EXPRESSION_TRANSITION_TIME)
        t = ease_in_out(self._t)
        blended = {"mouth_shape": self.params["mouth_shape"]}
        for key, to_val in self._to.items():
            if key == "mouth_shape":
                continue
            from_val = self._from[key]
            if key == "color":
                blended[key] = tuple(
                    from_val[i] + (to_val[i] - from_val[i]) * t for i in range(3)
                )
            else:
                blended[key] = from_val + (to_val - from_val) * t
        self.params = blended


def draw_eye(surface, center_x, center_y, blink_amount, gaze_offset, scale=1.0, size_wobble=1.0,
             is_left=True, eye_h_mult=1.0, pupil_mult=1.0, brow_inner=0.0, brow_outer=0.0,
             color=None, rotate_deg=0.0, closed_curve=0.0):
    """Draw one eye: rounded-rect eyelid shape with a black pupil inside.

    center_x/center_y and gaze_offset are in logical screen pixels; `scale`
    maps them onto the (possibly supersampled) drawing surface. When
    `rotate_deg` is non-zero the eye is built on its own transparent
    surface and rotated around its center before being blit onto
    `surface` (pygame's rotate() is counterclockwise for positive angles).
    `closed_curve` (0..1) crossfades from the normal eye to a wide "U"
    smile-like curve standing in for a closed eye (e.g. sleep) -- unlike
    `blink_amount`, which just flattens the normal eyelid shut.
    """
    color = EYE_COLOR if color is None else color

    eye_w = EYE_W * scale * size_wobble
    eye_h = EYE_H * scale * size_wobble * eye_h_mult
    eye_radius = EYE_RADIUS * scale * size_wobble
    cx, cy = center_x * scale, center_y * scale
    gx, gy = gaze_offset[0] * scale, gaze_offset[1] * scale

    # Draw onto a local surface sized to the eye's bounding box (plus a
    # small margin) so it can be rotated in place without affecting
    # anything else already drawn on `surface`.
    pad = 4
    local_w = int(eye_w) + pad * 2
    local_h = int(eye_h) + pad * 2
    local = pygame.Surface((local_w, local_h), pygame.SRCALPHA)
    lcx, lcy = local_w / 2, local_h / 2

    if closed_curve < 0.999:
        normal = pygame.Surface((local_w, local_h), pygame.SRCALPHA)

        # As closed_curve rises the lid shrinks toward a thin line too (not
        # just fading out at full size), so the eye reads as closing rather
        # than shrinking-then-vanishing before the "U" curve fades in.
        closing = max(blink_amount, closed_curve)
        open_h = max(6, int(eye_h * (1.0 - closing)))
        rect = pygame.Rect(0, 0, int(eye_w), open_h)
        rect.center = (lcx, lcy)
        radius = min(int(eye_radius), open_h // 2, int(eye_w) // 2)
        pygame.draw.rect(normal, color, rect, border_radius=radius)

        # Pupil size/position stay fixed; the closing eyelid clips it away,
        # so the blink reads as eyelids covering the eye (vertical only),
        # not the pupil itself shrinking/zooming.
        pupil_w = eye_w * PUPIL_SCALE * pupil_mult
        pupil_h = eye_h * PUPIL_SCALE * pupil_mult

        max_travel_x = (eye_w - pupil_w) / 2
        max_travel_y = (eye_h - pupil_h) / 2
        px = lcx + max(-max_travel_x, min(max_travel_x, gx))
        py = lcy + max(-max_travel_y, min(max_travel_y, gy))

        pupil_rect = pygame.Rect(0, 0, max(2, int(pupil_w)), max(2, int(pupil_h)))
        pupil_rect.center = (int(px), int(py))
        pupil_radius = min(int(eye_radius), pupil_rect.width // 2, pupil_rect.height // 2)

        normal.set_clip(rect)
        pygame.draw.rect(normal, PUPIL_COLOR, pupil_rect, border_radius=pupil_radius)

        if blink_amount < 0.6:
            highlight_r = max(2, int(pupil_rect.width * 0.18))
            hx = pupil_rect.centerx - pupil_rect.width * 0.22
            hy = pupil_rect.centery - pupil_rect.height * 0.22
            pygame.draw.circle(normal, (255, 255, 255), (int(hx), int(hy)), highlight_r)

        normal.set_clip(None)

        # Brow/eyelid wedge: paint a dark triangle over the top-inner
        # and/or top-outer corner of the eye to slant its top edge --
        # furrowed (angry: inner corner cut deep) or worried (sad: outer
        # corner cut deep). Mirrored between the left and right eye.
        if brow_inner > 0.0 or brow_outer > 0.0:
            inner_cut = eye_h * brow_inner
            outer_cut = eye_h * brow_outer
            cut_left, cut_right = (outer_cut, inner_cut) if is_left else (inner_cut, outer_cut)
            pygame.draw.polygon(normal, (0, 0, 0, 0), [
                (rect.left, rect.top),
                (rect.right, rect.top),
                (rect.right, int(rect.top + cut_right)),
                (rect.left, int(rect.top + cut_left)),
            ])

        if closed_curve > 0.0:
            normal.set_alpha(int(255 * (1.0 - closed_curve)))
        local.blit(normal, (0, 0))

    if closed_curve > 0.001:
        # A wide, shallow "U" (a closed, smiling eye), drawn the same way
        # the smile mouth arc is -- the bottom half of an ellipse.
        closed = pygame.Surface((local_w, local_h), pygame.SRCALPHA)
        curve_rect = pygame.Rect(0, 0, eye_w * 0.95, eye_h * 0.55)
        curve_rect.center = (lcx, lcy)
        curve_thickness = max(2, int(eye_radius * 0.32))
        pygame.draw.arc(closed, color, curve_rect, math.pi * 1.08, math.pi * 1.92, curve_thickness)
        if closed_curve < 1.0:
            closed.set_alpha(int(255 * closed_curve))
        local.blit(closed, (0, 0))

    if rotate_deg:
        local = pygame.transform.rotate(local, rotate_deg)

    dest = local.get_rect(center=(cx, cy))
    surface.blit(local, dest)


def draw_mouth(surface, center_x, center_y, scale=1.0, curve=1.0, width_mult=1.0, color=None,
               shape="curve"):
    """A simple mouth. shape="curve": an arc, positive curve = smile,
    negative = frown. shape="circle": a round, fully filled "o" mouth
    (e.g. curious/sleep). shape="flat": a plain straight horizontal line
    (e.g. bored). shape="bar": a filled rectangle whose height follows
    `curve`, so animating curve reads as the mouth opening/closing
    (e.g. talking).
    """
    color = MOUTH_COLOR if color is None else color

    width = MOUTH_W * scale * width_mult
    height = max(MOUTH_H * scale * abs(curve), MOUTH_THICKNESS * scale * 1.4)
    thickness = max(1, int(MOUTH_THICKNESS * scale))
    cx, cy = center_x * scale, center_y * scale

    if shape == "circle":
        diameter = height
        rect = pygame.Rect(0, 0, diameter, diameter)
        rect.center = (cx, cy)
        pygame.draw.ellipse(surface, color, rect)
        return

    if shape == "flat":
        half_w = width / 2
        pygame.draw.line(surface, color, (cx - half_w, cy), (cx + half_w, cy), thickness)
        return

    if shape == "bar":
        rect = pygame.Rect(0, 0, width, height)
        rect.center = (cx, cy)
        pygame.draw.rect(surface, color, rect, border_radius=thickness // 2)
        return

    rect = pygame.Rect(0, 0, width, height)
    rect.center = (cx, cy)

    # Bottom half of the ellipse reads as a smile (a "cup" curve); the top
    # half reads as a frown.
    if curve >= 0:
        start_angle, stop_angle = math.pi * 1.2, math.pi * 1.8
    else:
        start_angle, stop_angle = math.pi * 0.2, math.pi * 0.8

    pygame.draw.arc(surface, color, rect, start_angle, stop_angle, thickness)


def run_standalone():
    pygame.init()
    pygame.display.set_caption("Robot Face")

    flags = (pygame.FULLSCREEN if FULLSCREEN else 0) | pygame.DOUBLEBUF
    try:
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags, vsync=1)
    except pygame.error:
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
    if FULLSCREEN:
        pygame.mouse.set_visible(False)

    clock = pygame.time.Clock()

    canvas = pygame.Surface((SCREEN_W * RENDER_SCALE, SCREEN_H * RENDER_SCALE))

    left_blink = Blinker()
    right_blink = Blinker()
    left_blink.timer = random.uniform(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL)
    right_blink.timer = left_blink.timer  # blink both eyes together

    gaze = Gaze()
    eye_wobble = EyeWobble()
    eye_drift = EyeDrift()
    talker = Talker()
    auto_enabled = AUTO_SEQUENCE_ENABLED and len(AUTO_SEQUENCE) > 0
    auto_index = 0
    auto_timer = AUTO_SEQUENCE[0][1] if auto_enabled else 0.0
    expression = FaceExpression(AUTO_SEQUENCE[0][0] if auto_enabled else "idle")
    bored_timer = None

    eye_y = SCREEN_H / 2 - 30
    left_eye_x = SCREEN_W / 2 - EYE_GAP / 2 - EYE_W / 2
    right_eye_x = SCREEN_W / 2 + EYE_GAP / 2 + EYE_W / 2
    mouth_x = SCREEN_W / 2
    mouth_y = eye_y + EYE_H / 2 + MOUTH_GAP

    fullscreen = FULLSCREEN
    running = True
    total_t = 0.0
    while running:
        dt = clock.tick(FPS) / 1000.0
        total_t += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    fullscreen = not fullscreen
                    flags = (pygame.FULLSCREEN if fullscreen else 0) | pygame.DOUBLEBUF
                    try:
                        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags, vsync=1)
                    except pygame.error:
                        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
                    pygame.mouse.set_visible(not fullscreen)
                elif event.key in EXPRESSION_KEYS:
                    auto_enabled = False
                    expression.set(EXPRESSION_KEYS[event.key])

        left_blink.update(dt)
        right_blink.timer = left_blink.timer
        right_blink.state = left_blink.state
        right_blink.progress = left_blink.progress
        right_blink.amount = left_blink.amount
        gaze.update(dt, interval_mult=expression.params["gaze_interval_mult"])
        eye_wobble.update(dt)
        eye_drift.update(dt)
        talker.update(dt)
        expression.update(dt)
        p = expression.params

        # Automatic idle-mood cycling (stops for good once the user picks
        # an expression manually).
        if auto_enabled:
            auto_timer -= dt
            if auto_timer <= 0:
                auto_index = (auto_index + 1) % len(AUTO_SEQUENCE)
                name, duration = AUTO_SEQUENCE[auto_index]
                expression.set(name)
                auto_timer = duration

        # "bored" auto-falls asleep after sitting bored for a while.
        if expression.name == "bored":
            bored_timer = BORED_TO_SLEEP_TIME if bored_timer is None else bored_timer - dt
            if bored_timer <= 0:
                expression.set("sleep")
                bored_timer = None
        else:
            bored_timer = None

        drift_x, drift_y = eye_drift.offset
        bounce_y = p["bounce"] * HAPPY_BOUNCE_AMOUNT * math.sin(total_t * HAPPY_BOUNCE_SPEED)
        wobble_l = 1.0 + (eye_wobble.scale[0] - 1.0) * p["wobble_mult"]
        wobble_r = 1.0 + (eye_wobble.scale[1] - 1.0) * p["wobble_mult"]
        talk_curve = TALK_MOUTH_MIN + (TALK_MOUTH_MAX - TALK_MOUTH_MIN) * talker.value
        mouth_curve = p["mouth_curve"] + (talk_curve - p["mouth_curve"]) * p["talk_mult"]
        breathe_t = 0.5 + 0.5 * math.sin(total_t * SLEEP_BREATHE_SPEED)
        breathe_curve = SLEEP_MOUTH_MIN + (SLEEP_MOUTH_MAX - SLEEP_MOUTH_MIN) * breathe_t
        mouth_curve += (breathe_curve - mouth_curve) * p["breathe_mult"]

        canvas.fill(BG_COLOR)
        draw_eye(canvas, left_eye_x + drift_x, eye_y + drift_y + bounce_y, left_blink.amount,
                 gaze.offset, RENDER_SCALE, wobble_l,
                 is_left=True, eye_h_mult=p["eye_h"], pupil_mult=p["pupil_mult"],
                 brow_inner=p["brow_inner"], brow_outer=p["brow_outer"], color=p["color"],
                 rotate_deg=p["eye_rotate"], closed_curve=p["eyes_closed"])
        draw_eye(canvas, right_eye_x + drift_x, eye_y + drift_y + bounce_y, right_blink.amount,
                 gaze.offset, RENDER_SCALE, wobble_r,
                 is_left=False, eye_h_mult=p["eye_h"], pupil_mult=p["pupil_mult"],
                 brow_inner=p["brow_inner"], brow_outer=p["brow_outer"], color=p["color"],
                 rotate_deg=-p["eye_rotate"], closed_curve=p["eyes_closed"])
        if SHOW_MOUTH:
            draw_mouth(canvas, mouth_x, mouth_y + p["mouth_offset_y"], RENDER_SCALE,
                       curve=mouth_curve, width_mult=p["mouth_width"], color=p["color"],
                       shape=p["mouth_shape"])

        if RENDER_SCALE == 1:
            screen.blit(canvas, (0, 0))
        else:
            pygame.transform.smoothscale(canvas, (SCREEN_W, SCREEN_H), screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()



# ===========================================================================
#  Library interface
# ===========================================================================
class FaceRenderer:
    """The face, with the caller owning the frame loop.

    The original main() held the animation state in local variables and drove
    it from a while loop.  That is the right shape for a standalone script and
    the wrong shape for a ROS node, which needs to interleave rendering with
    spinning an executor.  This class holds exactly the same state and exposes
    it as `step(dt)` — every drawing call below is the original one, in the
    original order, with the original arguments.

    Usage:

        r = FaceRenderer(fullscreen=True)
        while ok:
            dt = r.tick()
            r.set_expression('happy')
            r.step(dt)
    """

    def __init__(self, width=None, height=None, fullscreen=None,
                 start_expression='idle', auto_sequence=False,
                 show_mouth=None, caption='A2R_bot face'):
        self.width = int(width or SCREEN_W)
        self.height = int(height or SCREEN_H)
        self.fullscreen = (FULLSCREEN if fullscreen is None
                           else bool(fullscreen))
        self.show_mouth = (SHOW_MOUTH if show_mouth is None
                           else bool(show_mouth))

        pygame.init()
        pygame.display.set_caption(caption)
        flags = (pygame.FULLSCREEN if self.fullscreen else 0) | pygame.DOUBLEBUF
        try:
            self.screen = pygame.display.set_mode(
                (self.width, self.height), flags, vsync=1)
        except pygame.error:
            self.screen = pygame.display.set_mode(
                (self.width, self.height), flags)
        if self.fullscreen:
            pygame.mouse.set_visible(False)

        self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface((self.width * RENDER_SCALE,
                                      self.height * RENDER_SCALE))

        self.left_blink = Blinker()
        self.right_blink = Blinker()
        self.left_blink.timer = random.uniform(BLINK_MIN_INTERVAL,
                                               BLINK_MAX_INTERVAL)
        self.right_blink.timer = self.left_blink.timer

        self.gaze = Gaze()
        self.eye_wobble = EyeWobble()
        self.eye_drift = EyeDrift()
        self.talker = Talker()

        if start_expression not in EXPRESSIONS:
            start_expression = 'idle'
        self.expression = FaceExpression(start_expression)

        self.auto_enabled = bool(auto_sequence) and len(AUTO_SEQUENCE) > 0
        self.auto_index = 0
        self.auto_timer = AUTO_SEQUENCE[0][1] if self.auto_enabled else 0.0

        self.total_t = 0.0
        self.running = True
        self.bored_timer = None

        # Layout, exactly as the original computed it.
        self.eye_y = self.height / 2 - 30
        self.left_eye_x = self.width / 2 - EYE_GAP / 2 - EYE_W / 2
        self.right_eye_x = self.width / 2 + EYE_GAP / 2 + EYE_W / 2
        self.mouth_x = self.width / 2
        self.mouth_y = self.eye_y + EYE_H / 2 + MOUTH_GAP

    # ------------------------------------------------------------------
    def tick(self, fps=None):
        """Advance the frame clock and return dt in seconds."""
        dt = self.clock.tick(fps or FPS) / 1000.0
        self.total_t += dt
        return dt

    def set_expression(self, name):
        """Switch expression.  Unknown names are ignored, not raised on.

        A face that crashes because a state machine somewhere sent an
        unexpected string is worse than a face that keeps smiling.
        """
        if name and name in EXPRESSIONS:
            self.expression.set(name)
            return True
        return False

    @property
    def expression_name(self):
        return self.expression.name

    def talk(self, on=True):
        """Start or stop the talking mouth.

        Note that Talker free-runs: it always animates, and what decides
        whether you can see it is the current expression's `talk_mult`. So
        "talking" is an expression here, not an event — which is why this
        just switches expression rather than triggering an animation.
        """
        if on:
            self._pre_talk = self.expression.name
            return self.set_expression('talking')
        back = getattr(self, '_pre_talk', 'idle')
        return self.set_expression(back)

    def pump_events(self):
        """Handle window events.  Returns False when the user asked to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_f:
                    self.toggle_fullscreen()
                elif event.key in EXPRESSION_KEYS:
                    # Manual override from the keyboard still works, which is
                    # invaluable when checking the screen without a Jetson
                    # attached.
                    self.auto_enabled = False
                    self.expression.set(EXPRESSION_KEYS[event.key])
        return self.running

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        flags = (pygame.FULLSCREEN if self.fullscreen else 0) | pygame.DOUBLEBUF
        try:
            self.screen = pygame.display.set_mode(
                (self.width, self.height), flags, vsync=1)
        except pygame.error:
            self.screen = pygame.display.set_mode(
                (self.width, self.height), flags)
        pygame.mouse.set_visible(not self.fullscreen)

    # ------------------------------------------------------------------
    def step(self, dt):
        """Update every animator and draw one frame.

        This body is the original render block from run_standalone(), moved
        here unchanged: same order, same constants, same draw_eye and
        draw_mouth arguments. The only substitutions are `self.` prefixes on
        what used to be the loop's local variables.

        It is copied rather than paraphrased on purpose. Every number in it —
        HAPPY_BOUNCE_SPEED, the talk and breathe curve blends, the sign flip
        on the right eye's rotation — was tuned by looking at the screen, and
        a rewrite that is 95% faithful is a rewrite that looks subtly wrong in
        a way nobody can put their finger on.
        """
        self.left_blink.update(dt)
        self.right_blink.timer = self.left_blink.timer
        self.right_blink.state = self.left_blink.state
        self.right_blink.progress = self.left_blink.progress
        self.right_blink.amount = self.left_blink.amount
        self.gaze.update(
            dt, interval_mult=self.expression.params["gaze_interval_mult"])
        self.eye_wobble.update(dt)
        self.eye_drift.update(dt)
        self.talker.update(dt)
        self.expression.update(dt)
        p = self.expression.params

        # Automatic idle-mood cycling. Off by default under ROS: when the
        # Jetson is choosing the expression, an auto sequence competing for
        # the same face is a bug that looks like a feature.
        if self.auto_enabled:
            self.auto_timer -= dt
            if self.auto_timer <= 0:
                self.auto_index = (self.auto_index + 1) % len(AUTO_SEQUENCE)
                name, duration = AUTO_SEQUENCE[self.auto_index]
                self.expression.set(name)
                self.auto_timer = duration

        # "bored" auto-falls asleep after sitting bored for a while.
        if self.expression.name == "bored":
            self.bored_timer = (BORED_TO_SLEEP_TIME
                                if self.bored_timer is None
                                else self.bored_timer - dt)
            if self.bored_timer <= 0:
                self.expression.set("sleep")
                self.bored_timer = None
        else:
            self.bored_timer = None

        drift_x, drift_y = self.eye_drift.offset
        bounce_y = p["bounce"] * HAPPY_BOUNCE_AMOUNT * \
            math.sin(self.total_t * HAPPY_BOUNCE_SPEED)
        wobble_l = 1.0 + (self.eye_wobble.scale[0] - 1.0) * p["wobble_mult"]
        wobble_r = 1.0 + (self.eye_wobble.scale[1] - 1.0) * p["wobble_mult"]
        talk_curve = TALK_MOUTH_MIN + \
            (TALK_MOUTH_MAX - TALK_MOUTH_MIN) * self.talker.value
        mouth_curve = p["mouth_curve"] + \
            (talk_curve - p["mouth_curve"]) * p["talk_mult"]
        breathe_t = 0.5 + 0.5 * math.sin(self.total_t * SLEEP_BREATHE_SPEED)
        breathe_curve = SLEEP_MOUTH_MIN + \
            (SLEEP_MOUTH_MAX - SLEEP_MOUTH_MIN) * breathe_t
        mouth_curve += (breathe_curve - mouth_curve) * p["breathe_mult"]

        self.canvas.fill(BG_COLOR)
        draw_eye(self.canvas, self.left_eye_x + drift_x,
                 self.eye_y + drift_y + bounce_y, self.left_blink.amount,
                 self.gaze.offset, RENDER_SCALE, wobble_l,
                 is_left=True, eye_h_mult=p["eye_h"],
                 pupil_mult=p["pupil_mult"],
                 brow_inner=p["brow_inner"], brow_outer=p["brow_outer"],
                 color=p["color"], rotate_deg=p["eye_rotate"],
                 closed_curve=p["eyes_closed"])
        draw_eye(self.canvas, self.right_eye_x + drift_x,
                 self.eye_y + drift_y + bounce_y, self.right_blink.amount,
                 self.gaze.offset, RENDER_SCALE, wobble_r,
                 is_left=False, eye_h_mult=p["eye_h"],
                 pupil_mult=p["pupil_mult"],
                 brow_inner=p["brow_inner"], brow_outer=p["brow_outer"],
                 color=p["color"], rotate_deg=-p["eye_rotate"],
                 closed_curve=p["eyes_closed"])
        if self.show_mouth:
            draw_mouth(self.canvas, self.mouth_x,
                       self.mouth_y + p["mouth_offset_y"], RENDER_SCALE,
                       curve=mouth_curve, width_mult=p["mouth_width"],
                       color=p["color"], shape=p["mouth_shape"])

        if RENDER_SCALE == 1:
            self.screen.blit(self.canvas, (0, 0))
        else:
            pygame.transform.smoothscale(
                self.canvas, (self.width, self.height), self.screen)
        pygame.display.flip()

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass


def list_expressions():
    """Every expression name the face knows."""
    return sorted(EXPRESSIONS.keys())


def main():
    """Kept so `python3 robot_face.py` behaves exactly as it always did."""
    run_standalone()


if __name__ == '__main__':
    main()
