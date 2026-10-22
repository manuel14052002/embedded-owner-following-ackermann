"""
Gesture Detector
- Owner ID: raise hand HIGH above head and hold 2 seconds
- Release:  open palm (5 fingers) held 1.5 seconds
"""
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time


class GestureDetector:
    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.5):
        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._raise_start  = None
        self._stop_start   = None
        self._handsup_start = None
        self.HANDSUP_HOLD  = 1.0   # both hands up held 1s = stop
        self.RAISE_HOLD    = 2.0   # seconds hand must be raised
        self.STOP_HOLD     = 1.5   # seconds stop sign held
        self._progress     = 0.0   # 0-1 fill for visual feedback
        self.reset()

    def reset(self):
        self._raise_start = None
        self._stop_start  = None
        self._handsup_start = None
        self._progress    = 0.0

    def _count_extended_fingers(self, lm) -> int:
        """Count extended fingers. Margins scale with apparent hand size so the
        test works at operating distance, not just close to the camera (F18)."""
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        count = 0
        # Hand scale proxy: wrist (0) to middle-finger MCP (9)
        _dx = lm[9].x - lm[0].x
        _dy = lm[9].y - lm[0].y
        hand_scale = max(0.02, (_dx*_dx + _dy*_dy) ** 0.5)   # floor avoids div-by-zero
        thumb_margin  = 0.20 * hand_scale    # was fixed 0.03
        finger_margin = 0.13 * hand_scale    # was fixed 0.02
        # Thumb
        if abs(lm[4].x - lm[3].x) > thumb_margin:
            count += 1
        # 4 fingers
        for tip, pip in zip(tips, pips):
            if lm[tip].y < lm[pip].y - finger_margin:
                count += 1
        return count

    def _hand_high(self, lm, h) -> bool:
        """Hand is raised HIGH — wrist above nose level."""
        wrist_y = lm[0].y * h
        nose_y  = h * 0.50   # F18: measured wrist sits 0.27-0.48 at 2.5m; 0.50 gives hold margin
        return wrist_y < nose_y

    def _hand_raised(self, lm, h) -> bool:
        """Hand is raised above shoulder level."""
        wrist_y    = lm[0].y * h
        shoulder_y = h * 0.55
        return wrist_y < shoulder_y

    def detect_gestures(self, frame: np.ndarray) -> dict:
        result = {
            'wave_detected':   False,
            'wave_confirmed':  False,
            'stop_sign':       False,
            'hands_up_stop':   False,   # both hands raised high = STOP
            'hand_bbox':       None,
            'wave_bbox':       None,
            'progress':        0.0,
            'annotated_frame': frame.copy(),
        }

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = self.hands.process(rgb)
        ann  = result['annotated_frame']

        if not res.multi_hand_landmarks:
            self._raise_start = None
            self._stop_start  = None
            self._handsup_start = None
            self._progress    = 0.0
            return result

        # ── HANDS-UP STOP: two hands both raised high, held 1s ──────────
        hands_high = 0
        for hand_lm in res.multi_hand_landmarks:
            if self._hand_high(hand_lm.landmark, h):
                hands_high += 1
        if hands_high >= 2:
            if self._handsup_start is None:
                self._handsup_start = time.time()
            elapsed = time.time() - self._handsup_start
            prog = min(1.0, elapsed / self.HANDSUP_HOLD)
            cv2.putText(ann, f"STOP (hands up)? {int(prog*100)}%", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,220), 2)
            if elapsed >= self.HANDSUP_HOLD:
                result['hands_up_stop'] = True
                cv2.putText(ann, "HANDS-UP STOP!", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
        else:
            self._handsup_start = None

        # Draw every detected hand, but only ONE hand drives the gesture logic.
        #
        # Iterating all hands was wrong: the enrollment timer is shared, so a
        # second hand hanging at the person's side hit the `else` branch every
        # frame and reset the progress of the hand that WAS raised. With two
        # hands visible, enrollment could never complete.
        #
        # The highest wrist wins. A deliberate raised hand is above every
        # other hand in frame, including a bystander's.
        for _hl in res.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(ann, _hl, self.mp_hands.HAND_CONNECTIONS)

        _primary = min(res.multi_hand_landmarks,
                       key=lambda hl: hl.landmark[0].y)   # smallest y = highest

        for hand_lm in (_primary,):
            lm = hand_lm.landmark

            xs = [p.x*w for p in lm]; ys = [p.y*h for p in lm]
            x1,y1 = int(min(xs)),int(min(ys))
            x2,y2 = int(max(xs)),int(max(ys))
            result['hand_bbox'] = (x1,y1,x2-x1,y2-y1)

            fingers = self._count_extended_fingers(lm)
            high    = self._hand_high(lm, h)
            raised  = self._hand_raised(lm, h)

            # ── STOP SIGN: 5 fingers open, raised ──────────────────
            if fingers >= 4 and raised and not high:
                if self._stop_start is None:
                    self._stop_start = time.time()
                elapsed = time.time() - self._stop_start
                prog    = min(1.0, elapsed / self.STOP_HOLD)

                # Draw red progress arc
                cx,cy = (x1+x2)//2, y1-30
                cv2.ellipse(ann,(cx,cy),(25,25),-90,0,int(360*prog),(0,0,220),3)
                cv2.putText(ann,f"STOP? {int(prog*100)}%",(x1,y1-40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,220),2)

                if elapsed >= self.STOP_HOLD:
                    result['stop_sign'] = True
                    cv2.rectangle(ann,(x1,y1),(x2,y2),(0,0,255),3)
                    cv2.putText(ann,"STOP SIGN",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            else:
                self._stop_start = None

            # ── OWNER ID: raise hand HIGH above head and hold ───────
            if high and fingers >= 2:
                result['wave_detected'] = True
                result['wave_bbox']     = (x1,y1,x2-x1,y2-y1)

                if self._raise_start is None:
                    self._raise_start = time.time()

                elapsed = time.time() - self._raise_start
                prog    = min(1.0, elapsed / self.RAISE_HOLD)
                self._progress = prog
                result['progress'] = prog

                # Draw green progress arc
                cx2,cy2 = (x1+x2)//2, y1-30
                cv2.ellipse(ann,(cx2,cy2),(25,25),-90,0,int(360*prog),(0,220,0),3)
                cv2.putText(ann,f"OWNER? {int(prog*100)}%",(x1,y1-40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,0),2)
                cv2.rectangle(ann,(x1,y1),(x2,y2),(0,200,0),2)

                if elapsed >= self.RAISE_HOLD:
                    result['wave_confirmed'] = True
                    cv2.rectangle(ann,(x1,y1),(x2,y2),(0,255,0),4)
                    cv2.putText(ann,"OWNER IDENTIFIED!",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            else:
                # Only reset if hand clearly not raised
                if not raised:
                    self._raise_start = None
                    self._progress    = 0.0

        return result
