"""
Advanced rule-based hand gesture classifier — flicker-resistant version.

Uses:
    - landmark distances
    - joint angles
    - palm geometry
    - finger extension
    - EMA landmark smoothing        (reduces raw jitter)
    - angle hysteresis              (prevents chatter at thresholds)
    - temporal label confirmation   (prevents single-frame label flips)

This is much more stable than simply checking whether
finger_tip.y < finger_pip.y, and far more stable than classifying
every frame independently.
"""

import math
from collections import deque


# ============================================================
# BASIC GEOMETRY
# ============================================================

def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def angle(a, b, c):
    """Angle ABC, in degrees."""

    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)

    dot_product = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]

    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0

    cosine = dot_product / (magnitude_ba * magnitude_bc)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


class _Point:
    """Tiny mutable x/y/z container so we can EMA-smooth in place."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


# ============================================================
# LANDMARK INDEXES
# ============================================================

WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

_FINGER_JOINTS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}


# ============================================================
# STABLE GESTURE CLASSIFIER
# ============================================================

class HandGestureClassifier:
    """
    Stateful classifier. Create ONE instance per tracked hand and call
    `.classify(landmarks)` every frame — don't call the module-level
    functions directly per frame, since the stability comes from the
    state this object keeps between calls.

    Parameters
    ----------
    smoothing_alpha : float
        EMA factor for landmark smoothing. Higher = more responsive,
        less smoothing. 0.4-0.6 is a good range.
    confirm_frames : int
        How many consecutive frames a NEW raw gesture must win before
        it replaces the currently displayed one. Higher = more stable
        but slightly more latency. 3-5 at 30fps feels near-instant
        while killing single-frame flicker.
    lost_hand_reset_frames : int
        How many consecutive missed detections before resetting state
        (so a brief 1-2 frame tracking dropout doesn't reset anything).
    """

    def __init__(
        self,
        smoothing_alpha=0.5,
        confirm_frames=4,
        lost_hand_reset_frames=6
    ):
        self.smoothing_alpha = smoothing_alpha
        self.confirm_frames = confirm_frames
        self.lost_hand_reset_frames = lost_hand_reset_frames

        self._smoothed = None
        self._finger_prev = {"thumb": False, "index": False, "middle": False,
                              "ring": False, "pinky": False}

        self._candidate_gesture = "Unknown"
        self._candidate_count = 0

        self.stable_gesture = "Unknown"
        self.stable_confidence = 0.0

        self._missed_frames = 0

    # --------------------------------------------------------
    # PUBLIC ENTRY POINT
    # --------------------------------------------------------

    def classify(self, landmarks):
        """
        landmarks: list/sequence of 21 objects with .x .y .z, or None
        if no hand was detected this frame.

        Returns (gesture_label, confidence) — the STABILIZED result,
        not the raw per-frame guess.
        """

        if landmarks is None:
            self._missed_frames += 1
            if self._missed_frames >= self.lost_hand_reset_frames:
                self._reset()
            return self.stable_gesture, self.stable_confidence

        self._missed_frames = 0

        smoothed = self._smooth(landmarks)
        fingers = self._get_finger_states(smoothed)
        raw_gesture, raw_confidence = self._classify_raw(fingers, smoothed)

        self._confirm(raw_gesture, raw_confidence)

        return self.stable_gesture, self.stable_confidence

    def reset(self):
        self._reset()

    # --------------------------------------------------------
    # LANDMARK SMOOTHING (EMA)
    # --------------------------------------------------------

    def _smooth(self, landmarks):

        if self._smoothed is None:
            self._smoothed = [_Point(p.x, p.y, p.z) for p in landmarks]
            return self._smoothed

        a = self.smoothing_alpha

        for i, p in enumerate(landmarks):
            s = self._smoothed[i]
            s.x = a * p.x + (1 - a) * s.x
            s.y = a * p.y + (1 - a) * s.y
            s.z = a * p.z + (1 - a) * s.z

        return self._smoothed

    # --------------------------------------------------------
    # FINGER EXTENSION (with hysteresis)
    # --------------------------------------------------------

    def _finger_extended_hysteresis(self, landmarks, mcp, pip, dip, tip, was_extended):
        """
        Two thresholds instead of one:
          - to BECOME extended, angles must clear the HIGH bar
          - once extended, it stays extended until angles drop below
            the LOW bar

        This deadband is what stops a finger sitting right at ~155°
        from flickering extended/bent every other frame.
        """

        pip_angle = angle(landmarks[mcp], landmarks[pip], landmarks[dip])
        dip_angle = angle(landmarks[pip], landmarks[dip], landmarks[tip])

        if was_extended:
            return pip_angle > 140 and dip_angle > 135
        else:
            return pip_angle > 158 and dip_angle > 153

    def _thumb_extended_hysteresis(self, landmarks, was_extended):

        thumb_angle = angle(
            landmarks[THUMB_MCP],
            landmarks[THUMB_IP],
            landmarks[THUMB_TIP]
        )

        if was_extended:
            return thumb_angle > 138
        else:
            return thumb_angle > 153

    def _get_finger_states(self, landmarks):

        prev = self._finger_prev

        thumb = self._thumb_extended_hysteresis(landmarks, prev["thumb"])

        states = {"thumb": thumb}

        for name, (mcp, pip, dip, tip) in _FINGER_JOINTS.items():
            states[name] = self._finger_extended_hysteresis(
                landmarks, mcp, pip, dip, tip, prev[name]
            )

        self._finger_prev = states
        return states

    # --------------------------------------------------------
    # RAW GESTURE (single-frame guess, from smoothed+hysteresis fingers)
    # --------------------------------------------------------

    def _classify_raw(self, fingers, landmarks):

        thumb = fingers["thumb"]
        index = fingers["index"]
        middle = fingers["middle"]
        ring = fingers["ring"]
        pinky = fingers["pinky"]

        count = sum(fingers.values())

        if count == 0:
            return "Fist", 0.90

        if count == 5:
            return "Open Palm", 0.95

        if not thumb and index and middle and not ring and not pinky:
            return "Peace", 0.92

        if not thumb and index and not middle and not ring and not pinky:
            return "Pointing", 0.90

        if thumb and not index and not middle and not ring and not pinky:
            return "Thumbs Up", 0.90

        if thumb and not index and not middle and not ring and pinky:
            return "Call Me", 0.88

        # OK / PINCH — checked before the generic finger-count fallbacks
        thumb_index_distance = distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
        palm_size = distance(landmarks[WRIST], landmarks[MIDDLE_MCP])

        if palm_size > 0:
            normalized_distance = thumb_index_distance / palm_size
            if normalized_distance < 0.28 and middle and ring and pinky:
                return "OK", 0.90

        if thumb and index and not middle and not ring and pinky:
            return "I Love You", 0.85

        if count == 1:
            return "1 Finger", 0.70
        if count == 2:
            return "2 Fingers", 0.70
        if count == 3:
            return "3 Fingers", 0.70
        if count == 4:
            return "4 Fingers", 0.70

        return "Unknown", 0.40

    # --------------------------------------------------------
    # TEMPORAL CONFIRMATION (kills single-frame flicker)
    # --------------------------------------------------------

    def _confirm(self, raw_gesture, raw_confidence):

        if raw_gesture == self._candidate_gesture:
            self._candidate_count += 1
        else:
            self._candidate_gesture = raw_gesture
            self._candidate_count = 1

        if (
            self._candidate_count >= self.confirm_frames
            and raw_gesture != self.stable_gesture
        ):
            self.stable_gesture = raw_gesture
            self.stable_confidence = raw_confidence
        elif raw_gesture == self.stable_gesture:
            # same gesture continuing — let confidence drift smoothly
            self.stable_confidence = 0.8 * self.stable_confidence + 0.2 * raw_confidence

    def _reset(self):
        self._smoothed = None
        self._finger_prev = {"thumb": False, "index": False, "middle": False,
                              "ring": False, "pinky": False}
        self._candidate_gesture = "Unknown"
        self._candidate_count = 0
        self.stable_gesture = "Unknown"
        self.stable_confidence = 0.0
        self._missed_frames = 0