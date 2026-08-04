"""
MediaPipe Hand Analyzer.
"""

import time

import cv2
import mediapipe as mp

from gesture_classifier import HandGestureClassifier


class HandAnalyzer:

    # Hoisted to class level — no reason to rebuild this list every draw() call.
    CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (5, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (9, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (13, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (0, 17)
    ]

    def __init__(
        self,
        model_path,
        max_hands=2,
        detection_confidence=0.60,
        presence_confidence=0.50,
        tracking_confidence=0.50
    ):

        self.mp = mp
        self.max_hands = max_hands

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),

            # VIDEO mode (not IMAGE) enables MediaPipe's own temporal
            # tracking between frames — this is what makes
            # min_tracking_confidence actually do something, and it
            # keeps hand identity stable frame-to-frame instead of
            # re-detecting from scratch every call.
            running_mode=RunningMode.VIDEO,

            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.landmarker = HandLandmarker.create_from_options(options)

        # One gesture classifier PER HAND, keyed by handedness label, kept
        # alive across frames. This is what actually removes flicker —
        # creating a fresh classifier every frame (or calling a stateless
        # function) throws away all the temporal smoothing.
        self._classifiers = {}

        self._last_timestamp_ms = 0

    # ========================================================
    # PROCESS
    # ========================================================

    def process(self, frame, timestamp_ms=None):
        """
        timestamp_ms: monotonically increasing timestamp in ms. If you're
        reading from a live camera, you can leave this as None and it will
        use the wall clock. If you're processing a video file, pass the
        frame's actual timestamp (e.g. frame_index * (1000 / fps)) so
        playback speed doesn't affect tracking quality.
        """

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # VIDEO mode requires strictly increasing timestamps.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        hands = []
        seen_labels = set()

        if result.hand_landmarks:

            for index, landmarks in enumerate(result.hand_landmarks):

                handedness = "Unknown"

                if result.handedness and index < len(result.handedness):
                    handedness = result.handedness[index][0].category_name

                # Disambiguate two hands with the same handedness label
                # (shouldn't normally happen, but keeps classifiers from
                # colliding if it does).
                label = handedness
                if label in seen_labels:
                    label = f"{handedness}_{index}"
                seen_labels.add(label)

                classifier = self._classifiers.get(label)
                if classifier is None:
                    classifier = HandGestureClassifier()
                    self._classifiers[label] = classifier

                gesture, confidence = classifier.classify(landmarks)

                hands.append({
                    "landmarks": landmarks,
                    "handedness": handedness,
                    "gesture": gesture,
                    "confidence": confidence
                })

        # Any previously-tracked hand that's missing this frame needs to
        # be told "no landmarks" so its lost-frame counter advances and
        # it resets cleanly instead of freezing on its last gesture forever.
        for label, classifier in self._classifiers.items():
            if label not in seen_labels:
                classifier.classify(None)

        return hands

    # ========================================================
    # DRAW
    # ========================================================

    def draw(self, frame, hands):

        h, w, _ = frame.shape

        for hand in hands:

            landmarks = hand["landmarks"]

            # ----------------------------------------------
            # Connections
            # ----------------------------------------------

            for start, end in self.CONNECTIONS:

                x1 = int(landmarks[start].x * w)
                y1 = int(landmarks[start].y * h)
                x2 = int(landmarks[end].x * w)
                y2 = int(landmarks[end].y * h)

                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # ----------------------------------------------
            # Points
            # ----------------------------------------------

            for point in landmarks:

                x = int(point.x * w)
                y = int(point.y * h)

                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # ----------------------------------------------
            # Gesture / handedness label
            # ----------------------------------------------

            wrist = landmarks[0]
            label_x = int(wrist.x * w)
            label_y = int(wrist.y * h) + 30

            text = f"{hand['handedness']}: {hand['gesture']} ({hand['confidence']:.2f})"

            cv2.putText(
                frame, text, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

        return frame

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        """Clear all per-hand gesture state (e.g. when switching video sources)."""
        self._classifiers.clear()
        self._last_timestamp_ms = 0

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        self.landmarker.close()