"""
Face detection + DeepFace analysis.

MediaPipe:
    Face detection

DeepFace:
    Gender estimation
    Emotion estimation
"""

import time

import cv2
import mediapipe as mp

from deepface import DeepFace


class FaceAnalyzer:

    def __init__(
        self,
        model_path,
        detection_confidence=0.60,
        min_face_width=80,
        min_face_height=80,
        padding_x=0.20,
        padding_y=0.25,
        analyze_every_n_frames=5,   # throttle DeepFace calls (expensive)
        target_face_size=224        # DeepFace's models expect ~224x224 input
    ):

        self.min_face_width = min_face_width
        self.min_face_height = min_face_height
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.analyze_every_n_frames = analyze_every_n_frames
        self.target_face_size = target_face_size

        self._frame_count = 0
        self._last_result = None

        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=detection_confidence
        )

        self.detector = FaceDetector.create_from_options(options)

    # ========================================================
    # DETECT FACE
    # ========================================================

    def detect(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_image)

        if not result.detections:
            return None

        detection = result.detections[0]
        box = detection.bounding_box

        h, w, _ = frame.shape

        x = max(0, box.origin_x)
        y = max(0, box.origin_y)
        width = min(box.width, w - x)
        height = min(box.height, h - y)

        if width < self.min_face_width or height < self.min_face_height:
            return None

        pad_x = int(width * self.padding_x)
        pad_y = int(height * self.padding_y)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + width + pad_x)
        y2 = min(h, y + height + pad_y)

        crop = frame[y1:y2, x1:x2].copy()

        return {
            "box": (x1, y1, x2, y2),
            "crop": crop
        }

    # ========================================================
    # DEEPFACE
    # ========================================================

    def analyze(self, face_crop):
        """
        Run DeepFace emotion + gender analysis on an already-detected,
        already-cropped face (from detect()).

        Throttled to self.analyze_every_n_frames — DeepFace inference is
        heavy, so re-running it on every single frame is both wasteful
        and unnecessary for something like emotion display.
        """

        if face_crop is None:
            return None

        h, w = face_crop.shape[:2]

        if w < self.min_face_width or h < self.min_face_height:
            return None

        self._frame_count += 1

        # Reuse the last result on skipped frames instead of returning None
        if (self._frame_count % self.analyze_every_n_frames) != 0 and self._last_result is not None:
            return self._last_result

        try:
            # Resize to a fixed, model-friendly size instead of blindly
            # upscaling 2x every time. cv2.INTER_AREA is correct for
            # shrinking, INTER_CUBIC for enlarging.
            interp = (
                cv2.INTER_AREA
                if max(w, h) > self.target_face_size
                else cv2.INTER_CUBIC
            )

            resized = cv2.resize(
                face_crop,
                (self.target_face_size, self.target_face_size),
                interpolation=interp
            )

            # KEY FIX: detector_backend="skip" tells DeepFace NOT to run
            # its own (weak) face detector on top of the crop MediaPipe
            # already found. Re-detecting with opencv on a tight padded
            # crop frequently fails or misaligns the face, which is why
            # emotion/gender output was wrong or unstable before.
            result = DeepFace.analyze(
                img_path=resized,
                actions=["emotion", "gender"],
                detector_backend="skip",
                enforce_detection=False,
                align=False,
                silent=True
            )

            if isinstance(result, list):
                if not result:
                    return self._last_result
                result = result[0]

            gender = result.get("dominant_gender")
            gender_scores = result.get("gender", {})

            emotion = result.get("dominant_emotion")
            emotion_scores = result.get("emotion", {})

            gender_confidence = float(gender_scores.get(gender, 0)) if gender_scores else 0.0
            emotion_confidence = float(emotion_scores.get(emotion, 0)) if emotion_scores else 0.0

            output = {
                "gender": gender,
                "gender_confidence": gender_confidence,
                "emotion": emotion,
                "emotion_confidence": emotion_confidence,
                "gender_scores": gender_scores,
                "emotion_scores": emotion_scores
            }

            self._last_result = output
            return output

        except Exception as e:
            print("\nDeepFace error:", e)
            return self._last_result

    # ========================================================
    # DRAW
    # ========================================================

    def draw_box(self, frame, face_data, analysis=None):

        if face_data is None:
            return

        x1, y1, x2, y2 = face_data["box"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if analysis:
            label = f"{analysis.get('emotion', '?')} "
            cv2.putText(
                frame, label, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        self.detector.close()