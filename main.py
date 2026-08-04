"""
Face + Hand Analyzer V2

Main application.

Run:
    python main.py

Press:
    Q -> Quit
"""

import os
import cv2
import mediapipe as mp

import config

from face_analyzer import FaceAnalyzer
from hand_analyzer import HandAnalyzer
from smoothing import (
    PredictionSmoother,
    ConfidenceSmoother
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

FACE_MODEL = os.path.join(
    MODEL_DIR,
    "face_detector.tflite"
)

HAND_MODEL = os.path.join(
    MODEL_DIR,
    "hand_landmarker.task"
)


# ============================================================
# CHECK MODELS
# ============================================================

if not os.path.exists(FACE_MODEL):

    raise FileNotFoundError(
        f"""
Face model not found:

{FACE_MODEL}
"""
    )


if not os.path.exists(HAND_MODEL):

    raise FileNotFoundError(
        f"""
Hand model not found:

{HAND_MODEL}
"""
    )


# ============================================================
# UI PANEL
# ============================================================

def draw_panel(
    frame,
    gender,
    gender_confidence,
    emotion,
    emotion_confidence,
    hand_results
):

    h, w, _ = frame.shape

    # --------------------------------------------------------
    # Panel
    # --------------------------------------------------------

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (480, 180),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
        0
    )

    # --------------------------------------------------------
    # Gender
    # --------------------------------------------------------

    if gender is None:

        gender_text = "Analyzing..."

    else:

        gender_text = (
            f"{gender} "
            f"({gender_confidence:.0f}%)"
        )

    # cv2.putText(
    #     frame,
    #     f"Gender  : {gender_text}",
    #     (15, 30),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.65,
    #     (0, 255, 255),
    #     2
    # )

    # --------------------------------------------------------
    # Emotion
    # --------------------------------------------------------

    if emotion is None:

        emotion_text = "Analyzing..."

    else:

        emotion_text = (
            f"{emotion} "
            f"({emotion_confidence:.0f}%)"
        )

    cv2.putText(
        frame,
        f"Emotion : {emotion_text}",
        (15, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Hands
    # --------------------------------------------------------

    if not hand_results:

        cv2.putText(
            frame,
            "Hands   : None",
            (15, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 180, 0),
            2
        )

    else:

        y = 94

        for hand in hand_results:

            label = hand[
                "handedness"
            ]

            gesture = hand[
                "gesture"
            ]

            confidence = hand[
                "confidence"
            ]

            cv2.putText(
                frame,
                (
                    f"{label}: "
                    f"{gesture} "
                    f"({confidence * 100:.0f}%)"
                ),
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 180, 0),
                2
            )

            y += 30

    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Q = Quit",
        (15, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("       FACE + HAND ANALYZER V2")
    print("=" * 60)
    print(
        "Python:",
        __import__("sys").version.split()[0]
    )
    print(
        "MediaPipe:",
        mp.__version__
    )
    print()
    print("Starting camera...")
    print("Press Q to quit.")
    print("=" * 60)

    # --------------------------------------------------------
    # Face analyzer
    # --------------------------------------------------------

    face_analyzer = FaceAnalyzer(

        model_path=FACE_MODEL,

        detection_confidence=(
            config.FACE_DETECTION_CONFIDENCE
        ),

        min_face_width=(
            config.MIN_FACE_WIDTH
        ),

        min_face_height=(
            config.MIN_FACE_HEIGHT
        ),

        padding_x=(
            config.FACE_PADDING_X
        ),

        padding_y=(
            config.FACE_PADDING_Y
        )
    )

    # --------------------------------------------------------
    # Hand analyzer
    # --------------------------------------------------------

    hand_analyzer = HandAnalyzer(

        model_path=HAND_MODEL,

        max_hands=(
            config.MAX_HANDS
        ),

        detection_confidence=(
            config.HAND_DETECTION_CONFIDENCE
        ),

        presence_confidence=(
            config.HAND_PRESENCE_CONFIDENCE
        ),

        tracking_confidence=(
            config.HAND_TRACKING_CONFIDENCE
        )
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        config.CAMERA_INDEX
    )

    if not cap.isOpened():

        print(
            "ERROR: Camera could not be opened."
        )

        face_analyzer.close()
        hand_analyzer.close()

        return

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        config.FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        config.FRAME_HEIGHT
    )

    # --------------------------------------------------------
    # Prediction smoothers
    # --------------------------------------------------------

    gender_smoother = (
        PredictionSmoother(
            config.GENDER_HISTORY_SIZE
        )
    )

    gender_confidence_smoother = (
        ConfidenceSmoother(
            config.GENDER_HISTORY_SIZE
        )
    )

    emotion_smoother = (
        PredictionSmoother(
            config.EMOTION_HISTORY_SIZE
        )
    )

    emotion_confidence_smoother = (
        ConfidenceSmoother(
            config.EMOTION_HISTORY_SIZE
        )
    )

    # Separate smoother for each hand
    left_gesture_smoother = (
        PredictionSmoother(
            config.GESTURE_HISTORY_SIZE
        )
    )

    right_gesture_smoother = (
        PredictionSmoother(
            config.GESTURE_HISTORY_SIZE
        )
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    frame_count = 0

    current_gender = None
    current_gender_confidence = 0

    current_emotion = None
    current_emotion_confidence = 0

    # --------------------------------------------------------
    # Camera loop
    # --------------------------------------------------------

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print(
                    "ERROR: Could not read camera frame."
                )

                break

            # ------------------------------------------------
            # Mirror
            # ------------------------------------------------

            frame = cv2.flip(
                frame,
                1
            )

            # =================================================
            # FACE
            # =================================================

            face_data = (
                face_analyzer.detect(
                    frame
                )
            )

            if face_data is not None:

                face_analyzer.draw_box(
                    frame,
                    face_data
                )

                # --------------------------------------------
                # DeepFace every N frames
                # --------------------------------------------

                if (
                    frame_count %
                    config.FACE_ANALYSIS_INTERVAL
                    == 0
                ):

                    result = (
                        face_analyzer.analyze(
                            face_data["crop"]
                        )
                    )

                    if result is not None:

                        # ====================================
                        # GENDER
                        # ====================================

                        gender = result[
                            "gender"
                        ]

                        gender_confidence = result[
                            "gender_confidence"
                        ]

                        if (
                            gender
                            and
                            gender_confidence >=
                            config.MIN_GENDER_CONFIDENCE
                        ):

                            current_gender = (
                                gender_smoother.update(
                                    gender
                                )
                            )

                            current_gender_confidence = (
                                gender_confidence_smoother.update(
                                    gender_confidence
                                )
                            )

                        # ====================================
                        # EMOTION
                        # ====================================

                        emotion = result[
                            "emotion"
                        ]

                        emotion_confidence = result[
                            "emotion_confidence"
                        ]

                        if (
                            emotion
                            and
                            emotion_confidence >=
                            config.MIN_EMOTION_CONFIDENCE
                        ):

                            current_emotion = (
                                emotion_smoother.update(
                                    emotion
                                )
                            )

                            current_emotion_confidence = (
                                emotion_confidence_smoother.update(
                                    emotion_confidence
                                )
                            )

            # =================================================
            # HANDS
            # =================================================

            hands = hand_analyzer.process(
                frame
            )

            # Draw
            hand_analyzer.draw(
                frame,
                hands
            )

            # -------------------------------------------------
            # Smooth individual hands
            # -------------------------------------------------

            displayed_hands = []

            for hand in hands:

                label = hand[
                    "handedness"
                ]

                gesture = hand[
                    "gesture"
                ]

                confidence = hand[
                    "confidence"
                ]

                # ---------------------------------------------
                # LEFT
                # ---------------------------------------------

                if label == "Left":

                    stable_gesture = (
                        left_gesture_smoother.update(
                            gesture
                        )
                    )

                # ---------------------------------------------
                # RIGHT
                # ---------------------------------------------

                elif label == "Right":

                    stable_gesture = (
                        right_gesture_smoother.update(
                            gesture
                        )
                    )

                else:

                    stable_gesture = gesture

                displayed_hands.append({

                    "handedness":
                        label,

                    "gesture":
                        stable_gesture,

                    "confidence":
                        confidence
                })

            # =================================================
            # THUMBNAIL
            # =================================================

            if face_data is not None:

                crop = face_data[
                    "crop"
                ]

                if crop is not None and crop.size > 0:

                    try:

                        thumbnail = cv2.resize(
                            crop,
                            (
                                config.THUMBNAIL_SIZE,
                                config.THUMBNAIL_SIZE
                            )
                        )

                        frame[
                            10:
                            10 + config.THUMBNAIL_SIZE,

                            frame.shape[1] -
                            config.THUMBNAIL_SIZE -
                            10:
                            frame.shape[1] -
                            10
                        ] = thumbnail

                        cv2.rectangle(

                            frame,

                            (
                                frame.shape[1] -
                                config.THUMBNAIL_SIZE -
                                10,

                                10
                            ),

                            (
                                frame.shape[1] - 10,

                                10 +
                                config.THUMBNAIL_SIZE
                            ),

                            (255, 255, 255),

                            2
                        )

                    except Exception:
                        pass

            # =================================================
            # UI
            # =================================================

            frame = draw_panel(

                frame,

                current_gender,

                current_gender_confidence,

                current_emotion,

                current_emotion_confidence,

                displayed_hands
            )

            # =================================================
            # DISPLAY
            # =================================================

            cv2.imshow(
                config.WINDOW_NAME,
                frame
            )

            frame_count += 1

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                break

    finally:

        print()
        print("Closing...")

        cap.release()

        cv2.destroyAllWindows()

        face_analyzer.close()

        hand_analyzer.close()

        print("Program stopped.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()