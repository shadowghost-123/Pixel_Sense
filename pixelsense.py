"""
Real-Time Face + Hand Analyzer
================================

Python:
    3.13

Tested environment:
    MediaPipe 0.10.35
    TensorFlow 2.21.0
    DeepFace 0.0.100
    OpenCV

Features:
    - Real-time face detection
    - Gender estimation
    - Emotion estimation
    - Hand landmark tracking
    - Stable gesture recognition
    - Face thumbnail
    - Temporal smoothing

Press:
    Q -> Quit
"""

import os
import cv2
import time
from collections import Counter, deque

import mediapipe as mp
from deepface import DeepFace


# ============================================================
# CONFIGURATION
# ============================================================

ANALYZE_EVERY_N_FRAMES = 30

THUMB_SIZE = 120

# Number of previous gesture predictions used for smoothing
GESTURE_HISTORY_SIZE = 8

# Number of previous emotion predictions used for smoothing
EMOTION_HISTORY_SIZE = 5

# Minimum face size for DeepFace analysis
MIN_FACE_SIZE = 80

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

FACE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "face_detector.tflite"
)

HAND_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hand_landmarker.task"
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.exists(FACE_MODEL_PATH):
    raise FileNotFoundError(
        f"""
Face detector model not found:

{FACE_MODEL_PATH}

Make sure this file exists:
models/face_detector.tflite
"""
    )

if not os.path.exists(HAND_MODEL_PATH):
    raise FileNotFoundError(
        f"""
Hand landmarker model not found:

{HAND_MODEL_PATH}

Make sure this file exists:
models/hand_landmarker.task
"""
    )


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

RunningMode = mp.tasks.vision.RunningMode


# ============================================================
# FACE DETECTOR
# ============================================================

face_options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    min_detection_confidence=0.6
)

face_detector = FaceDetector.create_from_options(
    face_options
)


# ============================================================
# HAND LANDMARKER
# ============================================================

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hand_landmarker = HandLandmarker.create_from_options(
    hand_options
)


# ============================================================
# GESTURE HISTORY
# ============================================================

gesture_history = deque(
    maxlen=GESTURE_HISTORY_SIZE
)

emotion_history = deque(
    maxlen=EMOTION_HISTORY_SIZE
)


# ============================================================
# STABLE VALUE
# ============================================================

def get_stable_value(history):
    """
    Returns the most common value in the recent history.
    """

    if not history:
        return None

    counter = Counter(history)

    return counter.most_common(1)[0][0]


# ============================================================
# GESTURE CLASSIFIER
# ============================================================

def classify_gesture(hand_landmarks, handedness_label):

    lm = hand_landmarks

    # --------------------------------------------------------
    # Thumb
    # --------------------------------------------------------

    if handedness_label == "Right":
        thumb_extended = lm[4].x < lm[3].x
    else:
        thumb_extended = lm[4].x > lm[3].x

    fingers = [
        1 if thumb_extended else 0
    ]

    # --------------------------------------------------------
    # Four fingers
    # --------------------------------------------------------

    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18)
    ]

    for tip_id, pip_id in finger_pairs:

        extended = (
            lm[tip_id].y <
            lm[pip_id].y
        )

        fingers.append(
            1 if extended else 0
        )

    total = sum(fingers)

    # --------------------------------------------------------
    # Gesture rules
    # --------------------------------------------------------

    if total == 0:
        return "Fist"

    if total == 5:
        return "Open Palm"

    if fingers == [0, 1, 1, 0, 0]:
        return "Peace / Victory"

    if fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"

    if fingers == [0, 1, 0, 0, 0]:
        return "Pointing"

    if fingers == [1, 0, 0, 0, 1]:
        return "Call Me"

    return f"{total} Fingers"


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand_landmarks(frame, landmarks):

    h, w, _ = frame.shape

    connections = [

        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        # Index
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        # Middle
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        # Ring
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        # Pinky
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        # Palm
        (0, 17)
    ]

    # --------------------------------------------------------
    # Lines
    # --------------------------------------------------------

    for start, end in connections:

        x1 = int(
            landmarks[start].x * w
        )

        y1 = int(
            landmarks[start].y * h
        )

        x2 = int(
            landmarks[end].x * w
        )

        y2 = int(
            landmarks[end].y * h
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

    # --------------------------------------------------------
    # Points
    # --------------------------------------------------------

    for landmark in landmarks:

        x = int(
            landmark.x * w
        )

        y = int(
            landmark.y * h
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )


# ============================================================
# EXPAND FACE BOX
# ============================================================

def expand_face_box(
    x,
    y,
    width,
    height,
    frame_width,
    frame_height
):
    """
    Expands the detected face region slightly.

    DeepFace performs better when it receives
    some surrounding facial context.
    """

    padding_x = int(width * 0.20)
    padding_y = int(height * 0.25)

    x1 = max(
        0,
        x - padding_x
    )

    y1 = max(
        0,
        y - padding_y
    )

    x2 = min(
        frame_width,
        x + width + padding_x
    )

    y2 = min(
        frame_height,
        y + height + padding_y
    )

    return x1, y1, x2, y2


# ============================================================
# DEEPFACE ANALYSIS
# ============================================================

def analyze_face(face_crop):

    try:

        if face_crop is None:
            return None, None

        h, w = face_crop.shape[:2]

        if (
            w < MIN_FACE_SIZE
            or h < MIN_FACE_SIZE
        ):
            return None, None

        # ----------------------------------------------------
        # Improve crop resolution
        # ----------------------------------------------------

        enlarged = cv2.resize(
            face_crop,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC
        )

        # ----------------------------------------------------
        # DeepFace
        # ----------------------------------------------------

        result = DeepFace.analyze(
            img_path=enlarged,
            actions=[
                "emotion",
                "gender"
            ],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True
        )

        if isinstance(result, list):
            result = result[0]

        # ----------------------------------------------------
        # Emotion
        # ----------------------------------------------------

        emotion = result.get(
            "dominant_emotion",
            None
        )

        # ----------------------------------------------------
        # Gender
        # ----------------------------------------------------

        gender = result.get(
            "dominant_gender",
            None
        )

        # ----------------------------------------------------
        # Confidence information
        # ----------------------------------------------------

        emotion_scores = result.get(
            "emotion",
            {}
        )

        gender_scores = result.get(
            "gender",
            {}
        )

        return (
            gender,
            emotion,
            gender_scores,
            emotion_scores
        )

    except Exception as e:

        print(
            "\nDeepFace error:",
            str(e)
        )

        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 55)
    print(" REAL-TIME FACE + HAND ANALYZER")
    print("=" * 55)
    print("Python     :", __import__("sys").version.split()[0])
    print("MediaPipe  :", mp.__version__)
    print()
    print("Starting camera...")
    print("Press Q to quit.")
    print("=" * 55)

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "ERROR: Camera could not be opened."
        )

        return

    # --------------------------------------------------------
    # Camera resolution
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    frame_count = 0

    # --------------------------------------------------------
    # Display values
    # --------------------------------------------------------

    current_gender = "Waiting..."

    current_emotion = "Waiting..."

    current_gesture = "No hand"

    last_analysis_time = 0

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Mirror
        frame = cv2.flip(
            frame,
            1
        )

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ====================================================
        # FACE DETECTION
        # ====================================================

        face_crop = None

        try:

            face_result = (
                face_detector.detect(
                    mp_image
                )
            )

            if face_result.detections:

                detection = (
                    face_result.detections[0]
                )

                box = detection.bounding_box

                x = max(
                    0,
                    box.origin_x
                )

                y = max(
                    0,
                    box.origin_y
                )

                bw = box.width
                bh = box.height

                # --------------------------------------------
                # Make sure dimensions are valid
                # --------------------------------------------

                bw = min(
                    bw,
                    w - x
                )

                bh = min(
                    bh,
                    h - y
                )

                if bw > 0 and bh > 0:

                    # ----------------------------------------
                    # Expanded face crop
                    # ----------------------------------------

                    x1, y1, x2, y2 = (
                        expand_face_box(
                            x,
                            y,
                            bw,
                            bh,
                            w,
                            h
                        )
                    )

                    face_crop = frame[
                        y1:y2,
                        x1:x2
                    ].copy()

                    # ----------------------------------------
                    # Face rectangle
                    # ----------------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # ----------------------------------------
                    # DeepFace analysis
                    # ----------------------------------------

                    if (
                        frame_count %
                        ANALYZE_EVERY_N_FRAMES
                        == 0
                    ):

                        analysis = (
                            analyze_face(
                                face_crop
                            )
                        )

                        if analysis is not None:

                            (
                                gender,
                                emotion,
                                gender_scores,
                                emotion_scores
                            ) = analysis

                            # ------------------------------
                            # Gender
                            # ------------------------------

                            if gender:

                                current_gender = (
                                    gender
                                )

                            # ------------------------------
                            # Emotion
                            # ------------------------------

                            if emotion:

                                emotion_history.append(
                                    emotion
                                )

                                stable_emotion = (
                                    get_stable_value(
                                        emotion_history
                                    )
                                )

                                if stable_emotion:

                                    current_emotion = (
                                        stable_emotion
                                    )

        except Exception as e:

            print(
                "\nFace error:",
                str(e)
            )

        # ====================================================
        # HAND DETECTION
        # ====================================================

        try:

            hand_result = (
                hand_landmarker.detect(
                    mp_image
                )
            )

            if hand_result.hand_landmarks:

                gesture_predictions = []

                for idx, landmarks in enumerate(
                    hand_result.hand_landmarks
                ):

                    # ----------------------------------------
                    # Draw
                    # ----------------------------------------

                    draw_hand_landmarks(
                        frame,
                        landmarks
                    )

                    # ----------------------------------------
                    # Handedness
                    # ----------------------------------------

                    handedness = "Right"

                    if (
                        hand_result.handedness
                        and
                        idx <
                        len(
                            hand_result.handedness
                        )
                    ):

                        handedness = (
                            hand_result
                            .handedness[idx][0]
                            .category_name
                        )

                    # ----------------------------------------
                    # Gesture
                    # ----------------------------------------

                    gesture = (
                        classify_gesture(
                            landmarks,
                            handedness
                        )
                    )

                    gesture_predictions.append(
                        f"{handedness}: {gesture}"
                    )

                # ------------------------------------------------
                # Smooth first detected hand
                # ------------------------------------------------

                if gesture_predictions:

                    # Remove handedness for history
                    raw_gesture = (
                        gesture_predictions[0]
                        .split(": ", 1)[-1]
                    )

                    gesture_history.append(
                        raw_gesture
                    )

                    stable_gesture = (
                        get_stable_value(
                            gesture_history
                        )
                    )

                    if stable_gesture:

                        current_gesture = (
                            stable_gesture
                        )

            else:

                # Don't instantly change the gesture.
                # This prevents flickering when MediaPipe
                # temporarily loses the hand.

                pass

        except Exception as e:

            print(
                "\nHand error:",
                str(e)
            )

        # ====================================================
        # FACE THUMBNAIL
        # ====================================================

        if (
            face_crop is not None
            and face_crop.size > 0
        ):

            try:

                thumb = cv2.resize(
                    face_crop,
                    (
                        THUMB_SIZE,
                        THUMB_SIZE
                    )
                )

                tx = (
                    w -
                    THUMB_SIZE -
                    10
                )

                ty = 10

                frame[
                    ty:ty + THUMB_SIZE,
                    tx:tx + THUMB_SIZE
                ] = thumb

                cv2.rectangle(
                    frame,
                    (
                        tx,
                        ty
                    ),
                    (
                        tx + THUMB_SIZE,
                        ty + THUMB_SIZE
                    ),
                    (255, 255, 255),
                    2
                )

            except Exception:
                pass

        # ====================================================
        # UI PANEL
        # ====================================================

        # Semi-transparent black panel
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (420, 125),
            (0, 0, 0),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.55,
            frame,
            0.45,
            0
        )

        # ====================================================
        # TEXT
        # ====================================================

        # # cv2.putText(
        #     frame,
        #     f"Gender  : {current_gender}",
        #     (15, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.65,
        #     (0, 255, 255),
        #     2
        # )

        cv2.putText(
            frame,
            f"Emotion : {current_emotion}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Gesture : {current_gesture}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 150, 0),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Face & Hand Analyzer",
            frame
        )

        frame_count += 1

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    face_detector.close()
    hand_landmarker.close()

    print()
    print("Program stopped.")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()