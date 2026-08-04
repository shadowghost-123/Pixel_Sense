"""
Configuration for Face + Hand Analyzer V2
"""

# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# ============================================================
# FACE DETECTION
# ============================================================

FACE_DETECTION_CONFIDENCE = 0.60

# Minimum face dimensions before DeepFace analysis
MIN_FACE_WIDTH = 80
MIN_FACE_HEIGHT = 80

# Expand detected face before sending to DeepFace
FACE_PADDING_X = 0.20
FACE_PADDING_Y = 0.25


# ============================================================
# DEEPFACE
# ============================================================

# Analyze face once every N frames
FACE_ANALYSIS_INTERVAL = 30

# DeepFace detector
DEEPFACE_DETECTOR = "opencv"

# Minimum confidence to display a prediction
MIN_GENDER_CONFIDENCE = 60.0
MIN_EMOTION_CONFIDENCE = 45.0


# ============================================================
# SMOOTHING
# ============================================================

GESTURE_HISTORY_SIZE = 7

EMOTION_HISTORY_SIZE = 5

GENDER_HISTORY_SIZE = 5


# ============================================================
# HAND DETECTION
# ============================================================

MAX_HANDS = 2

HAND_DETECTION_CONFIDENCE = 0.60

HAND_PRESENCE_CONFIDENCE = 0.50

HAND_TRACKING_CONFIDENCE = 0.50


# ============================================================
# UI
# ============================================================

THUMBNAIL_SIZE = 130

WINDOW_NAME = "Face + Hand Analyzer V2"


# ============================================================
# GESTURE
# ============================================================

# Minimum confidence required for gesture classification
GESTURE_CONFIDENCE_THRESHOLD = 0.55