"""Tunable configuration constants for VisionDrive-AI."""

# Gesture detection ratio thresholds
# Tip-to-wrist distance vs MCP-to-wrist distance ratio
CURL_RATIO = 0.75
EXTEND_RATIO = 1.15

# Minimum curled/extended fingers to trigger fist/palm (out of 4 non-thumb fingers)
FIST_MIN_CURLED = 3
PALM_MIN_EXTENDED = 3

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_INDEX = 0

# MediaPipe Hands detection & tracking settings
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# Driving & Steering controls
STEER_SENSITIVITY = 2.5
EMA_ALPHA = 0.35
LOSS_FRAMES_LIMIT = 8
STEER_DEADZONE = 0.08
STEER_FIRE_THRESHOLD = 0.18
