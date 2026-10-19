"""Gesture recognition module for VisionDrive-AI based on landmark distance ratios."""

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.config import (
        CURL_RATIO,
        EXTEND_RATIO,
        FIST_MIN_CURLED,
        PALM_MIN_EXTENDED,
    )
except ImportError:
    from config import (
        CURL_RATIO,
        EXTEND_RATIO,
        FIST_MIN_CURLED,
        PALM_MIN_EXTENDED,
    )

# MediaPipe landmark indices for 4 non-thumb fingers: (MCP_INDEX, TIP_INDEX)
WRIST_INDEX = 0
FINGER_LANDMARK_MAP: Dict[str, Tuple[int, int]] = {
    "index": (5, 8),
    "middle": (9, 12),
    "ring": (13, 16),
    "pinky": (17, 20),
}


def _extract_landmarks_list(hand_landmarks: Any) -> Optional[List[Any]]:
    """Safely extract landmark collection as an indexable list."""
    if hand_landmarks is None:
        return None
    if hasattr(hand_landmarks, "landmark"):
        return hand_landmarks.landmark
    if isinstance(hand_landmarks, (list, tuple)):
        return list(hand_landmarks)
    return None


def _calc_euclidean_dist(p1: Any, p2: Any) -> float:
    """Calculate 2D Euclidean distance between two landmarks."""
    x1 = p1.x if hasattr(p1, "x") else p1[0]
    y1 = p1.y if hasattr(p1, "y") else p1[1]
    x2 = p2.x if hasattr(p2, "x") else p2[0]
    y2 = p2.y if hasattr(p2, "y") else p2[1]
    return math.hypot(x1 - x2, y1 - y2)


def get_finger_ratios(hand_landmarks: Any) -> Dict[str, float]:
    """
    Compute tip-to-wrist distance vs mcp-to-wrist distance ratio
    for index, middle, ring, pinky fingers.

    Returns:
        Dict mapping finger name to distance ratio (tip_dist / mcp_dist).
        Empty dict if landmarks are invalid or fewer than 21 landmarks.
    """
    lms = _extract_landmarks_list(hand_landmarks)
    if not lms or len(lms) <= max(tip for _, tip in FINGER_LANDMARK_MAP.values()):
        return {}

    wrist = lms[WRIST_INDEX]
    ratios: Dict[str, float] = {}

    for finger_name, (mcp_idx, tip_idx) in FINGER_LANDMARK_MAP.items():
        mcp = lms[mcp_idx]
        tip = lms[tip_idx]

        d_wrist_mcp = _calc_euclidean_dist(wrist, mcp)
        d_wrist_tip = _calc_euclidean_dist(wrist, tip)

        if d_wrist_mcp < 1e-6:
            ratios[finger_name] = 0.0
        else:
            ratios[finger_name] = d_wrist_tip / d_wrist_mcp

    return ratios


def _is_degenerate_hand(lms: Optional[List[Any]]) -> bool:
    """Check if hand landmarks are collapsed to a single point or missing."""
    if not lms or len(lms) <= max(tip for _, tip in FINGER_LANDMARK_MAP.values()):
        return True
    wrist = lms[WRIST_INDEX]
    mcp_distances = [
        _calc_euclidean_dist(wrist, lms[mcp_idx])
        for mcp_idx, _ in FINGER_LANDMARK_MAP.values()
    ]
    return all(d < 1e-5 for d in mcp_distances)


def count_curled_fingers(hand_landmarks: Any) -> int:
    """
    Count curled fingers among index, middle, ring, pinky.
    A finger is curled when its tip-to-wrist distance ratio is less than CURL_RATIO.
    Degenerate hands (e.g. MCPs at wrist) return 0.
    """
    lms = _extract_landmarks_list(hand_landmarks)
    if _is_degenerate_hand(lms):
        return 0
    ratios = get_finger_ratios(hand_landmarks)
    if not ratios:
        return 0
    return sum(1 for ratio in ratios.values() if ratio < CURL_RATIO)


def count_extended_fingers(hand_landmarks: Any) -> int:
    """
    Count extended fingers among index, middle, ring, pinky.
    A finger is extended when its tip-to-wrist distance ratio is greater than EXTEND_RATIO.
    Degenerate hands (e.g. MCPs at wrist) return 0.
    """
    lms = _extract_landmarks_list(hand_landmarks)
    if _is_degenerate_hand(lms):
        return 0
    ratios = get_finger_ratios(hand_landmarks)
    if not ratios:
        return 0
    return sum(1 for ratio in ratios.values() if ratio > EXTEND_RATIO)


def is_fist(hand_landmarks: Any) -> bool:
    """
    Check if the given hand's landmarks represent a fist.
    True if curled-finger count >= FIST_MIN_CURLED.
    """
    if hand_landmarks is None:
        return False
    return count_curled_fingers(hand_landmarks) >= FIST_MIN_CURLED


def is_open_palm(hand_landmarks: Any) -> bool:
    """
    Check if the given hand's landmarks represent an open palm.
    True if extended-finger count >= PALM_MIN_EXTENDED.
    """
    if hand_landmarks is None:
        return False
    return count_extended_fingers(hand_landmarks) >= PALM_MIN_EXTENDED
