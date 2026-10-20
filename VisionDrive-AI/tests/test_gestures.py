"""Tests for gesture detection (fist, open palm, and ambiguous gestures)."""

import pytest
from src.config import CURL_RATIO, EXTEND_RATIO, FIST_MIN_CURLED, PALM_MIN_EXTENDED
from src.gestures import (
    count_curled_fingers,
    count_extended_fingers,
    get_finger_ratios,
    is_fist,
    is_open_palm,
)


class LandmarkPoint:
    """Mock MediaPipe NormalizedLandmark point."""
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class HandLandmarks:
    """Mock MediaPipe NormalizedLandmarkList."""
    def __init__(self, landmarks) -> None:
        self.landmark = landmarks


def build_mock_hand(finger_ratios: dict) -> HandLandmarks:
    """
    Construct a 21-landmark mock hand where wrist is at (0.5, 0.8),
    and each finger's MCP and TIP distances achieve the requested ratio:
        ratio = tip_distance / mcp_distance
    MediaPipe landmark indices:
        Wrist: 0
        Index: MCP 5, TIP 8
        Middle: MCP 9, TIP 12
        Ring: MCP 13, TIP 16
        Pinky: MCP 17, TIP 20
    """
    landmarks = [LandmarkPoint(0.5, 0.8, 0.0) for _ in range(21)]
    wrist_x, wrist_y = 0.5, 0.8
    mcp_dist = 0.25  # Exact binary floating point representation

    finger_map = {
        "index": (5, 8),
        "middle": (9, 12),
        "ring": (13, 16),
        "pinky": (17, 20),
    }

    for finger, (mcp_idx, tip_idx) in finger_map.items():
        ratio = finger_ratios.get(finger, 1.0)
        landmarks[mcp_idx] = LandmarkPoint(wrist_x, wrist_y - mcp_dist)
        landmarks[tip_idx] = LandmarkPoint(wrist_x, wrist_y - mcp_dist * ratio)

    return HandLandmarks(landmarks)


class TestGesturesClearFist:
    """Tests for clear fist gestures."""

    def test_all_four_fingers_curled(self):
        """When all 4 fingers are tightly curled, is_fist must be True and is_open_palm False."""
        hand = build_mock_hand({"index": 0.4, "middle": 0.4, "ring": 0.4, "pinky": 0.4})
        assert count_curled_fingers(hand) == 4
        assert count_extended_fingers(hand) == 0
        assert is_fist(hand) is True
        assert is_open_palm(hand) is False

    def test_three_fingers_curled_one_neutral(self):
        """3 fingers curled meets FIST_MIN_CURLED threshold."""
        hand = build_mock_hand({"index": 0.5, "middle": 0.5, "ring": 0.5, "pinky": 0.95})
        assert count_curled_fingers(hand) == 3
        assert is_fist(hand) is True
        assert is_open_palm(hand) is False


class TestGesturesClearOpenPalm:
    """Tests for clear open palm gestures."""

    def test_all_four_fingers_extended(self):
        """When all 4 fingers are fully extended, is_open_palm must be True and is_fist False."""
        hand = build_mock_hand({"index": 1.6, "middle": 1.7, "ring": 1.6, "pinky": 1.5})
        assert count_extended_fingers(hand) == 4
        assert count_curled_fingers(hand) == 0
        assert is_open_palm(hand) is True
        assert is_fist(hand) is False

    def test_three_fingers_extended_one_neutral(self):
        """3 fingers extended meets PALM_MIN_EXTENDED threshold."""
        hand = build_mock_hand({"index": 1.5, "middle": 1.6, "ring": 1.5, "pinky": 0.95})
        assert count_extended_fingers(hand) == 3
        assert is_open_palm(hand) is True
        assert is_fist(hand) is False


class TestGesturesAmbiguousAndEdgeCases:
    """Tests for ambiguous gestures, boundaries, and degenerate inputs."""

    def test_two_curled_two_extended(self):
        """2 curled and 2 extended should be ambiguous: neither fist nor palm."""
        hand = build_mock_hand({"index": 0.4, "middle": 0.4, "ring": 1.6, "pinky": 1.6})
        assert count_curled_fingers(hand) == 2
        assert count_extended_fingers(hand) == 2
        assert is_fist(hand) is False
        assert is_open_palm(hand) is False

    def test_relaxed_neutral_hand(self):
        """All fingers in resting position (ratio ~0.95) should be neither fist nor palm."""
        hand = build_mock_hand({"index": 0.95, "middle": 0.95, "ring": 0.95, "pinky": 0.95})
        assert count_curled_fingers(hand) == 0
        assert count_extended_fingers(hand) == 0
        assert is_fist(hand) is False
        assert is_open_palm(hand) is False

    def test_boundary_values_at_exact_thresholds(self):
        """Exact threshold boundary values (CURL_RATIO=0.75, EXTEND_RATIO=1.15)."""
        # Exactly 0.75 is not strictly < 0.75, so not curled
        hand_curl_boundary = build_mock_hand({
            "index": CURL_RATIO,
            "middle": CURL_RATIO,
            "ring": CURL_RATIO,
            "pinky": CURL_RATIO,
        })
        assert count_curled_fingers(hand_curl_boundary) == 0
        assert is_fist(hand_curl_boundary) is False

        # Exactly 1.15 is not strictly > 1.15, so not extended
        hand_extend_boundary = build_mock_hand({
            "index": EXTEND_RATIO,
            "middle": EXTEND_RATIO,
            "ring": EXTEND_RATIO,
            "pinky": EXTEND_RATIO,
        })
        assert count_extended_fingers(hand_extend_boundary) == 0
        assert is_open_palm(hand_extend_boundary) is False

    def test_just_inside_thresholds(self):
        """Values just past thresholds: 0.749 is curled, 1.151 is extended."""
        hand_curled = build_mock_hand({
            "index": CURL_RATIO - 0.001,
            "middle": CURL_RATIO - 0.001,
            "ring": CURL_RATIO - 0.001,
            "pinky": 0.95,
        })
        assert count_curled_fingers(hand_curled) == 3
        assert is_fist(hand_curled) is True

        hand_extended = build_mock_hand({
            "index": EXTEND_RATIO + 0.001,
            "middle": EXTEND_RATIO + 0.001,
            "ring": EXTEND_RATIO + 0.001,
            "pinky": 0.95,
        })
        assert count_extended_fingers(hand_extended) == 3
        assert is_open_palm(hand_extended) is True

    def test_none_and_empty_inputs(self):
        """None or empty landmark collections must return False gracefully."""
        assert is_fist(None) is False
        assert is_open_palm(None) is False
        assert count_curled_fingers(None) == 0
        assert count_extended_fingers(None) == 0
        assert get_finger_ratios(None) == {}

        empty_hand = HandLandmarks([])
        assert is_fist(empty_hand) is False
        assert is_open_palm(empty_hand) is False

    def test_insufficient_landmarks(self):
        """Hand with fewer than 21 landmarks (e.g. truncated) returns False."""
        short_hand = HandLandmarks([LandmarkPoint(0.5, 0.5) for _ in range(10)])
        assert is_fist(short_hand) is False
        assert is_open_palm(short_hand) is False
        assert get_finger_ratios(short_hand) == {}

    def test_collapsed_degenerate_hand(self):
        """Hand where all landmarks are collapsed to wrist position."""
        collapsed = HandLandmarks([LandmarkPoint(0.5, 0.5) for _ in range(21)])
        assert is_fist(collapsed) is False
        assert is_open_palm(collapsed) is False
        assert count_curled_fingers(collapsed) == 0
        assert count_extended_fingers(collapsed) == 0

    def test_raw_list_of_tuples_accepted(self):
        """Gestures module should also accept raw lists of (x, y) coordinates."""
        wrist_y = 0.8
        coords = [(0.5, 0.8)] * 21
        # Make index, middle, ring, pinky curled
        for mcp, tip in [(5, 8), (9, 12), (13, 16), (17, 20)]:
            coords[mcp] = (0.5, wrist_y - 0.2)
            coords[tip] = (0.5, wrist_y - 0.08)  # ratio = 0.08 / 0.2 = 0.4 < 0.75
        assert is_fist(coords) is True
        assert is_open_palm(coords) is False

    def test_two_curled_two_neutral_is_not_fist(self):
        """2 curled fingers is strictly below FIST_MIN_CURLED=3 -> is_fist must be False."""
        hand = build_mock_hand({"index": 0.4, "middle": 0.4, "ring": 0.95, "pinky": 0.95})
        assert count_curled_fingers(hand) == 2
        assert is_fist(hand) is False
        assert is_open_palm(hand) is False

    def test_two_extended_two_neutral_is_not_palm(self):
        """2 extended fingers is strictly below PALM_MIN_EXTENDED=3 -> is_open_palm must be False."""
        hand = build_mock_hand({"index": 1.6, "middle": 1.6, "ring": 0.95, "pinky": 0.95})
        assert count_extended_fingers(hand) == 2
        assert is_open_palm(hand) is False
        assert is_fist(hand) is False

    def test_one_curled_finger_is_not_fist(self):
        """1 curled finger -> is_fist must be False."""
        hand = build_mock_hand({"index": 0.4, "middle": 0.95, "ring": 0.95, "pinky": 0.95})
        assert count_curled_fingers(hand) == 1
        assert is_fist(hand) is False

    def test_literal_21_coordinates_fixtures(self):
        """Test with literal 21-coordinate fixed numeric lists for fist, palm, and ambiguous gestures."""
        # 1. Clear Fist (wrist at (0.5, 0.8), MCPs at 0.6, tips at 0.7 -> ratio 0.5 < 0.75)
        fist_coords = [(0.5, 0.8)] * 21
        for mcp, tip in [(5, 8), (9, 12), (13, 16), (17, 20)]:
            fist_coords[mcp] = (0.5, 0.6)  # dist = 0.2
            fist_coords[tip] = (0.5, 0.7)  # dist = 0.1, ratio = 0.5
        assert is_fist(fist_coords) is True
        assert is_open_palm(fist_coords) is False

        # 2. Clear Open Palm (wrist at (0.5, 0.8), MCPs at 0.6, tips at 0.3 -> ratio 2.5 > 1.15)
        palm_coords = [(0.5, 0.8)] * 21
        for mcp, tip in [(5, 8), (9, 12), (13, 16), (17, 20)]:
            palm_coords[mcp] = (0.5, 0.6)  # dist = 0.2
            palm_coords[tip] = (0.5, 0.3)  # dist = 0.5, ratio = 2.5
        assert is_open_palm(palm_coords) is True
        assert is_fist(palm_coords) is False

        # 3. Ambiguous Peace Sign (index & middle extended: tip 0.3; ring & pinky curled: tip 0.7)
        peace_coords = [(0.5, 0.8)] * 21
        for mcp in [5, 9, 13, 17]:
            peace_coords[mcp] = (0.5, 0.6)
        peace_coords[8] = (0.5, 0.3)   # index extended
        peace_coords[12] = (0.5, 0.3)  # middle extended
        peace_coords[16] = (0.5, 0.7)  # ring curled
        peace_coords[20] = (0.5, 0.7)  # pinky curled
        assert is_fist(peace_coords) is False
        assert is_open_palm(peace_coords) is False

    def test_3d_coordinates_tuples_accepted(self):
        """Gestures module should accept 3D coordinate tuples (x, y, z)."""
        coords_3d = [(0.5, 0.8, 0.0)] * 21
        for mcp, tip in [(5, 8), (9, 12), (13, 16), (17, 20)]:
            coords_3d[mcp] = (0.5, 0.6, 0.05)
            coords_3d[tip] = (0.5, 0.3, 0.10)
        assert is_open_palm(coords_3d) is True
        assert is_fist(coords_3d) is False
