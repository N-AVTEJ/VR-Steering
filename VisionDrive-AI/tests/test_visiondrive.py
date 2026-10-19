"""Unit tests for VisionDrive-AI."""

import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure VisionDrive-AI root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.camera import Camera
from src.vision import Vision
from src.gestures import (
    count_curled_fingers,
    count_extended_fingers,
    get_finger_ratios,
    is_fist,
    is_open_palm,
)


class LandmarkMock:
    """Mock for a MediaPipe landmark point."""
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def HasField(self, field: str) -> bool:
        return False


class HandLandmarksMock:
    """Mock for MediaPipe NormalizedLandmarkList."""
    def __init__(self, landmarks) -> None:
        self.landmark = landmarks


def create_dummy_hand(finger_ratios_target: dict) -> HandLandmarksMock:
    """
    Construct a dummy hand with 21 landmarks where wrist is at (0.5, 0.8),
    and each finger's MCP and TIP distances produce the requested ratio.
    Target ratio: tip_dist / mcp_dist.
    """
    landmarks = [LandmarkMock(0.5, 0.8, 0.0) for _ in range(21)]
    wrist_x, wrist_y = 0.5, 0.8

    finger_map = {
        "index": (5, 8, 0.45),
        "middle": (9, 12, 0.50),
        "ring": (13, 16, 0.55),
        "pinky": (17, 20, 0.60),
    }

    for finger, (mcp_idx, tip_idx, x_col) in finger_map.items():
        mcp_dist = 0.30
        target_ratio = finger_ratios_target.get(finger, 1.0)
        tip_dist = mcp_dist * target_ratio

        landmarks[mcp_idx] = LandmarkMock(x_col, wrist_y - mcp_dist)
        landmarks[tip_idx] = LandmarkMock(x_col, wrist_y - tip_dist)

    return HandLandmarksMock(landmarks)


class TestConfig(unittest.TestCase):
    """Test configuration constants."""

    def test_constants_exist_and_match_specs(self):
        self.assertEqual(config.CURL_RATIO, 0.75)
        self.assertEqual(config.EXTEND_RATIO, 1.15)
        self.assertEqual(config.FIST_MIN_CURLED, 3)
        self.assertEqual(config.PALM_MIN_EXTENDED, 3)
        self.assertEqual(config.CAMERA_WIDTH, 640)
        self.assertEqual(config.CAMERA_HEIGHT, 480)
        self.assertEqual(config.CAMERA_INDEX, 0)


class TestGestures(unittest.TestCase):
    """Test gesture recognition and landmark ratio calculations."""

    def test_open_palm_detection(self):
        # All 4 fingers extended (ratio = 2.0 > 1.15)
        open_hand = create_dummy_hand({
            "index": 2.0,
            "middle": 2.0,
            "ring": 2.0,
            "pinky": 2.0,
        })
        self.assertEqual(count_curled_fingers(open_hand), 0)
        self.assertEqual(count_extended_fingers(open_hand), 4)
        self.assertTrue(is_open_palm(open_hand))
        self.assertFalse(is_fist(open_hand))

    def test_fist_detection(self):
        # All 4 fingers curled (ratio = 0.5 < 0.75)
        fist_hand = create_dummy_hand({
            "index": 0.5,
            "middle": 0.5,
            "ring": 0.5,
            "pinky": 0.5,
        })
        self.assertEqual(count_curled_fingers(fist_hand), 4)
        self.assertEqual(count_extended_fingers(fist_hand), 0)
        self.assertTrue(is_fist(fist_hand))
        self.assertFalse(is_open_palm(fist_hand))

    def test_threshold_boundaries_fist(self):
        # Exactly 3 curled fingers, 1 extended -> Fist should be True
        hand_3_curled = create_dummy_hand({
            "index": 0.5,
            "middle": 0.5,
            "ring": 0.5,
            "pinky": 1.5,
        })
        self.assertEqual(count_curled_fingers(hand_3_curled), 3)
        self.assertTrue(is_fist(hand_3_curled))
        self.assertFalse(is_open_palm(hand_3_curled))

        # Only 2 curled fingers -> Fist should be False
        hand_2_curled = create_dummy_hand({
            "index": 0.5,
            "middle": 0.5,
            "ring": 1.0,
            "pinky": 1.0,
        })
        self.assertEqual(count_curled_fingers(hand_2_curled), 2)
        self.assertFalse(is_fist(hand_2_curled))

    def test_threshold_boundaries_palm(self):
        # Exactly 3 extended fingers, 1 curled -> Palm should be True
        hand_3_extended = create_dummy_hand({
            "index": 1.5,
            "middle": 1.5,
            "ring": 1.5,
            "pinky": 0.5,
        })
        self.assertEqual(count_extended_fingers(hand_3_extended), 3)
        self.assertTrue(is_open_palm(hand_3_extended))
        self.assertFalse(is_fist(hand_3_extended))

        # Only 2 extended fingers -> Palm should be False
        hand_2_extended = create_dummy_hand({
            "index": 1.5,
            "middle": 1.5,
            "ring": 1.0,
            "pinky": 1.0,
        })
        self.assertEqual(count_extended_fingers(hand_2_extended), 2)
        self.assertFalse(is_open_palm(hand_2_extended))

    def test_intermediate_deadzone(self):
        # Finger ratios strictly between 0.75 and 1.15
        neutral_hand = create_dummy_hand({
            "index": 0.95,
            "middle": 0.95,
            "ring": 0.95,
            "pinky": 0.95,
        })
        self.assertEqual(count_curled_fingers(neutral_hand), 0)
        self.assertEqual(count_extended_fingers(neutral_hand), 0)
        self.assertFalse(is_fist(neutral_hand))
        self.assertFalse(is_open_palm(neutral_hand))

    def test_boundary_values_exact_thresholds(self):
        # Exactly 0.75 is NOT curled (< 0.75)
        # Exactly 1.15 is NOT extended (> 1.15)
        boundary_hand = create_dummy_hand({
            "index": 0.75,
            "middle": 1.15,
            "ring": 0.75,
            "pinky": 1.15,
        })
        self.assertEqual(count_curled_fingers(boundary_hand), 0)
        self.assertEqual(count_extended_fingers(boundary_hand), 0)

    def test_empty_and_none_landmarks(self):
        self.assertFalse(is_fist(None))
        self.assertFalse(is_open_palm(None))
        self.assertEqual(count_curled_fingers(None), 0)
        self.assertEqual(count_extended_fingers(None), 0)
        self.assertEqual(get_finger_ratios(None), {})
        self.assertEqual(get_finger_ratios([]), {})

    def test_zero_division_safety(self):
        # MCP at wrist location (distance = 0)
        landmarks = [LandmarkMock(0.5, 0.8) for _ in range(21)]
        ratios = get_finger_ratios(HandLandmarksMock(landmarks))
        for ratio in ratios.values():
            self.assertEqual(ratio, 0.0)

    def test_degenerate_hand_not_fist_or_palm(self):
        # A collapsed/degenerate hand where all landmarks are at the same point
        # must NOT be classified as a fist or an open palm.
        collapsed_landmarks = [LandmarkMock(0.5, 0.8) for _ in range(21)]
        collapsed_hand = HandLandmarksMock(collapsed_landmarks)
        self.assertFalse(is_fist(collapsed_hand))
        self.assertFalse(is_open_palm(collapsed_hand))
        self.assertEqual(count_curled_fingers(collapsed_hand), 0)
        self.assertEqual(count_extended_fingers(collapsed_hand), 0)


class TestCamera(unittest.TestCase):
    """Test Camera class functionality."""

    @patch("cv2.VideoCapture")
    def test_camera_init_and_explicit_resolution(self, mock_cv_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cv_capture.return_value = mock_cap_instance

        cam = Camera(camera_index=0, width=640, height=480)
        mock_cv_capture.assert_called_once_with(0)
        # Verify cv2.CAP_PROP_FRAME_WIDTH/HEIGHT explicitly set
        calls = mock_cap_instance.set.call_args_list
        self.assertEqual(len(calls), 2)
        cam.release()

    @patch("cv2.VideoCapture")
    def test_camera_open_failure_raises_error(self, mock_cv_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False
        mock_cv_capture.return_value = mock_cap_instance

        with self.assertRaises(RuntimeError) as ctx:
            Camera(camera_index=99)
        self.assertIn("Failed to open camera", str(ctx.exception))

    @patch("cv2.VideoCapture")
    def test_camera_horizontal_flip(self, mock_cv_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        # Asymmetric test frame: left half 1, right half 2
        test_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        test_frame[:, :5] = 100
        test_frame[:, 5:] = 200
        mock_cap_instance.read.return_value = (True, test_frame)
        mock_cv_capture.return_value = mock_cap_instance

        cam = Camera()
        ret, flipped = cam.read()
        self.assertTrue(ret)
        # Flipped horizontally: left half should now be 200, right half 100
        self.assertEqual(flipped[0, 0, 0], 200)
        self.assertEqual(flipped[0, 9, 0], 100)
        cam.release()

    @patch("cv2.VideoCapture")
    def test_safe_release_and_context_manager(self, mock_cv_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cv_capture.return_value = mock_cap_instance

        with Camera() as cam:
            self.assertIsNotNone(cam.cap)
        # Verify released on exit
        mock_cap_instance.release.assert_called_once()
        self.assertIsNone(cam.cap)

        # Calling release again is safe and does not raise
        cam.release()


class TestVision(unittest.TestCase):
    """Test Vision MediaPipe wrapper."""

    @patch("mediapipe.solutions.hands.Hands")
    def test_sorting_by_landmark_x_ascending(self, mock_hands_class):
        mock_hands_instance = MagicMock()
        mock_hands_class.return_value = mock_hands_instance

        vision = Vision()

        # Two hands: Hand 1 has wrist x = 0.8, Hand 2 has wrist x = 0.2
        hand1 = HandLandmarksMock([LandmarkMock(0.8, 0.5) for _ in range(21)])
        hand2 = HandLandmarksMock([LandmarkMock(0.2, 0.5) for _ in range(21)])

        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [hand1, hand2]
        # Even if handedness labels say otherwise, process() sorts strictly by actual landmark x
        mock_results.multi_handedness = [MagicMock(classification=[MagicMock(label="Left")]),
                                         MagicMock(classification=[MagicMock(label="Right")])]
        mock_hands_instance.process.return_value = mock_results

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sorted_hands = vision.process(dummy_frame)

        self.assertEqual(len(sorted_hands), 2)
        # Leftmost hand (hand2 with x=0.2) must come first
        self.assertEqual(sorted_hands[0], hand2)
        self.assertEqual(sorted_hands[1], hand1)
        vision.close()

    @patch("mediapipe.solutions.hands.Hands")
    def test_sorting_by_centroid_when_wrists_cross(self, mock_hands_class):
        # When wrists cross or angle inwards:
        # Hand A is on the left (centroid x = 0.30), but its wrist is at x = 0.48.
        # Hand B is on the right (centroid x = 0.65), but its wrist is at x = 0.44.
        mock_hands_instance = MagicMock()
        mock_hands_class.return_value = mock_hands_instance

        vision = Vision()

        hand_a_lms = [LandmarkMock(0.29, 0.5) for _ in range(21)]
        hand_a_lms[0] = LandmarkMock(0.48, 0.8) # wrist is 0.48
        hand_a = HandLandmarksMock(hand_a_lms)  # centroid is ~0.30

        hand_b_lms = [LandmarkMock(0.66, 0.5) for _ in range(21)]
        hand_b_lms[0] = LandmarkMock(0.44, 0.8) # wrist is 0.44
        hand_b = HandLandmarksMock(hand_b_lms)  # centroid is ~0.65

        mock_results = MagicMock()
        # Pass in reverse order: [hand_b, hand_a]
        mock_results.multi_hand_landmarks = [hand_b, hand_a]
        mock_hands_instance.process.return_value = mock_results

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sorted_hands = vision.process(dummy_frame)

        self.assertEqual(len(sorted_hands), 2)
        # Hand A (leftmost by centroid) must be first, despite wrist x being larger
        self.assertEqual(sorted_hands[0], hand_a)
        self.assertEqual(sorted_hands[1], hand_b)
        vision.close()

    @patch("mediapipe.solutions.hands.Hands")
    def test_process_empty_or_none(self, mock_hands_class):
        mock_hands_class.return_value = MagicMock()
        vision = Vision()
        self.assertEqual(vision.process(None), [])
        self.assertEqual(vision.process(np.array([])), [])
        vision.close()

    def test_draw_landmarks_guards_none_and_empty(self):
        vision = Vision()
        self.assertIsNone(vision.draw_landmarks(None, []))
        dummy = np.zeros((10, 10, 3), dtype=np.uint8)
        out = vision.draw_landmarks(dummy, None)
        self.assertIsNotNone(out)
        vision.close()


class TestMainLogic(unittest.TestCase):
    """Test helper logic and workflow of main.py."""

    def test_get_hand_x(self):
        import main
        hand = HandLandmarksMock([LandmarkMock(0.3, 0.5) for _ in range(21)])
        self.assertAlmostEqual(main._get_hand_x(hand), 0.3)
        self.assertEqual(main._get_hand_x(None), 0.5)
        self.assertEqual(main._get_hand_x([]), 0.5)

    @patch("cv2.destroyAllWindows")
    @patch("cv2.imshow")
    @patch("cv2.waitKey", return_value=ord("q"))
    @patch("main.Camera")
    @patch("main.Vision")
    def test_main_execution_clean_exit(
        self, mock_vision_cls, mock_camera_cls, mock_waitkey, mock_imshow, mock_destroy
    ):
        import main

        mock_cam_inst = MagicMock()
        mock_cam_inst.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_camera_cls.return_value = mock_cam_inst

        mock_vis_inst = MagicMock()
        mock_vis_inst.process.return_value = []
        mock_vision_cls.return_value = mock_vis_inst

        # Running main should process 1 frame, detect 'q', and exit cleanly
        main.main()

        mock_cam_inst.release.assert_called_once()
        mock_vis_inst.close.assert_called_once()
        mock_destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
