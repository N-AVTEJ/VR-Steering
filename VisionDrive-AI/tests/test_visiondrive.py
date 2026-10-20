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
from src.driving import (
    DrivingController,
    DrivingState,
    NEUTRAL,
    READY,
    DRIVING,
    BRAKING,
    clamp,
    get_hand_x,
    apply_deadzone,
    apply_ema,
    calculate_raw_steering,
    calculate_steering,
    calibrate,
    update,
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


def create_test_hand(x: float = 0.5, gesture: str = "relaxed") -> HandLandmarksMock:
    """Construct a mock hand at horizontal position x with specified gesture."""
    if gesture == "fist":
        ratios = {"index": 0.5, "middle": 0.5, "ring": 0.5, "pinky": 0.5}
    elif gesture == "palm":
        ratios = {"index": 2.0, "middle": 2.0, "ring": 2.0, "pinky": 2.0}
    else:  # relaxed / neutral
        ratios = {"index": 0.95, "middle": 0.95, "ring": 0.95, "pinky": 0.95}

    hand = create_dummy_hand(ratios)
    current_avg_x = sum(lm.x for lm in hand.landmark) / len(hand.landmark)
    offset_x = x - current_avg_x
    for lm in hand.landmark:
        lm.x += offset_x
    return hand


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
        self.assertEqual(config.STEER_SENSITIVITY, 2.5)
        self.assertEqual(config.EMA_ALPHA, 0.35)
        self.assertEqual(config.LOSS_FRAMES_LIMIT, 8)
        self.assertEqual(config.STEER_DEADZONE, 0.08)
        self.assertEqual(config.STEER_FIRE_THRESHOLD, 0.18)


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


class TestDriving(unittest.TestCase):
    """Test driving control, steering calculation, EMA smoothing, tracking loss, and state machine."""

    def setUp(self):
        self.controller = DrivingController()

    def test_clamp_helper(self):
        self.assertEqual(clamp(0.5, -1.0, 1.0), 0.5)
        self.assertEqual(clamp(1.5, -1.0, 1.0), 1.0)
        self.assertEqual(clamp(-1.5, -1.0, 1.0), -1.0)
        self.assertEqual(clamp(5.0, 0.0, 10.0), 5.0)
        self.assertEqual(clamp(-1.0, 0.0, 10.0), 0.0)

    def test_get_hand_x_helper(self):
        self.assertEqual(get_hand_x(0.42), 0.42)
        self.assertEqual(get_hand_x(None), 0.5)
        self.assertEqual(get_hand_x([]), 0.5)
        hand = HandLandmarksMock([LandmarkMock(0.35, 0.5) for _ in range(21)])
        self.assertAlmostEqual(get_hand_x(hand), 0.35)

    def test_apply_deadzone(self):
        # Strictly below deadzone (0.08) -> ignored (0.0)
        self.assertEqual(apply_deadzone(0.05, 0.08), 0.0)
        self.assertEqual(apply_deadzone(-0.05, 0.08), 0.0)
        self.assertEqual(apply_deadzone(0.0, 0.08), 0.0)
        # Equal to or above deadzone -> preserved
        self.assertEqual(apply_deadzone(0.08, 0.08), 0.08)
        self.assertEqual(apply_deadzone(0.25, 0.08), 0.25)
        self.assertEqual(apply_deadzone(-0.25, 0.08), -0.25)

    def test_apply_ema(self):
        # smoothed = alpha * curr + (1 - alpha) * prev
        # 0.35 * 1.0 + 0.65 * 0.0 = 0.35
        self.assertAlmostEqual(apply_ema(1.0, 0.0, 0.35), 0.35)
        # Next frame: 0.35 * 1.0 + 0.65 * 0.35 = 0.35 + 0.2275 = 0.5775
        self.assertAlmostEqual(apply_ema(1.0, 0.35, 0.35), 0.5775)

    def test_steering_formula_and_clamping(self):
        # neutral_x = 0.5, STEER_SENSITIVITY = 2.5
        # hand_x = 0.5 -> raw = 0.0
        self.assertAlmostEqual(calculate_raw_steering(0.5, 0.5, 2.5), 0.0)
        # hand_x = 0.7 -> (0.7 - 0.5) * 2.5 = 0.50
        self.assertAlmostEqual(calculate_raw_steering(0.7, 0.5, 2.5), 0.50)
        # hand_x = 0.3 -> (0.3 - 0.5) * 2.5 = -0.50
        self.assertAlmostEqual(calculate_raw_steering(0.3, 0.5, 2.5), -0.50)
        # Clamping at +1.0 for large displacement
        self.assertEqual(calculate_raw_steering(1.0, 0.5, 2.5), 1.0)
        # Clamping at -1.0 for large displacement
        self.assertEqual(calculate_raw_steering(0.0, 0.5, 2.5), -1.0)

    def test_calibration_sets_neutral_and_recenters_steering(self):
        ctrl = DrivingController(neutral_x=0.5)
        # Give it some smoothed steering
        hand_steer = create_test_hand(x=0.75, gesture="relaxed")
        ctrl.update(leftmost_hand=hand_steer)
        self.assertGreater(ctrl.smoothed_steering, 0.0)

        # Calibrate with hand at x=0.75
        calibrated_val = ctrl.calibrate(hand_steer)
        self.assertAlmostEqual(calibrated_val, 0.75, places=2)
        self.assertAlmostEqual(ctrl.neutral_x, 0.75, places=2)
        # Steering must be recentered to 0 immediately
        self.assertEqual(ctrl.smoothed_steering, 0.0)

        # Calling calibrate with None does not crash or overwrite neutral
        ctrl.calibrate(None)
        self.assertAlmostEqual(ctrl.neutral_x, 0.75, places=2)

    def test_steering_smoothing_persists_across_frames(self):
        ctrl = DrivingController(neutral_x=0.5, ema_alpha=0.35)
        # Step input hand at x=0.7 -> raw clamped is (0.7 - 0.5) * 2.5 = 0.5
        hand = create_test_hand(x=0.7, gesture="relaxed")

        # Frame 1: smoothed = 0.35 * 0.5 + 0.65 * 0.0 = 0.175
        ctrl.update(leftmost_hand=hand)
        val1 = ctrl.smoothed_steering
        self.assertAlmostEqual(val1, 0.175, places=2)

        # Frame 2: smoothed = 0.35 * 0.5 + 0.65 * 0.175 = 0.28875
        ctrl.update(leftmost_hand=hand)
        val2 = ctrl.smoothed_steering
        self.assertAlmostEqual(val2, 0.28875, places=2)
        self.assertGreater(val2, val1)

        # Frame 3: continues smoothly towards 0.5
        ctrl.update(leftmost_hand=hand)
        val3 = ctrl.smoothed_steering
        self.assertGreater(val3, val2)

    def test_deadzone_and_virtual_keys_firing(self):
        ctrl = DrivingController(
            neutral_x=0.5,
            steer_sensitivity=2.5,
            steer_deadzone=0.08,
            steer_fire_threshold=0.18,
        )
        # 1. Hand within deadzone: hand_x = 0.52 -> (0.52 - 0.5) * 2.5 = 0.05 < 0.08
        hand_deadzone = create_test_hand(x=0.52, gesture="relaxed")
        ctrl.update(leftmost_hand=hand_deadzone)
        self.assertEqual(ctrl.smoothed_steering, 0.0)
        self.assertNotIn("d", ctrl.active_keys)
        self.assertNotIn("a", ctrl.active_keys)

        # 2. Hand beyond deadzone but smoothed value below fire threshold (0.18)
        # hand_x = 0.55 -> raw = (0.55 - 0.5) * 2.5 = 0.125 (> 0.08)
        hand_small = create_test_hand(x=0.55, gesture="relaxed")
        ctrl.update(leftmost_hand=hand_small)
        # On frame 1: smoothed = 0.35 * 0.125 = 0.04375 (< 0.18)
        self.assertGreater(ctrl.smoothed_steering, 0.0)
        self.assertLess(ctrl.smoothed_steering, 0.18)
        self.assertNotIn("d", ctrl.active_keys)

        # 3. Hand far right: hand_x = 0.8 -> raw = 0.75
        # Repeat until smoothed exceeds 0.18
        hand_right = create_test_hand(x=0.8, gesture="relaxed")
        for _ in range(5):
            ctrl.update(leftmost_hand=hand_right)
        self.assertGreater(ctrl.smoothed_steering, 0.18)
        self.assertIn("d", ctrl.active_keys)
        self.assertNotIn("a", ctrl.active_keys)

        # 4. Hand far left: hand_x = 0.2 -> raw = -0.75
        hand_left = create_test_hand(x=0.2, gesture="relaxed")
        for _ in range(10):
            ctrl.update(leftmost_hand=hand_left)
        self.assertLess(ctrl.smoothed_steering, -0.18)
        self.assertIn("a", ctrl.active_keys)
        self.assertNotIn("d", ctrl.active_keys)

    def test_tracking_loss_grace_period_and_limit(self):
        ctrl = DrivingController(loss_frames_limit=8)
        hand = create_test_hand(x=0.8, gesture="fist")
        ctrl.update(leftmost_hand=hand)
        self.assertEqual(ctrl.state, DrivingState.DRIVING)
        self.assertGreater(ctrl.smoothed_steering, 0.0)

        # Frames 1 through 7: tracking loss grace period
        for frame_idx in range(1, 8):
            ctrl.update(leftmost_hand=None)
            self.assertEqual(ctrl.loss_frame_count, frame_idx)
            self.assertEqual(ctrl.state, DrivingState.DRIVING)
            self.assertGreater(ctrl.smoothed_steering, 0.0)

        # Frame 8: reaches LOSS_FRAMES_LIMIT=8 consecutive frames
        ctrl.update(leftmost_hand=None)
        self.assertEqual(ctrl.loss_frame_count, 8)
        self.assertEqual(ctrl.state, DrivingState.NEUTRAL)
        self.assertEqual(ctrl.smoothed_steering, 0.0)
        self.assertEqual(ctrl.active_keys, [])

        # Recovery when hand returns
        ctrl.update(leftmost_hand=hand)
        self.assertEqual(ctrl.loss_frame_count, 0)
        self.assertEqual(ctrl.state, DrivingState.DRIVING)

    def test_state_machine_driving_fist(self):
        ctrl = DrivingController()
        left_fist = create_test_hand(x=0.5, gesture="fist")
        ctrl.update(leftmost_hand=left_fist, rightmost_hand=None)
        self.assertEqual(ctrl.state, DrivingState.DRIVING)
        self.assertIn("w", ctrl.active_keys)
        self.assertNotIn("s", ctrl.active_keys)

    def test_state_machine_braking_open_palm(self):
        ctrl = DrivingController()
        left_palm = create_test_hand(x=0.3, gesture="palm")
        right_palm = create_test_hand(x=0.7, gesture="palm")
        ctrl.update(leftmost_hand=left_palm, rightmost_hand=right_palm)
        self.assertEqual(ctrl.state, DrivingState.BRAKING)
        self.assertIn("s", ctrl.active_keys)
        self.assertNotIn("w", ctrl.active_keys)

        # Single-hand backwards compatibility: 1 palm -> BRAKING
        ctrl_single = DrivingController()
        single_palm = create_test_hand(x=0.5, gesture="palm")
        ctrl_single.update(leftmost_hand=single_palm, rightmost_hand=None)
        self.assertEqual(ctrl_single.state, DrivingState.BRAKING)
        self.assertIn("s", ctrl_single.active_keys)

    def test_state_machine_braking_overrides_driving(self):
        # Both fist condition (steering hand) and brake condition (brake hand) true
        ctrl = DrivingController()
        left_fist = create_test_hand(x=0.3, gesture="fist")
        right_palm = create_test_hand(x=0.7, gesture="palm")
        ctrl.update(leftmost_hand=left_fist, rightmost_hand=right_palm)
        # Braking MUST override driving!
        self.assertEqual(ctrl.state, DrivingState.BRAKING)
        self.assertIn("s", ctrl.active_keys)
        self.assertNotIn("w", ctrl.active_keys)

    def test_state_machine_both_fists_shows_driving(self):
        ctrl = DrivingController()
        left_fist = create_test_hand(x=0.3, gesture="fist")
        right_fist = create_test_hand(x=0.7, gesture="fist")
        ctrl.update(leftmost_hand=left_fist, rightmost_hand=right_fist)
        # Both fists -> dual-hand drive -> DRIVING (presses 'w')
        self.assertEqual(ctrl.state, DrivingState.DRIVING)
        self.assertIn("w", ctrl.active_keys)
        self.assertNotIn("s", ctrl.active_keys)

    def test_state_machine_both_palms_shows_braking(self):
        ctrl = DrivingController()
        left_palm = create_test_hand(x=0.3, gesture="palm")
        right_palm = create_test_hand(x=0.7, gesture="palm")
        ctrl.update(leftmost_hand=left_palm, rightmost_hand=right_palm)
        # Both palms -> dual-hand brake -> BRAKING (presses 's')
        self.assertEqual(ctrl.state, DrivingState.BRAKING)
        self.assertIn("s", ctrl.active_keys)
        self.assertNotIn("w", ctrl.active_keys)

    def test_state_machine_initial_neutral_before_any_hand(self):
        ctrl = DrivingController()
        self.assertEqual(ctrl.state, DrivingState.NEUTRAL)
        ctrl.update(leftmost_hand=None, rightmost_hand=None)
        self.assertEqual(ctrl.state, DrivingState.NEUTRAL)
        self.assertEqual(ctrl.active_keys, [])

    def test_state_transitions_between_driving_braking_and_ready(self):
        ctrl = DrivingController()
        left_fist = create_test_hand(x=0.3, gesture="fist")
        right_palm = create_test_hand(x=0.7, gesture="palm")
        right_fist = create_test_hand(x=0.7, gesture="fist")
        left_relaxed = create_test_hand(x=0.3, gesture="relaxed")

        # 1. Start in READY
        ctrl.update(leftmost_hand=left_relaxed)
        self.assertEqual(ctrl.state, DrivingState.READY)

        # 2. Transition READY -> DRIVING
        ctrl.update(leftmost_hand=left_fist)
        self.assertEqual(ctrl.state, DrivingState.DRIVING)

        # 3. Transition DRIVING -> BRAKING directly (right hand raises palm)
        ctrl.update(leftmost_hand=left_fist, rightmost_hand=right_palm)
        self.assertEqual(ctrl.state, DrivingState.BRAKING)

        # 4. Transition BRAKING -> DRIVING directly (right hand lowers)
        ctrl.update(leftmost_hand=left_fist, rightmost_hand=None)
        self.assertEqual(ctrl.state, DrivingState.DRIVING)

        # 5. Transition DRIVING (right hand closes into fist -> both fists)
        ctrl.update(leftmost_hand=left_fist, rightmost_hand=right_fist)
        self.assertEqual(ctrl.state, DrivingState.DRIVING)

        # 6. Transition DRIVING -> READY (relaxing hand to coast)
        ctrl.update(leftmost_hand=left_relaxed, rightmost_hand=right_fist)
        self.assertEqual(ctrl.state, DrivingState.READY)

    def test_camera_covered_30_frames_forces_neutral(self):
        ctrl = DrivingController(loss_frames_limit=8)
        left_fist = create_test_hand(x=0.8, gesture="fist")
        # Drive for a few frames
        for _ in range(5):
            ctrl.update(leftmost_hand=left_fist)
        self.assertEqual(ctrl.state, DrivingState.DRIVING)
        self.assertGreater(ctrl.smoothed_steering, 0.0)

        # Cover camera for ~1 second (30 frames at 30 fps)
        for _ in range(30):
            ctrl.update(leftmost_hand=None, rightmost_hand=None)

        self.assertEqual(ctrl.state, DrivingState.NEUTRAL)
        self.assertEqual(ctrl.smoothed_steering, 0.0)
        self.assertEqual(ctrl.active_keys, [])

        # Uncover camera -> hand appears -> recovers to DRIVING
        ctrl.update(leftmost_hand=left_fist)
        self.assertEqual(ctrl.state, DrivingState.DRIVING)

    def test_extreme_hand_positions_strictly_clamped(self):
        ctrl = DrivingController(neutral_x=0.5, steer_sensitivity=2.5)
        # Position far to the right: x = 2.0
        hand_extreme_right = create_test_hand(x=2.0, gesture="relaxed")
        for _ in range(15):
            ctrl.update(leftmost_hand=hand_extreme_right)
        self.assertAlmostEqual(ctrl.smoothed_steering, 1.0, places=2)
        self.assertLessEqual(ctrl.smoothed_steering, 1.0)

        # Position far to the left: x = -1.0
        hand_extreme_left = create_test_hand(x=-1.0, gesture="relaxed")
        for _ in range(25):
            ctrl.update(leftmost_hand=hand_extreme_left)
        self.assertAlmostEqual(ctrl.smoothed_steering, -1.0, places=2)
        self.assertGreaterEqual(ctrl.smoothed_steering, -1.0)

    def test_driving_and_braking_combined_with_steering_keys(self):
        ctrl = DrivingController(neutral_x=0.5, steer_sensitivity=2.5, steer_fire_threshold=0.18)

        # DRIVING + steer right -> ['d', 'w']
        hand_fist_right = create_test_hand(x=0.8, gesture="fist")
        for _ in range(10):
            ctrl.update(leftmost_hand=hand_fist_right)
        self.assertEqual(ctrl.state, DrivingState.DRIVING)
        self.assertEqual(ctrl.active_keys, ["d", "w"])

        # DRIVING + steer left -> ['a', 'w']
        hand_fist_left = create_test_hand(x=0.2, gesture="fist")
        for _ in range(15):
            ctrl.update(leftmost_hand=hand_fist_left)
        self.assertEqual(ctrl.state, DrivingState.DRIVING)
        self.assertEqual(ctrl.active_keys, ["a", "w"])

        # BRAKING + steer right -> ['d', 's']
        hand_brake_left = create_test_hand(x=0.8, gesture="palm")
        hand_brake_right = create_test_hand(x=0.8, gesture="palm")
        for _ in range(15):
            ctrl.update(leftmost_hand=hand_brake_left, rightmost_hand=hand_brake_right)
        self.assertEqual(ctrl.state, DrivingState.BRAKING)
        self.assertEqual(ctrl.active_keys, ["d", "s"])

    def test_module_level_convenience_functions(self):
        from src import driving
        driving.reset_controller()

        self.assertEqual(driving.get_driving_state(), DrivingState.NEUTRAL)
        self.assertEqual(driving.get_smoothed_steering(), 0.0)

        # Calibrate
        driving.calibrate(0.4)
        self.assertAlmostEqual(driving._default_controller.neutral_x, 0.4)

        driving.set_neutral_x(0.35)
        self.assertAlmostEqual(driving._default_controller.neutral_x, 0.35)

        hand = create_test_hand(x=0.35, gesture="fist")
        st = driving.update(leftmost_hand=hand)
        self.assertEqual(st, DrivingState.DRIVING)
        self.assertIn("w", driving.get_active_keys())

    def test_single_hand_across_entire_x_range_retains_driving(self):
        # A single hand with a fist should be recognized as DRIVING anywhere on screen (x in 0.1 to 0.9)
        # without being dropped into NEUTRAL or BRAKING.
        for test_x in [0.1, 0.25, 0.5, 0.75, 0.9]:
            ctrl = DrivingController(neutral_x=0.5)
            hand = create_test_hand(x=test_x, gesture="fist")
            state = ctrl.update(hands=[hand])
            self.assertEqual(
                state,
                DrivingState.DRIVING,
                f"Failed for x={test_x}: expected DRIVING, got {state}",
            )
            self.assertIn("w", ctrl.active_keys)
            self.assertEqual(ctrl.loss_frame_count, 0)

    def test_single_hand_across_entire_x_range_relaxed_is_ready(self):
        # A single hand with relaxed/neutral posture anywhere on screen should be READY
        for test_x in [0.1, 0.5, 0.9]:
            ctrl = DrivingController(neutral_x=0.5)
            hand = create_test_hand(x=test_x, gesture="relaxed")
            state = ctrl.update(hands=[hand])
            self.assertEqual(state, DrivingState.READY)
            self.assertNotIn("w", ctrl.active_keys)
            self.assertNotIn("s", ctrl.active_keys)
            self.assertEqual(ctrl.loss_frame_count, 0)

    def test_hands_parameter_list_resolution(self):
        ctrl = DrivingController()
        # 0 hands -> tracking loss
        st_empty = ctrl.update(hands=[])
        self.assertEqual(st_empty, DrivingState.NEUTRAL)

        # 1 hand -> leftmost is hands[0], rightmost is None
        h1 = create_test_hand(x=0.4, gesture="fist")
        st_single = ctrl.update(hands=[h1])
        self.assertEqual(st_single, DrivingState.DRIVING)

        # 2 hands -> leftmost is hands[0], rightmost is hands[-1]
        h2 = create_test_hand(x=0.7, gesture="palm")
        st_dual = ctrl.update(hands=[h1, h2])
        self.assertEqual(st_dual, DrivingState.BRAKING)

        # 3 hands -> leftmost is hands[0], rightmost is hands[-1]
        h_mid = create_test_hand(x=0.55, gesture="relaxed")
        st_triple = ctrl.update(hands=[h1, h_mid, h2])
        self.assertEqual(st_triple, DrivingState.BRAKING)

    def test_calibrate_clears_steering_keys(self):
        ctrl = DrivingController(neutral_x=0.5, steer_fire_threshold=0.18)
        # Drive and steer right until 'd' is active
        hand = create_test_hand(x=0.8, gesture="fist")
        for _ in range(5):
            ctrl.update(leftmost_hand=hand)
        self.assertEqual(ctrl.state, DrivingState.DRIVING)
        self.assertIn("d", ctrl.active_keys)
        self.assertIn("w", ctrl.active_keys)

        # Calibrate with current hand position -> steering recenters to 0
        ctrl.calibrate(hand)
        self.assertEqual(ctrl.smoothed_steering, 0.0)
        self.assertNotIn("d", ctrl.active_keys)
        self.assertNotIn("a", ctrl.active_keys)
        self.assertIn("w", ctrl.active_keys)

    def test_initial_loss_frame_count_stays_zero_before_any_hand(self):
        ctrl = DrivingController()
        for _ in range(10):
            st = ctrl.update(leftmost_hand=None, rightmost_hand=None)
            self.assertEqual(st, DrivingState.NEUTRAL)
            self.assertEqual(ctrl.loss_frame_count, 0)

    def test_steer_left_and_right_calibrated_at_extremes(self):
        # Calibrated at 0.25, steer right to 0.75
        ctrl = DrivingController(neutral_x=0.25)
        hand_r = create_test_hand(x=0.75, gesture="fist")
        for _ in range(10):
            st = ctrl.update(hands=[hand_r])
        self.assertEqual(st, DrivingState.DRIVING)
        self.assertIn("d", ctrl.active_keys)
        self.assertIn("w", ctrl.active_keys)

        # Calibrated at 0.75, steer left to 0.25
        ctrl2 = DrivingController(neutral_x=0.75)
        hand_l = create_test_hand(x=0.25, gesture="fist")
        for _ in range(10):
            st = ctrl2.update(hands=[hand_l])
        self.assertEqual(st, DrivingState.DRIVING)
        self.assertIn("a", ctrl2.active_keys)
        self.assertIn("w", ctrl2.active_keys)


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

    @patch("cv2.destroyAllWindows")
    @patch("cv2.imshow")
    @patch("cv2.waitKey", side_effect=[ord("c"), ord("q")])
    @patch("main.Camera")
    @patch("main.Vision")
    def test_main_keypress_c_calibrates(
        self, mock_vision_cls, mock_camera_cls, mock_waitkey, mock_imshow, mock_destroy
    ):
        import main

        mock_cam_inst = MagicMock()
        mock_cam_inst.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_camera_cls.return_value = mock_cam_inst

        mock_vis_inst = MagicMock()
        hand = HandLandmarksMock([LandmarkMock(0.35, 0.5) for _ in range(21)])
        mock_vis_inst.process.return_value = [hand]
        mock_vision_cls.return_value = mock_vis_inst

        with patch("main.DrivingController") as mock_ctrl_cls:
            mock_ctrl_inst = MagicMock()
            mock_ctrl_inst.is_calibrated = False
            mock_ctrl_inst.neutral_x = 0.5
            mock_ctrl_inst.smoothed_steering = 0.0
            mock_ctrl_inst.state = DrivingState.READY
            mock_ctrl_inst.update.return_value = DrivingState.READY
            mock_ctrl_inst.get_active_keys.return_value = []
            mock_ctrl_inst.calibrate.return_value = 0.35
            mock_ctrl_cls.return_value = mock_ctrl_inst

            main.main()
            mock_ctrl_inst.calibrate.assert_called_once_with(hand)

    @patch("cv2.destroyAllWindows")
    @patch("cv2.imshow")
    @patch("cv2.waitKey", side_effect=[ord("C"), ord("Q")])
    @patch("main.Camera")
    @patch("main.Vision")
    def test_main_keypress_capital_c_calibrates(
        self, mock_vision_cls, mock_camera_cls, mock_waitkey, mock_imshow, mock_destroy
    ):
        import main

        mock_cam_inst = MagicMock()
        mock_cam_inst.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_camera_cls.return_value = mock_cam_inst

        mock_vis_inst = MagicMock()
        hand = HandLandmarksMock([LandmarkMock(0.65, 0.5) for _ in range(21)])
        mock_vis_inst.process.return_value = [hand]
        mock_vision_cls.return_value = mock_vis_inst

        with patch("main.DrivingController") as mock_ctrl_cls:
            mock_ctrl_inst = MagicMock()
            mock_ctrl_inst.is_calibrated = False
            mock_ctrl_inst.neutral_x = 0.5
            mock_ctrl_inst.smoothed_steering = 0.0
            mock_ctrl_inst.state = DrivingState.READY
            mock_ctrl_inst.update.return_value = DrivingState.READY
            mock_ctrl_inst.get_active_keys.return_value = []
            mock_ctrl_inst.calibrate.return_value = 0.65
            mock_ctrl_cls.return_value = mock_ctrl_inst

            main.main()
            mock_ctrl_inst.calibrate.assert_called_once_with(hand)

    @patch("cv2.destroyAllWindows")
    @patch("cv2.imshow")
    @patch("cv2.waitKey", return_value=ord("q"))
    @patch("main.Camera")
    @patch("main.Vision")
    def test_main_single_hand_on_right_side_is_leftmost(
        self, mock_vision_cls, mock_camera_cls, mock_waitkey, mock_imshow, mock_destroy
    ):
        import main

        mock_cam_inst = MagicMock()
        mock_cam_inst.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_camera_cls.return_value = mock_cam_inst

        mock_vis_inst = MagicMock()
        # Hand on right half of screen: x = 0.75
        hand = HandLandmarksMock([LandmarkMock(0.75, 0.5) for _ in range(21)])
        mock_vis_inst.process.return_value = [hand]
        mock_vision_cls.return_value = mock_vis_inst

        with patch("main.DrivingController") as mock_ctrl_cls:
            mock_ctrl_inst = MagicMock()
            mock_ctrl_inst.is_calibrated = False
            mock_ctrl_inst.neutral_x = 0.5
            mock_ctrl_inst.smoothed_steering = 0.0
            mock_ctrl_inst.state = DrivingState.READY
            mock_ctrl_inst.update.return_value = DrivingState.READY
            mock_ctrl_inst.get_active_keys.return_value = []
            mock_ctrl_cls.return_value = mock_ctrl_inst

            main.main()
            # Must be passed as leftmost_hand=hand, rightmost_hand=None
            mock_ctrl_inst.update.assert_called_once_with(leftmost_hand=hand, rightmost_hand=None)


if __name__ == "__main__":
    unittest.main()
