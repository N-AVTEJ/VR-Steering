"""Tests for driving controller, steering calibration math, thresholds, and tracking loss."""

import sys
from pathlib import Path
import pytest

# Ensure VisionDrive-AI root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.config import (
    EMA_ALPHA,
    LOSS_FRAMES_LIMIT,
    STEER_DEADZONE,
    STEER_FIRE_THRESHOLD,
    STEER_SENSITIVITY,
)
from src.driving import (
    BRAKING,
    DRIVING,
    NEUTRAL,
    READY,
    DrivingController,
    DrivingState,
    apply_deadzone,
    apply_ema,
    calculate_raw_steering,
    calculate_steering,
    clamp,
    get_hand_x,
)


class MockPoint:
    def __init__(self, x: float, y: float = 0.5) -> None:
        self.x = x
        self.y = y


class MockHand:
    def __init__(self, x: float = 0.5, gesture: str = "relaxed") -> None:
        # 21 points
        self.landmark = [MockPoint(x, 0.8) for _ in range(21)]
        wrist_y = 0.8
        mcp_dist = 0.20
        ratio = 0.4 if gesture == "fist" else (1.6 if gesture == "palm" else 0.95)
        tip_dist = mcp_dist * ratio

        for mcp_idx, tip_idx in [(5, 8), (9, 12), (13, 16), (17, 20)]:
            self.landmark[mcp_idx] = MockPoint(x, wrist_y - mcp_dist)
            self.landmark[tip_idx] = MockPoint(x, wrist_y - tip_dist)


class TestSteeringCalibrationMath:
    """Tests for steering calculation, normalization, clamping, and calibration."""

    def test_clamp_fixed_values(self):
        assert clamp(0.5, -1.0, 1.0) == 0.5
        assert clamp(1.5, -1.0, 1.0) == 1.0
        assert clamp(-1.5, -1.0, 1.0) == -1.0
        assert clamp(-0.0, -1.0, 1.0) == 0.0
        assert clamp(10.0, 0.0, 5.0) == 5.0
        assert clamp(-10.0, 0.0, 5.0) == 0.0

    def test_calculate_raw_steering_math(self):
        # raw = clamp((hand_x - neutral_x) * sensitivity, -1.0, 1.0)
        neutral = 0.5
        sens = 2.5

        # Centered: 0
        assert calculate_raw_steering(0.5, neutral, sens) == pytest.approx(0.0)

        # Steering right: hand_x = 0.6 -> (0.6 - 0.5) * 2.5 = 0.25
        assert calculate_raw_steering(0.6, neutral, sens) == pytest.approx(0.25)

        # Steering left: hand_x = 0.4 -> (0.4 - 0.5) * 2.5 = -0.25
        assert calculate_raw_steering(0.4, neutral, sens) == pytest.approx(-0.25)

        # Clamp right saturation: hand_x = 1.0 -> (1.0 - 0.5) * 2.5 = 1.25 -> 1.0
        assert calculate_raw_steering(1.0, neutral, sens) == 1.0

        # Clamp left saturation: hand_x = 0.0 -> (0.0 - 0.5) * 2.5 = -1.25 -> -1.0
        assert calculate_raw_steering(0.0, neutral, sens) == -1.0

    def test_calibrate_method(self):
        controller = DrivingController(neutral_x=0.5)
        assert controller.neutral_x == 0.5
        assert controller.is_calibrated is False

        # Calibrate with numeric float
        new_x = controller.calibrate(0.35)
        assert new_x == pytest.approx(0.35)
        assert controller.neutral_x == pytest.approx(0.35)
        assert controller.is_calibrated is True
        assert controller.smoothed_steering == 0.0

        # Calibrate with mock hand object
        hand = MockHand(x=0.62)
        controller.calibrate(hand)
        assert controller.neutral_x == pytest.approx(0.62)
        assert controller.smoothed_steering == 0.0


class TestDeadzoneAndThresholdBoundaries:
    """Tests for deadzone filtering, EMA smoothing, and steering threshold boundaries."""

    def test_apply_deadzone_boundaries(self):
        deadzone = 0.08
        # Strictly within deadzone (-0.08, 0.08) -> returns 0.0
        assert apply_deadzone(0.0, deadzone) == 0.0
        assert apply_deadzone(0.079, deadzone) == 0.0
        assert apply_deadzone(-0.079, deadzone) == 0.0

        # At boundary or beyond -> retains value
        assert apply_deadzone(0.08, deadzone) == 0.08
        assert apply_deadzone(-0.08, deadzone) == -0.08
        assert apply_deadzone(0.5, deadzone) == 0.5
        assert apply_deadzone(-0.5, deadzone) == -0.5

    def test_calculate_steering_combined(self):
        # hand_x = 0.52 -> (0.52 - 0.5) * 2.5 = 0.05 < deadzone 0.08 -> 0.0
        assert calculate_steering(0.52, 0.5, 2.5, 0.08) == 0.0

        # hand_x = 0.54 -> (0.54 - 0.5) * 2.5 = 0.10 >= deadzone 0.08 -> 0.10
        assert calculate_steering(0.54, 0.5, 2.5, 0.08) == pytest.approx(0.10)

        # hand_x = 0.46 -> (0.46 - 0.5) * 2.5 = -0.10 <= -deadzone -> -0.10
        assert calculate_steering(0.46, 0.5, 2.5, 0.08) == pytest.approx(-0.10)

    def test_apply_ema_smoothing(self):
        alpha = 0.35
        # 1st step from 0 towards 1.0: 0.35 * 1.0 + 0.65 * 0.0 = 0.35
        s1 = apply_ema(1.0, 0.0, alpha)
        assert s1 == pytest.approx(0.35)

        # 2nd step: 0.35 * 1.0 + 0.65 * 0.35 = 0.5775
        s2 = apply_ema(1.0, s1, alpha)
        assert s2 == pytest.approx(0.5775)

    def test_steer_fire_threshold_boundaries(self):
        """Verify 'a' and 'd' virtual keys fire only when strictly beyond STEER_FIRE_THRESHOLD."""
        # Sensitivity 1.0, alpha 1.0 (no EMA delay) for direct boundary inspection
        controller = DrivingController(
            neutral_x=0.5,
            steer_sensitivity=1.0,
            ema_alpha=1.0,
            steer_deadzone=0.05,
            steer_fire_threshold=0.18,
        )

        # Hand in DRIVING mode (fist), steer = 0.5 + 0.18 = 0.68
        # raw = 0.18 -> smoothed = 0.18.
        # Boundary condition: smoothed_steering > 0.18 is False (0.18 is NOT > 0.18)
        hand_exact_right = MockHand(x=0.68, gesture="fist")
        controller.update(leftmost_hand=hand_exact_right)
        assert controller.state == DRIVING
        assert "d" not in controller.get_active_keys()

        # Beyond boundary: steer = 0.5 + 0.181 = 0.681
        hand_over_right = MockHand(x=0.681, gesture="fist")
        controller.update(leftmost_hand=hand_over_right)
        assert "d" in controller.get_active_keys()
        assert "w" in controller.get_active_keys()

        # Boundary left: steer = 0.5 - 0.18 = 0.32
        hand_exact_left = MockHand(x=0.32, gesture="fist")
        controller.update(leftmost_hand=hand_exact_left)
        assert "a" not in controller.get_active_keys()

        # Beyond boundary left: steer = 0.5 - 0.181 = 0.319
        hand_over_left = MockHand(x=0.319, gesture="fist")
        controller.update(leftmost_hand=hand_over_left)
        assert "a" in controller.get_active_keys()
        assert "w" in controller.get_active_keys()


class TestTrackingLossHandling:
    """Tests for tracking-loss behavior and reset to NEUTRAL after LOSS_FRAMES_LIMIT frames."""

    def test_initial_no_hand_remains_neutral(self):
        """Before any hand is ever seen, no hand remains in NEUTRAL without incrementing frames."""
        controller = DrivingController()
        state = controller.update(leftmost_hand=None)
        assert state == NEUTRAL
        assert controller.loss_frame_count == 0
        assert controller.get_active_keys() == []

    def test_tracking_loss_grace_period_and_reset(self):
        """
        After hand is established in DRIVING:
        - Frames 1 to LOSS_FRAMES_LIMIT-1 maintain state (grace period).
        - At exactly frame LOSS_FRAMES_LIMIT: resets to NEUTRAL, steering=0, keys=[].
        """
        limit = 8
        controller = DrivingController(loss_frames_limit=limit, neutral_x=0.5)

        # Establish DRIVING state with fist
        hand = MockHand(x=0.5, gesture="fist")
        state = controller.update(leftmost_hand=hand)
        assert state == DRIVING
        assert controller.get_active_keys() == ["w"]
        assert controller.loss_frame_count == 0

        # Frames 1 to 7: loss grace period
        for frame in range(1, limit):
            state = controller.update(leftmost_hand=None)
            assert state == DRIVING
            assert controller.loss_frame_count == frame

        # Frame 8: reaches LOSS_FRAMES_LIMIT -> reset to NEUTRAL
        state = controller.update(leftmost_hand=None)
        assert state == NEUTRAL
        assert controller.loss_frame_count == limit
        assert controller.smoothed_steering == 0.0
        assert controller.raw_steering == 0.0
        assert controller.get_active_keys() == []

    def test_hand_recovery_after_tracking_loss(self):
        """If hand returns after tracking loss, loss_frame_count resets to 0 and state updates."""
        controller = DrivingController(loss_frames_limit=5, neutral_x=0.5)

        # Drive
        hand = MockHand(x=0.5, gesture="fist")
        controller.update(leftmost_hand=hand)
        assert controller.state == DRIVING

        # 5 frames of loss -> NEUTRAL
        for _ in range(5):
            controller.update(leftmost_hand=None)
        assert controller.state == NEUTRAL

        # Hand reappears as fist
        state = controller.update(leftmost_hand=hand)
        assert state == DRIVING
        assert controller.loss_frame_count == 0
        assert controller.get_active_keys() == ["w"]

    def test_tracking_loss_grace_period_recovery_midway(self):
        """If hand returns during grace period (before limit), state continues smoothly."""
        controller = DrivingController(loss_frames_limit=8, neutral_x=0.5)
        hand = MockHand(x=0.5, gesture="fist")
        controller.update(leftmost_hand=hand)
        assert controller.state == DRIVING

        # Hand dropped for 3 frames
        for f in range(1, 4):
            controller.update(leftmost_hand=None)
            assert controller.state == DRIVING
            assert controller.loss_frame_count == f
            assert controller.get_active_keys() == ["w"]

        # Hand restored on frame 4
        state = controller.update(leftmost_hand=hand)
        assert state == DRIVING
        assert controller.loss_frame_count == 0
        assert controller.get_active_keys() == ["w"]

    def test_tracking_loss_with_empty_hands_list(self):
        """Passing hands=[] resolves to None for leftmost/rightmost and triggers tracking loss."""
        controller = DrivingController(loss_frames_limit=3, neutral_x=0.5)
        hand = MockHand(x=0.5, gesture="fist")
        controller.update(hands=[hand])
        assert controller.state == DRIVING

        # 3 frames of hands=[]
        for _ in range(3):
            controller.update(hands=[])
        assert controller.state == NEUTRAL
        assert controller.get_active_keys() == []

    def test_steer_fire_threshold_with_braking_and_ready(self):
        """Virtual steering keys 'a'/'d' fire alongside 's' (BRAKING) and alone (READY)."""
        controller = DrivingController(
            neutral_x=0.5,
            steer_sensitivity=1.0,
            ema_alpha=1.0,
            steer_deadzone=0.05,
            steer_fire_threshold=0.18,
        )

        # 1. BRAKING + Steer Right: both hands at 0.75 (open palms)
        steer_hand_right = MockHand(x=0.75, gesture="palm")
        brake_hand_palm_right = MockHand(x=0.75, gesture="palm")
        controller.update(leftmost_hand=steer_hand_right, rightmost_hand=brake_hand_palm_right)
        assert controller.state == BRAKING
        assert controller.get_active_keys() == ["d", "s"]

        # 2. BRAKING + Steer Left: both hands at 0.25 (open palms)
        steer_hand_left = MockHand(x=0.25, gesture="palm")
        brake_hand_palm_left = MockHand(x=0.25, gesture="palm")
        controller.update(leftmost_hand=steer_hand_left, rightmost_hand=brake_hand_palm_left)
        assert controller.state == BRAKING
        assert controller.get_active_keys() == ["a", "s"]

        # 3. READY + Steer Left: steer hand at 0.25, no brake hand (single hand)
        steer_hand_left_rel = MockHand(x=0.25, gesture="relaxed")
        controller.update(leftmost_hand=steer_hand_left_rel, rightmost_hand=None)
        assert controller.state == READY
        assert controller.get_active_keys() == ["a"]

        # 4. READY + Steer Right: steer hand at 0.75, no brake hand (single hand)
        steer_hand_right_rel = MockHand(x=0.75, gesture="relaxed")
        controller.update(leftmost_hand=steer_hand_right_rel, rightmost_hand=None)
        assert controller.state == READY
        assert controller.get_active_keys() == ["d"]

    def test_calibration_resets_steering_keys_immediately(self):
        """Calibrating while steering immediately drops steering keys 'a'/'d'."""
        controller = DrivingController(
            neutral_x=0.5,
            steer_sensitivity=1.0,
            ema_alpha=1.0,
            steer_deadzone=0.05,
            steer_fire_threshold=0.18,
        )
        # Drive right
        hand = MockHand(x=0.75, gesture="fist")
        controller.update(leftmost_hand=hand)
        assert controller.get_active_keys() == ["d", "w"]

        # Calibrate to current hand position (0.75 becomes neutral)
        controller.calibrate(hand)
        assert "d" not in controller.get_active_keys()
        assert controller.smoothed_steering == 0.0


class TestTwoHandSteeringAndGestures:
    """Tests for two-hand midpoint steering calculation, calibration, dual-hand gestures, and single-hand compatibility."""

    def test_two_hand_midpoint_calculation(self):
        controller = DrivingController(neutral_x=0.5, steer_sensitivity=2.5, ema_alpha=1.0, steer_deadzone=0.0)

        # Centered: left at 0.35, right at 0.65 -> midpoint is 0.50 -> steer 0.0
        h_left_center = MockHand(x=0.35, gesture="fist")
        h_right_center = MockHand(x=0.65, gesture="fist")
        controller.update(leftmost_hand=h_left_center, rightmost_hand=h_right_center)
        assert controller.hand_x == pytest.approx(0.50)
        assert controller.smoothed_steering == pytest.approx(0.0)

        # Moving both hands right: left at 0.45, right at 0.75 -> midpoint 0.60 -> steer +0.25
        h_left_right = MockHand(x=0.45, gesture="fist")
        h_right_right = MockHand(x=0.75, gesture="fist")
        controller.update(leftmost_hand=h_left_right, rightmost_hand=h_right_right)
        assert controller.hand_x == pytest.approx(0.60)
        assert controller.smoothed_steering == pytest.approx(0.25)

        # Moving both hands left: left at 0.25, right at 0.55 -> midpoint 0.40 -> steer -0.25
        h_left_left = MockHand(x=0.25, gesture="fist")
        h_right_left = MockHand(x=0.55, gesture="fist")
        controller.update(leftmost_hand=h_left_left, rightmost_hand=h_right_left)
        assert controller.hand_x == pytest.approx(0.40)
        assert controller.smoothed_steering == pytest.approx(-0.25)

    def test_two_hand_midpoint_steering_keys(self):
        controller = DrivingController(
            neutral_x=0.5,
            steer_sensitivity=1.0,
            ema_alpha=1.0,
            steer_deadzone=0.05,
            steer_fire_threshold=0.18,
        )

        # Both hands shifted right: midpoint = (0.60 + 0.80) / 2 = 0.70.
        # Steering = 0.70 - 0.50 = +0.20 > 0.18 -> fires 'd'
        h_l_r = MockHand(x=0.60, gesture="fist")
        h_r_r = MockHand(x=0.80, gesture="fist")
        controller.update(leftmost_hand=h_l_r, rightmost_hand=h_r_r)
        assert controller.state == DRIVING
        assert "d" in controller.get_active_keys()
        assert "w" in controller.get_active_keys()
        assert "a" not in controller.get_active_keys()

        # Both hands shifted left: midpoint = (0.20 + 0.40) / 2 = 0.30.
        # Steering = 0.30 - 0.50 = -0.20 < -0.18 -> fires 'a'
        h_l_l = MockHand(x=0.20, gesture="fist")
        h_r_l = MockHand(x=0.40, gesture="fist")
        controller.update(leftmost_hand=h_l_l, rightmost_hand=h_r_l)
        assert controller.state == DRIVING
        assert "a" in controller.get_active_keys()
        assert "w" in controller.get_active_keys()
        assert "d" not in controller.get_active_keys()

    def test_two_hand_calibration(self):
        controller = DrivingController(neutral_x=0.5)

        # 1. Calibrate with two mock hands directly
        h1 = MockHand(x=0.20)
        h2 = MockHand(x=0.60)
        new_neutral = controller.calibrate(h1, h2)
        assert new_neutral == pytest.approx(0.40)
        assert controller.neutral_x == pytest.approx(0.40)
        assert controller.is_calibrated is True
        assert controller.smoothed_steering == 0.0

        # 2. Calibrate with hands sequence parameter
        h3 = MockHand(x=0.35)
        h4 = MockHand(x=0.75)
        new_neutral2 = controller.calibrate(hands=[h3, h4])
        assert new_neutral2 == pytest.approx(0.55)
        assert controller.neutral_x == pytest.approx(0.55)

        # 3. Calibrate with list of hands as first parameter
        new_neutral3 = controller.calibrate([MockHand(x=0.30), MockHand(x=0.70)])
        assert new_neutral3 == pytest.approx(0.50)
        assert controller.neutral_x == pytest.approx(0.50)

        # 4. Calibrate with single hand
        new_neutral4 = controller.calibrate(MockHand(x=0.42))
        assert new_neutral4 == pytest.approx(0.42)
        assert controller.neutral_x == pytest.approx(0.42)

    def test_two_hand_gestures_both_fists_is_driving(self):
        controller = DrivingController()
        h_l = MockHand(x=0.4, gesture="fist")
        h_r = MockHand(x=0.6, gesture="fist")
        state = controller.update(leftmost_hand=h_l, rightmost_hand=h_r)
        assert state == DRIVING
        assert "w" in controller.get_active_keys()
        assert "s" not in controller.get_active_keys()

    def test_two_hand_gestures_both_palms_is_braking(self):
        controller = DrivingController()
        h_l = MockHand(x=0.4, gesture="palm")
        h_r = MockHand(x=0.6, gesture="palm")
        state = controller.update(leftmost_hand=h_l, rightmost_hand=h_r)
        assert state == BRAKING
        assert "s" in controller.get_active_keys()
        assert "w" not in controller.get_active_keys()

    def test_two_hand_gestures_braking_overrides_driving(self):
        controller = DrivingController()

        # Left palm, right fist -> BRAKING
        state1 = controller.update(
            leftmost_hand=MockHand(x=0.4, gesture="palm"),
            rightmost_hand=MockHand(x=0.6, gesture="fist"),
        )
        assert state1 == BRAKING
        assert "s" in controller.get_active_keys()
        assert "w" not in controller.get_active_keys()

        # Left fist, right palm -> BRAKING
        state2 = controller.update(
            leftmost_hand=MockHand(x=0.4, gesture="fist"),
            rightmost_hand=MockHand(x=0.6, gesture="palm"),
        )
        assert state2 == BRAKING
        assert "s" in controller.get_active_keys()
        assert "w" not in controller.get_active_keys()

    def test_two_hand_gestures_relaxed_coasting_is_ready(self):
        controller = DrivingController()

        # Both relaxed -> READY
        s1 = controller.update(
            leftmost_hand=MockHand(x=0.4, gesture="relaxed"),
            rightmost_hand=MockHand(x=0.6, gesture="relaxed"),
        )
        assert s1 == READY
        assert controller.get_active_keys() == []

        # Left fist, right relaxed -> READY (coasting)
        s2 = controller.update(
            leftmost_hand=MockHand(x=0.4, gesture="fist"),
            rightmost_hand=MockHand(x=0.6, gesture="relaxed"),
        )
        assert s2 == READY
        assert controller.get_active_keys() == []

        # Left relaxed, right fist -> READY (coasting)
        s3 = controller.update(
            leftmost_hand=MockHand(x=0.4, gesture="relaxed"),
            rightmost_hand=MockHand(x=0.6, gesture="fist"),
        )
        assert s3 == READY
        assert controller.get_active_keys() == []

        # Left palm, right relaxed -> READY (coasting)
        s4 = controller.update(
            leftmost_hand=MockHand(x=0.4, gesture="palm"),
            rightmost_hand=MockHand(x=0.6, gesture="relaxed"),
        )
        assert s4 == READY
        assert controller.get_active_keys() == []

        # Left relaxed, right palm -> READY (coasting)
        s5 = controller.update(
            leftmost_hand=MockHand(x=0.4, gesture="relaxed"),
            rightmost_hand=MockHand(x=0.6, gesture="palm"),
        )
        assert s5 == READY
        assert controller.get_active_keys() == []

    def test_single_hand_backwards_compatibility(self):
        controller = DrivingController(neutral_x=0.5, steer_sensitivity=2.5, ema_alpha=1.0, steer_deadzone=0.0)

        # Single hand fist -> DRIVING
        s_fist = controller.update(leftmost_hand=MockHand(x=0.5, gesture="fist"), rightmost_hand=None)
        assert s_fist == DRIVING
        assert "w" in controller.get_active_keys()
        assert "s" not in controller.get_active_keys()

        # Single hand palm -> BRAKING
        s_palm = controller.update(leftmost_hand=MockHand(x=0.5, gesture="palm"), rightmost_hand=None)
        assert s_palm == BRAKING
        assert "s" in controller.get_active_keys()
        assert "w" not in controller.get_active_keys()

        # Single hand relaxed -> READY
        s_rel = controller.update(leftmost_hand=MockHand(x=0.5, gesture="relaxed"), rightmost_hand=None)
        assert s_rel == READY
        assert controller.get_active_keys() == []

        # Single hand steer right: x = 0.65 -> (0.65 - 0.5) * 2.5 = 0.375
        controller.update(leftmost_hand=MockHand(x=0.65, gesture="fist"), rightmost_hand=None)
        assert controller.hand_x == pytest.approx(0.65)
        assert controller.smoothed_steering == pytest.approx(0.375)

    def test_hands_sequence_two_hands_vs_one_hand(self):
        controller = DrivingController(neutral_x=0.5, steer_sensitivity=1.0, ema_alpha=1.0, steer_deadzone=0.0)

        # Two hands in sequence: midpoint (0.3 + 0.7) / 2 = 0.5
        h1 = MockHand(x=0.3, gesture="fist")
        h2 = MockHand(x=0.7, gesture="fist")
        st = controller.update(hands=[h1, h2])
        assert st == DRIVING
        assert controller.hand_x == pytest.approx(0.5)
        assert controller.smoothed_steering == pytest.approx(0.0)

        # One hand in sequence: uses h1.x = 0.3 -> steering = 0.3 - 0.5 = -0.2
        st_single = controller.update(hands=[h1])
        assert st_single == DRIVING
        assert controller.hand_x == pytest.approx(0.3)
        assert controller.smoothed_steering == pytest.approx(-0.2)

    def test_calibrate_empty_sequence_preserves_neutral_x(self):
        controller = DrivingController(neutral_x=0.35)
        # Positional empty sequence
        res1 = controller.calibrate([])
        assert res1 == pytest.approx(0.35)
        assert controller.neutral_x == pytest.approx(0.35)

        # Keyword empty sequence
        res2 = controller.calibrate(hands=[])
        assert res2 == pytest.approx(0.35)
        assert controller.neutral_x == pytest.approx(0.35)

    def test_calibrate_single_hand_in_list(self):
        controller = DrivingController(neutral_x=0.5)
        res = controller.calibrate([MockHand(x=0.65)])
        assert res == pytest.approx(0.65)
        assert controller.neutral_x == pytest.approx(0.65)

        # Single float in list
        res_float = controller.calibrate([0.42])
        assert res_float == pytest.approx(0.42)
        assert controller.neutral_x == pytest.approx(0.42)

    def test_calibrate_coordinates_list(self):
        controller = DrivingController(neutral_x=0.5)
        res = controller.calibrate([0.30, 0.70])
        assert res == pytest.approx(0.50)
        assert controller.neutral_x == pytest.approx(0.50)

    def test_update_positional_hand_sequence(self):
        controller = DrivingController(neutral_x=0.5, steer_sensitivity=1.0, ema_alpha=1.0, steer_deadzone=0.0)

        # Positional 2-hand sequence
        h1 = MockHand(x=0.3, gesture="fist")
        h2 = MockHand(x=0.7, gesture="fist")
        st = controller.update([h1, h2])
        assert st == DRIVING
        assert controller.hand_x == pytest.approx(0.5)
        assert "w" in controller.get_active_keys()

        # Positional 1-hand sequence
        st_palm = controller.update([MockHand(x=0.4, gesture="palm")])
        assert st_palm == BRAKING
        assert controller.hand_x == pytest.approx(0.4)
        assert "s" in controller.get_active_keys()

    def test_update_positional_empty_list_triggers_tracking_loss(self):
        controller = DrivingController(loss_frames_limit=5, neutral_x=0.5)
        # Establish driving state
        controller.update(MockHand(x=0.5, gesture="fist"))
        assert controller.state == DRIVING

        # Update with empty list -> increments loss frame count
        for i in range(1, 5):
            controller.update([])
            assert controller.loss_frame_count == i
            assert controller.state == DRIVING

        # 5th frame reaches limit -> NEUTRAL
        controller.update([])
        assert controller.state == NEUTRAL
        assert controller.loss_frame_count == 5
        assert controller.get_active_keys() == []

    def test_update_single_hand_passed_as_rightmost(self):
        controller = DrivingController(neutral_x=0.5, steer_sensitivity=2.0, ema_alpha=1.0, steer_deadzone=0.0)
        # Pass hand as rightmost_hand with leftmost_hand=None
        st = controller.update(rightmost_hand=MockHand(x=0.65, gesture="fist"))
        assert st == DRIVING
        assert controller.hand_x == pytest.approx(0.65)
        assert controller.smoothed_steering == pytest.approx(0.30)
        assert "w" in controller.get_active_keys()

    def test_get_hand_x_numeric_list(self):
        from src.driving import get_hand_x
        assert get_hand_x([0.3, 0.7]) == pytest.approx(0.5)
        assert get_hand_x([0.45]) == pytest.approx(0.45)
        assert get_hand_x([]) == 0.5


