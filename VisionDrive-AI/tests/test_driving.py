"""Tests for driving controller, steering calibration math, thresholds, and tracking loss."""

import pytest
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

        # 1. BRAKING + Steer Right: steer hand at 0.75, brake hand (open palm)
        steer_hand_right = MockHand(x=0.75, gesture="relaxed")
        brake_hand_palm = MockHand(x=0.75, gesture="palm")
        controller.update(leftmost_hand=steer_hand_right, rightmost_hand=brake_hand_palm)
        assert controller.state == BRAKING
        assert controller.get_active_keys() == ["d", "s"]

        # 2. BRAKING + Steer Left: steer hand at 0.25, brake hand (open palm)
        steer_hand_left = MockHand(x=0.25, gesture="relaxed")
        controller.update(leftmost_hand=steer_hand_left, rightmost_hand=brake_hand_palm)
        assert controller.state == BRAKING
        assert controller.get_active_keys() == ["a", "s"]

        # 3. READY + Steer Left: steer hand at 0.25, no brake hand
        controller.update(leftmost_hand=steer_hand_left, rightmost_hand=None)
        assert controller.state == READY
        assert controller.get_active_keys() == ["a"]

        # 4. READY + Steer Right: steer hand at 0.75, no brake hand
        controller.update(leftmost_hand=steer_hand_right, rightmost_hand=None)
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
