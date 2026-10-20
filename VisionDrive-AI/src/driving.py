"""Driving control module for VisionDrive-AI.

Implements steering calculation, neutral calibration, exponential moving average smoothing,
tracking loss handling, virtual keyboard outputs, and state machine.
"""

from enum import Enum
from typing import Any, List, Optional, Sequence, Union

try:
    from src.config import (
        EMA_ALPHA,
        LOSS_FRAMES_LIMIT,
        STEER_DEADZONE,
        STEER_FIRE_THRESHOLD,
        STEER_SENSITIVITY,
    )
    from src.gestures import is_fist, is_open_palm
except ImportError:
    from config import (
        EMA_ALPHA,
        LOSS_FRAMES_LIMIT,
        STEER_DEADZONE,
        STEER_FIRE_THRESHOLD,
        STEER_SENSITIVITY,
    )
    from gestures import is_fist, is_open_palm


class DrivingState(str, Enum):
    """Possible states for the driving system."""

    NEUTRAL = "NEUTRAL"
    READY = "READY"
    DRIVING = "DRIVING"
    BRAKING = "BRAKING"


# Module-level aliases for convenience
NEUTRAL = DrivingState.NEUTRAL
READY = DrivingState.READY
DRIVING = DrivingState.DRIVING
BRAKING = DrivingState.BRAKING


def clamp(val: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """Clamp a numerical value within [min_val, max_val]."""
    return max(min_val, min(max_val, val))


def _extract_landmark_coord_x(p: Any) -> float:
    """Safely extract the horizontal x-coordinate from a landmark or point."""
    if hasattr(p, "x"):
        return float(p.x)
    if isinstance(p, (int, float)):
        return float(p)
    if isinstance(p, (list, tuple)) and len(p) > 0:
        return float(p[0])
    return 0.5


def get_hand_x(hand_landmarks: Any) -> float:
    """
    Extract horizontal centroid (average x-coordinate) of hand landmarks.
    Accepts float/int, MediaPipe NormalizedLandmarkList, or list/tuple of points.
    """
    if hand_landmarks is None:
        return 0.5
    if isinstance(hand_landmarks, (int, float)):
        return float(hand_landmarks)

    lms = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
    if not lms:
        return 0.5

    return sum(_extract_landmark_coord_x(lm) for lm in lms) / len(lms)


def _is_hand_sequence(obj: Any) -> bool:
    """Check if obj is a sequence of hand representations rather than a single hand."""
    if not isinstance(obj, (list, tuple)):
        return False
    if len(obj) == 0:
        return True
    first = obj[0]
    # If the first element is a hand object (has landmark attribute)
    if hasattr(first, "landmark"):
        return True
    # If it contains <= 2 elements (MediaPipe hands have 21 landmarks, so 1 or 2 items are hand representations)
    if len(obj) <= 2:
        return True
    return False


def _resolve_hands(
    leftmost_hand: Any = None,
    rightmost_hand: Any = None,
    hands: Optional[Sequence[Any]] = None,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Resolve leftmost and rightmost hand representations from various call patterns:
    - update(leftmost, rightmost)
    - update(leftmost)
    - update(hands=[h1, h2]) or update(hands=[h1]) or update(hands=[])
    - update([h1, h2]) or update([h1]) or update([])
    - update(rightmost_hand=h)
    """
    hand_seq = None
    if hands is not None:
        hand_seq = hands
    elif _is_hand_sequence(leftmost_hand):
        hand_seq = leftmost_hand

    if hand_seq is not None:
        l_hand = hand_seq[0] if len(hand_seq) >= 1 else None
        r_hand = hand_seq[-1] if len(hand_seq) >= 2 else None
        return l_hand, r_hand

    if leftmost_hand is None and rightmost_hand is not None:
        return rightmost_hand, None

    return leftmost_hand, rightmost_hand


def apply_deadzone(val: float, deadzone: float = STEER_DEADZONE) -> float:
    """Ignore values whose absolute magnitude is strictly below deadzone."""
    if abs(val) < deadzone:
        return 0.0
    return val


def apply_ema(current_val: float, prev_val: float, alpha: float = EMA_ALPHA) -> float:
    """Calculate exponential moving average (EMA)."""
    return alpha * current_val + (1.0 - alpha) * prev_val


def calculate_raw_steering(
    hand_x: float,
    neutral_x: float,
    sensitivity: float = STEER_SENSITIVITY,
) -> float:
    """
    Compute clamped normalized steering value:
    clamp((hand_x - neutral_x) * sensitivity, -1, 1).
    """
    return clamp((hand_x - neutral_x) * sensitivity, -1.0, 1.0)


def calculate_steering(
    hand_x: float,
    neutral_x: float,
    sensitivity: float = STEER_SENSITIVITY,
    deadzone: float = STEER_DEADZONE,
) -> float:
    """Compute steering with clamp and deadzone applied."""
    raw = calculate_raw_steering(hand_x, neutral_x, sensitivity)
    return apply_deadzone(raw, deadzone)


class DrivingController:
    """
    Driving controller tracking state machine, smoothed steering,
    tracking loss, and active virtual key outputs.
    """

    def __init__(
        self,
        neutral_x: float = 0.5,
        steer_sensitivity: float = STEER_SENSITIVITY,
        ema_alpha: float = EMA_ALPHA,
        loss_frames_limit: int = LOSS_FRAMES_LIMIT,
        steer_deadzone: float = STEER_DEADZONE,
        steer_fire_threshold: float = STEER_FIRE_THRESHOLD,
    ) -> None:
        self.neutral_x = neutral_x
        self.is_calibrated = False
        self.steer_sensitivity = steer_sensitivity
        self.ema_alpha = ema_alpha
        self.loss_frames_limit = loss_frames_limit
        self.steer_deadzone = steer_deadzone
        self.steer_fire_threshold = steer_fire_threshold

        self.smoothed_steering: float = 0.0
        self.raw_steering: float = 0.0
        self.hand_x: float = neutral_x
        self.state: DrivingState = DrivingState.NEUTRAL
        self.loss_frame_count: int = 0
        self.active_keys: List[str] = []
        self._has_seen_hand: bool = False

    def calibrate(
        self,
        hand_or_x: Optional[Any] = None,
        rightmost_hand: Optional[Any] = None,
        hands: Optional[Sequence[Any]] = None,
    ) -> float:
        """
        Calibrate neutral_x from hand(s) or horizontal position.
        When two hands are present, calibrates to the midpoint:
            hand_x = (get_hand_x(leftmost_hand) + get_hand_x(rightmost_hand)) / 2.0.
        When one hand is present, calibrates to that hand's position:
            hand_x = get_hand_x(leftmost_hand).
        Immediately recenters smoothed steering to 0 and clears steering keys.
        If no hands are present, returns current neutral_x unchanged.
        """
        # Check if empty sequence was passed -> keep existing neutral_x
        if (hands is not None and len(hands) == 0) or (
            hand_or_x is not None and isinstance(hand_or_x, (list, tuple)) and len(hand_or_x) == 0
        ):
            return self.neutral_x

        left_h, right_h = _resolve_hands(hand_or_x, rightmost_hand, hands)

        if left_h is None and right_h is None:
            return self.neutral_x

        if left_h is not None and right_h is not None:
            hand_x = (get_hand_x(left_h) + get_hand_x(right_h)) / 2.0
        elif left_h is not None:
            hand_x = get_hand_x(left_h)
        else:
            hand_x = get_hand_x(right_h)

        self.neutral_x = float(hand_x)
        self.hand_x = float(hand_x)
        self.is_calibrated = True
        self.smoothed_steering = 0.0
        self.raw_steering = 0.0
        self.loss_frame_count = 0
        self._has_seen_hand = True
        self.active_keys = [k for k in self.active_keys if k not in ("a", "d")]
        return self.neutral_x

    def update(
        self,
        leftmost_hand: Any = None,
        rightmost_hand: Any = None,
        hands: Optional[Sequence[Any]] = None,
    ) -> DrivingState:
        """
        Update controller state for current frame.

        Args:
            leftmost_hand: Detected leftmost hand landmarks (or single hand, or sequence of hands).
            rightmost_hand: Detected rightmost hand landmarks.
            hands: Optional sequence of hands if not pre-separated.

        Returns:
            Current DrivingState.
        """
        leftmost_hand, rightmost_hand = _resolve_hands(leftmost_hand, rightmost_hand, hands)

        # Tracking-loss handling (when 0 hands are detected)
        if leftmost_hand is None and rightmost_hand is None:
            if not self._has_seen_hand:
                self.smoothed_steering = 0.0
                self.raw_steering = 0.0
                self.hand_x = self.neutral_x
                self.active_keys = []
                self.state = DrivingState.NEUTRAL
                self.loss_frame_count = 0
                return self.state

            self.loss_frame_count += 1
            if self.loss_frame_count >= self.loss_frames_limit:
                # Force smoothed steering to 0 and clear all active outputs
                self.smoothed_steering = 0.0
                self.raw_steering = 0.0
                self.hand_x = self.neutral_x
                self.active_keys = []
                self.state = DrivingState.NEUTRAL
                return self.state

            return self.state

        # At least one hand is present
        self.loss_frame_count = 0
        self._has_seen_hand = True

        # Calculate steering position
        if leftmost_hand is not None and rightmost_hand is not None:
            # Two hands detected: horizontal midpoint of both hands
            hand_x = (get_hand_x(leftmost_hand) + get_hand_x(rightmost_hand)) / 2.0
        elif leftmost_hand is not None:
            # Single hand detected
            hand_x = get_hand_x(leftmost_hand)
        else:
            hand_x = get_hand_x(rightmost_hand)
        self.hand_x = hand_x

        self.raw_steering = calculate_raw_steering(
            hand_x, self.neutral_x, self.steer_sensitivity
        )
        deadzoned_input = apply_deadzone(self.raw_steering, self.steer_deadzone)

        # Exponential moving average smoothing
        self.smoothed_steering = apply_ema(
            deadzoned_input, self.smoothed_steering, self.ema_alpha
        )

        # Clean zeroing in deadzone or micro-residuals
        if deadzoned_input == 0.0 and abs(self.smoothed_steering) < self.steer_deadzone:
            self.smoothed_steering = 0.0
        elif abs(self.smoothed_steering) < 1e-4:
            self.smoothed_steering = 0.0

        # Evaluate gestures
        left_fist = is_fist(leftmost_hand) if leftmost_hand is not None else False
        left_palm = is_open_palm(leftmost_hand) if leftmost_hand is not None else False
        right_fist = is_fist(rightmost_hand) if rightmost_hand is not None else False
        right_palm = is_open_palm(rightmost_hand) if rightmost_hand is not None else False

        # State machine resolution
        if leftmost_hand is not None and rightmost_hand is not None:
            # Two hands detected:
            # 1. Both open palms OR one open palm and one fist -> BRAKING (braking priority rule)
            if (left_palm and right_palm) or (left_palm and right_fist) or (left_fist and right_palm):
                self.state = DrivingState.BRAKING
            # 2. Both closed fists -> DRIVING
            elif left_fist and right_fist:
                self.state = DrivingState.DRIVING
            # 3. Relaxed hands / coasting: neither both fists nor both palms -> READY
            else:
                self.state = DrivingState.READY
        else:
            # Single hand detected (backwards compatible):
            hand = leftmost_hand if leftmost_hand is not None else rightmost_hand
            if is_open_palm(hand):
                self.state = DrivingState.BRAKING
            elif is_fist(hand):
                self.state = DrivingState.DRIVING
            else:
                self.state = DrivingState.READY

        # Active virtual keys calculation
        active = []
        if self.state == DrivingState.DRIVING:
            active.append("w")
        elif self.state == DrivingState.BRAKING:
            active.append("s")

        if self.state != DrivingState.NEUTRAL:
            if self.smoothed_steering < -self.steer_fire_threshold:
                active.append("a")
            elif self.smoothed_steering > self.steer_fire_threshold:
                active.append("d")

        self.active_keys = sorted(active)
        return self.state

    def get_active_keys(self) -> List[str]:
        """Return list of active virtual keys."""
        return list(self.active_keys)

    def is_key_active(self, key: str) -> bool:
        """Check if a specific virtual key is active."""
        return key in self.active_keys

    def reset(self) -> None:
        """Reset all tracking and state."""
        self.smoothed_steering = 0.0
        self.raw_steering = 0.0
        self.hand_x = self.neutral_x
        self.state = DrivingState.NEUTRAL
        self.loss_frame_count = 0
        self.active_keys = []
        self._has_seen_hand = False


# Default controller singleton and helper functions
_default_controller = DrivingController()


def calibrate(
    hand_or_x: Optional[Any] = None,
    rightmost_hand: Optional[Any] = None,
    hands: Optional[Sequence[Any]] = None,
) -> float:
    """Set neutral_x from hand(s) or x-position using default controller."""
    return _default_controller.calibrate(hand_or_x, rightmost_hand, hands)


def set_neutral_x(neutral_x: float) -> float:
    """Set neutral_x directly on default controller."""
    return _default_controller.calibrate(neutral_x)


def update(
    leftmost_hand: Any = None,
    rightmost_hand: Any = None,
    hands: Optional[Sequence[Any]] = None,
) -> DrivingState:
    """Update driving state using default controller."""
    return _default_controller.update(leftmost_hand, rightmost_hand, hands)


def get_active_keys() -> List[str]:
    """Get active virtual keys from default controller."""
    return _default_controller.get_active_keys()


def get_smoothed_steering() -> float:
    """Get smoothed steering value from default controller."""
    return _default_controller.smoothed_steering


def get_driving_state() -> DrivingState:
    """Get current state from default controller."""
    return _default_controller.state


def reset_controller() -> None:
    """Reset default controller."""
    _default_controller.reset()
