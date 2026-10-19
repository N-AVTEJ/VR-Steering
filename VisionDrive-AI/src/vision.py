"""MediaPipe Hands wrapper module for VisionDrive-AI."""

from typing import Any, List, Optional
import cv2
import mediapipe as mp
import numpy as np

try:
    from src.config import (
        MAX_NUM_HANDS,
        MIN_DETECTION_CONFIDENCE,
        MIN_TRACKING_CONFIDENCE,
    )
except ImportError:
    from config import (
        MAX_NUM_HANDS,
        MIN_DETECTION_CONFIDENCE,
        MIN_TRACKING_CONFIDENCE,
    )


class Vision:
    """Wraps MediaPipe Hands and extracts sorted hand landmarks."""

    def __init__(
        self,
        max_num_hands: int = MAX_NUM_HANDS,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.max_num_hands = max_num_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _calc_hand_centroid_x(hand: Any) -> float:
        """Compute the horizontal centroid of hand landmarks."""
        lms = hand.landmark if hasattr(hand, "landmark") else hand
        if not lms:
            return 0.0
        return sum(float(lm.x) if hasattr(lm, "x") else float(lm[0]) for lm in lms) / len(lms)

    def process(self, frame: Optional[np.ndarray]) -> List:
        """
        Process a horizontally flipped BGR frame and return raw hand landmarks
        sorted by actual landmark x-position ascending (leftmost hand first, rightmost last).

        Do NOT use MediaPipe's handedness label.

        Args:
            frame: Horizontally flipped BGR frame from camera.

        Returns:
            List of raw hand landmark collections sorted by x-position ascending.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not results or not results.multi_hand_landmarks:
            return []

        # Sort by actual landmark horizontal centroid ascending.
        # Leftmost hand (lowest average x in flipped frame) comes first, rightmost last.
        sorted_hands = sorted(
            results.multi_hand_landmarks,
            key=self._calc_hand_centroid_x,
        )
        return sorted_hands[: self.max_num_hands]

    def draw_landmarks(
        self,
        frame: Optional[np.ndarray],
        hand_landmarks_list: Optional[List],
    ) -> Optional[np.ndarray]:
        """
        Draw hand landmarks and hand connections on the given frame.

        Args:
            frame: BGR frame to draw on (modified in-place).
            hand_landmarks_list: List of hand landmarks returned by process().

        Returns:
            The modified frame with landmarks drawn.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame

        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                if hand_landmarks is not None:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )
        return frame

    def close(self) -> None:
        """Safely release MediaPipe resources."""
        if hasattr(self, "hands") and self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass
            finally:
                self.hands = None

    def __enter__(self) -> "Vision":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
