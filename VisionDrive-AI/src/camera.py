"""Camera capture module for VisionDrive-AI."""

from typing import Optional, Tuple
import cv2
import numpy as np

try:
    from src.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
except ImportError:
    from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


class Camera:
    """Manages webcam capture with explicit resolution and horizontal flipping."""

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self.cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.release()
            raise RuntimeError(
                f"Failed to open camera at index {self.camera_index}."
            )

        # Explicitly configure resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the camera, horizontally flipped.

        Returns:
            Tuple of (success_flag, flipped_bgr_frame).
        """
        if self.cap is None:
            return False, None

        try:
            if not self.cap.isOpened():
                return False, None

            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None

            # Horizontal flip (mirror view)
            flipped = cv2.flip(frame, 1)
            return True, flipped
        except Exception:
            return False, None

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Convenience wrapper returning horizontally flipped frame or None.
        """
        ret, frame = self.read()
        return frame if ret else None

    def release(self) -> None:
        """Safely release the camera device."""
        cap = getattr(self, "cap", None)
        if cap is not None:
            try:
                if hasattr(cap, "isOpened") and cap.isOpened():
                    cap.release()
            except Exception:
                pass
            finally:
                self.cap = None

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
