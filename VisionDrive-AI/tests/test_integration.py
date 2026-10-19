import sys
from pathlib import Path
import unittest
import numpy as np

# Ensure VisionDrive-AI root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision import Vision
from src.gestures import is_fist, is_open_palm, count_curled_fingers, count_extended_fingers


class TestMediaPipeIntegration(unittest.TestCase):
    """Integration test with actual MediaPipe Hands engine initialized."""

    def setUp(self):
        self.vision = Vision()

    def tearDown(self):
        self.vision.close()

    def test_real_mediapipe_blank_frame(self):
        # A blank frame should be processed by real MediaPipe and return [] without error
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        hands = self.vision.process(blank)
        self.assertEqual(hands, [])

    def test_draw_landmarks_blank_frame(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        out = self.vision.draw_landmarks(blank, [])
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (480, 640, 3))

    def test_real_image_open_palm(self):
        import cv2
        from pathlib import Path
        img_path = Path(__file__).parent / "test_data" / "open_palm.jpg"
        if not img_path.exists():
            self.skipTest("Test image open_palm.jpg not found")
        img = cv2.imread(str(img_path))
        hands = self.vision.process(img)
        self.assertGreaterEqual(len(hands), 1)
        hand = hands[0]
        self.assertTrue(is_open_palm(hand))
        self.assertFalse(is_fist(hand))
        self.assertGreaterEqual(count_extended_fingers(hand), 3)

    def test_real_image_closed_fist(self):
        import cv2
        from pathlib import Path
        img_path = Path(__file__).parent / "test_data" / "closed_fist.jpg"
        if not img_path.exists():
            self.skipTest("Test image closed_fist.jpg not found")
        img = cv2.imread(str(img_path))
        hands = self.vision.process(img)
        self.assertGreaterEqual(len(hands), 1)
        hand = hands[0]
        self.assertTrue(is_fist(hand))
        self.assertFalse(is_open_palm(hand))
        self.assertGreaterEqual(count_curled_fingers(hand), 3)


if __name__ == "__main__":
    unittest.main()
