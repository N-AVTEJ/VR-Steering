"""Main entry point for VisionDrive-AI."""

import sys
import time
from pathlib import Path
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import Camera
from src.vision import Vision
from src.gestures import is_fist, is_open_palm
from src.driving import DrivingController, DrivingState


def _get_hand_x(hand_landmarks) -> float:
    """Compute average horizontal position of hand landmarks."""
    lms = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
    if not lms:
        return 0.5
    return sum(float(lm.x) if hasattr(lm, "x") else float(lm[0]) for lm in lms) / len(lms)


def main() -> None:
    camera = None
    vision = None
    driving_controller = DrivingController()

    try:
        camera = Camera()
        vision = Vision()

        print("VisionDrive-AI started.")
        print("Controls: 'c' = Calibrate neutral steering, 'q' = Quit")

        prev_time = time.time()
        fps = 0.0
        consecutive_read_failures = 0

        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures % 30 == 1:
                    print("Warning: Failed to grab frame from camera.")
                time.sleep(0.03)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quit requested by user.")
                    break
                if consecutive_read_failures > 100:
                    print("Camera connection lost or stream ended. Exiting.")
                    break
                continue
            consecutive_read_failures = 0

            # Calculate FPS
            curr_time = time.time()
            time_delta = curr_time - prev_time
            prev_time = curr_time
            if time_delta > 0:
                instant_fps = 1.0 / time_delta
                fps = 0.9 * fps + 0.1 * instant_fps if fps > 0 else instant_fps

            # Vision: process frame (returns hands sorted leftmost to rightmost)
            hands = vision.process(frame)

            # Determine leftmost and rightmost hands based on detections and horizontal position
            leftmost = None
            rightmost = None
            if len(hands) == 1:
                hand_x = _get_hand_x(hands[0])
                if driving_controller.is_calibrated and abs(hand_x - driving_controller.neutral_x) < 0.35:
                    leftmost = hands[0]
                elif hand_x < 0.5:
                    leftmost = hands[0]
                else:
                    rightmost = hands[0]
            elif len(hands) >= 2:
                leftmost = hands[0]
                rightmost = hands[-1]

            # Update driving controller (steering, smoothing, tracking-loss, state, active keys)
            state = driving_controller.update(leftmost_hand=leftmost, rightmost_hand=rightmost)
            smoothed_steer = driving_controller.smoothed_steering
            active_keys = driving_controller.get_active_keys()
            keys_str = ", ".join(active_keys) if active_keys else "None"

            # Console output
            left_str = (
                f"Fist: {is_fist(leftmost)!s:<5} Palm: {is_open_palm(leftmost)!s:<5}"
                if leftmost is not None
                else "None"
            )
            right_str = (
                f"Fist: {is_fist(rightmost)!s:<5} Palm: {is_open_palm(rightmost)!s:<5}"
                if rightmost is not None
                else "None"
            )
            print(
                f"[Hands: {len(hands)}] State: {state.value:<7} | "
                f"Steer: {smoothed_steer:+.2f} | "
                f"Keys: [{keys_str}] | "
                f"Left -> {left_str} | Right -> {right_str} | FPS: {fps:.1f}"
            )

            # Draw landmarks on frame
            vision.draw_landmarks(frame, hands)

            # Live FPS overlay on the video window
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # State color coding
            state_colors = {
                DrivingState.DRIVING: (0, 255, 0),      # Green
                DrivingState.BRAKING: (0, 0, 255),      # Red
                DrivingState.READY: (0, 255, 255),      # Yellow
                DrivingState.NEUTRAL: (180, 180, 180),  # Gray
            }
            state_color = state_colors.get(state, (255, 255, 255))

            # On-screen HUD overlays
            state_text = f"STATE: {state.value}"
            cv2.putText(
                frame,
                state_text,
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                state_color,
                2,
                cv2.LINE_AA,
            )

            steer_text = f"STEER: {smoothed_steer:+.2f} (Neutral: {driving_controller.neutral_x:.2f})"
            cv2.putText(
                frame,
                steer_text,
                (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            keys_display = f"VIRTUAL KEYS: [{keys_str}]"
            cv2.putText(
                frame,
                keys_display,
                (15, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255) if active_keys else (180, 180, 180),
                2,
                cv2.LINE_AA,
            )

            # Gesture status overlay on the video window
            if leftmost is not None:
                l_f = is_fist(leftmost)
                l_p = is_open_palm(leftmost)
                status_l = f"L (Steer): {'FIST' if l_f else ('PALM' if l_p else 'NONE')}"
                cv2.putText(
                    frame,
                    status_l,
                    (15, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

            if rightmost is not None:
                r_f = is_fist(rightmost)
                r_p = is_open_palm(rightmost)
                status_r = f"R (Brake): {'FIST' if r_f else ('PALM' if r_p else 'NONE')}"
                y_pos = 195 if leftmost is not None else 170
                cv2.putText(
                    frame,
                    status_r,
                    (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

            # Instructions overlay at bottom
            cv2.putText(
                frame,
                "'c': Calibrate neutral | 'q': Quit",
                (15, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            # Display output
            cv2.imshow("VisionDrive-AI", frame)

            # Process key events: 'q' to quit, 'c' to calibrate neutral_x
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested by user.")
                break
            elif key == ord("c"):
                calib_hand = leftmost if leftmost is not None else (hands[0] if hands else None)
                if calib_hand is not None:
                    new_neutral = driving_controller.calibrate(calib_hand)
                    print(f"[CALIBRATION] Neutral X calibrated to: {new_neutral:.3f}")
                else:
                    print("[CALIBRATION] Warning: No hand detected to calibrate neutral position.")

    except Exception as exc:
        print(f"Error in VisionDrive-AI: {exc}", file=sys.stderr)
        raise
    finally:
        print("Cleaning up resources...")
        if camera is not None:
            camera.release()
        if vision is not None:
            vision.close()
        cv2.destroyAllWindows()
        print("Camera released and windows destroyed.")


if __name__ == "__main__":
    main()
