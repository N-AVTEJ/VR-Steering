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
from src.driving import DrivingController, DrivingState, get_hand_x
from src.controller import KeyboardController


_get_hand_x = get_hand_x


def main() -> None:
    camera = None
    vision = None
    controller = None
    driving_controller = DrivingController()

    try:
        controller = KeyboardController()
        camera = Camera()
        vision = Vision()

        print("VisionDrive-AI started.")
        print("Controls: 'c' = Calibrate neutral steering, 'q' = Quit")

        prev_time = time.time()
        fps = 0.0

        while True:
            try:
                ret, frame = camera.read()
            except Exception:
                ret, frame = False, None

            if not ret or frame is None:
                print("Camera disconnected or failed to read frame. Exiting cleanly...")
                if controller is not None:
                    controller.release_all()
                if camera is not None:
                    camera.release()
                break

            # Calculate FPS
            curr_time = time.time()
            time_delta = curr_time - prev_time
            prev_time = curr_time
            if time_delta > 0:
                instant_fps = 1.0 / time_delta
                fps = 0.9 * fps + 0.1 * instant_fps if fps > 0 else instant_fps

            # Vision: process frame (returns hands sorted leftmost to rightmost)
            hands = vision.process(frame)

            # Determine leftmost and rightmost hands from sorted detections
            leftmost = hands[0] if len(hands) >= 1 else None
            rightmost = hands[-1] if len(hands) >= 2 else None

            # Update driving controller (steering, smoothing, tracking-loss, state, active keys)
            state = driving_controller.update(leftmost_hand=leftmost, rightmost_hand=rightmost)
            smoothed_steer = driving_controller.smoothed_steering
            active_keys = driving_controller.get_active_keys()
            controller.apply(active_keys)
            keys_str = ", ".join(active_keys) if active_keys else "None"

            # Hand tracking information and dual gesture status
            num_hands = len(hands)
            left_f = is_fist(leftmost) if leftmost is not None else False
            left_p = is_open_palm(leftmost) if leftmost is not None else False
            right_f = is_fist(rightmost) if rightmost is not None else False
            right_p = is_open_palm(rightmost) if rightmost is not None else False

            if num_hands >= 2:
                mid_x = (get_hand_x(leftmost) + get_hand_x(rightmost)) / 2.0
                left_stat = "FIST" if left_f else ("PALM" if left_p else "RELAXED")
                right_stat = "FIST" if right_f else ("PALM" if right_p else "RELAXED")
                if (left_p and right_p) or (left_p and right_f) or (left_f and right_p):
                    dual_gesture_desc = "BRAKE"
                elif left_f and right_f:
                    dual_gesture_desc = "DRIVE"
                else:
                    dual_gesture_desc = "COAST"

                print(
                    f"[Hands: {num_hands}] Midpoint X: {mid_x:.3f} | State: {state.value:<7} | "
                    f"Steer: {smoothed_steer:+.2f} | Keys: [{keys_str}] | "
                    f"Dual: {dual_gesture_desc} (L:{left_stat}, R:{right_stat}) | FPS: {fps:.1f}"
                )
            elif num_hands == 1:
                h_x = get_hand_x(leftmost)
                h_stat = "FIST" if left_f else ("PALM" if left_p else "RELAXED")
                print(
                    f"[Hands: 1] Hand X: {h_x:.3f} | State: {state.value:<7} | "
                    f"Steer: {smoothed_steer:+.2f} | Keys: [{keys_str}] | "
                    f"Gesture: {h_stat} | FPS: {fps:.1f}"
                )
            else:
                print(
                    f"[Hands: 0] State: {state.value:<7} | "
                    f"Steer: {smoothed_steer:+.2f} | Keys: [{keys_str}] | FPS: {fps:.1f}"
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

            if num_hands >= 2:
                mid_x = (get_hand_x(leftmost) + get_hand_x(rightmost)) / 2.0
                steer_text = f"STEER: {smoothed_steer:+.2f} (Midpoint: {mid_x:.2f}, Neutral: {driving_controller.neutral_x:.2f})"
            elif num_hands == 1:
                h_x = get_hand_x(leftmost)
                steer_text = f"STEER: {smoothed_steer:+.2f} (Hand: {h_x:.2f}, Neutral: {driving_controller.neutral_x:.2f})"
            else:
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

            # Dual-hand / single-hand gesture status overlay on video window
            if num_hands >= 2:
                left_stat = "FIST" if left_f else ("PALM" if left_p else "RELAXED")
                right_stat = "FIST" if right_f else ("PALM" if right_p else "RELAXED")
                status_dual = f"DUAL: {dual_gesture_desc} (L: {left_stat} | R: {right_stat})"
                cv2.putText(
                    frame,
                    status_dual,
                    (15, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    state_color,
                    2,
                    cv2.LINE_AA,
                )

                # Draw tracking midpoint marker and neutral guide on frame
                h_frame, w_frame = frame.shape[:2]
                mid_px = int(mid_x * w_frame)
                neut_px = int(driving_controller.neutral_x * w_frame)
                # Neutral center guide tick mark (top)
                cv2.line(frame, (neut_px, 0), (neut_px, 25), (100, 255, 100), 2)
                # Current midpoint marker
                cv2.circle(frame, (mid_px, int(h_frame / 2)), 8, (0, 255, 255), -1)
                cv2.line(frame, (mid_px, int(h_frame / 2) - 15), (mid_px, int(h_frame / 2) + 15), (0, 255, 255), 2)

            elif num_hands == 1:
                h_stat = "FIST (DRIVE)" if left_f else ("PALM (BRAKE)" if left_p else "RELAXED (COAST)")
                status_single = f"HAND: {h_stat}"
                cv2.putText(
                    frame,
                    status_single,
                    (15, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    state_color,
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
            if key in (ord("q"), ord("Q")):
                print("Quit requested by user.")
                break
            elif key in (ord("c"), ord("C")):
                if len(hands) >= 2:
                    new_neutral = driving_controller.calibrate(leftmost, rightmost)
                    if controller is not None:
                        controller.apply(driving_controller.get_active_keys())
                    print(f"[CALIBRATION] Neutral X calibrated to: {new_neutral:.3f}")
                elif len(hands) == 1:
                    calib_hand = leftmost if leftmost is not None else hands[0]
                    new_neutral = driving_controller.calibrate(calib_hand)
                    if controller is not None:
                        controller.apply(driving_controller.get_active_keys())
                    print(f"[CALIBRATION] Neutral X calibrated to: {new_neutral:.3f}")
                else:
                    print("[CALIBRATION] Warning: No hand detected to calibrate neutral position.")

    except KeyboardInterrupt:
        print("\nShutdown requested by user (Ctrl+C). Exiting cleanly...")
    except Exception as exc:
        print(f"Error in VisionDrive-AI: {exc}", file=sys.stderr)
        raise
    finally:
        print("Cleaning up resources...")
        if controller is not None:
            try:
                controller.release_all()
            except Exception:
                pass
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Camera released, keys released, and windows destroyed.")


if __name__ == "__main__":
    main()
