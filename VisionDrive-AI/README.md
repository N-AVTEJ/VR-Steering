# VisionDrive-AI

AI-powered vision-based vehicle steering and throttle control system using a standard webcam, OpenCV, and MediaPipe Hands. VisionDrive-AI translates real-time hand gestures and spatial positioning into hardware-level Windows DirectInput keyboard signals (`W`, `A`, `S`, `D`) for driving games.

---

## Features

- **Gesture-Based Throttle & Brake**: Accelerate with a closed fist on your steering hand; brake instantaneously by raising an open palm with your secondary hand (braking always overrides acceleration).
- **Spatial Steering with EMA Smoothing**: Steer left and right by moving your hand horizontally across the camera frame, smoothed by an Exponential Moving Average filter with deadzone rejection.
- **DirectInput Hardware Key Pressing**: Powered by `pydirectinput` for game compatibility, with automatic stuck-key prevention (`release_all` on normal exit, camera disconnect, or `Ctrl+C`).
- **One-Key Neutral Calibration**: Center your steering hand anywhere in the camera frame and press `c` to calibrate your neutral position on the fly.
- **Fail-Safe & Grace Periods**: Grace period for temporary hand occlusion, resetting safely to `NEUTRAL` and releasing all held keys if tracking is lost or the camera disconnects.

---

## Control Mapping Table

| Hand / Gesture | Condition | Driving State | Virtual Key | Action |
|---|---|---|---|---|
| **Left Hand: Closed Fist** | Curled finger count $\ge 3$ | `DRIVING` | `W` | Accelerate / Drive Forward |
| **Right Hand: Open Palm** | Extended finger count $\ge 3$ | `BRAKING` | `S` | Brake / Reverse (overrides Drive) |
| **Left Hand: Steer Left** | Position $< \text{neutral\_x} - \text{threshold}$ | `DRIVING` / `READY` / `BRAKING` | `A` | Steer Left |
| **Left Hand: Steer Right** | Position $> \text{neutral\_x} + \text{threshold}$ | `DRIVING` / `READY` / `BRAKING` | `D` | Steer Right |
| **Both Fists or Both Palms** | Conflicting / ambiguous inputs | `READY` | None | Coast / Neutral Throttle & Brake |
| **Left Hand Relaxed** | Neither fist nor open palm | `READY` | None | Coast / Ready |
| **Tracking Lost** | $> 8$ consecutive frames lost | `NEUTRAL` | None | All keys released immediately |
| **Keyboard 'c' / 'C'** | Active video window | Calibration | — | Set current hand $x$ as neutral center |
| **Keyboard 'q' / 'Q'** | Active video window | Shutdown | — | Clean exit (releases all keys & camera) |

---

## Installation

### Prerequisites
- Windows 10 or 11
- Python 3.10 or 3.11
- Standard USB webcam or integrated laptop camera

### Setup Steps

1. **Clone the repository and enter the directory**:
   ```bash
   git clone <repository_url>
   cd VisionDrive-AI
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install pinned dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## How to Run

1. Make sure your webcam is plugged in and not in use by another application.
2. Launch VisionDrive-AI:
   ```powershell
   python main.py
   ```
3. A video window titled **VisionDrive-AI** will open showing your mirrored webcam feed with an on-screen HUD (FPS, current driving state, steering offset, active keys, and hand landmark skeletons).
4. Launch your desired keyboard-compatible driving game in windowed or borderless-fullscreen mode alongside VisionDrive-AI.

---

## Calibration Instructions

1. Sit in your normal, comfortable driving position in front of your camera.
2. Raise your steering hand (leftmost hand in your camera view) to where you naturally rest it when driving straight.
3. Tap the **`c`** key on your keyboard while the VisionDrive-AI window is focused.
4. The HUD and terminal will confirm:
   ```
   [CALIBRATION] Neutral X calibrated to: 0.485
   ```
5. Your steering offset is now zeroed to your natural rest position. You can recalibrate at any time during gameplay without restarting the application.

---

## Verification & Tests

To execute the test suite (83 unit and integration tests covering gesture recognition, steering math, deadzone boundaries, tracking-loss reset, DirectInput controller safety, and OpenCV pipeline):

```powershell
pytest -v
```

---

## Known Limitations

- **Lighting Sensitivity**: MediaPipe landmark detection relies on monocular RGB feature extraction. Low lighting, severe backlighting, or rapid lighting changes can cause tracking jitter or momentary loss of hand detection.
- **2D Camera Tracking**: A standard 2D monocular webcam cannot measure true 3D depth or rotational torque like a physical steering wheel. Steering is mapped from the horizontal centroid ($x$-axis) displacement relative to the calibrated center point.
- **Keyboard-Only (Digital Inputs)**: VisionDrive-AI generates discrete DirectInput keystrokes (`W`, `A`, `S`, `D`). It does not simulate analog axes (such as smooth $0\text{--}100\%$ throttle pedals or variable gamepad analog sticks). Keystrokes act as full on/off inputs.
- **Game Compatibility Note**: This software was tested and designed strictly for games that accept standard Windows keyboard inputs (`WASD`). Games requiring native DirectInput / XInput analog gamepad axes or steering wheel drivers will only respond if configured to use keyboard controls.
- **No Anti-Cheat Bypass Claims**: VisionDrive-AI uses standard Windows OS user-mode input synthesis via `pydirectinput`. It does not contain kernel drivers or evasion techniques, and makes no claim or attempt to bypass anti-cheat systems. Use caution in multiplayer games with strict third-party anti-cheat software.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
