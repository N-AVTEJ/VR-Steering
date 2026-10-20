"""Tests for KeyboardController and DirectInput key mapping."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Ensure VisionDrive-AI root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.controller import KeyboardController


class TestKeyboardController:
    """Unit tests for the KeyboardController."""

    def test_init_simulate_mode(self):
        controller = KeyboardController(simulate=True)
        assert controller.simulate is True
        assert controller.held_keys == set()

    def test_apply_adds_and_releases_keys(self, capsys):
        controller = KeyboardController(simulate=True)

        # Apply 'w'
        controller.apply(["w"])
        assert controller.held_keys == {"w"}
        out = capsys.readouterr().out
        assert "[SIM] keyDown w" in out

        # Apply 'w' again -> should NOT double press 'w'
        controller.apply(["w"])
        out = capsys.readouterr().out
        assert out == ""

        # Apply 'w' and 'd' -> should only press 'd', not re-press 'w'
        controller.apply(["w", "d"])
        assert controller.held_keys == {"w", "d"}
        out = capsys.readouterr().out
        assert "[SIM] keyDown d" in out
        assert "keyDown w" not in out

        # Apply 's' -> should release 'w' and 'd' before pressing 's'
        controller.apply(["s"])
        assert controller.held_keys == {"s"}
        out = capsys.readouterr().out
        assert "[SIM] keyUp w" in out or "[SIM] keyUp d" in out
        assert "[SIM] keyDown s" in out
        # Ensure keyUp happens before keyDown
        first_keyup = min(out.find("[SIM] keyUp w"), out.find("[SIM] keyUp d"))
        keydown_pos = out.find("[SIM] keyDown s")
        assert first_keyup < keydown_pos

        # Apply empty -> releases 's'
        controller.apply([])
        assert controller.held_keys == set()
        out = capsys.readouterr().out
        assert "[SIM] keyUp s" in out

    def test_release_all(self, capsys):
        controller = KeyboardController(simulate=True)
        controller.apply(["w", "a"])
        assert controller.held_keys == {"w", "a"}

        controller.release_all()
        assert controller.held_keys == set()
        out = capsys.readouterr().out
        assert "[SIM] keyUp w" in out
        assert "[SIM] keyUp a" in out

        # Second release_all call should be idempotent and do nothing
        controller.release_all()
        out2 = capsys.readouterr().out
        assert out2 == ""

    def test_mocked_pydirectinput_calls(self):
        controller = KeyboardController(simulate=False)
        with patch("src.controller.pydirectinput") as mock_pdi:
            mock_pdi.keyDown = MagicMock()
            mock_pdi.keyUp = MagicMock()

            controller.apply(["w", "d"])
            mock_pdi.keyDown.assert_any_call("w", _pause=False)
            mock_pdi.keyDown.assert_any_call("d", _pause=False)

            controller.apply(["s"])
            mock_pdi.keyUp.assert_any_call("w", _pause=False)
            mock_pdi.keyUp.assert_any_call("d", _pause=False)
            mock_pdi.keyDown.assert_any_call("s", _pause=False)

            controller.release_all()
            mock_pdi.keyUp.assert_any_call("s", _pause=False)
            assert controller.held_keys == set()

    def test_case_insensitivity(self):
        controller = KeyboardController(simulate=True)
        controller.apply(["W", "A"])
        assert controller.held_keys == {"w", "a"}
        controller.apply(["w", "a"])
        # Should not re-trigger because lowercase matches
        assert controller.held_keys == {"w", "a"}
        controller.release_all()

    def test_apply_none_releases_all(self, capsys):
        controller = KeyboardController(simulate=True)
        controller.apply(["w", "d"])
        assert controller.held_keys == {"w", "d"}
        capsys.readouterr()

        controller.apply(None)
        assert controller.held_keys == set()
        out = capsys.readouterr().out
        assert "[SIM] keyUp w" in out
        assert "[SIM] keyUp d" in out

    def test_apply_filters_none_and_empty(self, capsys):
        controller = KeyboardController(simulate=True)
        controller.apply(["w", None, "", "  "])
        assert controller.held_keys == {"w"}
        out = capsys.readouterr().out
        assert "[SIM] keyDown w" in out
        assert "none" not in out
        controller.release_all()

    def test_camel_case_aliases(self, capsys):
        controller = KeyboardController(simulate=True)
        controller.keyDown("w")
        assert "[SIM] keyDown w" in capsys.readouterr().out
        controller.keyUp("w")
        assert "[SIM] keyUp w" in capsys.readouterr().out

    def test_key_down_up_ignore_none_and_empty(self, capsys):
        controller = KeyboardController(simulate=True)
        controller.key_down(None)
        controller.key_down("")
        controller.key_down("   ")
        controller.key_up(None)
        controller.key_up("")
        controller.key_up("   ")
        out = capsys.readouterr().out
        assert out == ""

    def test_context_manager(self, capsys):
        with KeyboardController(simulate=True) as ctrl:
            ctrl.apply(["w"])
            assert ctrl.held_keys == {"w"}
        assert ctrl.held_keys == set()
        assert "[SIM] keyUp w" in capsys.readouterr().out

    def test_release_all_handles_exceptions(self):
        controller = KeyboardController(simulate=False)
        controller.held_keys = {"w", "a"}
        with patch.object(controller, "key_up", side_effect=[Exception("fail"), None]):
            controller.release_all()
            assert controller.held_keys == set()

    def test_apply_tracks_keys_incrementally_on_error(self):
        controller = KeyboardController(simulate=False)
        # If key_down fails on the second key, the first key should still be recorded in held_keys
        with patch.object(controller, "key_down", side_effect=[None, RuntimeError("OS error")]):
            with pytest.raises(RuntimeError):
                controller.apply(["w", "d"])
            # 'w' succeeded and should be in held_keys
            assert "w" in controller.held_keys
            assert "d" not in controller.held_keys
        controller.release_all()

    def test_fallback_when_pydirectinput_is_none(self, capsys):
        with patch("src.controller.pydirectinput", None):
            controller = KeyboardController(simulate=None)
            assert controller.simulate is True
            controller.key_down("w")
            out = capsys.readouterr().out
            assert "[SIM] keyDown w" in out

            controller.key_up("w")
            out2 = capsys.readouterr().out
            assert "[SIM] keyUp w" in out2

    def test_main_camera_disconnect_exits_cleanly(self, capsys):
        """Camera read failure should print disconnect message and exit cleanly."""
        import main

        with patch("main.Camera") as mock_cam_cls, \
             patch("main.Vision") as mock_vis_cls, \
             patch("main.KeyboardController") as mock_ctrl_cls, \
             patch("cv2.destroyAllWindows"):

            mock_cam_inst = MagicMock()
            # camera.read() fails on first frame
            mock_cam_inst.read.return_value = (False, None)
            mock_cam_cls.return_value = mock_cam_inst

            mock_ctrl_inst = MagicMock()
            mock_ctrl_cls.return_value = mock_ctrl_inst

            main.main()

            out = capsys.readouterr().out
            assert "Camera disconnected or failed to read frame. Exiting cleanly..." in out
            mock_ctrl_inst.release_all.assert_called()
            mock_cam_inst.release.assert_called()

    def test_main_camera_read_exception_exits_cleanly(self, capsys):
        """Camera read raising an exception mid-run should exit cleanly."""
        import main

        with patch("main.Camera") as mock_cam_cls, \
             patch("main.Vision") as mock_vis_cls, \
             patch("main.KeyboardController") as mock_ctrl_cls, \
             patch("cv2.destroyAllWindows"):

            mock_cam_inst = MagicMock()
            mock_cam_inst.read.side_effect = RuntimeError("USB disconnected")
            mock_cam_cls.return_value = mock_cam_inst

            mock_ctrl_inst = MagicMock()
            mock_ctrl_cls.return_value = mock_ctrl_inst

            main.main()

            out = capsys.readouterr().out
            assert "Camera disconnected or failed to read frame. Exiting cleanly..." in out
            mock_ctrl_inst.release_all.assert_called()

    def test_main_wires_controller_apply(self):
        """Verify main calls controller.apply with driving controller's active keys."""
        import main
        import numpy as np
        from src.driving import DrivingState

        with patch("main.Camera") as mock_cam_cls, \
             patch("main.Vision") as mock_vis_cls, \
             patch("main.KeyboardController") as mock_ctrl_cls, \
             patch("main.DrivingController") as mock_driving_cls, \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", return_value=ord("q")), \
             patch("cv2.destroyAllWindows"):

            mock_cam_inst = MagicMock()
            mock_cam_inst.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cam_cls.return_value = mock_cam_inst

            mock_vis_inst = MagicMock()
            mock_vis_inst.process.return_value = []
            mock_vis_cls.return_value = mock_vis_inst

            mock_ctrl_inst = MagicMock()
            mock_ctrl_cls.return_value = mock_ctrl_inst

            mock_driving_inst = MagicMock()
            mock_driving_inst.update.return_value = DrivingState.DRIVING
            mock_driving_inst.smoothed_steering = 0.25
            mock_driving_inst.get_active_keys.return_value = ["w", "d"]
            mock_driving_inst.neutral_x = 0.5
            mock_driving_cls.return_value = mock_driving_inst

            main.main()

            mock_ctrl_inst.apply.assert_called_with(["w", "d"])
            mock_ctrl_inst.release_all.assert_called()

