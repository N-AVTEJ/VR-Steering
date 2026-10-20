"""Virtual keyboard controller wrapping pydirectinput with graceful fallback."""

import atexit
from typing import Any, Iterable, Set, Optional

try:
    import pydirectinput

    # Disable default 100ms pause and failsafe for direct low-latency game input
    pydirectinput.PAUSE = 0.0
    pydirectinput.FAILSAFE = False
    PYDIRECTINPUT_AVAILABLE = True
except (ImportError, Exception):
    pydirectinput = None
    PYDIRECTINPUT_AVAILABLE = False


class KeyboardController:
    """
    Virtual keyboard controller managing hardware-level key presses via DirectInput.
    Maintains active held keys set to prevent double-presses and ensure clean releases.
    """

    def __init__(self, simulate: Optional[bool] = None) -> None:
        """
        Initialize the KeyboardController.

        Args:
            simulate: If True, forces simulation mode (prints [SIM] messages).
                      If None, auto-detects based on pydirectinput availability.
        """
        if simulate is None:
            self.simulate: bool = (pydirectinput is None) or (not PYDIRECTINPUT_AVAILABLE)
        else:
            self.simulate = bool(simulate)

        self.held_keys: Set[str] = set()
        self._atexit_registered: bool = False

        # Register release_all at process exit as an extra safety measure
        try:
            atexit.register(self.release_all)
            self._atexit_registered = True
        except Exception:
            pass

    def key_down(self, key: str) -> None:
        """Press down a key using DirectInput or simulated output."""
        if key is None:
            return
        key_str = str(key).strip().lower()
        if not key_str:
            return

        if self.simulate or pydirectinput is None:
            print(f"[SIM] keyDown {key_str}")
        else:
            try:
                pydirectinput.keyDown(key_str, _pause=False)
            except Exception:
                print(f"[SIM] keyDown {key_str}")

    def key_up(self, key: str) -> None:
        """Release a key using DirectInput or simulated output."""
        if key is None:
            return
        key_str = str(key).strip().lower()
        if not key_str:
            return

        if self.simulate or pydirectinput is None:
            print(f"[SIM] keyUp {key_str}")
        else:
            try:
                pydirectinput.keyUp(key_str, _pause=False)
            except Exception:
                print(f"[SIM] keyUp {key_str}")

    # Aliases matching pydirectinput camelCase naming
    keyDown = key_down
    keyUp = key_up

    def apply(self, wanted_keys_set: Optional[Iterable[str]]) -> None:
        """
        Synchronize currently held keys with the desired set of keys.

        Releases any held key not in the new wanted set before pressing new ones,
        and never double-presses an already-held key.
        """
        if wanted_keys_set is None:
            wanted: Set[str] = set()
        else:
            wanted = {
                str(k).strip().lower()
                for k in wanted_keys_set
                if k is not None and str(k).strip()
            }

        # Release keys not in the wanted set first, updating held_keys as each is released
        to_release = list(self.held_keys - wanted)
        for key in to_release:
            self.key_up(key)
            self.held_keys.discard(key)

        # Press keys newly requested that are not already held, updating held_keys as each is pressed
        to_press = list(wanted - self.held_keys)
        for key in to_press:
            self.key_down(key)
            self.held_keys.add(key)

    def release_all(self) -> None:
        """
        Release all currently held keys to prevent stuck keys on exit.
        Safe to call multiple times.
        """
        if not self.held_keys:
            return

        to_release = list(self.held_keys)
        self.held_keys.clear()
        for key in to_release:
            try:
                self.key_up(key)
            except Exception:
                pass

    def close(self) -> None:
        """Release all held keys and unregister atexit hook."""
        self.release_all()
        if getattr(self, "_atexit_registered", False):
            try:
                atexit.unregister(self.release_all)
                self._atexit_registered = False
            except Exception:
                pass

    def __enter__(self) -> "KeyboardController":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# Default singleton controller instance and convenience alias
Controller = KeyboardController
