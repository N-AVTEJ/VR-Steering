"""Root entry point delegating to VisionDrive-AI/main.py."""

import sys
import runpy
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "VisionDrive-AI"
TARGET_MAIN = PROJECT_DIR / "main.py"


def main() -> None:
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    # Execute VisionDrive-AI/main.py cleanly
    runpy.run_path(str(TARGET_MAIN), run_name="__main__")


if __name__ == "__main__":
    main()
