import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "unittests"


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(TEST_DIR), pattern="*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
