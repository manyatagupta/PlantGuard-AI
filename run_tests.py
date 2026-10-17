"""
PlantGuard AI - Test Suite Runner
Runs unit and integration tests across all platform modules.
"""

import unittest
import sys

# Windows cp1252 safety
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    print("=" * 65)
    print("PlantGuard AI - Automated Agricultural Test Pipeline")
    print("=" * 65)
    
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\nAll automated tests passed successfully!")
        sys.exit(0)
    else:
        print(f"\nTest suite failed with {len(result.failures)} failure(s) and {len(result.errors)} error(s).")
        sys.exit(1)

if __name__ == "__main__":
    main()
