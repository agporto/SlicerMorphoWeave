"""Run each module's standalone regressions in an isolated interpreter.

Usage: python Testing/run_headless_tests.py [--require-native]
This does not replace live Slicer/CMake or biological validation.
"""
import argparse
import importlib
from pathlib import Path
import subprocess
import sys

MODULES = (
    "MorphoWeaveAtlasBuilder",
    "MorphoWeaveModelLibrary",
    "MorphoWeaveLandmarkTransfer",
    "MorphoWeaveShapeCompletion",
    "MorphoWeaveSurfaceSegmentation",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-native", action="store_true",
                        help="Fail if NumPy, SciPy, VTK or pinned rustcpd is unavailable.")
    args = parser.parse_args()
    if args.require_native:
        for name in ("numpy", "scipy", "vtk", "rustcpd"):
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"{name}: {version}", flush=True)
            if name == "rustcpd" and version != "4.0.0":
                parser.error(f"Expected rustcpd 4.0.0, imported {version}")
    root = Path(__file__).resolve().parents[1]
    failures = []
    for module in MODULES:
        print(f"\n=== {module} ===", flush=True)
        result = subprocess.run([
            sys.executable, "-m", "unittest", "discover", "-s",
            str(root / module / "Testing" / "Python"), "-p", "*Test.py", "-v",
        ], cwd=root, check=False)
        if result.returncode:
            failures.append(module)
    if failures:
        print("Failed modules: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("\nAll module test commands passed. Inspect the output for skipped tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
