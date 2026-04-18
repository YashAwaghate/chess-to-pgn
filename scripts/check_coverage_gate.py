"""
check_coverage_gate.py — CI gate: fail if any src/ module is below MIN_COVERAGE %.

Usage:
    python scripts/check_coverage_gate.py            # default 80 %
    python scripts/check_coverage_gate.py --min 90

Reads the .coverage file produced by pytest-cov.
Intended for use in CI (Railway, GitHub Actions, etc.).
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=int, default=80, dest="min_coverage",
                        help="Minimum acceptable coverage percent (default: 80)")
    args = parser.parse_args()

    result = subprocess.run(
        ["python", "-m", "coverage", "report", "--include=src/*"],
        capture_output=True, text=True
    )

    lines = result.stdout.splitlines()
    failures = []

    for line in lines:
        parts = line.split()
        # Lines look like: src/pipeline/fen_generator.py   56    3    95%
        if not parts or not parts[0].startswith("src/"):
            continue
        pct_str = parts[-1].rstrip("%")
        try:
            pct = int(pct_str)
        except ValueError:
            continue
        if pct < args.min_coverage:
            failures.append((parts[0], pct))

    if failures:
        print(f"\n[FAIL] Coverage gate ({args.min_coverage}%) not met:")
        for path, pct in failures:
            print(f"  {path}: {pct}%")
        sys.exit(1)
    else:
        print(f"\n[PASS] All modules meet the {args.min_coverage}% coverage gate.")
        sys.exit(0)


if __name__ == "__main__":
    main()
