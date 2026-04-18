#!/usr/bin/env bash
# run_coverage.sh — run tests + coverage, open HTML report
# Usage: bash scripts/run_coverage.sh [--open]

set -euo pipefail

OPEN_REPORT=false
if [[ "${1:-}" == "--open" ]]; then
  OPEN_REPORT=true
fi

echo "==> Running pytest with coverage..."
python -m pytest tests/ \
  --cov=src \
  --cov-report=term-missing \
  --cov-report=html:htmlcov \
  -v

echo ""
echo "==> Coverage report written to htmlcov/index.html"

if $OPEN_REPORT; then
  if command -v xdg-open &>/dev/null; then
    xdg-open htmlcov/index.html
  elif command -v open &>/dev/null; then
    open htmlcov/index.html
  elif command -v start &>/dev/null; then
    start htmlcov/index.html
  fi
fi
