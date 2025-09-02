#!/usr/bin/env bash
set -Eeuo pipefail
rm -rf logs outputs .pytest_cache __pycache__
echo "[OK] cleaned logs/outputs caches"
