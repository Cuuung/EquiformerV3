#!/usr/bin/env bash
# 复用 esen 的 torch 2.11 venv 跑本 repo（PYTHONPATH 覆盖 fairchem.core），零重装。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/mnt/afs/home/maoruicong/esen/.venv-torch211"
exec env PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}" "$VENV/bin/python" "$@"
