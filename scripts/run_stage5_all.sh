#!/usr/bin/env bash
# Stage 5 sequential runner: 5A → 5B → 5C → 5D (5E gated).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
LOG="$ROOT/runs/stage5_console.log"
mkdir -p "$ROOT/runs"

{
  echo "==== Stage5 start $(date -Is) ===="
  echo "cwd=$ROOT branch=$(git branch --show-current) sha=$(git rev-parse HEAD)"

  echo "==== 5A confirmatory ===="
  python scripts/run_stage5a_confirm.py

  echo "==== 5B budget-matched ===="
  python scripts/run_stage5b_budget.py

  echo "==== 5C structured baselines ===="
  python scripts/run_stage5c_baselines.py

  echo "==== 5D gradient path ===="
  python scripts/run_stage5d_gradpath.py

  echo "==== Stage5 experiments done $(date -Is) ===="
  echo "Write reports next; Stage 5E only if 5A–5C pass stop-conditions."
} 2>&1 | tee -a "$LOG"
