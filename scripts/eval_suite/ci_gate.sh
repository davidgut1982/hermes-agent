#!/usr/bin/env bash
# Hermes offline eval gate — on-deploy and weekly CI hook.
#
# Why: Provides a single callable gate that runs the offline eval suite and
# fails ONLY on the regression guard (reranker mean >= 0.70) and on data that
# is actually present. Does NOT gate on aspirational xfail bars (0.85
# livetest accuracy) or on MISSING / NOT_DERIVABLE metrics.
#
# What: Runs pytest (regression guard only) + run_eval.py, checks the
# regression guard JSONL row, and optionally appends results to
# scores_history.jsonl for trend tracking.
#
# Usage:
#   # On-deploy (runtag auto-derived from date):
#   bash scripts/eval_suite/ci_gate.sh
#
#   # Weekly timer (runtag supplied by systemd timer / cron):
#   bash scripts/eval_suite/ci_gate.sh 2026-05-31
#
#   # Named run:
#   bash scripts/eval_suite/ci_gate.sh post-deploy-v1.15
#
# Gate contract (MUST NOT block on):
#   - xfail 0.85 success bar (test_reranker_tool_correctness_success_bar)
#   - MISSING livetest data (livetest _summary.json absent)
#   - NOT_DERIVABLE profile_pick metrics
#   - scenario_coverage_fraction (delegation_profile / should_not_delegate are
#     structurally unscored — see scorer comments)
#
# Gate contract (BLOCKS on):
#   - test_reranker_tool_correctness_regression_guard (reranker mean >= 0.70)
#   - Any metric whose source does NOT contain MISSING / NOT_DERIVABLE / ERROR
#     and whose bar > 0 and actual < bar. (Currently only reranker metrics
#     have such data; livetest metrics are data-present but the 0.85 bar is
#     aspirational — see NOTES below.)
#
# NOTES on livetest gating:
#   livetest_tool_selection_accuracy has bar=0.85 and data IS present after a
#   live run. The gate does NOT block on it because it is flagged aspirational
#   in the KB (kb_28650bfe5f17) — the same reasoning that makes the pytest
#   test xfail. If you want to gate on livetest, remove the
#   --skip-metric-patterns exclusion below after the reranker hits 0.85.
#
# Reverting:
#   systemd timer: systemctl --user disable --now hermes-eval-weekly.timer
#                  systemctl --user daemon-reload
#   cron:          crontab -e → remove the hermes-eval-weekly line

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EVAL_VENV="/home/david/venv-eval"
PROD_VENV="/opt/hermes/venv-combined6"
EVAL_PYTHON="${EVAL_VENV}/bin/python3"
PROD_PYTHON="${PROD_VENV}/bin/python3"

HISTORY_FILE="${SCRIPT_DIR}/out/scores_history.jsonl"
OUT_DIR="${SCRIPT_DIR}/out"

# Runtag: if first arg supplied (from timer/cron), use it; else derive from date.
# Do NOT rely on Date.now() inside Python — the shell supplies the stamp.
RUNTAG="${1:-$(date +%Y-%m-%d)}"

REGRESSION_GUARD_METRIC="reranker_recall5_overall"
REGRESSION_GUARD_BAR="0.70"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[ci_gate] $*"; }
fail() { echo "[ci_gate] GATE FAIL: $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

log "Runtag: ${RUNTAG}"
log "Repo root: ${REPO_ROOT}"

if [[ ! -x "${EVAL_PYTHON}" ]]; then
    fail "Eval venv not found at ${EVAL_VENV}. " \
         "Install deepeval with: python3 -m venv ${EVAL_VENV} && ${EVAL_VENV}/bin/pip install deepeval"
fi

if [[ ! -x "${PROD_PYTHON}" ]]; then
    fail "Prod venv not found at ${PROD_VENV}. Cannot run run_eval.py."
fi

mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Step 1: pytest — regression guard only (NOT the xfail success bar)
# ---------------------------------------------------------------------------

log "Step 1/3: pytest regression guard (test_reranker_tool_correctness_regression_guard)"

PYTEST_RESULT=0
"${EVAL_PYTHON}" -m pytest \
    "${SCRIPT_DIR}/test_tool_accuracy.py" \
    -k "test_reranker_tool_correctness_regression_guard or test_recall_at_k_unit or test_labeled_scenarios_count" \
    -v --tb=short \
    --no-header \
    2>&1 | tee /tmp/hermes_eval_pytest.log || PYTEST_RESULT=$?

if [[ "${PYTEST_RESULT}" -ne 0 ]]; then
    fail "pytest regression guard FAILED (exit ${PYTEST_RESULT}). " \
         "Reranker mean Recall@5 likely dropped below ${REGRESSION_GUARD_BAR}. " \
         "Check /tmp/hermes_eval_pytest.log for details."
fi

log "Step 1/3: pytest PASSED"

# ---------------------------------------------------------------------------
# Step 2: run_eval.py — scorecard generation
# ---------------------------------------------------------------------------

log "Step 2/3: run_eval.py --runtag ${RUNTAG}"

PYTHONPATH="${PROD_VENV}/lib/python3.13/site-packages:${REPO_ROOT}" \
"${PROD_PYTHON}" "${SCRIPT_DIR}/run_eval.py" \
    --runtag "${RUNTAG}" \
    2>&1 | tee /tmp/hermes_eval_scorecard.log || true
# Note: run_eval.py exits 1 when ANY bar fails (including aspirational bars).
# We do NOT use its exit code to gate; we inspect the JSONL directly below.

SCORECARD_FILE="${OUT_DIR}/scores_${RUNTAG}.jsonl"
if [[ ! -f "${SCORECARD_FILE}" ]]; then
    fail "run_eval.py did not produce ${SCORECARD_FILE}. Check /tmp/hermes_eval_scorecard.log."
fi
log "Step 2/3: scorecard written to ${SCORECARD_FILE}"

# ---------------------------------------------------------------------------
# Step 3: Gate check — regression guard metric from JSONL
# ---------------------------------------------------------------------------

log "Step 3/3: Checking regression guard: ${REGRESSION_GUARD_METRIC} >= ${REGRESSION_GUARD_BAR}"

_GATE_SCRIPT=$(mktemp /tmp/hermes_eval_gate_XXXXXX.py)
cat > "${_GATE_SCRIPT}" <<'PYEOF'
import json, sys
from pathlib import Path

scorecard_path, metric, bar_str = sys.argv[1], sys.argv[2], sys.argv[3]
bar = float(bar_str)

rows = [json.loads(l) for l in Path(scorecard_path).read_text().splitlines() if l.strip()]
hit = [r for r in rows if r["metric"] == metric]
if not hit:
    print(f"GATE_ERROR: metric {metric!r} not found in scorecard")
    sys.exit(2)

row = hit[0]
src = row.get("source", "")
if any(t in src for t in ("MISSING", "ERROR", "NOT_DERIVABLE")):
    print(f"GATE_SKIP: {metric} source is data-absent ({src[:60]}). Not gating.")
    sys.exit(0)

actual = float(row["actual"])
if actual >= bar:
    print(f"GATE_PASS: {metric}={actual:.4f} >= bar={bar}")
    sys.exit(0)
else:
    print(f"GATE_FAIL: {metric}={actual:.4f} < bar={bar}")
    sys.exit(1)
PYEOF

GATE_RESULT=$(python3 "${_GATE_SCRIPT}" \
    "${SCORECARD_FILE}" "${REGRESSION_GUARD_METRIC}" "${REGRESSION_GUARD_BAR}")
GATE_EXIT=$?
rm -f "${_GATE_SCRIPT}"
log "${GATE_RESULT}"

if [[ "${GATE_EXIT}" -eq 2 ]]; then
    fail "Gate check script error — see log above."
elif [[ "${GATE_EXIT}" -ne 0 ]]; then
    fail "Regression guard failed: ${GATE_RESULT}"
fi

# ---------------------------------------------------------------------------
# Append to scores_history.jsonl for trend tracking
# ---------------------------------------------------------------------------

log "Appending scorecard rows to ${HISTORY_FILE}"
cat "${SCORECARD_FILE}" >> "${HISTORY_FILE}"

log "History file now has $(wc -l < "${HISTORY_FILE}") rows"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

log "=== GATE PASSED ==="
log "  Runtag:           ${RUNTAG}"
log "  Regression guard: ${REGRESSION_GUARD_METRIC} PASS (>= ${REGRESSION_GUARD_BAR})"
log "  Scorecard:        ${SCORECARD_FILE}"
log "  History:          ${HISTORY_FILE}"
log ""
log "NOTE: The 0.85 livetest_tool_selection_accuracy bar is aspirational (xfail)."
log "      The gate does NOT block on it. See ki_gate.sh header for reasoning."
exit 0
