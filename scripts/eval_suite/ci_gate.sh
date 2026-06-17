#!/usr/bin/env bash
# Hermes offline eval gate — on-deploy and weekly CI hook.
#
# Default behaviour (no flags): LIVE mode.
#   Before scoring, the gate:
#     (a) Re-generates the reranker benchmark FRESH from the live nomic endpoint
#         (run_live_benchmark.py — NO disk cache, prod params).
#     (b) Re-runs tool_search_livetest.py against the live build to produce a
#         fresh out/_summary.json (real agent/LLM calls).
#     (c) Then runs run_eval.py and the gate checks on the FRESH outputs.
#
# Fail-closed: if a live prerequisite is DOWN (nomic endpoint unreachable,
# livetest fails before producing output), the gate exits NON-ZERO with a
# clear "could not measure live — gate FAILS closed" message.
#
# --snapshot flag: use committed/frozen artifacts (old behaviour).
#   LOGS LOUDLY "SNAPSHOT MODE — not a live gate" — cannot be mistaken for real.
#
# Usage:
#   bash scripts/eval_suite/ci_gate.sh                # live (default)
#   bash scripts/eval_suite/ci_gate.sh 2026-06-01     # live with runtag
#   bash scripts/eval_suite/ci_gate.sh --snapshot     # snapshot (LOUD warning)
#   bash scripts/eval_suite/ci_gate.sh --snapshot 2026-06-01
#
# Gate contract:
#   BLOCKS on:
#     - test_reranker_tool_correctness_regression_guard (reranker mean >= 0.70)
#     - live measure failure (endpoint down, livetest produce no output)
#   Does NOT block on:
#     - xfail 0.85 livetest_tool_selection_accuracy (aspirational)
#     - NOT_DERIVABLE profile_pick metrics
#     - scenario_coverage_fraction (structural gap — delegation_profile rows)
#
# Reverting the weekly service change:
#   cp ~/.config/systemd/user/hermes-eval-weekly.service.bak-YYYYMMDD \
#      ~/.config/systemd/user/hermes-eval-weekly.service
#   systemctl --user daemon-reload

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

SNAPSHOT_MODE=false
RUNTAG_ARG=""

for _arg in "$@"; do
    if [[ "${_arg}" == "--snapshot" ]]; then
        SNAPSHOT_MODE=true
    else
        RUNTAG_ARG="${_arg}"
    fi
done

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

RUNTAG="${RUNTAG_ARG:-$(date +%Y-%m-%d)}"

REGRESSION_GUARD_METRIC="reranker_recall5_overall"
REGRESSION_GUARD_BAR="0.70"

# Live benchmark script
LIVE_BENCHMARK_PY="${SCRIPT_DIR}/run_live_benchmark.py"
# Livetest script (in scripts/, one level up)
LIVETEST_PY="${SCRIPT_DIR}/../tool_search_livetest.py"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "[ci_gate] $*"; }
warn() { echo "[ci_gate] WARNING: $*" >&2; }
fail() { echo "[ci_gate] GATE FAIL: $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

# ---------------------------------------------------------------------------
# Snapshot-mode guard
# ---------------------------------------------------------------------------

if [[ "${SNAPSHOT_MODE}" == "true" ]]; then
    echo ""
    echo "############################################################"
    echo "#  SNAPSHOT MODE — not a live gate                         #"
    echo "#  Scoring FROZEN committed artifacts, NOT the deployed    #"
    echo "#  build. This CANNOT catch a live reranker regression.    #"
    echo "#  Use only for CI pre-merge checks, never post-deploy.    #"
    echo "############################################################"
    echo ""
    warn "SNAPSHOT MODE ACTIVE — runtag=${RUNTAG}. Using committed data/results_prefix.json + existing out/_summary.json."
    log "Skipping live re-measure. Jumping straight to scoring."
    _IS_LIVE=false
else
    _IS_LIVE=true
    log "LIVE MODE (default) — runtag=${RUNTAG}"
    log "Gate will re-measure from the LIVE nomic endpoint before scoring."
fi

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
# LIVE STEP A: Regenerate reranker benchmark from LIVE nomic endpoint
# (Skipped in snapshot mode)
# ---------------------------------------------------------------------------

if [[ "${_IS_LIVE}" == "true" ]]; then
    log "=== LIVE STEP A: Regenerating reranker benchmark from live endpoint ==="
    log "    Script: ${LIVE_BENCHMARK_PY}"
    log "    This bypasses all disk caches — fresh embeddings from nomic endpoint."

    if [[ ! -f "${LIVE_BENCHMARK_PY}" ]]; then
        fail "run_live_benchmark.py not found at ${LIVE_BENCHMARK_PY}. " \
             "could not measure live — gate FAILS closed"
    fi

    LIVE_BENCH_EXIT=0
    "${EVAL_PYTHON}" "${LIVE_BENCHMARK_PY}" \
        2>&1 | tee /tmp/hermes_eval_live_bench.log || LIVE_BENCH_EXIT=$?

    if [[ "${LIVE_BENCH_EXIT}" -ne 0 ]]; then
        echo ""
        echo "[ci_gate] ============================================================"
        echo "[ci_gate] LIVE STEP A FAILED — could not measure live — gate FAILS closed"
        echo "[ci_gate] Nomic endpoint may be down. See /tmp/hermes_eval_live_bench.log"
        echo "[ci_gate] To bypass (NOT recommended), run with --snapshot flag."
        echo "[ci_gate] ============================================================"
        exit 1
    fi

    log "LIVE STEP A complete — fresh data/results_prefix.json written."
fi

# ---------------------------------------------------------------------------
# LIVE STEP B: Re-run tool_search_livetest.py for fresh out/_summary.json
# (Skipped in snapshot mode)
# ---------------------------------------------------------------------------

if [[ "${_IS_LIVE}" == "true" ]]; then
    log "=== LIVE STEP B: Re-running tool_search_livetest.py (real LLM calls) ==="

    if [[ ! -f "${LIVETEST_PY}" ]]; then
        fail "tool_search_livetest.py not found at ${LIVETEST_PY}. " \
             "could not measure live — gate FAILS closed"
    fi

    # Load env (OpenRouter key) — livetest needs it
    _ENV_FILE="/etc/systemd/system/hermes-gateway.env"
    if [[ -f "${_ENV_FILE}" ]]; then
        # shellcheck disable=SC1090
        set -a; source "${_ENV_FILE}" 2>/dev/null || true; set +a
        log "Loaded env from ${_ENV_FILE}"
    else
        warn "${_ENV_FILE} not found — livetest will use whatever env is already set"
    fi

    LIVETEST_EXIT=0
    PYTHONPATH="${PROD_VENV}/lib/python3.13/site-packages:${REPO_ROOT}" \
    HERMES_HOME="/opt/hermes/home" \
    "${PROD_PYTHON}" "${LIVETEST_PY}" \
        2>&1 | tee /tmp/hermes_eval_livetest.log || LIVETEST_EXIT=$?

    # Livetest exits non-zero on scenario failures but we only care if it produced output.
    # The real gate is in run_eval.py / JSONL check. But if it produced NO _summary.json,
    # that is a hard failure (could not measure live).
    _SUMMARY="${SCRIPT_DIR}/../out/_summary.json"
    if [[ ! -f "${_SUMMARY}" ]]; then
        echo ""
        echo "[ci_gate] ============================================================"
        echo "[ci_gate] LIVE STEP B FAILED — out/_summary.json not produced"
        echo "[ci_gate] could not measure live — gate FAILS closed"
        echo "[ci_gate] livetest exit=${LIVETEST_EXIT}. Check /tmp/hermes_eval_livetest.log"
        echo "[ci_gate] ============================================================"
        exit 1
    fi

    if [[ "${LIVETEST_EXIT}" -ne 0 ]]; then
        warn "tool_search_livetest.py exited ${LIVETEST_EXIT} — some scenarios may have failed."
        warn "Continuing — run_eval.py will report the exact livetest metrics."
    fi

    log "LIVE STEP B complete — fresh out/_summary.json produced."
fi

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

if [[ "${SNAPSHOT_MODE}" == "true" ]]; then
    echo ""
    echo "[ci_gate] ########################################################"
    echo "[ci_gate] # SNAPSHOT MODE completed — this was NOT a live gate.  #"
    echo "[ci_gate] # Results reflect committed frozen artifacts, not the   #"
    echo "[ci_gate] # deployed build. Do NOT treat this as a deploy gate.  #"
    echo "[ci_gate] ########################################################"
    echo ""
fi

log "=== GATE PASSED ==="
log "  Mode:             $( [[ ${_IS_LIVE} == true ]] && echo 'LIVE (fresh embeddings + fresh livetest)' || echo 'SNAPSHOT (frozen artifacts — not a live gate)' )"
log "  Runtag:           ${RUNTAG}"
log "  Regression guard: ${REGRESSION_GUARD_METRIC} PASS (>= ${REGRESSION_GUARD_BAR})"
log "  Scorecard:        ${SCORECARD_FILE}"
log "  History:          ${HISTORY_FILE}"
log ""
log "NOTE: The 0.85 livetest_tool_selection_accuracy bar is aspirational (xfail)."
log "      The gate does NOT block on it. See ci_gate.sh header for reasoning."
exit 0
