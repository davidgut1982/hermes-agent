# Hermes Offline Eval Suite

Offline quality harness for the Hermes tool-search and agent-orchestration stack.
Measures Recall@5 and MRR without re-running the live agent for every check.

## Files

| File | Purpose |
|------|---------|
| `run_eval.py` | One-command scorecard runner. Calls all four scorers and writes a JSONL report. |
| `scorers.py` | Recall@K / MRR scorers for the reranker benchmark and livetest outputs. |
| `score_profiles.py` | Delegation-profile accuracy scorer (reads livetest bridge_calls). |
| `test_tool_accuracy.py` | DeepEval-based pytest gate; contains `test_reranker_tool_correctness_regression_guard`. |
| `ci_gate.sh` | On-deploy CI gate. Defaults to **live mode** (fresh embeddings + fresh livetest). |
| `run_live_benchmark.py` | Forces fresh embeddings from the live nomic endpoint; writes `data/results_prefix.json`. |
| `gen_prefix_benchmark.py` | One-off generator for `data/results_prefix.json` using labeled queries. |
| `labeled_scenarios.jsonl` | 30 labeled evaluation scenarios (ground-truth prompts + expected tools). |
| `data/results_prefix.json` | Committed benchmark snapshot — 98 queries, prefix-correct nomic config, R@5 ≈ 0.810. |
| `out/scores_baseline.jsonl` | Baseline scorecard (v2). |
| `out/scores_baseline_v3.jsonl` | Baseline scorecard (v3, prefix-correct artifact). |
| `out/scores_history.jsonl` | Append-only history — every `ci_gate.sh` run appends its scorecard here. |

## Quick start

```bash
# Run the full offline scorecard (no live calls)
python3 scripts/eval_suite/run_eval.py --runtag 2026-06-17
```

The runtag is required and must be a meaningful string (e.g. a date or PR name).
Output is written to `out/scores_<runtag>.jsonl` and a summary table is printed.

## Requirements

Two separate Python environments are expected:

| Env path | Contents | Used by |
|----------|----------|---------|
| `/home/david/venv-eval` | `deepeval` + dependencies | `test_tool_accuracy.py`, `ci_gate.sh` (pytest step) |
| `/opt/hermes/venv-combined6` | Production hermes install | `run_eval.py`, `ci_gate.sh` (scoring step) |

To create the eval venv if missing:
```bash
python3 -m venv /home/david/venv-eval
/home/david/venv-eval/bin/pip install deepeval
```

## CI gate

`ci_gate.sh` is the on-deploy gate. It has two modes:

**Live (default)** — re-measures from the live nomic endpoint before scoring:
```bash
bash scripts/eval_suite/ci_gate.sh                 # live, date-stamped runtag
bash scripts/eval_suite/ci_gate.sh 2026-06-17      # live with explicit runtag
```

**Snapshot** — scores the committed `data/results_prefix.json` without re-fetching:
```bash
bash scripts/eval_suite/ci_gate.sh --snapshot      # NOT a live gate — logs loudly
bash scripts/eval_suite/ci_gate.sh --snapshot 2026-06-17
```

Snapshot mode is for pre-merge CI checks only. The live gate is required for post-deploy verification.

### What the gate blocks on

- `reranker_recall5_overall` falling below **0.70** (hard regression guard).
- Live re-measure failure (nomic endpoint unreachable, or livetest produces no output).

### What the gate does NOT block on

- `livetest_tool_selection_accuracy` (aspirational 0.85 bar — marked xfail).
- `NOT_DERIVABLE` profile-pick metrics.
- `scenario_coverage_fraction` (structural gap while delegation_profile rows lack instrumentation).

## Success bars

| Metric | Bar |
|--------|-----|
| `reranker_recall5_overall` | ≥ 0.80 |
| `reranker_recall5_semantic` | ≥ 0.84 |
| `reranker_recall5_lexical` | ≥ 0.95 |
| `livetest_tool_selection_accuracy` | ≥ 0.85 (aspirational / xfail) |
| `livetest_recall5` | ≥ 0.80 |
| `profile_pick_accuracy` | ≥ 0.90 |
| `scenario_recall5_covered` | ≥ 0.80 |

## Benchmark artifact

`data/results_prefix.json` is the prefix-correct benchmark snapshot (98 queries, full 194-tool catalog, nomic task-prefixes applied, R@5 ≈ 0.810). It is committed to the repo so the offline suite is self-contained. The live gate regenerates this file on every run via `run_live_benchmark.py`; snapshot mode uses the committed copy.

Do **not** replace this file with the old `/tmp/tool-rerank-poc/results.json` artifact — that was generated without nomic task-prefixes (R@5 ≈ 0.757) and must not be used.
