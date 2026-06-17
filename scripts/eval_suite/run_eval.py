#!/usr/bin/env python3
"""Hermes Offline Eval Suite — single entrypoint.

Why: Provides a one-command way to run all offline scorers, write a JSONL
scorecard, and print a summary table vs the success bars. Designed for
minimal resource footprint on a saturated machine — no live inference,
no Docker, no SaaS calls.

What: Runs score_reranker_benchmark(), score_livetest(), score_labeled_scenarios(),
and score_profile_pick(); collects ScorecardRow objects; writes them to
scripts/eval_suite/out/scores_<runtag>.jsonl; prints a formatted summary table.

Usage:
    python3 scripts/eval_suite/run_eval.py --runtag baseline
    python3 scripts/eval_suite/run_eval.py --runtag 2026-06-01

Do NOT run with random/time-based runtags — always supply a meaningful name.
The runtag must be provided; there is no default.

Test: Run with --runtag baseline, verify out/scores_baseline.jsonl is created
and the summary table prints without error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SUITE_DIR = Path(__file__).resolve().parent
if str(_SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(_SUITE_DIR))

from score_profiles import score_profile_pick  # noqa: E402
from scorers import (  # noqa: E402
    ScorecardRow,
    score_labeled_scenarios,
    score_livetest,
    score_reranker_benchmark,
)

# ---------------------------------------------------------------------------
# Success bars from kb_28650bfe5f17
# ---------------------------------------------------------------------------

SUCCESS_BARS: dict[str, float] = {
    # Reranker retrieval accuracy
    "reranker_recall5_overall": 0.80,
    "reranker_recall5_semantic": 0.84,
    "reranker_recall5_lexical": 0.95,
    # Tool-selection end-to-end
    "livetest_tool_selection_accuracy": 0.85,
    "livetest_recall5": 0.80,
    # Profile routing
    "profile_pick_accuracy": 0.90,
    # Scenario coverage
    "scenario_recall5_covered": 0.80,
}


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------


def print_summary_table(rows: list[ScorecardRow]) -> None:
    """Print a formatted summary table comparing actual vs bar.

    Why: Quick human-readable overview of pass/fail status.
    What: Prints a fixed-width table with metric | actual | bar | status.
    Test: Verify no exceptions when rows contains a mix of pass/fail rows.
    """
    # Primary metrics to highlight (in display order)
    primary = [
        "reranker_recall5_overall",
        "reranker_recall5_semantic",
        "reranker_recall5_lexical",
        "reranker_recall5_ambiguous",
        "reranker_mrr_overall",
        "reranker_bm25_recall5_overall",
        "livetest_tool_selection_accuracy",
        "livetest_recall5",
        "livetest_mrr",
        "profile_pick_accuracy",
        "profile_pick_coverage_fraction",
        "scenario_coverage_fraction",
        "scenario_recall5_covered",
    ]

    by_metric = {r.metric: r for r in rows}
    ordered = [by_metric[m] for m in primary if m in by_metric]
    rest = [r for r in rows if r.metric not in set(primary)]

    print()
    print("=" * 78)
    print("  Hermes Offline Eval Suite — Scorecard")
    print("=" * 78)
    print(f"  {'Metric':<45}  {'Actual':>7}  {'Bar':>6}  Status")
    print("-" * 78)

    def _print_row(r: ScorecardRow) -> None:
        status = "PASS" if r.pass_ else "FAIL" if r.bar > 0 else "INFO"
        bar_str = f"{r.bar:.3f}" if r.bar > 0 else "  n/a"
        print(f"  {r.metric:<45}  {r.actual:>7.3f}  {bar_str:>6}  [{status}]")
        if r.source:
            # Print source on second line, indented
            src = r.source[:120]
            print(f"    {'':>45}  source: {src}")

    for r in ordered:
        _print_row(r)

    if rest:
        print("-" * 78)
        print("  Additional metrics:")
        for r in rest:
            _print_row(r)

    print("=" * 78)

    # Summary counts
    bars_set = [r for r in rows if r.bar > 0]
    n_pass = sum(1 for r in bars_set if r.pass_)
    n_fail = sum(1 for r in bars_set if not r.pass_)
    print(f"  Scored metrics with bars: {len(bars_set)} | "
          f"PASS: {n_pass} | FAIL: {n_fail}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run all offline scorers, write JSONL scorecard, print summary.

    Why: Single entrypoint keeps the workflow simple — one command, one output.
    What: Collects all ScorecardRow objects from the four scorer functions,
    writes them to out/scores_<runtag>.jsonl, prints a summary table.
    Test: Run with --runtag baseline; verify the output file is created and
    the summary table contains the expected metric names.
    """
    parser = argparse.ArgumentParser(
        description="Hermes offline eval suite runner",
    )
    parser.add_argument(
        "--runtag",
        required=True,
        help="Tag identifying this run (e.g. 'baseline', '2026-06-01'). "
             "Used as the output file suffix. Do not use random or time-based values.",
    )
    args = parser.parse_args()
    runtag = args.runtag.strip()
    if not runtag:
        print("ERROR: --runtag must be non-empty", file=sys.stderr)
        return 1

    out_dir = _SUITE_DIR / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scores_{runtag}.jsonl"

    print(f"\nRunning Hermes offline eval suite (runtag={runtag!r}) ...")
    print(f"Output: {out_path}")

    all_rows: list[ScorecardRow] = []

    # --- Reranker benchmark ---
    print(  # noqa: E501
        "\n[1/4] Scoring reranker benchmark "
        "(98 queries, scripts/eval_suite/data/results_prefix.json — prefix-correct) ..."
    )
    try:
        rr_rows = score_reranker_benchmark()
        all_rows.extend(rr_rows)
        n_present = sum(1 for r in rr_rows if "MISSING" not in r.source)
        print(f"      OK: {len(rr_rows)} rows computed ({n_present} with data)")
    except Exception as exc:
        print(f"      ERROR: {exc}", file=sys.stderr)
        all_rows.append(ScorecardRow("reranker_recall5_overall", 0.0, 0.80,
                                      source=f"ERROR: {exc}"))

    # --- Livetest ---
    print("\n[2/4] Scoring livetest output (scripts/out/_summary.json) ...")
    try:
        lt_rows = score_livetest()
        all_rows.extend(lt_rows)
        n_present = sum(1 for r in lt_rows if "MISSING" not in r.source)
        print(f"      OK: {len(lt_rows)} rows computed ({n_present} with data)")
    except Exception as exc:
        print(f"      ERROR: {exc}", file=sys.stderr)
        all_rows.append(ScorecardRow("livetest_tool_selection_accuracy", 0.0, 0.85,
                                      source=f"ERROR: {exc}"))

    # --- Labeled scenarios ---
    print("\n[3/4] Scoring labeled scenarios (labeled_scenarios.jsonl cross-ref) ...")
    try:
        sc_rows = score_labeled_scenarios()
        all_rows.extend(sc_rows)
        print(f"      OK: {len(sc_rows)} rows computed")
    except Exception as exc:
        print(f"      ERROR: {exc}", file=sys.stderr)
        all_rows.append(ScorecardRow("scenario_recall5_covered", 0.0, 0.80,
                                      source=f"ERROR: {exc}"))

    # --- Profile pick ---
    print("\n[4/4] Scoring profile-pick accuracy (from livetest bridge_calls) ...")
    try:
        pp_rows = score_profile_pick()
        all_rows.extend(pp_rows)
        print(f"      OK: {len(pp_rows)} rows computed")
    except Exception as exc:
        print(f"      ERROR: {exc}", file=sys.stderr)
        all_rows.append(ScorecardRow("profile_pick_accuracy", 0.0, 0.90,
                                      source=f"ERROR: {exc}"))

    # --- Write JSONL ---
    with out_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row.to_dict(runtag), ensure_ascii=False) + "\n")

    print(f"\nScorecard written: {out_path} ({len(all_rows)} rows)")

    # --- Print summary ---
    print_summary_table(all_rows)

    # --- Print instrumentation gap reminder ---
    print("INSTRUMENTATION GAP NOTE:")
    print("  profile_pick_accuracy is NOT fully derivable from existing outputs.")
    print("  To enable full profile-pick scoring, add ONE log line to:")
    print("    /opt/hermes/build-combined6/tools/delegate_tool.py ~line 2074")
    print("  (immediately after `resolved_profile_name = profile`)")
    print()
    print('    logger.info(')
    print('        "delegate_profile_event: resolved=%r goal_chars=%d",')
    print('        resolved_profile_name,')
    print('        len(str(goal)) if goal else 0,')
    print('    )')
    print()
    print("  Batch into next prod deploy — do NOT redeploy prod now.")
    print()

    # Return exit code: 0 if all bars pass, 1 if any fail
    bars_set = [r for r in all_rows if r.bar > 0 and "MISSING" not in r.source
                and "ERROR" not in r.source and "NOT_DERIVABLE" not in r.source]
    any_fail = any(not r.pass_ for r in bars_set)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
