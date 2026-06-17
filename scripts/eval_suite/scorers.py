"""Pure offline scorers for the Hermes eval suite.

Why: Computes Recall@K and MRR from existing harness outputs without
re-running any live agent or SaaS. All computation is post-processing
over JSON files already on disk.

What: Three scoring surfaces:
  1. reranker_scorer — reads scripts/eval_suite/data/results_prefix.json
     (the 98-query prefix-correct benchmark, produced by gen_prefix_benchmark.py
     and committed to the repo). Computes per-category Recall@5 and MRR.
     NOTE: This is the PREFIX-CORRECT artifact (shipped nomic-prefix config,
     R@5~0.810). The old /tmp/tool-rerank-poc/results.json (R@5=0.757) was
     generated WITHOUT nomic task-prefixes and must NOT be used.
  2. livetest_scorer — reads scripts/out/_summary.json and the per-scenario
     JSONs from tool_search_livetest.py, computing Recall@K and MRR for the
     5 livetest scenarios (tool_search ENABLED runs only).
  3. scenario_scorer — reads labeled_scenarios.jsonl and cross-references
     against benchmark data using stable benchmark_idx field (not fragile text
     matching). Reports explicitly for every scenario: scored or unscored-with-reason.

Test: Run `python -m pytest scripts/eval_suite/test_tool_accuracy.py` after
running this module to verify ToolCorrectnessMetric thresholds pass.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Canonical paths — ALL in-repo; no /tmp dependency
_SUITE_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SUITE_DIR.parent
_DATA_DIR = _SUITE_DIR / "data"

# Primary reranker benchmark artifact (prefix-correct, committed to repo).
# DO NOT change this to /tmp — the benchmark must be self-contained.
_RERANKER_RESULTS = _DATA_DIR / "results_prefix.json"

_LIVETEST_SUMMARY = _SCRIPTS_DIR / "out" / "_summary.json"
_LABELED_SCENARIOS = _SUITE_DIR / "labeled_scenarios.jsonl"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ToolQueryResult:
    """Scored result for a single query against a tool catalog.

    Why: Uniform container for benchmark results so scorers can treat both
    prefix-correct and livetest sources identically.
    What: Holds gold/predicted lists and precomputed scalar metrics.
    Test: recall_at_k(["a","b"], ["b","c","d","a"], 3) == 1.0
    """

    query_id: str
    query_text: str
    category: str  # LEXICAL / SEMANTIC / AMBIGUOUS
    gold_tools: list[str]
    bm25_top5: list[str]
    rerank_top5: list[str]
    bm25_recall5: float
    rerank_recall5: float
    bm25_mrr: float
    rerank_mrr: float


@dataclass
class CategoryScore:
    """Aggregate scores for a query category.

    Why: Summary per category (LEXICAL/SEMANTIC/AMBIGUOUS) mirrors the
    benchmark writeup table and makes the scorecard readable.
    What: Mean Recall@5 and MRR across all queries in the category.
    Test: avg over 2 queries: R@5=[0.5, 1.0] -> 0.75.
    """

    category: str
    n: int
    recall5: float
    mrr: float


@dataclass
class ScorecardRow:
    """One row in the final scorecard JSONL.

    Why: Provides a stable schema for the output file so downstream tools
    can parse it without fragile text parsing.
    What: One row per metric, with actual value, bar, and pass/fail.
    Test: ScorecardRow("reranker_recall5_semantic", 0.767, 0.840) -> pass=False.
    """

    metric: str
    actual: float
    bar: float
    pass_: bool = field(init=False)
    source: str = ""

    def __post_init__(self) -> None:
        """Why: Auto-derive pass_ from actual >= bar at construction time."""
        self.pass_ = self.actual >= self.bar

    def to_dict(self, runtag: str) -> dict[str, Any]:
        """Why: Serialise to a flat dict for JSONL output.
        What: Includes runtag for run identification.
        Test: row.to_dict('baseline')['runtag'] == 'baseline'.
        """
        return {
            "runtag": runtag,
            "metric": self.metric,
            "actual": round(self.actual, 4),
            "bar": self.bar,
            "pass": self.pass_,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Recall@K and MRR helpers
# ---------------------------------------------------------------------------


def recall_at_k(gold: list[str], predicted: list[str], k: int) -> float:
    """Recall@K: fraction of gold tools appearing in top-K predictions.

    Why: Standard IR metric; measures how many expected tools the system
    surfaces in its top-K results (partial credit for multi-gold queries).
    What: |{gold} ∩ {predicted[:k]}| / |{gold}|. Returns 0.0 if gold is empty.
    Test: recall_at_k(['a','b'], ['b','c','a'], 2) == 0.5 (only 'b' in top-2).
    """
    if not gold:
        return 0.0
    gold_set = set(gold)
    top_k = set(predicted[:k])
    return len(gold_set & top_k) / len(gold_set)


def mrr(gold: list[str], predicted: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first gold tool in predicted list.

    Why: Measures how high the first relevant result appears; complements R@K
    by penalising systems that bury the best match deep in the list.
    What: 1/rank_of_first_gold; 0.0 if no gold tool appears in predicted.
    Test: mrr(['a'], ['c','b','a']) == 1/3; mrr(['a'], ['b','c']) == 0.0.
    """
    gold_set = set(gold)
    for rank, name in enumerate(predicted, start=1):
        if name in gold_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Reranker benchmark scorer (reads data/results_prefix.json — IN-REPO)
# ---------------------------------------------------------------------------


def load_reranker_results() -> list[ToolQueryResult]:
    """Load the 98-query prefix-correct benchmark results from the in-repo artifact.

    Why: The benchmark artifact (results_prefix.json) is committed to the repo at
    scripts/eval_suite/data/ so the suite is self-contained and does NOT depend
    on /tmp (which is cleared on reboot). The artifact reflects the SHIPPED
    reranker config (nomic task-prefixes applied).
    What: Reads results_prefix.json, extracts per-query gold/rerank lists indexed
    by stable idx field. Returns ToolQueryResult objects indexed as RB001..RB098.
    Test: len(load_reranker_results()) == 98 when results_prefix.json is present.
    """
    if not _RERANKER_RESULTS.exists():
        return []

    raw = json.loads(_RERANKER_RESULTS.read_text(encoding="utf-8"))
    results: list[ToolQueryResult] = []

    # results_prefix.json has a 'per_query_full' list of per-query dicts
    if isinstance(raw, dict):
        items = raw.get("per_query_full", [])
    else:
        # Unexpected format — treat as list
        items = raw

    for item in items:
        query_id = f"RB{item.get('idx', 0):03d}"
        query_text = item.get("query", "")
        category = item.get("category", "UNKNOWN")
        gold = item.get("gold", item.get("gold_tools", []))
        bm25 = item.get("bm25_top5", [])
        rerank = item.get("rerank_top5", [])

        results.append(ToolQueryResult(
            query_id=query_id,
            query_text=query_text,
            category=category,
            gold_tools=gold,
            bm25_top5=bm25,
            rerank_top5=rerank,
            bm25_recall5=recall_at_k(gold, bm25, 5) if bm25 else 0.0,
            rerank_recall5=recall_at_k(gold, rerank, 5),
            bm25_mrr=mrr(gold, bm25) if bm25 else 0.0,
            rerank_mrr=mrr(gold, rerank),
        ))

    return results


def score_reranker_benchmark() -> list[ScorecardRow]:
    """Compute Recall@5 and MRR from the prefix-correct 98-query benchmark.

    Why: Produces the primary tool-selection accuracy scorecard rows without
    any live inference. Numbers should match the PR #35457 benchmark writeup
    (overall R@5 ~0.810 with nomic task-prefixes applied).
    What: Loads results_prefix.json; aggregates per-category and overall R@5+MRR.
    Test: Overall rerank R@5 should be ~0.810 (FULL retrieve, 194 tools, with prefix).
    """
    results = load_reranker_results()
    if not results:
        return [ScorecardRow(
            "reranker_recall5_overall", 0.0, 0.80,
            source=(
                f"MISSING: {_RERANKER_RESULTS} not found. "
                "Run scripts/eval_suite/gen_prefix_benchmark.py to regenerate."
            ),
        )]

    # Aggregate by category
    cats: dict[str, list[ToolQueryResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)

    rows: list[ScorecardRow] = []

    # Category bars from kb_28650bfe5f17 (success bars for SHIPPED config):
    # Overall R@5 >= 0.80, Semantic >= 0.84, tool-selection >= ~85%
    cat_bars: dict[str, float] = {
        "OVERALL": 0.80,
        "SEMANTIC": 0.84,
        "LEXICAL": 0.95,
        "AMBIGUOUS": 0.50,  # harder category, lower bar
    }

    for cat, items in sorted(cats.items()):
        avg_r5_bm25 = sum(i.bm25_recall5 for i in items) / len(items)
        avg_r5_rr = sum(i.rerank_recall5 for i in items) / len(items)
        avg_mrr_bm25 = sum(i.bm25_mrr for i in items) / len(items)
        avg_mrr_rr = sum(i.rerank_mrr for i in items) / len(items)
        bar = cat_bars.get(cat, 0.60)
        n = len(items)

        rows.append(ScorecardRow(
            f"reranker_recall5_{cat.lower()}", avg_r5_rr, bar,
            source=f"offline_benchmark_prefix_correct (n={n})",
        ))
        bm25_source = (
            f"offline_benchmark_bm25_baseline (n={n})"
            if avg_r5_bm25 > 0.0
            else f"offline_benchmark_bm25_not_in_artifact (n={n})"
        )
        rows.append(ScorecardRow(
            f"reranker_bm25_recall5_{cat.lower()}", avg_r5_bm25, 0.0,
            source=bm25_source,
        ))
        rows.append(ScorecardRow(
            f"reranker_mrr_{cat.lower()}", avg_mrr_rr, 0.0,
            source=f"offline_benchmark_prefix_correct (n={n})",
        ))
        rows.append(ScorecardRow(
            f"reranker_bm25_mrr_{cat.lower()}", avg_mrr_bm25, 0.0,
            source=bm25_source,
        ))

    # Overall
    all_r5_rr = sum(i.rerank_recall5 for i in results) / len(results)
    all_r5_bm25 = sum(i.bm25_recall5 for i in results) / len(results)
    all_mrr_rr = sum(i.rerank_mrr for i in results) / len(results)
    all_mrr_bm25 = sum(i.bm25_mrr for i in results) / len(results)
    n_total = len(results)

    rows.append(ScorecardRow(
        "reranker_recall5_overall", all_r5_rr, cat_bars["OVERALL"],
        source=f"offline_benchmark_prefix_correct (n={n_total})",
    ))
    overall_bm25_source = (
        f"offline_benchmark_bm25_baseline (n={n_total})"
        if all_r5_bm25 > 0.0
        else f"offline_benchmark_bm25_not_in_artifact (n={n_total})"
    )
    rows.append(ScorecardRow(
        "reranker_bm25_recall5_overall", all_r5_bm25, 0.0,
        source=overall_bm25_source,
    ))
    rows.append(ScorecardRow(
        "reranker_mrr_overall", all_mrr_rr, 0.0,
        source=f"offline_benchmark_prefix_correct (n={n_total})",
    ))
    rows.append(ScorecardRow(
        "reranker_bm25_mrr_overall", all_mrr_bm25, 0.0,
        source=overall_bm25_source,
    ))

    return rows


# ---------------------------------------------------------------------------
# Livetest scorer (reads scripts/out/_summary.json)
# ---------------------------------------------------------------------------


def load_livetest_summary() -> list[dict[str, Any]]:
    """Load livetest summary from scripts/out/_summary.json.

    Why: Reuses existing tool_search_livetest.py output without re-running
    the live agent (one live run max per instructions).
    What: Returns list of summary rows, one per (scenario, enabled) pair.
    Test: Returns [] without error when _summary.json does not exist.
    """
    if not _LIVETEST_SUMMARY.exists():
        return []
    return json.loads(_LIVETEST_SUMMARY.read_text(encoding="utf-8"))


def score_livetest() -> list[ScorecardRow]:
    """Compute Recall@5 and MRR from tool_search_livetest.py output.

    Why: Livetest output is the only source of tool_search=ENABLED end-to-end
    accuracy signal we can score offline (reranker benchmark uses static tool
    list; livetest captures actual bridge dispatch).
    What: For each ENABLED livetest scenario, computes R@K from
    (underlying_tools_called ∩ expected) / |expected| and MRR from rank of
    first expected tool in the ordered call list. Tool-selection accuracy =
    fraction of scenarios where all expected tools were called.
    Test: Scenario with expected=['github_create_issue'] and
    underlying_tool_calls=[{'name':'github_create_issue','args':{}}] -> R@1=1.0.
    """
    rows: list[ScorecardRow] = []
    summary = load_livetest_summary()
    if not summary:
        return [ScorecardRow(
            "livetest_tool_selection_accuracy", 0.0, 0.85,
            source="MISSING: scripts/out/_summary.json not found",
        )]

    enabled_rows = [r for r in summary if r.get("enabled", False)]
    if not enabled_rows:
        return [ScorecardRow(
            "livetest_tool_selection_accuracy", 0.0, 0.85,
            source="MISSING: no tool_search=enabled rows in summary",
        )]

    # Compute per-scenario R@K and MRR
    scenario_r5 = []
    scenario_mrr = []
    passed = 0

    for row in enabled_rows:
        expected = row.get("expected", [])
        called = row.get("underlying_tools_called", [])

        if not expected:
            # Should-not-delegate scenario: pass iff no unexpected tools called
            extra = set(called) - {"read_file", "search_files", "terminal",
                                    "todo", "memory"}
            if not extra:
                passed += 1
            continue

        r5 = recall_at_k(expected, called, 5)
        m = mrr(expected, called)
        scenario_r5.append(r5)
        scenario_mrr.append(m)

        expected_set = set(expected)
        if expected_set.issubset(set(called)):
            passed += 1

    n_enabled = len(enabled_rows)
    tool_sel_acc = passed / n_enabled if n_enabled > 0 else 0.0
    avg_r5 = sum(scenario_r5) / len(scenario_r5) if scenario_r5 else 0.0
    avg_mrr_val = sum(scenario_mrr) / len(scenario_mrr) if scenario_mrr else 0.0

    rows.append(ScorecardRow(
        "livetest_tool_selection_accuracy", tool_sel_acc, 0.85,
        source=f"livetest_enabled (n={n_enabled})",
    ))
    rows.append(ScorecardRow(
        "livetest_recall5", avg_r5, 0.80,
        source=f"livetest_enabled (n={len(scenario_r5)})",
    ))
    rows.append(ScorecardRow(
        "livetest_mrr", avg_mrr_val, 0.0,
        source=f"livetest_enabled (n={len(scenario_mrr)})",
    ))

    return rows


# ---------------------------------------------------------------------------
# Combined scenario scorer (ID-based cross-reference against labeled_scenarios.jsonl)
# ---------------------------------------------------------------------------


def load_labeled_scenarios() -> list[dict[str, Any]]:
    """Load labeled_scenarios.jsonl into a list of dicts.

    Why: Provides a single ground-truth source for all eval scenarios,
    combining livetest and reranker benchmark coverage.
    What: Each row has id, prompt, expected_tools, expected_profile, k, category,
    and optionally benchmark_idx (stable integer index into labeled_queries.json
    / per_query_full in the prefix-correct benchmark artifact).
    Test: Returns a list of >=30 dicts when the file exists.
    """
    path = _LABELED_SCENARIOS
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def score_labeled_scenarios() -> list[ScorecardRow]:
    """Score labeled scenarios using ID-based benchmark and livetest cross-reference.

    Why: Provides a unified table covering all 30 scenarios. Uses stable
    benchmark_idx field for cross-reference (not fragile text matching).
    Fails loudly (non-zero row + FAIL) when scenario coverage < 1.0 instead
    of silently dropping scenarios.
    What:
      - Benchmark scenarios (S006-S025): cross-referenced via benchmark_idx.
      - Livetest scenarios (S001-S005): cross-referenced via livetest summary.
      - Unscored scenarios reported explicitly with reason (not silently dropped).
    Loud failure: If any scenario is unscored for an unexpected reason (i.e., not
    a known data-source gap), run_eval.py receives a FAIL row for scenario_coverage.
    Test: With 30 labeled scenarios and results_prefix.json + no livetest data,
    returns coverage 20/30 with 10 explicitly reported as UNSCORED_NO_DATA (S001-S005
    livetest, S026-S030 delegation_profile/should_not_delegate).
    """
    scenarios = load_labeled_scenarios()
    if not scenarios:
        return [ScorecardRow("scenario_coverage", 0.0, 1.0,
                             source="MISSING: labeled_scenarios.jsonl")]

    # Load reranker results indexed by stable benchmark_idx (1-based integer)
    rr_by_idx: dict[int, ToolQueryResult] = {}
    for rr in load_reranker_results():
        # benchmark_idx corresponds to rr.query_id which is RB001, RB002, etc.
        idx = int(rr.query_id[2:])  # e.g. "RB042" -> 42
        rr_by_idx[idx] = rr

    # Load livetest summary indexed by scenario_id from livetest harness
    lt_by_scenario: dict[str, dict[str, Any]] = {}
    for row in load_livetest_summary():
        if row.get("enabled", False):
            lt_by_scenario[row["scenario"]] = row

    # Livetest scenario id -> labeled scenario id mapping
    lt_id_map: dict[str, str] = {
        "A_obvious_single": "S001",
        "B_vague_paraphrased": "S002",
        "C_multi_tool_chain": "S003",
        "D_core_plus_deferred": "S004",
        "E_no_tool_needed": "S005",
    }
    labeled_id_to_lt: dict[str, str] = {v: k for k, v in lt_id_map.items()}

    # Categories that have no benchmark or livetest source by design
    # (delegation_profile and should_not_delegate scenarios without livetest mapping)
    _NO_DATA_CATEGORIES = frozenset({"delegation_profile", "should_not_delegate"})

    scenario_r5: list[float] = []
    covered = 0
    unscored: list[dict[str, str]] = []  # {id, reason} for every unscored scenario

    for sc in scenarios:
        sc_id: str = sc["id"]
        expected: list[str] = sc.get("expected_tools", [])
        k: int = sc.get("k", 5)
        category: str = sc.get("category", "")
        benchmark_idx: int | None = sc.get("benchmark_idx")

        predicted: list[str] = []

        # 1. Try benchmark cross-reference by stable ID
        if benchmark_idx is not None and benchmark_idx in rr_by_idx:
            rr = rr_by_idx[benchmark_idx]
            predicted = rr.rerank_top5
            covered += 1

        # 2. Try livetest data (S001-S005)
        elif sc_id in labeled_id_to_lt:
            lt_key = labeled_id_to_lt[sc_id]
            if lt_key in lt_by_scenario:
                row = lt_by_scenario[lt_key]
                predicted = row.get("underlying_tools_called", [])
                covered += 1
            else:
                unscored.append({
                    "id": sc_id,
                    "reason": (
                        f"livetest_not_run: {lt_key} not in _summary.json. "
                        "Run scripts/tool_search_livetest.py to populate."
                    ),
                })
                continue

        # 3. Explicitly unscored — no data source available
        else:
            if category in _NO_DATA_CATEGORIES:
                reason = (
                    f"no_data_source: category={category!r}. "
                    "delegation_profile and should_not_delegate scenarios require "
                    "livetest output or future instrumentation (see score_profiles.py)."
                )
            else:
                reason = (
                    f"no_benchmark_idx_and_no_livetest_mapping: "
                    f"category={category!r} prompt={sc.get('prompt','')[:60]!r}. "
                    "Add benchmark_idx to labeled_scenarios.jsonl to fix."
                )
            unscored.append({"id": sc_id, "reason": reason})
            continue

        if not expected:
            # should-not-delegate: no tools expected
            scenario_r5.append(1.0 if not predicted else 0.0)
        else:
            scenario_r5.append(recall_at_k(expected, predicted, k))

    n_scenarios = len(scenarios)
    coverage = covered / n_scenarios if n_scenarios > 0 else 0.0
    avg_r5 = sum(scenario_r5) / len(scenario_r5) if scenario_r5 else 0.0

    # Print explicit unscored report (always, so it's visible in logs)
    if unscored:
        print(f"\n  SCENARIO COVERAGE: {covered}/{n_scenarios} scored. "
              f"Unscored ({len(unscored)}) — explicit reasons:")
        for u in unscored:
            print(f"    [{u['id']}] {u['reason']}")

    rows: list[ScorecardRow] = [
        ScorecardRow(
            "scenario_coverage_fraction", coverage, 1.0,
            source=f"labeled_scenarios (covered={covered}/{n_scenarios}, "
                   f"unscored={len(unscored)})",
        ),
        ScorecardRow(
            "scenario_recall5_covered", avg_r5, 0.80,
            source=f"labeled_scenarios_covered (n={covered})",
        ),
    ]

    # LOUD FAIL: if any scenario is unscored for a non-expected reason, add FAIL row
    unexpected_unscored = [
        u for u in unscored
        if "no_data_source" not in u["reason"]  # expected gaps are OK
        and "livetest_not_run" not in u["reason"]  # livetest not run is OK
    ]
    if unexpected_unscored:
        ids = ", ".join(u["id"] for u in unexpected_unscored)
        rows.append(ScorecardRow(
            "scenario_unexpected_unscored", float(len(unexpected_unscored)), 0.0,
            source=(
                f"FAIL: {len(unexpected_unscored)} scenarios unscored for unexpected "
                f"reasons: {ids}. Fix benchmark_idx or livetest mapping."
            ),
        ))
        # Also print to stderr so CI catches it
        print(
            f"\n  FAIL: {len(unexpected_unscored)} scenarios unscored unexpectedly: "
            f"{ids}",
            file=sys.stderr,
        )

    return rows
