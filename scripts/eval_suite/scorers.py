"""Pure offline scorers for the Hermes eval suite.

Why: Computes Recall@K and MRR from existing harness outputs without
re-running any live agent or SaaS. All computation is post-processing
over JSON files already on disk.

What: Two scoring surfaces:
  1. reranker_scorer — reads /tmp/tool-rerank-poc/results.json (the 98-query
     benchmark that produced PR #35457 numbers) and recomputes per-category
     Recall@5 and MRR.
  2. livetest_scorer — reads scripts/out/_summary.json and the per-scenario
     JSONs from tool_search_livetest.py, computing Recall@K and MRR for the
     5 livetest scenarios (tool_search ENABLED runs only).
  3. scenario_scorer — reads labeled_scenarios.jsonl and cross-references
     against available benchmark/livetest data to produce a unified per-
     scenario table.

Test: Run `python -m pytest scripts/eval_suite/test_tool_accuracy.py` after
running this module to verify ToolCorrectnessMetric thresholds pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Canonical paths
_SUITE_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SUITE_DIR.parent
_RERANKER_RESULTS = Path("/tmp/tool-rerank-poc/results.json")
_LIVETEST_SUMMARY = _SCRIPTS_DIR / "out" / "_summary.json"
_LABELED_SCENARIOS = _SUITE_DIR / "labeled_scenarios.jsonl"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ToolQueryResult:
    """Scored result for a single query against a tool catalog.

    Why: Uniform container for BM25 and reranker results so scorers
    can treat both sources identically.
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
    Test: avg over 2 queries: R@5=[0.5, 1.0] → 0.75.
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
    Test: ScorecardRow("reranker_recall5_semantic", 0.767, 0.840) → pass=False.
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
# Reranker benchmark scorer (reads /tmp/tool-rerank-poc/results.json)
# ---------------------------------------------------------------------------


def load_reranker_results() -> list[ToolQueryResult]:
    """Load the 98-query reranker benchmark results from disk.

    Why: The benchmark already ran when PR #35457 was authored; reusing it
    avoids re-embedding 194 tools (expensive, slow on saturated box).
    What: Reads results.json, extracts per-query gold/bm25/rerank lists.
    Test: len(load_reranker_results()) == 98 when results.json is present.
    """
    if not _RERANKER_RESULTS.exists():
        return []

    raw = json.loads(_RERANKER_RESULTS.read_text(encoding="utf-8"))
    results: list[ToolQueryResult] = []

    # results.json is a dict with a 'per_query_full' list of per-query dicts
    # from bench.py / bench_tiers.py (key 'gold' not 'gold_tools')
    if isinstance(raw, dict):
        items = raw.get("per_query_full", [])
    else:
        # Fallback: treat as list if legacy format
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
            bm25_recall5=recall_at_k(gold, bm25, 5),
            rerank_recall5=recall_at_k(gold, rerank, 5),
            bm25_mrr=mrr(gold, bm25),
            rerank_mrr=mrr(gold, rerank),
        ))

    return results


def score_reranker_benchmark() -> list[ScorecardRow]:
    """Compute Recall@5 and MRR from the existing 98-query benchmark.

    Why: Produces the primary tool-selection accuracy scorecard rows without
    any live inference. Numbers should match the PR #35457 benchmark writeup.
    What: Loads results.json; aggregates per-category and overall R@5 + MRR.
    Test: Overall rerank R@5 should be ~0.757 (FULL retrieve, 194 tools).
    """
    results = load_reranker_results()
    if not results:
        return [ScorecardRow(
            "reranker_recall5_overall", 0.0, 0.80,
            source="MISSING: /tmp/tool-rerank-poc/results.json not found",
        )]

    # Aggregate by category
    cats: dict[str, list[ToolQueryResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)

    rows: list[ScorecardRow] = []

    # Category bars from kb_28650bfe5f17:
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
            source=f"offline_benchmark (n={n})",
        ))
        rows.append(ScorecardRow(
            f"reranker_bm25_recall5_{cat.lower()}", avg_r5_bm25, 0.0,
            source=f"offline_benchmark_bm25_baseline (n={n})",
        ))
        rows.append(ScorecardRow(
            f"reranker_mrr_{cat.lower()}", avg_mrr_rr, 0.0,
            source=f"offline_benchmark (n={n})",
        ))
        rows.append(ScorecardRow(
            f"reranker_bm25_mrr_{cat.lower()}", avg_mrr_bm25, 0.0,
            source=f"offline_benchmark_bm25_baseline (n={n})",
        ))

    # Overall
    all_r5_rr = sum(i.rerank_recall5 for i in results) / len(results)
    all_r5_bm25 = sum(i.bm25_recall5 for i in results) / len(results)
    all_mrr_rr = sum(i.rerank_mrr for i in results) / len(results)
    all_mrr_bm25 = sum(i.bm25_mrr for i in results) / len(results)
    n_total = len(results)

    rows.append(ScorecardRow(
        "reranker_recall5_overall", all_r5_rr, cat_bars["OVERALL"],
        source=f"offline_benchmark (n={n_total})",
    ))
    rows.append(ScorecardRow(
        "reranker_bm25_recall5_overall", all_r5_bm25, 0.0,
        source=f"offline_benchmark_bm25_baseline (n={n_total})",
    ))
    rows.append(ScorecardRow(
        "reranker_mrr_overall", all_mrr_rr, 0.0,
        source=f"offline_benchmark (n={n_total})",
    ))
    rows.append(ScorecardRow(
        "reranker_bm25_mrr_overall", all_mrr_bm25, 0.0,
        source=f"offline_benchmark_bm25_baseline (n={n_total})",
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
    underlying_tool_calls=[{'name':'github_create_issue','args':{}}] → R@1=1.0.
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
# Combined scenario scorer (cross-references labeled_scenarios.jsonl)
# ---------------------------------------------------------------------------


def load_labeled_scenarios() -> list[dict[str, Any]]:
    """Load labeled_scenarios.jsonl into a list of dicts.

    Why: Provides a single ground-truth source for all eval scenarios,
    combining livetest and reranker benchmark coverage.
    What: Each row has id, prompt, expected_tools, expected_profile, k, category.
    Test: Returns a list of >=30 dicts when the file exists.
    """
    path = _LABELED_SCENARIOS
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def score_labeled_scenarios() -> list[ScorecardRow]:
    """Score labeled scenarios using available benchmark and livetest data.

    Why: Provides a unified table covering all 30 scenarios, mapping each
    to its source of truth (reranker benchmark or livetest output).
    What: For livetest scenarios (S001-S005), reads scripts/out/*.json.
    For reranker scenarios (S006+), cross-references results.json by query text.
    Returns one summary row covering all scenarios.
    Test: With 30 labeled scenarios and results.json present, returns at least
    one passing overall row.
    """
    scenarios = load_labeled_scenarios()
    if not scenarios:
        return [ScorecardRow("scenario_coverage", 0.0, 1.0,
                             source="MISSING: labeled_scenarios.jsonl")]

    # Load reranker results indexed by query text
    rr_by_query: dict[str, ToolQueryResult] = {}
    for rr in load_reranker_results():
        rr_by_query[rr.query_text.strip().lower()] = rr

    # Load livetest summary indexed by scenario_id
    lt_by_scenario: dict[str, dict[str, Any]] = {}
    for row in load_livetest_summary():
        if row.get("enabled", False):
            lt_by_scenario[row["scenario"]] = row

    scenario_r5 = []
    covered = 0

    for sc in scenarios:
        sc_id: str = sc["id"]
        query: str = sc["prompt"].strip().lower()
        expected: list[str] = sc.get("expected_tools", [])
        k: int = sc.get("k", 5)

        predicted: list[str] = []

        # Try livetest data first (S001-S005)
        lt_key = None
        if sc_id == "S001":
            lt_key = "A_obvious_single"
        elif sc_id == "S002":
            lt_key = "B_vague_paraphrased"
        elif sc_id == "S003":
            lt_key = "C_multi_tool_chain"
        elif sc_id == "S004":
            lt_key = "D_core_plus_deferred"
        elif sc_id == "S005":
            lt_key = "E_no_tool_needed"

        if lt_key and lt_key in lt_by_scenario:
            row = lt_by_scenario[lt_key]
            predicted = row.get("underlying_tools_called", [])
            covered += 1
        elif query in rr_by_query:
            rr = rr_by_query[query]
            predicted = rr.rerank_top5
            covered += 1
        else:
            # Not covered by existing outputs — mark as uncovered
            continue

        if not expected:
            # should-not-delegate: no tools expected
            scenario_r5.append(1.0 if not predicted else 0.0)
        else:
            scenario_r5.append(recall_at_k(expected, predicted, k))

    n_scenarios = len(scenarios)
    coverage = covered / n_scenarios if n_scenarios > 0 else 0.0
    avg_r5 = sum(scenario_r5) / len(scenario_r5) if scenario_r5 else 0.0

    return [
        ScorecardRow("scenario_coverage_fraction", coverage, 0.50,
                     source=f"labeled_scenarios (n={n_scenarios})"),
        ScorecardRow("scenario_recall5_covered", avg_r5, 0.80,
                     source=f"labeled_scenarios_covered (n={covered})"),
    ]
