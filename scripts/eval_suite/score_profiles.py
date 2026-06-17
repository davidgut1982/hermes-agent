"""Profile-pick accuracy scorer for the Hermes delegation pipeline.

Why: The KB critic (kb_b0a34e28b29b) identified that profile-pick accuracy
is not emitted by any existing telemetry — delegate_task runs inside the
agent loop and the chosen profile is only visible as a DEBUG log line, never
as a structured event. This module:
  1. Attempts to derive predicted profile from any available signals:
     - tool_search_livetest.py output (underlying_tool_calls → infer profile)
     - 'delegate_task' bridge call args if present in livetest records
     - config.yaml agent_profiles keys (registry of valid profile names)
  2. Computes profile-pick accuracy = correct / total on the subset where
     a predicted profile can be derived.
  3. Documents the exact one-line instrumentation needed for full coverage.

INSTRUMENTATION GAP (documented per spec): Profile choice cannot be derived
from existing outputs for scenarios where delegate_task is not exercised or
where the tool call args were not recorded with a 'profile' field.

REQUIRED FUTURE INSTRUMENTATION:
  File: /opt/hermes/build-combined6/tools/delegate_tool.py
  Location: ~line 2074, immediately after `resolved_profile_name = profile`
  Log line to add (batch into next deploy — do NOT redeploy prod now):

    logger.info(
        "delegate_profile_event: resolved=%r goal_chars=%d",
        resolved_profile_name,
        len(str(goal)) if goal else 0,
    )

  This produces one structured log line per delegation at INFO level.
  Pattern to grep: `delegate_profile_event: resolved=`
  After adding, profile-pick accuracy can be derived from logs without
  any live rerun: grep the log for the pattern, parse resolved=, compare
  to labeled_scenarios.jsonl expected_profile.

Test: Run with existing livetest output; verify partial coverage is reported
and the instrumentation gap message is printed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scorers import (
    ScorecardRow,
    load_labeled_scenarios,
    load_livetest_summary,
)

# Canonical path to HERMES_HOME config
_HERMES_HOME = Path("/opt/hermes/home")
_CONFIG_YAML = _HERMES_HOME / "config.yaml"
_SUITE_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SUITE_DIR.parent


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------


def load_profile_registry() -> set[str]:
    """Load valid profile names from config.yaml agent_profiles section.

    Why: Validates that expected_profile values in labeled_scenarios.jsonl
    are real profiles, and provides the universe for prediction matching.
    What: Parses config.yaml YAML keys under agent_profiles.
    Test: Returns a non-empty set containing 'documents', 'mail', 'calendar'.
    """
    if not _CONFIG_YAML.exists():
        # Fallback: known profiles from delegation.md inspection
        return {"documents", "mail", "calendar", "homelab", "search",
                "files", "memory", "think", "weather", "dealfinder"}
    try:
        import yaml  # type: ignore[import-untyped]
        cfg = yaml.safe_load(_CONFIG_YAML.read_text(encoding="utf-8"))
        profiles = cfg.get("agent_profiles", {})
        return set(profiles.keys()) if isinstance(profiles, dict) else set()
    except Exception:
        # yaml not available or parse error — use known set
        return {"documents", "mail", "calendar", "homelab", "search"}


# ---------------------------------------------------------------------------
# Profile inference from livetest data
# ---------------------------------------------------------------------------


def _infer_profile_from_livetest(row: dict[str, Any],
                                  registry: set[str]) -> str | None:
    """Attempt to infer a delegation profile from a livetest summary row.

    Why: The livetest harness does not record the 'profile' arg passed to
    delegate_task, so we must infer from the tool calls made.
    What: Looks for a 'delegate_task' bridge call with a 'profile' arg in
    the bridge_calls list of the per-scenario JSON file; falls back to None
    when the tool was not used.
    Test: Row with bridge_calls=[{name:'tool_call', args:{name:'delegate_task',
    arguments:{profile:'mail'}}}] → returns 'mail'.
    """
    # Load the per-scenario JSON for richer data (bridge_calls field)
    scenario_id = row.get("scenario", "")
    enabled = row.get("enabled", False)
    suffix = "enabled" if enabled else "disabled"
    scenario_file = _SCRIPTS_DIR / "out" / f"{scenario_id}__{suffix}.json"

    if not scenario_file.exists():
        return None

    try:
        rec = json.loads(scenario_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    # Check bridge_calls for a tool_call → delegate_task with profile arg
    for bridge in rec.get("bridge_calls", []):
        if bridge.get("name") == "tool_call":
            inner = bridge.get("args", {})
            if inner.get("name") == "delegate_task":
                arguments = inner.get("arguments", {})
                if isinstance(arguments, dict):
                    profile = arguments.get("profile")
                    if profile and profile in registry:
                        return str(profile)
    return None


# ---------------------------------------------------------------------------
# Main profile-pick scorer
# ---------------------------------------------------------------------------


def score_profile_pick() -> list[ScorecardRow]:
    """Compute profile-pick accuracy from available signals.

    Why: Profile-pick accuracy (target >= 0.90 per kb_28650bfe5f17) is the
    delegation routing correctness metric. Without inline instrumentation it
    can only be partially computed from livetest bridge_calls data.
    What: Iterates scenarios with expected_profile != null, tries to infer
    predicted_profile from existing data, computes accuracy on the covered
    subset, and documents the gap for the uncovered subset.
    Test: With 5 livetest scenarios, at most 5 can have profiles inferred.
    The function returns a ScorecardRow whether or not any are covered.
    """
    scenarios = load_labeled_scenarios()
    registry = load_profile_registry()
    summary = load_livetest_summary()

    # Build enabled livetest lookup by scenario id
    lt_enabled: dict[str, dict[str, Any]] = {}
    for row in summary:
        if row.get("enabled", False):
            lt_enabled[row["scenario"]] = row

    # Livetest scenario id → labeled scenario id mapping
    lt_id_map = {
        "A_obvious_single": "S001",
        "B_vague_paraphrased": "S002",
        "C_multi_tool_chain": "S003",
        "D_core_plus_deferred": "S004",
        "E_no_tool_needed": "S005",
    }
    labeled_id_to_lt = {v: k for k, v in lt_id_map.items()}

    profile_scenarios = [s for s in scenarios if s.get("expected_profile") is not None]

    correct = 0
    covered = 0
    uncovered = 0

    for sc in profile_scenarios:
        sc_id = sc["id"]
        expected_profile = sc["expected_profile"]

        # Try to get a predicted profile
        predicted_profile: str | None = None

        # Check if this scenario maps to a livetest run
        lt_key = labeled_id_to_lt.get(sc_id)
        if lt_key and lt_key in lt_enabled:
            row = lt_enabled[lt_key]
            predicted_profile = _infer_profile_from_livetest(row, registry)

        if predicted_profile is not None:
            covered += 1
            if predicted_profile == expected_profile:
                correct += 1
        else:
            uncovered += 1

    # Build result rows
    rows: list[ScorecardRow] = []

    if covered == 0:
        # No profile predictions derivable from existing outputs
        rows.append(ScorecardRow(
            "profile_pick_accuracy", 0.0, 0.90,
            source=(
                f"NOT_DERIVABLE: {len(profile_scenarios)} profile scenarios, "
                f"0 covered by existing outputs. "
                "See REQUIRED FUTURE INSTRUMENTATION in score_profiles.py docstring."
            ),
        ))
    else:
        accuracy = correct / covered
        rows.append(ScorecardRow(
            "profile_pick_accuracy", accuracy, 0.90,
            source=(
                f"livetest_bridge_calls (covered={covered}/{len(profile_scenarios)}, "
                f"uncovered={uncovered})"
            ),
        ))

    rows.append(ScorecardRow(
        "profile_pick_coverage_fraction",
        covered / len(profile_scenarios) if profile_scenarios else 0.0,
        0.50,
        source=(
            f"labeled_scenarios profile scenarios={len(profile_scenarios)}, "
            f"covered={covered}"
        ),
    ))

    return rows


# ---------------------------------------------------------------------------
# Standalone runner for debugging
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    rows = score_profile_pick()
    print("\nProfile-pick scorer results:")
    print("-" * 60)
    for row in rows:
        status = "PASS" if row.pass_ else "FAIL"
        print(f"  [{status}] {row.metric}: {row.actual:.3f} (bar={row.bar:.2f})")
        if row.source:
            print(f"         source: {row.source}")
    print()
    print("INSTRUMENTATION GAP NOTE:")
    print(  # noqa: E501
        "  profile choice requires adding ONE log line to delegate_tool.py ~line 2074:"
    )
    print("    logger.info(")
    print('        "delegate_profile_event: resolved=%r goal_chars=%d",')
    print("        resolved_profile_name,")
    print("        len(str(goal)) if goal else 0,")
    print("    )")
