#!/usr/bin/env python3
"""Live re-measure of the reranker benchmark — ALWAYS calls the live nomic endpoint.

Why: The committed data/results_prefix.json is a SNAPSHOT that cannot catch a
deployed reranker regression. This script forces fresh embeddings from the live
nomic endpoint on every run, so the gate actually measures the deployed system.

Fidelity: FULL prod-reranker fidelity for mode=rerank.
  - Uses the SAME embedding endpoint, model, and task prefixes as prod
    (read from /opt/hermes/home/config.yaml if parseable, else hardcoded defaults
    matching the known prod config).
  - Uses mode=rerank (pure cosine over the full tool catalog), which is prod's
    configured mode. The cosine-rerank logic is replicated here in self-contained
    pure Python (the EmbeddingReranker class that previously lived in tool_search.py
    was extracted from that module; this script reimplements the same math so the
    gate does not depend on hermes internals at import time).
  - embed cache is IN-MEMORY ONLY (no disk read/write) so every gate run
    re-fetches all embeddings from the live endpoint. This is intentional —
    it is what makes the gate "live".
  - Does NOT invoke the prod Python module directly (that would require importing
    hermes internals and dealing with the registry). Instead, it replicates the
    cosine-rerank math directly, which is self-contained pure Python.

Fail-closed: If the nomic endpoint is unreachable, exits NON-ZERO with a clear
message. The gate MUST NOT pass when the endpoint is down.

What: Reads real_tools.json + labeled_queries.json from /tmp/tool-rerank-poc/,
embeds with prod nomic prefixes via LIVE endpoint (no disk cache), runs cosine
rerank (mode=rerank, full catalog), writes fresh scripts/eval_suite/data/results_prefix.json.

Usage:
    python3 scripts/eval_suite/run_live_benchmark.py [--tools-file F] [--queries-file F]
    # All args optional — defaults to /tmp/tool-rerank-poc/ files.

Exits 0 on success, non-zero on any failure (endpoint down, file missing, etc.).

Test: Run once; assert data/results_prefix.json updated, overall_recall5 >= 0.70.
      Run with bad endpoint URL; assert exit non-zero with "could not measure live" message.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Prod reranker defaults (mirror /opt/hermes/home/config.yaml)
# These are the KNOWN prod values as of 2026-05-31.
# We attempt to read them from the live config file; if that fails we use
# these hardcoded defaults so the gate still reflects the prod config.
# ---------------------------------------------------------------------------

_PROD_DEFAULTS = {
    "endpoint": "http://192.168.1.36:11434/v1/embeddings",
    "model": "nomic-embed-text-v2-moe",
    "mode": "rerank",
    "query_prefix": "search_query: ",
    "doc_prefix": "search_document: ",
    "top_k": 5,
    "timeout": 5.0,
}

_CONFIG_FILE = Path("/opt/hermes/home/config.yaml")
_BENCH_DIR = Path("/tmp/tool-rerank-poc")
_DEFAULT_TOOLS_FILE = _BENCH_DIR / "real_tools.json"
_DEFAULT_QUERIES_FILE = _BENCH_DIR / "labeled_queries.json"

_SUITE_DIR = Path(__file__).resolve().parent
_OUT_FILE = _SUITE_DIR / "data" / "results_prefix.json"

# Embed batch size — nomic handles up to 512 at once; use 64 to stay safe.
_EMBED_BATCH = 64


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_prod_reranker_config() -> dict[str, Any]:
    """Load prod reranker config from /opt/hermes/home/config.yaml.

    Why: Using prod config ensures the gate measures the EXACT deployed params.
    What: Parses YAML manually (no pyyaml dep) to extract tools.tool_search.reranker.
          Falls back to hardcoded defaults if file missing or parse error.
    Test: With config.yaml present, returned dict should have endpoint matching
          the known nomic URL; with file absent, returns _PROD_DEFAULTS.
    """
    cfg = dict(_PROD_DEFAULTS)

    if not _CONFIG_FILE.exists():
        print(f"[config] {_CONFIG_FILE} not found — using hardcoded prod defaults",
              flush=True)
        return cfg

    try:
        # Minimal YAML parsing (tools.tool_search.reranker section only).
        # We use a simple state-machine approach to avoid the pyyaml dep.
        text = _CONFIG_FILE.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Find "reranker:" under "tools:" -> "tool_search:"
        in_tools = False
        in_tool_search = False
        in_reranker = False
        extracted: dict[str, str] = {}

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())

            if indent == 0:
                in_tools = stripped.startswith("tools:")
                in_tool_search = False
                in_reranker = False
            elif in_tools and indent == 2:
                in_tool_search = stripped.startswith("tool_search:")
                in_reranker = False
            elif in_tool_search and indent == 4:
                in_reranker = stripped.startswith("reranker:")
            elif in_reranker and indent == 6 and ":" in stripped:
                key, _, val = stripped.partition(":")
                extracted[key.strip()] = val.strip().strip("'\"")

        if "endpoint" in extracted:
            cfg["endpoint"] = extracted["endpoint"]
        if "model" in extracted:
            cfg["model"] = extracted["model"]
        if "mode" in extracted:
            cfg["mode"] = extracted["mode"]
        if "query_prefix" in extracted:
            cfg["query_prefix"] = extracted["query_prefix"]
        if "doc_prefix" in extracted:
            cfg["doc_prefix"] = extracted["doc_prefix"]
        if "timeout" in extracted:
            try:
                cfg["timeout"] = float(extracted["timeout"])
            except ValueError:
                pass

        # top_k from search_default_limit (reranker.top_k defaults to search_default_limit in prod)
        # We scan for search_default_limit in the tool_search section.
        in_tools2 = False
        in_ts2 = False
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if indent == 0:
                in_tools2 = stripped.startswith("tools:")
            elif in_tools2 and indent == 2:
                in_ts2 = stripped.startswith("tool_search:")
            elif in_ts2 and indent == 4 and stripped.startswith("search_default_limit:"):
                _, _, val = stripped.partition(":")
                try:
                    cfg["top_k"] = int(val.strip())
                except ValueError:
                    pass

        print(f"[config] Loaded prod reranker config from {_CONFIG_FILE}:", flush=True)
        print(f"         endpoint={cfg['endpoint']} model={cfg['model']} "
              f"mode={cfg['mode']} top_k={cfg['top_k']}", flush=True)
        print(f"         query_prefix={cfg['query_prefix']!r} "
              f"doc_prefix={cfg['doc_prefix']!r}", flush=True)

    except Exception as exc:
        print(f"[config] WARNING: could not parse {_CONFIG_FILE}: {exc} "
              f"— using hardcoded defaults", flush=True)

    return cfg


# ---------------------------------------------------------------------------
# Embedding (live endpoint, in-memory cache only)
# ---------------------------------------------------------------------------


def _embed_batch_live(
    texts: list[str],
    endpoint: str,
    model: str,
    timeout: float,
) -> list[list[float]]:
    """Embed a batch of texts via the live nomic endpoint. No caching.

    Why: No disk cache means every gate run fetches fresh embeddings, which
    is what makes the gate actually measure the deployed system.
    What: POST to /v1/embeddings; returns ordered list of float vectors.
    Fail-closed: raises urllib.error.URLError or OSError on network failure
    so the caller can exit non-zero with a clear message.
    Test: Call with endpoint down; assert URLError raised (not silently ignored).
    """
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


def embed_all_live(
    texts: list[str],
    endpoint: str,
    model: str,
    timeout: float,
) -> list[list[float]]:
    """Embed all texts in batches using the live endpoint. IN-MEMORY cache only.

    Why: Batching avoids per-text HTTP overhead; in-memory dedup avoids
    re-embedding identical texts within a single gate run.
    What: Splits texts into batches of _EMBED_BATCH, calls _embed_batch_live,
    assembles results in order. Cache is per-call (dict), discarded on return.
    Fail-closed: propagates endpoint errors up — never silently swallows them.
    Test: Call with 200 identical texts; assert only 1 batch of 64 unique texts
    hits the endpoint (in-memory dedup working).
    """
    # In-memory dedup for this run only
    cache: dict[str, list[float]] = {}
    result: list[list[float]] = [[] for _ in texts]

    # Gather unique texts (preserving first-occurrence order)
    unique_texts: list[str] = []
    unique_idx_map: dict[str, int] = {}
    for t in texts:
        if t not in unique_idx_map:
            unique_idx_map[t] = len(unique_texts)
            unique_texts.append(t)

    # Batch-embed unique texts
    n_batches = (len(unique_texts) + _EMBED_BATCH - 1) // _EMBED_BATCH
    for batch_i in range(n_batches):
        batch = unique_texts[batch_i * _EMBED_BATCH:(batch_i + 1) * _EMBED_BATCH]
        print(f"  [embed] batch {batch_i + 1}/{n_batches} "
              f"({len(batch)} texts) → {endpoint}", flush=True)
        vecs = _embed_batch_live(batch, endpoint, model, timeout)
        if len(vecs) != len(batch):
            raise ValueError(
                f"embedding endpoint returned {len(vecs)} vectors for {len(batch)} texts"
            )
        for t, vec in zip(batch, vecs):
            cache[t] = vec

    # Assemble output (in original order, with in-run dedup)
    for i, t in enumerate(texts):
        result[i] = cache[t]

    return result


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Why: Core distance metric for the nomic embedding reranker. Same formula
    as prod tool_search.py:_cosine().
    What: dot(a, b) / (norm(a) * norm(b)); returns 0.0 if either norm is zero.
    Test: cosine([1,0], [1,0]) == 1.0; cosine([1,0], [0,1]) == 0.0.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Recall@K and MRR
# ---------------------------------------------------------------------------


def recall_at_k(gold: list[str], predicted: list[str], k: int) -> float:
    """Standard IR Recall@K. Test: recall_at_k(['a','b'],['b','c','a'],2)==0.5."""
    if not gold:
        return 0.0
    return len(set(gold) & set(predicted[:k])) / len(set(gold))


def mrr_score(gold: list[str], predicted: list[str]) -> float:
    """MRR: 1/rank of first gold tool. Test: mrr(['a'],['c','b','a'])==1/3."""
    gold_set = set(gold)
    for rank, name in enumerate(predicted, start=1):
        if name in gold_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(tools_file: Path, queries_file: Path, cfg: dict[str, Any]) -> int:
    """Run live benchmark and write fresh data/results_prefix.json.

    Why: Core logic — embeds all tools + queries from live endpoint, ranks by
    cosine (mode=rerank, matching prod), writes output for the gate to score.
    What: Loads tools + queries, embeds with prod prefixes via live endpoint,
    computes R@5/MRR, writes results_prefix.json.
    Fail-closed: returns non-zero on any live-endpoint failure.
    Test: After successful run, data/results_prefix.json should have
    overall_recall5 >= 0.70 and queries_count == 98.
    """
    endpoint: str = cfg["endpoint"]
    model: str = cfg["model"]
    query_prefix: str = cfg["query_prefix"]
    doc_prefix: str = cfg["doc_prefix"]
    top_k: int = cfg["top_k"]
    timeout: float = cfg["timeout"]

    # ---- Load input files ----
    if not tools_file.exists():
        print(
            f"\n[LIVE GATE FAIL] Tools file not found: {tools_file}\n"
            "could not measure live — gate FAILS closed\n"
            f"Hint: populate {_BENCH_DIR} by running bench_tiers.py once.",
            file=sys.stderr,
        )
        return 1

    if not queries_file.exists():
        print(
            f"\n[LIVE GATE FAIL] Queries file not found: {queries_file}\n"
            "could not measure live — gate FAILS closed",
            file=sys.stderr,
        )
        return 1

    tools_raw: list[dict[str, Any]] = json.loads(
        tools_file.read_text(encoding="utf-8")
    )
    queries: list[dict[str, Any]] = json.loads(
        queries_file.read_text(encoding="utf-8")
    )
    print(f"[corpus] {len(tools_raw)} tools, {len(queries)} queries loaded",
          flush=True)

    # ---- Build (name, embed_text) for each tool ----
    docs: list[tuple[str, str]] = []
    for t in tools_raw:
        sch = t.get("schema", {})
        fn = sch.get("function") or {}
        name = str(sch.get("name") or fn.get("name") or "")
        if not name:
            continue
        desc = str(sch.get("description") or fn.get("description") or "")
        embed_text = f"{name}: {desc}"
        docs.append((name, embed_text))

    print(f"[corpus] {len(docs)} tool documents built", flush=True)

    # ---- Verify endpoint is reachable BEFORE spending time on embedding ----
    print(f"\n[probe] Checking live endpoint {endpoint} ...", flush=True)
    try:
        probe_vecs = _embed_batch_live(
            [f"{query_prefix}hermes eval gate probe"], endpoint, model, timeout
        )
        if not probe_vecs or not probe_vecs[0]:
            raise ValueError("probe returned empty vector")
        print(f"[probe] OK — dim={len(probe_vecs[0])}", flush=True)
    except (urllib.error.URLError, OSError, Exception) as exc:
        print(
            f"\n[LIVE GATE FAIL] Embedding endpoint unreachable: {exc}\n"
            f"  endpoint: {endpoint}\n"
            "could not measure live — gate FAILS closed\n"
            "Bring up the nomic endpoint before running the gate in live mode.\n"
            "If you must use stale data (not recommended), pass --snapshot to ci_gate.sh.",
            file=sys.stderr,
        )
        return 1

    # ---- Embed tool documents (live, no disk cache) ----
    doc_names = [name for name, _ in docs]
    doc_pfx_texts = [f"{doc_prefix}{et}" for _, et in docs]

    print(f"\n[step 1/2] Embedding {len(doc_pfx_texts)} tool docs "
          f"(prefix={doc_prefix!r}) via live endpoint ...", flush=True)
    t0 = time.perf_counter()
    try:
        doc_vecs = embed_all_live(doc_pfx_texts, endpoint, model, timeout)
    except (urllib.error.URLError, OSError, Exception) as exc:
        print(
            f"\n[LIVE GATE FAIL] Tool embedding failed: {exc}\n"
            "could not measure live — gate FAILS closed",
            file=sys.stderr,
        )
        return 1
    print(f"  Done in {time.perf_counter() - t0:.1f}s", flush=True)

    doc_vec_map: dict[str, list[float]] = {
        name: vec for name, vec in zip(doc_names, doc_vecs)
    }

    # ---- Embed all queries in one batch pass ----
    print(f"\n[step 2/2] Embedding {len(queries)} queries (batch) and scoring ...",
          flush=True)

    pfx_queries = [f"{query_prefix}{q['query']}" for q in queries]
    try:
        query_vecs = embed_all_live(pfx_queries, endpoint, model, timeout)
    except (urllib.error.URLError, OSError, Exception) as exc:
        print(
            f"\n[LIVE GATE FAIL] Query embedding batch failed: {exc}\n"
            "could not measure live — gate FAILS closed",
            file=sys.stderr,
        )
        return 1

    per_query_full: list[dict[str, Any]] = []
    total_recall5 = 0.0
    cat_recall: dict[str, list[float]] = {}
    cat_mrr: dict[str, list[float]] = {}

    for idx, (q, q_vec) in enumerate(zip(queries, query_vecs), start=1):
        query_text: str = q["query"]
        category: str = q.get("category", "UNKNOWN")
        gold: list[str] = q.get("gold_tools", [])

        # Pure cosine rerank over full catalog (mode=rerank, matching prod exactly)
        scored: list[tuple[float, str]] = [
            (cosine(q_vec, doc_vec_map[name]), name)
            for name in doc_names
        ]
        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        rerank_top = [name for _, name in ranked[:top_k]]

        r5 = recall_at_k(gold, rerank_top, top_k)
        m = mrr_score(gold, rerank_top)

        total_recall5 += r5
        cat_recall.setdefault(category, []).append(r5)
        cat_mrr.setdefault(category, []).append(m)

        per_query_full.append({
            "idx": idx,
            "category": category,
            "query": query_text,
            "gold": gold,
            "bm25_top5": q.get("bm25_top5", []),
            "rerank_top5": rerank_top,
            "rerank_recall5": round(r5, 4),
            "rerank_mrr": round(m, 4),
        })

    n = len(queries)
    overall_r5 = total_recall5 / n if n > 0 else 0.0
    print(f"\n[result] Overall R@5 (live, mode={cfg['mode']}) = {overall_r5:.4f}",
          flush=True)
    for cat in sorted(cat_recall):
        vals = cat_recall[cat]
        print(f"  {cat:<12}: R@5={sum(vals)/len(vals):.4f} (n={len(vals)})",
              flush=True)

    if overall_r5 < 0.70:
        print(
            f"\n[WARN] Overall R@5={overall_r5:.4f} is BELOW regression guard (0.70).",
            flush=True,
        )
    else:
        print(f"\n[OK] R@5={overall_r5:.4f} >= regression guard 0.70", flush=True)

    # ---- Write output ----
    _OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "description": (
            "LIVE-MEASURED reranker benchmark — fresh embeddings from live nomic endpoint. "
            f"Generated by run_live_benchmark.py at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. "
            "NO disk cache used. Embeddings pulled fresh every gate run."
        ),
        "live_measured": True,
        "tools_count": len(docs),
        "queries_count": n,
        "embed_model": model,
        "embed_endpoint": endpoint,
        "prefix_config": {
            "query_prefix": query_prefix,
            "doc_prefix": doc_prefix,
            "note": "Prod config (matches /opt/hermes/home/config.yaml). "
                    "Prefixes required by nomic-embed-text-v2-moe.",
        },
        "reranker_mode": cfg["mode"],
        "overall_recall5": round(overall_r5, 4),
        "per_category": {
            cat: {
                "recall5": round(sum(cat_recall[cat]) / len(cat_recall[cat]), 4),
                "mrr": round(sum(cat_mrr[cat]) / len(cat_mrr[cat]), 4),
                "n": len(cat_recall[cat]),
            }
            for cat in sorted(cat_recall)
        },
        "per_query_full": per_query_full,
    }

    _OUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(
        f"\n[written] {_OUT_FILE}  ({_OUT_FILE.stat().st_size:,} bytes, {n} queries)",
        flush=True,
    )
    return 0


def main() -> int:
    """Entry point for live benchmark re-measure.

    Why: Allows standalone invocation and parameterised paths for testing.
    What: Parses args, loads prod config, calls run().
    Test: python3 run_live_benchmark.py; assert exit 0 and results_prefix.json updated.
    """
    parser = argparse.ArgumentParser(description="Live reranker benchmark re-measure")
    parser.add_argument(
        "--tools-file",
        type=Path,
        default=_DEFAULT_TOOLS_FILE,
        help=f"Path to real_tools.json (default: {_DEFAULT_TOOLS_FILE})",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=_DEFAULT_QUERIES_FILE,
        help=f"Path to labeled_queries.json (default: {_DEFAULT_QUERIES_FILE})",
    )
    args = parser.parse_args()

    cfg = _load_prod_reranker_config()
    return run(args.tools_file, args.queries_file, cfg)


if __name__ == "__main__":
    sys.exit(main())
