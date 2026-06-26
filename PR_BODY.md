# Add `terminal.parallel_safe_prefixes` config allowlist for parallel terminal batches

## Summary

The parallel-dispatch gate (`_should_parallelize_tool_batch`) treats every
`terminal` tool call as not-parallel-safe, so **any batch containing a
`terminal` call is forced to run sequentially** — even when the model emits
several independent, read-only one-shot commands in a single turn.

This PR adds an **operator-configurable allowlist of read-only command
prefixes**, `terminal.parallel_safe_prefixes`. When a batch's `terminal` calls
all start with a configured prefix, the batch is dispatched concurrently via the
existing `ThreadPoolExecutor` path instead of the sequential for-loop.

The feature is **off by default** (empty allowlist → current behavior exactly).
It is config-driven with **no command names baked into the engine**.

## Precedent

This mirrors the existing per-server MCP opt-in: `tools/mcp_tool.py` reads
`supports_parallel_tool_calls` from each server's config into
`_parallel_safe_servers`, which `is_mcp_tool_parallel_safe()` consults in the
same gate. Terminal commands get the analogous, prefix-based opt-in.

## Design

- **`agent/tool_dispatch_helpers.py`** — new `_is_terminal_call_parallel_safe()`
  reads the allowlist from the `TERMINAL_PARALLEL_SAFE_PREFIXES` env var (a JSON
  list, bridged from `terminal.parallel_safe_prefixes`), and returns `True` iff
  the call's `command` string starts with a configured prefix. Empty/unset/
  malformed → `False`.
- **`_should_parallelize_tool_batch`** — adds a `terminal` branch gated on that
  helper. All existing short-circuits (`_NEVER_PARALLEL_TOOLS`, `len <= 1`,
  arg-parse failures, non-dict args, path-scoped overlap) remain ordered ahead
  of it, so behavior outside the allowlist is unchanged.
- **`cli.py` / `gateway/run.py`** — add `parallel_safe_prefixes: []` to the
  terminal config defaults and bridge it to `TERMINAL_PARALLEL_SAFE_PREFIXES`
  (the existing bridge already JSON-encodes list values), matching every other
  `terminal.* → TERMINAL_*` mapping.

## Before / After

A turn where the model batches three independent read-only lookups:

**Before** — `terminal` is never parallel-safe → whole batch serial:

```
[parallel-gate] evaluating batch of 3 tools: ['terminal', 'terminal', 'terminal']
[parallel-gate] SEQUENTIAL — tool terminal is not parallel-safe
```

**After** — with `parallel_safe_prefixes` configured and all three commands
matching → concurrent dispatch:

```
[parallel-gate] evaluating batch of 3 tools: ['terminal', 'terminal', 'terminal']
[parallel-gate] CONCURRENT — all tools cleared
tool terminal completed (1.64s, ...)
tool terminal completed (1.85s, ...)   # overlapping wall-clock → ThreadPoolExecutor
tool terminal completed (1.91s, ...)
```

The three completions land within ~270 ms of each other with overlapping
durations — they ran on parallel workers, not back-to-back.

## Operator configuration (example)

This config lives in the **operator's** `config.yaml`, never in engine code.
Only opt in commands that are stateless and read-only — batching already implies
the model considers them independent:

```yaml
terminal:
  # Read-only one-shot CLI lookups: safe to run concurrently in a batch.
  parallel_safe_prefixes: [gh, kubectl get, aws s3 ls, curl -s]
```

Leaving it empty (the default) preserves today's fully-sequential terminal
behavior.

## Tests

Added `TestTerminalPrefixParallelToolBatch` driving the public
`_should_parallelize_tool_batch`, mocking the allowlist at the env-var boundary:

- 2 matching `terminal` calls, prefix configured → parallel (True)
- prefixes unset / empty `[]` → serial (False) — regression guard
- command not matching any prefix → serial (False)
- matching `terminal` + path-scoped `write_file` on a non-overlapping path →
  parallel (True); overlapping path → serial (False) — exercises the terminal
  branch together with the `_PATH_SCOPED_TOOLS` reserved-paths logic
- matching `terminal` + a `_NEVER_PARALLEL` tool → serial (False)
- single call (`len <= 1`) → serial (False)

```
$ pytest tests/run_agent/test_run_agent.py::TestTerminalPrefixParallelToolBatch \
         tests/run_agent/test_run_agent.py::TestMcpParallelToolBatch \
         tests/agent/test_tool_dispatch_helpers.py -q
.......................................                                  [100%]
39 passed in 5.50s
```

No behavior change when the allowlist is empty; the existing MCP and
path-overlap gate tests are unaffected.
