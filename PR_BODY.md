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
  the call's `command` matches a configured prefix on a **word boundary** AND is
  free of shell metacharacters. Empty/unset/malformed → `False`.
  - **Word-boundary match (not bare `startswith`):** the command must equal a
    prefix or be `prefix` followed by whitespace, so prefix `ls` matches `ls -l`
    but **not** `lsof`, and `git` does not match the mutating `git push` unless
    `git` itself is allowlisted with intent.
  - **Metacharacter rejection:** any of `&  ;  |  \`  $(  >  <` or a newline
    (covering `&&` / `||`) in the command means the prefix can no longer bound
    what actually runs (e.g. `mytool x && rm -rf /`), so the call is rejected
    from the parallel path and runs **serially** — the safer default.
- **`_should_parallelize_tool_batch`** — adds a `terminal` branch gated on that
  helper. All existing short-circuits (`_NEVER_PARALLEL_TOOLS`, `len <= 1`,
  arg-parse failures, non-dict args, path-scoped overlap) remain ordered ahead
  of it, so behavior outside the allowlist is unchanged.
- **`cli.py` / `gateway/run.py`** — add `parallel_safe_prefixes: []` to the
  terminal config defaults and bridge it to `TERMINAL_PARALLEL_SAFE_PREFIXES`
  (the existing bridge already JSON-encodes list values), matching every other
  `terminal.* → TERMINAL_*` mapping.

## Concurrency safety — persistent-shell snapshot race (upstream #38249)

Concurrent `terminal` calls share one persistent-shell environment (resolved by
`task_id`). `BaseEnvironment.execute()` persists session state by
**read-modify-write of per-session snapshot/cwd files** (`_snapshot_path` /
`_cwd_file`): each call sources the snapshot, runs, then re-dumps `export -p`
back to the snapshot and writes `pwd` to the cwd file. Two batched calls running
on the existing `ThreadPoolExecutor` would race that read-modify-write and
corrupt the session `PATH` — this is upstream bug **#38249**. Because this
feature is what *introduces* the concurrency, it must mitigate the race.

**Mitigation (stateless execution path):** when a `terminal` call is admitted to
a concurrent batch (i.e. it is a declared stateless read-only lookup matching the
allowlist), it executes on a **snapshot-free path** instead of the
session-persisting one:

- `tool_executor.execute_tool_calls_concurrent` marks the batch with a
  `parallel_batch_scope()` ContextVar **before** submitting work, so each worker
  thread inherits the flag when it copies the parent context.
- `terminal_tool` reads `parallel_batch_active()` and calls
  `env.execute(..., persist_session=False)`.
- `BaseEnvironment.execute(persist_session=False)` builds a wrapped script that
  **neither sources nor rewrites** the snapshot and **does not write** the cwd
  file, and it **skips the cwd read-back** so `self.cwd` is not mutated. The
  shared session state is never touched, so the race is removed **by
  construction** — no lock, no serialization of the command itself.

This was chosen over a snapshot-I/O lock because it matches the feature's
semantics exactly: allowlisted commands are declared stateless read-only
lookups that have no need for session cwd/env persistence, so opting them out of
the shared snapshot is correct, not merely a workaround. The path is off by
default and only engages when prefixes are configured AND a call is in a
concurrent batch; remote per-call transports (Modal) accept and ignore the flag.

**Accepted scope (documented in code):** terminal calls are not path-scoped and
reserve no `reserved_paths` entry, so an allowlisted read-only command could in
principle observe a torn read of a file a batched `write_file` is concurrently
writing. This is accepted by design — terminal commands have no declarable path
footprint and operators allowlist only read-only lookups; forcing every
terminal+write_file batch serial would defeat the feature's main use case.

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

`TestTerminalPrefixParallelToolBatch` drives the public
`_should_parallelize_tool_batch`, mocking the allowlist at the env-var boundary:

- 2 matching `terminal` calls, prefix configured → parallel (True)
- prefixes unset / empty `[]` → serial (False) — regression guard
- command not matching any prefix → serial (False)
- matching `terminal` + path-scoped `write_file` on a non-overlapping path →
  parallel (True); overlapping path → serial (False) — exercises the terminal
  branch together with the `_PATH_SCOPED_TOOLS` reserved-paths logic
- matching `terminal` + a `_NEVER_PARALLEL` tool → serial (False)
- single call (`len <= 1`) → serial (False)

Added for the prefix-match tightening:

- **word boundary** — prefix `ls` does **not** match `lsof` (serial); matches
  `ls -l` and bare `ls` (parallel)
- **metacharacter rejection** — `gh-axi issue view N` parallelizes, but
  `gh-axi x && rm -rf /` is forced serial; a unit matrix asserts each of
  `&& || ; | \` $( > <` and newline individually rejects an otherwise-matching
  command

Added for the concurrency-safety mitigation (#38249):

- `parallel_batch_scope()` sets the `parallel_batch_active()` ContextVar and it
  propagates into a copied (worker-thread) context, then resets on exit
- `BaseEnvironment._wrap_command(persist_session=False)` emits a script that
  references **neither** `_snapshot_path` **nor** `_cwd_file`; the default still
  references both
- `execute(persist_session=False)` does **not** mutate `self.cwd` (default path
  does), and 8 concurrent stateless calls reporting different cwds leave the
  shared `self.cwd` unchanged — proving the snapshot/cwd race is removed

```
$ pytest tests/run_agent/test_run_agent.py::TestTerminalPrefixParallelToolBatch \
         tests/tools/test_base_environment.py -q
.................................................                        [100%]
35 passed
```

No behavior change when the allowlist is empty; the existing MCP, path-overlap,
and environment tests are unaffected (full `tests/run_agent/test_run_agent.py`:
400 passed; touched environment suites: 135 passed).
