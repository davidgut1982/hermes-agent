"""Context-truncation detection + warning for the local-Ollama path.

Why: Ollama loads each model with a fixed context window. When the assembled
request (system prompt + conversation + tool schemas) exceeds that window, the
``/v1`` OpenAI-compatible endpoint **silently truncates the input** — it does
not error, it does not warn. The model then receives a mangled prompt and emits
garbled / confabulated output: it invents tool names and leaks tool-call JSON as
plain text. Ollama's ``/v1`` endpoint also ignores per-request ``num_ctx`` (see
upstream NousResearch/hermes-agent #43900), so even Hermes setting ``num_ctx``
cannot grow a window that was loaded too small. This module makes that
truncation **loud and actionable** instead of silent.

What: Two checks, scoped strictly to the ``provider == "custom"`` local/Ollama
path (cloud providers untouched):
  * Pre-flight — before the chat-completion call, compare the estimated request
    size against the model's *actually-loaded* context (Ollama ``/api/ps``); if
    it won't fit, emit a clear WARNING with counts + remedy.
  * Post-flight — when a local response returns ``finish_reason == "length"``
    with empty/near-empty content (the input-truncation signature), emit the
    same diagnostic instead of silently entering the garbled retry/concat path.

How to test: see ``tests/test_ollama_context_truncation.py`` — request exceeds
loaded context → warning; request fits → no warning; ``finish_reason="length"``
+ empty content → diagnostic; ``/api/ps`` unreachable → graceful ``None``.

Complements upstream PR #43995 (which fixes ``num_ctx`` passing) by *detecting*
the case where the request still cannot fit the loaded window.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.model_metadata import is_local_endpoint, query_ollama_loaded_context

logger = logging.getLogger(__name__)


# ── String resources (no magic strings) ────────────────────────────────────
CUSTOM_PROVIDER = "custom"
FINISH_REASON_LENGTH = "length"

# Reserve headroom so we warn *before* the model silently drops tokens, not
# only once we're already over. ~10% of the window or 256 tokens, whichever is
# larger, covers generation reserve + estimator slack.
SAFETY_MARGIN_FRACTION = 0.10
SAFETY_MARGIN_MIN_TOKENS = 256

# A truncated-input response is short: the model burned its window on the
# (mangled) prompt and produced almost nothing coherent.
POSTFLIGHT_SHORT_CONTENT_CHARS = 64

# How long to trust a cached /api/ps reading (seconds). The loaded window only
# changes on model reload, so a short cache removes per-call HTTP overhead
# without going stale in practice.
_PS_CACHE_TTL_SECONDS = 30.0

_REMEDY = (
    "Raise the loaded Ollama context window: set OLLAMA_CONTEXT_LENGTH (env on "
    "the Ollama server) or bake `PARAMETER num_ctx <N>` into the Modelfile and "
    "reload the model. Ollama's /v1 endpoint ignores per-request num_ctx, so "
    "Hermes cannot grow the window from the client side — see "
    "NousResearch/hermes-agent#43900."
)

# Process-local cache: {(server_url, model): (timestamp, loaded_ctx_or_None)}
_ps_cache: Dict[tuple[str, str], tuple[float, Optional[int]]] = {}


@dataclass(frozen=True)
class TruncationWarning:
    """Structured, surfaceable context-truncation warning.

    Why: WARNING logs are for operators; this struct lets the conversation /
    gateway layer relay the same diagnostic to the end user (e.g. Telegram).
    What: Carries the numbers and the human-readable message.
    Test: Construct one and assert ``.message`` contains the token counts.
    """

    estimated_tokens: int
    loaded_context: int
    phase: str  # "preflight" | "postflight"
    message: str

    def as_dict(self) -> Dict[str, Any]:
        """Why: gateways serialize result dicts; expose a JSON-safe view.
        What: Returns the warning as a plain dict.
        Test: Assert the returned dict round-trips the four fields.
        """
        return {
            "kind": "context_truncation",
            "phase": self.phase,
            "estimated_tokens": self.estimated_tokens,
            "loaded_context": self.loaded_context,
            "message": self.message,
        }


def is_local_custom_target(provider: Optional[str], base_url: Optional[str]) -> bool:
    """Why: scope every check to local Ollama only — never touch cloud providers.
    What: True iff provider is ``custom`` and base_url is a local endpoint.
    Test: ``("custom","http://localhost:11434/v1")`` → True;
    ``("openrouter","https://openrouter.ai/api/v1")`` → False;
    ``("custom","https://api.example.com")`` → False.
    """
    if (provider or "").strip().lower() != CUSTOM_PROVIDER:
        return False
    return bool(base_url) and is_local_endpoint(base_url)


def _safety_margin(loaded_context: int) -> int:
    """Why: warn before the silent drop, not after. What: headroom in tokens.
    Test: ``_safety_margin(8192)`` → 819; ``_safety_margin(1000)`` → 256 (floor).
    """
    return max(SAFETY_MARGIN_MIN_TOKENS, int(loaded_context * SAFETY_MARGIN_FRACTION))


def get_loaded_context(
    model: str,
    base_url: str,
    api_key: str = "",
    *,
    _now: Optional[float] = None,
) -> Optional[int]:
    """Why: per-call ``/api/ps`` HTTP would tax every turn; cache it briefly.
    What: Returns the loaded context window (int) for ``model`` on the Ollama
    server, cached for ``_PS_CACHE_TTL_SECONDS``; ``None`` if unreachable/unknown.
    Test: First call hits ``query_ollama_loaded_context``; an immediate second
    call within the TTL is served from cache (no second HTTP call).
    """
    now = _now if _now is not None else time.monotonic()
    server = (base_url or "").rstrip("/")
    if server.endswith("/v1"):
        server = server[:-3]
    key = (server, model)

    cached = _ps_cache.get(key)
    if cached is not None and (now - cached[0]) < _PS_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        loaded = query_ollama_loaded_context(model, base_url, api_key=api_key)
    except Exception as exc:  # graceful fallback — never crash the turn
        logger.debug("Ollama /api/ps loaded-context query failed: %s", exc)
        loaded = None

    _ps_cache[key] = (now, loaded)
    return loaded


def clear_cache() -> None:
    """Why: deterministic tests. What: drops the /api/ps cache. Test: call, then
    assert the next ``get_loaded_context`` re-queries."""
    _ps_cache.clear()


def _format_preflight_message(estimated: int, loaded: int) -> str:
    return (
        "Local Ollama request (~%d tokens) exceeds the model's loaded context "
        "window (%d tokens). Ollama will SILENTLY TRUNCATE the input, which "
        "causes garbled output (invented tool names, leaked tool-call JSON). %s"
        % (estimated, loaded, _REMEDY)
    )


def _format_postflight_message(estimated: Optional[int], loaded: int) -> str:
    est_part = (
        "~%d tokens" % estimated if estimated is not None else "an unknown number of tokens"
    )
    return (
        "Local Ollama response came back finish_reason='length' with empty/short "
        "content while the request was %s against a loaded context window of %d "
        "tokens. This is the input-truncation signature, NOT an output cap: the "
        "prompt overflowed the window and was silently truncated. %s"
        % (est_part, loaded, _REMEDY)
    )


def check_preflight(
    *,
    provider: Optional[str],
    base_url: Optional[str],
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    system_prompt: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Optional[TruncationWarning]:
    """Pre-flight check: will this request fit the loaded Ollama window?

    Why: catch silent input-truncation *before* sending, while we can still tell
    the user exactly why their model is about to confabulate.
    What: For a local custom target only, estimate request tokens (reusing
    ``estimate_request_tokens_rough``) and compare against the ``/api/ps`` loaded
    context minus a safety margin. Returns a ``TruncationWarning`` (and logs at
    WARNING) when it won't fit, else ``None``. Returns ``None`` for non-local
    targets or when the loaded window can't be determined (graceful fallback).
    Test: request 9000 est tokens vs 4096 loaded → warning; 1000 vs 8192 → None;
    ``/api/ps`` unreachable → None and no crash.
    """
    if not is_local_custom_target(provider, base_url):
        return None

    loaded = get_loaded_context(model, base_url or "", api_key or "")
    if not loaded or loaded <= 0:
        return None  # unknown window → cannot judge; stay silent (no false alarms)

    # Local import avoids a heavy import at module load and keeps the estimator
    # the single source of truth for request sizing.
    from agent.model_metadata import estimate_request_tokens_rough

    estimated = estimate_request_tokens_rough(
        messages, system_prompt=system_prompt, tools=tools
    )

    if estimated <= (loaded - _safety_margin(loaded)):
        return None

    message = _format_preflight_message(estimated, loaded)
    logger.warning("⚠️  CONTEXT TRUNCATION RISK — %s", message)
    return TruncationWarning(
        estimated_tokens=estimated,
        loaded_context=loaded,
        phase="preflight",
        message=message,
    )


def is_truncation_signature(finish_reason: Optional[str], content: Optional[str]) -> bool:
    """Why: distinguish input-truncation from a genuine output cap.
    What: True when finish_reason is ``length`` AND content is empty/very short.
    Test: ``("length","")`` → True; ``("length","x"*200)`` → False;
    ``("stop","")`` → False.
    """
    if (finish_reason or "") != FINISH_REASON_LENGTH:
        return False
    text = (content or "").strip()
    return len(text) <= POSTFLIGHT_SHORT_CONTENT_CHARS


def check_postflight(
    *,
    provider: Optional[str],
    base_url: Optional[str],
    api_key: str,
    model: str,
    finish_reason: Optional[str],
    content: Optional[str],
    estimated_tokens: Optional[int] = None,
) -> Optional[TruncationWarning]:
    """Post-flight check: did a local response come back truncated-at-input?

    Why: the existing length-handling path blames "max output tokens" and feeds
    broken partials into a continuation/concatenation loop. For local Ollama the
    real cause is usually input truncation; surface that instead of stitching
    garbage.
    What: For a local custom target with the truncation signature
    (finish_reason='length' + empty/short content), emit the diagnostic (log
    WARNING + return a ``TruncationWarning``). Returns ``None`` otherwise.
    Test: local + ('length','') → warning; local + ('length','full text') →
    None; cloud provider → None.
    """
    if not is_local_custom_target(provider, base_url):
        return None
    if not is_truncation_signature(finish_reason, content):
        return None

    loaded = get_loaded_context(model, base_url or "", api_key or "")
    if not loaded or loaded <= 0:
        # Window unknown — still surface a (slightly weaker) diagnostic so the
        # truncation cause is logged rather than silently retried.
        loaded = 0

    message = _format_postflight_message(estimated_tokens, loaded)
    logger.warning("⚠️  LOCAL RESPONSE TRUNCATED AT INPUT — %s", message)
    return TruncationWarning(
        estimated_tokens=estimated_tokens if estimated_tokens is not None else 0,
        loaded_context=loaded,
        phase="postflight",
        message=message,
    )
