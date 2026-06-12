"""Tests for the local-Ollama context-truncation detection + warning guard.

Covers ``agent/ollama_context_guard.py`` and
``agent/model_metadata.query_ollama_loaded_context``.

The defect (NousResearch/hermes-agent#43900): when a local Ollama model's loaded
context window is smaller than the assembled request, the /v1 endpoint silently
truncates the input and the model confabulates. This guard makes that loud.

IMPORTANT (anti-pattern avoidance): the prior Ollama test only asserted that
``num_ctx`` landed in ``extra_body`` — it passed while the real, user-visible
behavior (truncation surfacing) stayed broken. These tests instead assert the
USER-VISIBLE warning actually fires (a ``TruncationWarning`` is returned and a
WARNING is logged), not merely that a value was placed in a dict.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent import ollama_context_guard as guard
from agent.model_metadata import query_ollama_loaded_context


LOCAL_URL = "http://localhost:11434/v1"
CLOUD_URL = "https://openrouter.ai/api/v1"


@pytest.fixture(autouse=True)
def _clear_ps_cache():
    """Why: the /api/ps cache is process-local; isolate every test.
    What: clears it before and after each test. Test: implicit (fixture)."""
    guard.clear_cache()
    yield
    guard.clear_cache()


def _mock_httpx_get(ps_data, status_code=200, raise_exc=None):
    """Build a mock httpx.Client whose .get() returns the given /api/ps body."""
    mock_resp = MagicMock(status_code=status_code)
    mock_resp.json.return_value = ps_data
    mock_client = MagicMock()
    if raise_exc is not None:
        mock_client.get.side_effect = raise_exc
    else:
        mock_client.get.return_value = mock_resp
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_client)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx, mock_client


# ═══════════════════════════════════════════════════════════════════════════
# Level 1: query_ollama_loaded_context — /api/ps interaction
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryOllamaLoadedContext:
    def test_returns_loaded_context_for_matching_model(self):
        ps = {"models": [{"name": "qwen3:8b", "context_length": 4096}]}
        mock_ctx, _ = _mock_httpx_get(ps)
        import httpx

        with patch.object(httpx, "Client", return_value=mock_ctx):
            assert query_ollama_loaded_context("qwen3:8b", LOCAL_URL) == 4096

    def test_picks_matching_model_among_several(self):
        ps = {
            "models": [
                {"name": "llama3:8b", "context_length": 8192},
                {"name": "qwen3:8b", "context_length": 4096},
            ]
        }
        mock_ctx, _ = _mock_httpx_get(ps)
        import httpx

        with patch.object(httpx, "Client", return_value=mock_ctx):
            assert query_ollama_loaded_context("qwen3:8b", LOCAL_URL) == 4096

    def test_falls_back_to_single_loaded_model(self):
        # Requested name differs but only one model is loaded → use it.
        ps = {"models": [{"name": "qwen3:8b-instruct-q4", "context_length": 2048}]}
        mock_ctx, _ = _mock_httpx_get(ps)
        import httpx

        with patch.object(httpx, "Client", return_value=mock_ctx):
            assert query_ollama_loaded_context("something-else", LOCAL_URL) == 2048

    def test_returns_none_on_connection_error(self):
        mock_ctx, _ = _mock_httpx_get(None, raise_exc=Exception("connection refused"))
        import httpx

        with patch.object(httpx, "Client", return_value=mock_ctx):
            assert query_ollama_loaded_context("qwen3:8b", LOCAL_URL) is None

    def test_returns_none_on_non_200(self):
        mock_ctx, _ = _mock_httpx_get({}, status_code=404)
        import httpx

        with patch.object(httpx, "Client", return_value=mock_ctx):
            assert query_ollama_loaded_context("qwen3:8b", LOCAL_URL) is None

    def test_returns_none_when_no_models_loaded(self):
        mock_ctx, _ = _mock_httpx_get({"models": []})
        import httpx

        with patch.object(httpx, "Client", return_value=mock_ctx):
            assert query_ollama_loaded_context("qwen3:8b", LOCAL_URL) is None

    def test_rejects_bool_context_length(self):
        ps = {"models": [{"name": "qwen3:8b", "context_length": True}]}
        mock_ctx, _ = _mock_httpx_get(ps)
        import httpx

        with patch.object(httpx, "Client", return_value=mock_ctx):
            assert query_ollama_loaded_context("qwen3:8b", LOCAL_URL) is None


# ═══════════════════════════════════════════════════════════════════════════
# Level 2: scoping — local custom only, cloud untouched
# ═══════════════════════════════════════════════════════════════════════════


class TestScoping:
    def test_local_custom_is_target(self):
        assert guard.is_local_custom_target("custom", LOCAL_URL) is True

    def test_cloud_provider_not_target(self):
        assert guard.is_local_custom_target("openrouter", CLOUD_URL) is False

    def test_custom_but_remote_not_target(self):
        assert guard.is_local_custom_target("custom", "https://api.example.com") is False

    def test_non_custom_local_not_target(self):
        # provider must be exactly "custom"; lmstudio local is out of scope.
        assert guard.is_local_custom_target("lmstudio", LOCAL_URL) is False


# ═══════════════════════════════════════════════════════════════════════════
# Level 3: pre-flight check (a) exceeds, (b) fits
# ═══════════════════════════════════════════════════════════════════════════


def _big_messages(n_chars: int):
    """One user message of ~n_chars (≈ n_chars/4 tokens)."""
    return [{"role": "user", "content": "x" * n_chars}]


class TestPreflight:
    def test_warns_when_request_exceeds_loaded_context(self, caplog):
        # ~9000 chars ≈ 2250 tokens of message; tiny 1024 window → overflow.
        with patch.object(guard, "get_loaded_context", return_value=1024):
            with caplog.at_level("WARNING"):
                warning = guard.check_preflight(
                    provider="custom",
                    base_url=LOCAL_URL,
                    api_key="",
                    model="qwen3:8b",
                    messages=_big_messages(9000),
                    tools=None,
                )
        assert warning is not None
        assert warning.phase == "preflight"
        assert warning.loaded_context == 1024
        assert warning.estimated_tokens > 1024
        # USER-VISIBLE: the warning message carries the numbers + remedy.
        assert "1024" in warning.message
        assert "OLLAMA_CONTEXT_LENGTH" in warning.message
        assert "43900" in warning.message
        # And it was logged at WARNING (impossible to miss in logs).
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_no_warning_when_request_fits(self):
        with patch.object(guard, "get_loaded_context", return_value=131072):
            warning = guard.check_preflight(
                provider="custom",
                base_url=LOCAL_URL,
                api_key="",
                model="qwen3:8b",
                messages=_big_messages(400),  # ~100 tokens
                tools=None,
            )
        assert warning is None

    def test_tools_counted_toward_request_size(self):
        # Messages alone fit 4096; large tool schemas push it over.
        fat_tools = [
            {"type": "function", "function": {"name": f"tool_{i}", "description": "d" * 800}}
            for i in range(30)
        ]
        with patch.object(guard, "get_loaded_context", return_value=4096):
            warning = guard.check_preflight(
                provider="custom",
                base_url=LOCAL_URL,
                api_key="",
                model="qwen3:8b",
                messages=_big_messages(200),
                tools=fat_tools,
            )
        assert warning is not None
        assert warning.estimated_tokens > 4096

    def test_cloud_provider_never_warns(self):
        # Even with an absurd request, a non-local provider is out of scope and
        # get_loaded_context must not even be consulted.
        with patch.object(guard, "get_loaded_context") as mock_get:
            warning = guard.check_preflight(
                provider="openrouter",
                base_url=CLOUD_URL,
                api_key="sk-x",
                model="anthropic/claude",
                messages=_big_messages(100000),
                tools=None,
            )
        assert warning is None
        mock_get.assert_not_called()

    def test_unreachable_ps_falls_back_gracefully(self):
        # (d) /api/ps unreachable → get_loaded_context returns None → no warning,
        # no crash.
        with patch.object(guard, "get_loaded_context", return_value=None):
            warning = guard.check_preflight(
                provider="custom",
                base_url=LOCAL_URL,
                api_key="",
                model="qwen3:8b",
                messages=_big_messages(9000),
                tools=None,
            )
        assert warning is None


# ═══════════════════════════════════════════════════════════════════════════
# Level 4: post-flight check (c) finish_reason=length + empty content
# ═══════════════════════════════════════════════════════════════════════════


class TestPostflight:
    def test_diagnostic_on_length_plus_empty_content(self, caplog):
        with patch.object(guard, "get_loaded_context", return_value=4096):
            with caplog.at_level("WARNING"):
                warning = guard.check_postflight(
                    provider="custom",
                    base_url=LOCAL_URL,
                    api_key="",
                    model="qwen3:8b",
                    finish_reason="length",
                    content="",
                    estimated_tokens=9001,
                )
        assert warning is not None
        assert warning.phase == "postflight"
        assert warning.loaded_context == 4096
        assert warning.estimated_tokens == 9001
        assert "43900" in warning.message
        assert "input-truncation signature" in warning.message
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_no_diagnostic_on_length_with_full_content(self):
        # Genuine output cap (real, long content) → not the truncation signature.
        with patch.object(guard, "get_loaded_context", return_value=4096):
            warning = guard.check_postflight(
                provider="custom",
                base_url=LOCAL_URL,
                api_key="",
                model="qwen3:8b",
                finish_reason="length",
                content="A real, substantive answer that is clearly longer than the "
                "short-content threshold and represents a genuine output-cap truncation.",
                estimated_tokens=None,
            )
        assert warning is None

    def test_no_diagnostic_for_normal_stop(self):
        with patch.object(guard, "get_loaded_context", return_value=4096):
            warning = guard.check_postflight(
                provider="custom",
                base_url=LOCAL_URL,
                api_key="",
                model="qwen3:8b",
                finish_reason="stop",
                content="",
                estimated_tokens=None,
            )
        assert warning is None

    def test_cloud_provider_no_postflight_diagnostic(self):
        with patch.object(guard, "get_loaded_context") as mock_get:
            warning = guard.check_postflight(
                provider="anthropic",
                base_url=CLOUD_URL,
                api_key="sk-x",
                model="claude",
                finish_reason="length",
                content="",
                estimated_tokens=None,
            )
        assert warning is None
        mock_get.assert_not_called()

    def test_postflight_surfaces_even_when_window_unknown(self, caplog):
        # /api/ps unreachable but the signature is unmistakable → still log the
        # cause (loaded=0) rather than silently retrying garbage.
        with patch.object(guard, "get_loaded_context", return_value=None):
            with caplog.at_level("WARNING"):
                warning = guard.check_postflight(
                    provider="custom",
                    base_url=LOCAL_URL,
                    api_key="",
                    model="qwen3:8b",
                    finish_reason="length",
                    content="",
                    estimated_tokens=None,
                )
        assert warning is not None
        assert warning.loaded_context == 0
        assert any(r.levelname == "WARNING" for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════
# Level 5: signature + caching
# ═══════════════════════════════════════════════════════════════════════════


class TestSignatureAndCaching:
    def test_is_truncation_signature(self):
        assert guard.is_truncation_signature("length", "") is True
        assert guard.is_truncation_signature("length", "   ") is True
        assert guard.is_truncation_signature("length", "x" * 200) is False
        assert guard.is_truncation_signature("stop", "") is False
        assert guard.is_truncation_signature(None, None) is False

    def test_loaded_context_is_cached(self):
        ps = {"models": [{"name": "qwen3:8b", "context_length": 4096}]}
        mock_ctx, mock_client = _mock_httpx_get(ps)
        import httpx

        with patch.object(httpx, "Client", return_value=mock_ctx):
            first = guard.get_loaded_context("qwen3:8b", LOCAL_URL, _now=1000.0)
            second = guard.get_loaded_context("qwen3:8b", LOCAL_URL, _now=1005.0)

        assert first == 4096
        assert second == 4096
        # Within TTL → only ONE underlying HTTP GET.
        assert mock_client.get.call_count == 1

    def test_cache_expires_after_ttl(self):
        ps = {"models": [{"name": "qwen3:8b", "context_length": 4096}]}
        mock_ctx, mock_client = _mock_httpx_get(ps)
        import httpx

        with patch.object(httpx, "Client", return_value=mock_ctx):
            guard.get_loaded_context("qwen3:8b", LOCAL_URL, _now=1000.0)
            # Past the TTL → re-query.
            guard.get_loaded_context(
                "qwen3:8b", LOCAL_URL, _now=1000.0 + guard._PS_CACHE_TTL_SECONDS + 1
            )

        assert mock_client.get.call_count == 2
