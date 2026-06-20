"""Regression tests for #39550 — token-only compression success.

``compress_context()`` can materially shrink a request (tool-result pruning,
in-place summarization) WITHOUT reducing ``len(messages)``.  Before the fix,
both overflow-recovery checks only looked at ``len(messages) < original_len``,
so a same-count / fewer-tokens compression was misread as "cannot compress
further" and the turn aborted with a false context-exhaustion error.

These tests drive the 413 and context-overflow paths with a compression that
keeps the message COUNT identical but slashes the token estimate, and assert
the loop now retries to success.  They also pin the two guards the author
added: the 5% token-reduction threshold and the ``new_tokens > 0`` empty-
transcript guard.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
import run_agent


@pytest.fixture(autouse=True)
def _no_compression_sleep(monkeypatch):
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


def _mock_response(content="Hello", finish_reason="stop"):
    msg = SimpleNamespace(
        content=content, tool_calls=None, reasoning_content=None, reasoning=None
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = None
    return resp


def _make_413_error(message="Request entity too large"):
    err = Exception(message)
    err.status_code = 413
    return err


def _make_context_overflow_error():
    err = Exception(
        "Error code: 400 - {'error': {'message': "
        "\"This endpoint's maximum context length is 204800 tokens. "
        "However, you requested about 270460 tokens.\", 'code': 400}}"
    )
    err.status_code = 400
    return err


@pytest.fixture()
def agent():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "web_search tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = True
        a.save_trajectories = False
        return a


# A large prefill so the pre-compression token estimate is substantial; the
# fake compression below collapses each message to a single char, so the new
# estimate is well under the 95% threshold while the COUNT is unchanged.
def _big_prefill():
    return [
        {"role": "user", "content": "Q" * 8000},
        {"role": "assistant", "content": "A" * 8000},
    ]


def _same_count_token_shrinking_compress(replacement_char="x"):
    """Return a _compress_context stand-in that keeps message count identical
    but shrinks every message's content — i.e. token-only compression."""

    def _compress(messages, system_message, **_kwargs):
        compressed = [
            {"role": m.get("role", "user"), "content": replacement_char}
            for m in messages
        ]
        return compressed, "compressed prompt"

    return _compress


class TestTokenOnlyCompressionSuccess:
    """Same message count + materially fewer tokens must count as success."""

    def test_413_token_only_compression_retries_and_succeeds(self, agent):
        agent.client.chat.completions.create.side_effect = [
            _make_413_error(),
            _mock_response(content="Recovered via token-only compression"),
        ]
        statuses = []

        with (
            patch.object(
                agent, "_compress_context",
                side_effect=_same_count_token_shrinking_compress(),
            ) as mock_compress,
            patch.object(agent, "_buffer_status", side_effect=statuses.append),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=_big_prefill())

        mock_compress.assert_called_once()
        # The bug: this used to abort as "cannot compress further".
        assert result.get("failed") is not True
        assert result.get("compression_exhausted") is not True
        assert result["completed"] is True
        assert result["final_response"] == "Recovered via token-only compression"
        # User-visible evidence of the new token-based success branch.
        assert any("tokens, retrying" in s for s in statuses), statuses

    def test_context_overflow_token_only_compression_retries_and_succeeds(self, agent):
        agent.client.chat.completions.create.side_effect = [
            _make_context_overflow_error(),
            _mock_response(content="Recovered after context-overflow compression"),
        ]
        statuses = []

        with (
            patch.object(
                agent, "_compress_context",
                side_effect=_same_count_token_shrinking_compress(),
            ) as mock_compress,
            patch.object(agent, "_buffer_status", side_effect=statuses.append),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=_big_prefill())

        mock_compress.assert_called_once()
        assert result.get("failed") is not True
        assert result.get("compression_exhausted") is not True
        assert result["completed"] is True
        assert result["final_response"] == "Recovered after context-overflow compression"
        assert any("tokens, retrying" in s for s in statuses), statuses


class TestTokenOnlyGuards:
    """The two guard rails the author added around the token check."""

    def test_below_threshold_token_reduction_is_not_success(self, agent):
        """A <5% token reduction with unchanged count must NOT be a success.

        Tool returns 413 once; if the loop wrongly treated a tiny reduction as
        progress it would retry and consume the (never-provided) success
        response.  Instead it must abort with a terminal 413 error.
        """
        agent.client.chat.completions.create.side_effect = [_make_413_error()]

        def _barely_shrink(messages, system_message, **_kwargs):
            # Drop ~1 char from the last message only — well under 5%.
            out = [dict(m) for m in messages]
            if out and isinstance(out[-1].get("content"), str) and out[-1]["content"]:
                out[-1]["content"] = out[-1]["content"][:-1]
            return out, "barely changed prompt"

        with (
            patch.object(agent, "_compress_context", side_effect=_barely_shrink),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=_big_prefill())

        assert result["completed"] is False
        assert result.get("partial") is True
        assert "413" in result["error"]
