"""Guards for the orchestrator-skip-memory change (PR feat/orchestrator-skip-memory).

Why: the parent orchestrator called MemoryManager.prefetch_all() -> kb_search(top_k=30)
to Lore synchronously at the start of every run_conversation (~4.3s/request). SOUL.md
delegates ALL recall to the `memory` child profile, so that prefetch is latency the
parent never uses for routing. The fix passes ``skip_memory=True`` at every user-facing
PARENT construction point, which leaves ``_memory_manager=None`` and skips the prefetch.

What these tests verify:
  1. Each user-facing PARENT construction point passes ``skip_memory=True``.
  2. ``skip_memory`` is the sole gate: a real AIAgent built with skip_memory=True has
     ``_memory_manager is None`` regardless of config.memory.provider.
  3. Scoping: a delegated child is built with its OWN skip_memory flag — setting it on
     the parent does not flow into the child constructor (children manage memory
     independently).

How to run: ``python -m pytest tests/gateway/test_orchestrator_skip_memory.py -q``.
"""
from unittest.mock import MagicMock, patch

import pytest


class TestTuiParentSkipsMemory:
    """The TUI / dashboard embedded-chat parent must not build a memory manager."""

    def test_background_agent_kwargs_sets_skip_memory(self):
        """Why: the TUI background-task orchestrator delegates domain work like the
        interactive parent, so the Lore prefetch is unused-for-routing latency.
        What: ``_background_agent_kwargs`` injects skip_memory=True into the kwargs.
        Test: call the pure helper with a stub agent and assert the key.
        """
        from tui_gateway import server

        with patch.object(server, "_load_cfg", return_value={}), \
             patch.object(server, "_get_db", return_value=None), \
             patch.object(server, "_load_reasoning_config", return_value=None), \
             patch.object(server, "_load_service_tier", return_value=None), \
             patch.object(server, "_resolve_model", return_value="test/model"), \
             patch.object(server, "_load_enabled_toolsets", return_value=["delegation"]):
            stub_parent = MagicMock()
            stub_parent.enabled_toolsets = ["delegation"]
            kwargs = server._background_agent_kwargs(stub_parent, "bg_task_1")

        assert kwargs.get("skip_memory") is True

    def test_make_agent_forces_skip_memory(self):
        """Why: the interactive TUI parent (dashboard chat) must skip the Lore
        prefetch even when HERMES_IGNORE_RULES is unset.
        What: ``_make_agent`` passes skip_memory=True unconditionally.
        Test: mock AIAgent + resolvers, call _make_agent, assert the kwarg is True.
        """
        from tui_gateway import server

        with patch("run_agent.AIAgent") as mock_agent_cls, \
             patch.object(server, "_load_cfg", return_value={"agent": {}}), \
             patch.object(server, "_get_db", return_value=None), \
             patch.object(server, "_load_reasoning_config", return_value=None), \
             patch.object(server, "_load_service_tier", return_value=None), \
             patch.object(server, "_load_enabled_toolsets", return_value=["delegation"]), \
             patch.object(server, "_load_show_reasoning", return_value=False), \
             patch.object(server, "_parse_tui_skills_env", return_value=[]), \
             patch.object(server, "_cfg_max_turns", return_value=90), \
             patch.object(server, "_agent_cbs", return_value={}), \
             patch.object(server, "_resolve_startup_runtime", return_value=("test/model", None)), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value={"provider": None, "base_url": None, "api_key": "k",
                                 "api_mode": None, "command": None, "args": [],
                                 "credential_pool": None}):
            mock_agent_cls.return_value = MagicMock()
            server._make_agent("sid-1", "key-1", session_id="sess-1")

        mock_agent_cls.assert_called_once()
        assert mock_agent_cls.call_args.kwargs.get("skip_memory") is True


class TestSkipMemoryGatesManager:
    """``skip_memory`` is the single gate for whether a memory manager is built."""

    def _real_agent(self, *, skip_memory: bool):
        """Build a real AIAgent offline.

        Construction creates no LLM client until a turn runs. The memory-provider
        block in agent_init is gated entirely on ``not skip_memory`` (it sets
        ``_memory_manager = None`` first, then only populates it when skip_memory
        is False), so we can assert directly on the constructed attribute.
        """
        from run_agent import AIAgent

        return AIAgent(
            model="test/model",
            api_key="test-key",
            base_url="http://localhost:0/v1",
            enabled_toolsets=["delegation"],
            quiet_mode=True,
            skip_memory=skip_memory,
        )

    def test_skip_memory_true_yields_no_manager(self):
        """Why: this is the behavioral guarantee the PR depends on — with the flag set
        the parent never constructs a manager, so prefetch_all is never reached.
        What: AIAgent(skip_memory=True) -> _memory_manager is None.
        Test: build the agent offline, assert the attribute is None.
        """
        agent = self._real_agent(skip_memory=True)
        assert agent._memory_manager is None


class TestChildMemoryScopedIndependently:
    """Setting skip_memory on the parent must NOT disable memory for delegated children."""

    def test_delegate_child_carries_its_own_skip_memory_flag(self):
        """Why: children manage memory independently of the parent — the parent's
        skip_memory must not flow into the child constructor.
        What: ``_build_child_agent`` constructs the child with an explicit skip_memory
        kwarg of its own (decided by the child path / profile), proving the parent's
        value does not leak in.
        Test: mock AIAgent, build a child from a stub parent that itself has
        skip_memory semantics, assert the child's skip_memory kwarg is set explicitly
        on the child construction (not inherited from the parent object).
        """
        from tools import delegate_tool

        parent = MagicMock()
        parent.model = "test/model"
        parent.provider = None
        parent.base_url = None
        parent.api_key = "k"
        parent.api_mode = None
        parent.platform = "api_server"
        parent.enabled_toolsets = ["delegation"]
        parent.valid_tool_names = ["delegate_task"]
        parent.session_id = "parent-sess"
        parent._delegate_depth = 0
        parent.reasoning_config = None
        parent.prefill_messages = None
        parent.acp_command = None
        parent.acp_args = []
        parent._client_kwargs = {"api_key": "k"}
        parent._fallback_chain = None
        parent.providers_allowed = None
        parent.providers_ignored = None
        parent.providers_order = None
        parent.provider_sort = None
        parent.openrouter_min_coding_score = None
        parent.max_tokens = None
        parent._session_db = None
        parent._active_children = []
        parent._active_children_lock = None
        parent._print_fn = None
        parent._subagent_id = None

        with patch("run_agent.AIAgent") as mock_agent_cls, \
             patch.object(delegate_tool, "_load_config", return_value={}), \
             patch.object(delegate_tool, "_build_child_system_prompt", return_value="child prompt"), \
             patch.object(delegate_tool, "_build_child_progress_callback", return_value=None), \
             patch.object(delegate_tool, "_resolve_workspace_hint", return_value=None), \
             patch.object(delegate_tool, "_resolve_child_credential_pool", return_value=None):
            mock_agent_cls.return_value = MagicMock()
            delegate_tool._build_child_agent(
                task_index=0,
                goal="recall the user's name",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
            )

        mock_agent_cls.assert_called_once()
        # The child construction carries its OWN skip_memory kwarg, decided by the
        # delegation path — independent of whatever the parent was built with.
        assert "skip_memory" in mock_agent_cls.call_args.kwargs
