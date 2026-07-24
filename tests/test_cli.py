"""Tests for CLI — command imports and basic smoke tests."""

from __future__ import annotations

import pytest


class TestCLI:
    """Smoke tests for CLI commands (no API calls)."""

    def test_cli_module_imports(self):
        """Verify CLI module can be imported."""
        from knowledge_agent.cli import cli, main
        assert callable(cli)
        assert callable(main)

    def test_cli_has_commands(self):
        """Verify all expected CLI commands exist."""
        from knowledge_agent.cli import cli
        commands = {cmd.name for cmd in cli.commands.values()}
        assert "ingest" in commands
        assert "query" in commands
        assert "serve" in commands
        assert "webui" in commands

    def test_ingest_command_has_required_params(self):
        """Verify ingest command has path argument."""
        from knowledge_agent.cli import cli
        cmd = cli.commands["ingest"]
        params = [p.name for p in cmd.params]
        assert "path" in params
        assert "chunk_size" in params
        assert "chunk_overlap" in params

    def test_query_command_has_required_params(self):
        """Verify query command has question argument."""
        from knowledge_agent.cli import cli
        cmd = cli.commands["query"]
        params = [p.name for p in cmd.params]
        assert "question" in params
        assert "top_k" in params

    def test_serve_command_has_options(self):
        """Verify serve command has host/port options."""
        from knowledge_agent.cli import cli
        cmd = cli.commands["serve"]
        param_names = {p.name for p in cmd.params}
        assert "host" in param_names
        assert "port" in param_names

    def test_webui_command_has_options(self):
        """Verify webui command has host/port/share options."""
        from knowledge_agent.cli import cli
        cmd = cli.commands["webui"]
        param_names = {p.name for p in cmd.params}
        assert "host" in param_names
        assert "port" in param_names
        assert "share" in param_names
