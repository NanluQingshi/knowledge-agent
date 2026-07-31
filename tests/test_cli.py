"""Tests for CLI — command imports and basic smoke tests."""

from __future__ import annotations


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
        assert commands >= {
            "delete",
            "eval",
            "eval-dataset",
            "ingest",
            "query",
            "serve",
            "webui",
        }

    def test_ingest_command_has_required_params(self):
        """Verify ingest command has path argument."""
        from knowledge_agent.cli import cli

        cmd = cli.commands["ingest"]
        params = [p.name for p in cmd.params]
        assert "path" in params
        assert "chunk_size" in params
        assert "chunk_overlap" in params
        assert "extract" in params
        assert "quality" in params

    def test_query_command_has_required_params(self):
        """Verify query command has question argument."""
        from knowledge_agent.cli import cli

        cmd = cli.commands["query"]
        params = [p.name for p in cmd.params]
        assert "question" in params
        assert "top_k" in params
        assert "graphrag" in params

    def test_delete_command_has_document_id(self):
        """Verify delete was registered as a top-level command."""
        from knowledge_agent.cli import cli

        cmd = cli.commands["delete"]
        assert [p.name for p in cmd.params] == ["doc_id"]

    def test_eval_command_has_expected_params(self):
        """Verify the evaluation command exposes its mode and filters."""
        from knowledge_agent.cli import cli

        cmd = cli.commands["eval"]
        assert {p.name for p in cmd.params} == {
            "mode",
            "top_k",
            "category",
            "dataset",
        }

    def test_eval_dataset_has_expected_subcommands(self):
        """Verify all evaluation dataset operations are registered."""
        from knowledge_agent.cli import cli

        group = cli.commands["eval-dataset"]
        assert set(group.commands) == {"add", "list", "clear", "export"}

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
