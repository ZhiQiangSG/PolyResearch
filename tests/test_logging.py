"""Logging setup and redaction regression tests."""

import asyncio
import logging

from polyresearch import cli
from polyresearch.retrieval import mcp_utils


def test_configure_logging_uses_requested_level(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.update(kwargs))

    cli.configure_logging("debug")

    assert captured["level"] == logging.DEBUG


def test_bailian_loader_logs_failure_without_authorization_secret(caplog, monkeypatch) -> None:
    secret = "very-secret-bailian-token"

    class FailingMcpClient:
        def __init__(self, config):
            self.config = config

        async def get_tools(self):
            raise RuntimeError(f"Authorization: Bearer {secret}")

    monkeypatch.setattr(mcp_utils, "MultiServerMCPClient", FailingMcpClient)
    with caplog.at_level(logging.WARNING, logger="polyresearch.retrieval.mcp_utils"):
        tools = asyncio.run(
            mcp_utils.load_bailian_web_search_tool(
                {
                    "configurable": {
                        "bailian_web_search": {
                            "authentication": {"api_key": secret},
                        }
                    }
                },
                existing_tool_names=set(),
            )
        )

    assert tools == []
    assert "Bailian MCP tool loading failed" in caplog.text
    assert secret not in caplog.text
    assert "[REDACTED_SECRET]" in caplog.text
