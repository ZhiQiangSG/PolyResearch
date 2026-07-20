"""Tests for default Bailian activation across runtime entry points."""

import os
from unittest.mock import patch

from polyresearch.configuration import BailianWebSearchConfig, Configuration


def test_configuration_omits_inert_provider_and_bailian_language_fields() -> None:
    assert "model_provider" not in Configuration.model_fields
    assert "locale" not in BailianWebSearchConfig.model_fields
    assert "query_language" not in BailianWebSearchConfig.model_fields


def test_dashscope_key_enables_default_bailian_configuration() -> None:
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
        configuration = Configuration.from_runnable_config()

    assert configuration.bailian_web_search is not None
    assert configuration.bailian_web_search.authentication.api_key_env_var == "DASHSCOPE_API_KEY"


def test_no_dashscope_key_leaves_bailian_disabled() -> None:
    with patch.dict(os.environ, {}, clear=True):
        configuration = Configuration.from_runnable_config()

    assert configuration.bailian_web_search is None


def test_explicit_bailian_none_disables_environment_default() -> None:
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
        configuration = Configuration.from_runnable_config(
            {"configurable": {"bailian_web_search": None}}
        )

    assert configuration.bailian_web_search is None


def test_explicit_bailian_configuration_overrides_environment_default() -> None:
    override = {
        "server_url": "https://bailian.example.test/mcp",
        "timeout_seconds": 12,
        "max_requests_per_second": 3,
        "authentication": {"api_key": "override-key"},
    }
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "environment-key"}, clear=True):
        configuration = Configuration.from_runnable_config(
            {"configurable": {"bailian_web_search": override}}
        )

    assert configuration.bailian_web_search is not None
    assert configuration.bailian_web_search.server_url == override["server_url"]
    assert configuration.bailian_web_search.timeout_seconds == 12
    assert configuration.bailian_web_search.max_requests_per_second == 3
    assert configuration.bailian_web_search.authentication.api_key == "override-key"
