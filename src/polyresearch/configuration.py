import os
import re
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, field_validator

DEFAULT_QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL_ID_PATTERN = re.compile(r"qwen[a-z0-9._-]*", re.IGNORECASE)


class ModelProvider(str, Enum):
    """Model transport supported by this application."""

    QWEN_OPENAI_COMPATIBLE = "qwen_openai_compatible"


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """Deprecated legacy MCP configuration; not exposed to research agents."""
    
    # The URL of the MCP server
    url: Optional[str] = Field(default=None)
    # The tools to make available to the LLM
    tools: Optional[list[str]] = Field(default=None)
    # Whether the MCP server requires authentication
    auth_required: Optional[bool] = Field(default=False)


BAILIAN_WEB_SEARCH_MCP_URL = (
    "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
)


class BailianWebSearchConfig(BaseModel):
    """Narrow configuration for Bailian's Chinese web-search MCP service only."""

    server_url: str = Field(default=BAILIAN_WEB_SEARCH_MCP_URL)
    tool_name: str = Field(default="web_search")
    api_key: str | None = Field(default=None, repr=False)
    locale: str = Field(default="zh-CN")
    query_language: str = Field(default="zh")
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_requests_per_second: float = Field(default=15, gt=0, le=15)

    @field_validator("server_url")
    @classmethod
    def validate_server_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc or not parsed.path.endswith("/mcp"):
            raise ValueError("Bailian MCP server_url must be an absolute HTTPS /mcp URL")
        return value

    @field_validator("tool_name")
    @classmethod
    def validate_allowlisted_tool(cls, value: str) -> str:
        if value != "web_search":
            raise ValueError("Bailian Web Search may only load the 'web_search' MCP tool")
        return value

    @field_validator("locale")
    @classmethod
    def validate_chinese_locale(cls, value: str) -> str:
        if not value.lower().startswith("zh"):
            raise ValueError("Bailian Web Search locale must be Chinese")
        return value

    @field_validator("query_language")
    @classmethod
    def validate_chinese_query_language(cls, value: str) -> str:
        if value.lower() not in {"zh", "zh-cn", "zh-hans"}:
            raise ValueError("Bailian Web Search query_language must be Chinese")
        return value

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(default=3)
    allow_clarification: bool = Field(default=True)
    max_concurrent_research_units: int = Field(default=5)

    # Research Configuration
    search_api: SearchAPI = Field(default=SearchAPI.TAVILY)
    max_researcher_iterations: int = Field(default=3)
    max_react_tool_calls: int = Field(default=5)

    # Model Configuration
    model_provider: ModelProvider = Field(default=ModelProvider.QWEN_OPENAI_COMPATIBLE)
    qwen_base_url: str = Field(default=DEFAULT_QWEN_BASE_URL)
    research_model: str = Field(default="qwen3.7-max")
    research_model_max_tokens: int = Field(default=10000)
    compression_model: str = Field(default="qwen3.7-plus")
    compression_model_max_tokens: int = Field(default=8192)
    final_report_model: str = Field(default="qwen3.7-plus")
    final_report_model_max_tokens: int = Field(default=10000)

    # Bailian is the only MCP integration exposed during Milestone 3.
    bailian_web_search: Optional[BailianWebSearchConfig] = Field(default=None)

    # Deprecated legacy MCP configuration retained for config-file compatibility.
    # It is intentionally not loaded by ``get_all_tools``.
    mcp_config: Optional[MCPConfig] = Field(default=None)
    mcp_prompt: Optional[str] = Field(default=None)

    @field_validator("qwen_base_url")
    @classmethod
    def validate_qwen_base_url(cls, value: str) -> str:
        """Require an HTTPS Model Studio OpenAI-compatible API base URL."""
        normalized = value.rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("qwen_base_url must be an absolute HTTPS URL")
        if not parsed.path.endswith("/compatible-mode/v1"):
            raise ValueError(
                "qwen_base_url must end with '/compatible-mode/v1' for the "
                "Model Studio OpenAI-compatible API"
            )
        return normalized

    @field_validator(
        "research_model",
        "compression_model",
        "final_report_model",
    )
    @classmethod
    def validate_qwen_model_id(cls, value: str) -> str:
        """Reject provider prefixes, whitespace, and non-Qwen model identifiers."""
        if value != value.strip() or not QWEN_MODEL_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "Qwen model IDs must be bare identifiers beginning with 'qwen' "
                "(for example, 'qwen3.7-plus'); do not include a provider prefix"
            )
        return value

    def chat_model_config(self, model: str, max_tokens: int, api_key: Optional[str]) -> dict[str, Any]:
        """Return the complete LangChain configuration for a Qwen chat model."""
        return {
            "model": model,
            # This selects the OpenAI-compatible transport, not an OpenAI model.
            "model_provider": "openai",
            "base_url": self.qwen_base_url,
            "max_tokens": max_tokens,
            "api_key": api_key,
            "tags": ["langsmith:nostream"],
        }


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True
