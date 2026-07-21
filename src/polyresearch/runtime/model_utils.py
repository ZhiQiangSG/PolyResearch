"""Qwen model configuration, credentials, and context-limit helpers."""

import logging
import os
from time import monotonic
from typing import Any, Optional

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from polyresearch.configuration import Configuration
from polyresearch.security import redacted_exception_info

logger = logging.getLogger(__name__)


class QwenInvocationLoggingCallback(BaseCallbackHandler):
    """Log Qwen invocation lifecycle data without recording prompt content."""

    def __init__(self, model: str, max_tokens: int) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._started_at: dict[str, float] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, tags=None, **kwargs) -> None:
        self._started_at[str(run_id)] = monotonic()
        logger.info(
            "Qwen model invocation started",
            extra=self._log_context(run_id, tags),
        )

    def on_llm_end(self, response, *, run_id, tags=None, **kwargs) -> None:
        request_id = _provider_request_id(response)
        logger.info(
            "Qwen model invocation completed",
            extra={**self._log_context(run_id, tags), "provider_request_id": request_id},
        )

    def on_llm_error(self, error, *, run_id, tags=None, **kwargs) -> None:
        logger.warning(
            "Qwen model invocation failed",
            extra={
                **self._log_context(run_id, tags),
                "error_type": type(error).__name__,
                "status_code": getattr(error, "status_code", None),
            },
            exc_info=redacted_exception_info(error),
        )

    def _log_context(self, run_id: Any, tags: list[str] | None) -> dict[str, Any]:
        key = str(run_id)
        started_at = self._started_at.pop(key, None)
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "model_run_id": key,
            "attempt": _retry_attempt(tags),
            "elapsed_seconds": round(monotonic() - started_at, 3) if started_at else None,
        }


def _retry_attempt(tags: list[str] | None) -> int:
    for tag in tags or []:
        if tag.startswith("retry:attempt:"):
            return int(tag.rsplit(":", maxsplit=1)[1])
    return 1


def _provider_request_id(response: Any) -> str | None:
    for generations in getattr(response, "generations", []):
        for generation in generations:
            response_metadata = getattr(generation.message, "response_metadata", {})
            if isinstance(response_metadata, dict) and response_metadata.get("request_id"):
                return str(response_metadata["request_id"])
    return None

def is_token_limit_exceeded(exception: Exception) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded.
    
    Args:
        exception: The exception to analyze
    Returns:
        True if the exception indicates a token limit was exceeded, False otherwise
    """
    error_str = str(exception).lower()
    return _check_qwen_token_limit(exception, error_str)

def _check_qwen_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Qwen token limit exceeded."""
    # Check for Qwen-specific token limit patterns
    token_indicators = [
        'token', 'context', 'length', 'maximum context', 
        'reduce', 'exceed', 'limit', 'too long'
    ]
    
    # Check error message for token-related keywords
    if any(indicator in error_str.lower() for indicator in token_indicators):
        return True
    
    # Check exception attributes for Qwen-specific patterns
    if hasattr(exception, 'code'):
        error_code = getattr(exception, 'code', '').lower()
        if 'context_length' in error_code or 'token' in error_code:
            return True
    
    if hasattr(exception, 'type'):
        error_type = getattr(exception, 'type', '').lower()
        if 'token' in error_type or 'context' in error_type:
            return True
    
    return False


MODEL_TOKEN_LIMITS = {
    "qwen3.7-max": 100000,
    "qwen3.7-plus": 100000,
    "qwen3.6-plus": 100000
}

def get_model_token_limit(model_string) -> int:
    """Look up the token limit for a specific model.
    
    Args:
        model_string: The model identifier string to look up
        
    Returns:
        Token limit as integer if found, None if model not in lookup table
    """
    # Search through known model token limits
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    
    # Model not found in lookup table
    return None

def get_qwen_api_key(config: RunnableConfig) -> Optional[str]:
    """Get the Model Studio API key from runtime configuration or the environment."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        return api_keys.get("QWEN_API_KEY") if api_keys else None
    return os.getenv("QWEN_API_KEY")


def create_qwen_chat_model(
    configuration: Configuration,
    model: str,
    max_tokens: int,
    config: RunnableConfig,
) -> BaseChatModel:
    """Create a Qwen model using the configured DashScope-compatible transport."""
    api_key = get_qwen_api_key(config)
    if not api_key:
        raise ValueError(
            "Qwen credentials are missing. Set QWEN_API_KEY or provide it through "
            "configurable.apiKeys when GET_API_KEYS_FROM_CONFIG=true."
        )
    return init_chat_model(
        **configuration.chat_model_config(
            model=model,
            max_tokens=max_tokens,
            api_key=api_key,
        ),
        callbacks=[QwenInvocationLoggingCallback(model, max_tokens)],
    )

def get_tavily_api_key(config: RunnableConfig):
    """Get Tavily API key from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        return api_keys.get("TAVILY_API_KEY") if api_keys else None
    return os.getenv("TAVILY_API_KEY")
