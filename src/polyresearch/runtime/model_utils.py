"""Qwen model configuration, credentials, and context-limit helpers."""

import os
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from polyresearch.configuration import Configuration

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded.
    
    Args:
        exception: The exception to analyze
        model_name: Optional model name to optimize provider detection
        
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
        )
    )

def get_tavily_api_key(config: RunnableConfig):
    """Get Tavily API key from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        return api_keys.get("TAVILY_API_KEY") if api_keys else None
    return os.getenv("TAVILY_API_KEY")
