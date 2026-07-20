"""Security helpers for untrusted retrieval data and runtime policy enforcement."""

import re
from urllib.parse import urlparse

_SECRET = re.compile(r"(?i)\b(?:api[_-]?key|authorization|bearer|token|password)\b\s*[:=]\s*[^\s,;]+")
_PROMPT_INJECTION = re.compile(r"(?i)\b(?:ignore (?:previous|all) instructions|system prompt|developer message|tool call)\b")


def redact_secrets(value: str) -> str:
    """Keep credentials out of durable logs, attachments, and exports."""
    return _SECRET.sub("[REDACTED_SECRET]", value)


def is_allowed_domain(url: str, *, allowed: list[str], blocked: list[str]) -> bool:
    host = (urlparse(url).hostname or "").casefold()
    allowed_domains = [item.casefold().lstrip(".") for item in allowed]
    blocked_domains = [item.casefold().lstrip(".") for item in blocked]
    matches = lambda domain: host == domain or host.endswith("." + domain)
    return bool(host) and not any(matches(domain) for domain in blocked_domains) and (
        not allowed_domains or any(matches(domain) for domain in allowed_domains)
    )


def redact_prompt_injection(text: str) -> str:
    """Label instruction-like source text before it is placed in an LLM context."""
    return _PROMPT_INJECTION.sub("[UNTRUSTED_INSTRUCTION_REDACTED]", text)
