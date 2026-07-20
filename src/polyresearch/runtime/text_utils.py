"""Dependency-free formatting helpers."""

from datetime import datetime


def get_today_str() -> str:
    """Get the current date formatted for prompts and outputs."""
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"
