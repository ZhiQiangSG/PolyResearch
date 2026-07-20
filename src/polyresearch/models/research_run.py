"""Typed metadata for a durable research run."""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ResearchRun(BaseModel):
    """The durable identity and lifecycle metadata of a research run."""

    id: UUID = Field(default_factory=uuid4)
    question: str = Field(min_length=1)
    output_language: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["active", "completed", "failed"] = "active"
