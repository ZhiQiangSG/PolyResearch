"""Runtime context shared by graph nodes during one durable research run."""

from dataclasses import dataclass
from uuid import UUID

from langchain_core.runnables import RunnableConfig

from polyresearch.repositories.base import EvidenceRepository


@dataclass(frozen=True)
class RunContext:
    """The run identity and repository injected into LangGraph configuration."""

    run_id: UUID
    repository: EvidenceRepository
    research_unit_id: UUID | None = None

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "RunContext":
        """Load the explicitly injected run context or fail with a clear error."""
        configurable = config.get("configurable", {})
        run_id = configurable.get("run_id")
        repository = configurable.get("evidence_repository")
        research_unit_id = configurable.get("research_unit_id")
        if run_id is None or repository is None:
            raise ValueError(
                "LangGraph requires configurable.run_id and "
                "configurable.evidence_repository for durable research runs."
            )
        return cls(
            run_id=UUID(str(run_id)),
            repository=repository,
            research_unit_id=(UUID(str(research_unit_id)) if research_unit_id else None),
        )
