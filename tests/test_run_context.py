import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage

from polyresearch.graph import initialize_research_run
from polyresearch.repositories import SqliteEvidenceRepository


class RunContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_start_creates_the_configured_durable_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run_id = uuid4()
            try:
                command = await initialize_research_run(
                    {"messages": [HumanMessage(content="Research this topic.")]},
                    {
                        "configurable": {
                            "run_id": str(run_id),
                            "evidence_repository": repository,
                            "output_language": "en",
                        }
                    },
                )

                run = await repository.get_run(run_id)
                self.assertEqual(command.goto, "clarify_with_user")
                self.assertEqual(command.update["run_id"], run_id)
                self.assertEqual(run.question, "Human: Research this topic.")
                self.assertEqual(run.output_language, "en")
            finally:
                repository.close()
