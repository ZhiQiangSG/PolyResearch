import importlib
import unittest

from langchain_core.messages import AIMessage

supervisor_module = importlib.import_module("polyresearch.workflows.supervisor")
researcher_module = importlib.import_module("polyresearch.workflows.researcher")


class SupervisorLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_researcher_tool_returns_a_recoverable_tool_message(self) -> None:
        original_get_all_tools = researcher_module.get_all_tools
        researcher_module.get_all_tools = lambda config: _empty_tool_list()
        try:
            command = await researcher_module.researcher_tools(
                {
                    "researcher_messages": [
                        AIMessage(
                            content="Call an unavailable tool.",
                            tool_calls=[
                                {
                                    "name": "unknown_tool",
                                    "args": {},
                                    "id": "call-unknown",
                                }
                            ],
                        )
                    ],
                    "tool_call_iterations": 0,
                },
                {"configurable": {}},
            )

            self.assertEqual(command.goto, "researcher")
            self.assertEqual(len(command.update["researcher_messages"]), 1)
            self.assertIn(
                "'unknown_tool' is unavailable",
                command.update["researcher_messages"][0].content,
            )
        finally:
            researcher_module.get_all_tools = original_get_all_tools

    async def test_stops_when_iteration_count_reaches_configured_limit(self) -> None:
        command = await supervisor_module.supervisor_tools(
            {
                "supervisor_messages": [
                    AIMessage(
                        content="Planning another task.",
                        tool_calls=[
                            {
                                "name": "think_tool",
                                "args": {"reflection": "Continue."},
                                "id": "call-1",
                            }
                        ],
                    )
                ],
                "research_iterations": 3,
            },
            {"configurable": {"max_researcher_iterations": 3}},
        )

        self.assertEqual(command.goto, "__end__")

    async def test_parallel_researchers_receive_distinct_research_unit_ids(self) -> None:
        class FakeResearcherSubgraph:
            def __init__(self) -> None:
                self.unit_ids = []
                self.tasks = []

            async def ainvoke(self, state, config):
                self.unit_ids.append(config["configurable"]["research_unit_id"])
                self.tasks.append(state["evidence_task"])
                return {
                    "sources": [],
                    "passages": [],
                    "claims": [],
                    "verification_results": [],
                }

        fake_subgraph = FakeResearcherSubgraph()
        original_subgraph = supervisor_module.researcher_subgraph
        supervisor_module.researcher_subgraph = fake_subgraph
        try:
            await supervisor_module.supervisor_tools(
                {
                    "supervisor_messages": [
                        AIMessage(
                            content="Delegate work.",
                            tool_calls=[
                                {
                                    "name": "ConductResearch",
                                    "args": {"task": {"subquestion": "Topic A", "language": "en", "target_source_type": "official", "evidence_goal": "Find a citable official statement.", "query_rationale": "Primary evidence."}},
                                    "id": "call-a",
                                },
                                {
                                    "name": "ConductResearch",
                                    "args": {"task": {"subquestion": "Topic B", "language": "en", "target_source_type": "official", "evidence_goal": "Find a citable official statement.", "query_rationale": "Primary evidence."}},
                                    "id": "call-b",
                                },
                            ],
                        )
                    ],
                    "research_iterations": 1,
                },
                {"configurable": {"max_concurrent_research_units": 2}},
            )

            self.assertEqual(len(fake_subgraph.unit_ids), 2)
            self.assertEqual(len(set(fake_subgraph.unit_ids)), 2)
            self.assertTrue(all(task.target_source_type == "official" for task in fake_subgraph.tasks))
        finally:
            supervisor_module.researcher_subgraph = original_subgraph


async def _empty_tool_list():
    return []
