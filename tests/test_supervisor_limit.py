import importlib
import unittest

from langchain_core.messages import AIMessage

graph_module = importlib.import_module("polyresearch.graph")


class SupervisorLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_stops_when_iteration_count_reaches_configured_limit(self) -> None:
        command = await graph_module.supervisor_tools(
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

            async def ainvoke(self, state, config):
                self.unit_ids.append(config["configurable"]["research_unit_id"])
                return {
                    "sources": [],
                    "passages": [],
                    "claims": [],
                    "verification_results": [],
                }

        fake_subgraph = FakeResearcherSubgraph()
        original_subgraph = graph_module.researcher_subgraph
        graph_module.researcher_subgraph = fake_subgraph
        try:
            await graph_module.supervisor_tools(
                {
                    "supervisor_messages": [
                        AIMessage(
                            content="Delegate work.",
                            tool_calls=[
                                {
                                    "name": "ConductResearch",
                                    "args": {"research_topic": "Topic A"},
                                    "id": "call-a",
                                },
                                {
                                    "name": "ConductResearch",
                                    "args": {"research_topic": "Topic B"},
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
        finally:
            graph_module.researcher_subgraph = original_subgraph
