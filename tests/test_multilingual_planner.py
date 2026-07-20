import importlib
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from polyresearch.models import (
    AtomicSubquestion,
    LanguageSelectionAssessment,
    ResearchEntity,
    ResearchLanguage,
    ResearchPlan,
    ResearchRun,
)
from polyresearch.repositories import SqliteEvidenceRepository

graph_module = importlib.import_module("polyresearch.graph")


def _selection_assessment() -> LanguageSelectionAssessment:
    return LanguageSelectionAssessment(
        place_and_institutional_jurisdiction="The policy was issued in China.",
        primary_actors_and_official_records="The issuing authority publishes in Chinese.",
        scholarly_technical_and_media_ecosystems="Chinese policy analysis supplies local context.",
        diasporic_or_regional_coverage="Not applicable: domestic policy is the focus.",
        primary_source_availability="Official Chinese records are expected to be available.",
        marginal_information_gain="Adds primary records unavailable in English coverage.",
    )


class _PlannerStub:
    def __init__(self, plan: ResearchPlan) -> None:
        self.plan = plan
        self.messages = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def with_retry(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return self.plan


class MultilingualPlannerTests(unittest.IsolatedAsyncioTestCase):
    def test_plan_requires_atomic_subquestions_and_queries_for_ranked_languages(self):
        with self.assertRaises(ValueError):
            ResearchPlan(
                run_id=uuid4(),
                subquestions=[
                    AtomicSubquestion(
                        question="What changed?", answer_scope="Identify the change."
                    )
                ],
                ranked_languages=[
                    ResearchLanguage(
                        language="zh",
                        priority=1,
                        query_budget=1,
                        expected_unique_value="Official records.",
                        selection_rationale="The policy was issued in China.",
                        selection_assessment=_selection_assessment(),
                    )
                ],
                language_rationale={"zh": "Primary records."},
            )

        with self.assertRaises(ValueError):
            AtomicSubquestion(question="What changed?", answer_scope="Scope", extra="no")

    async def test_planner_persists_plan_and_initializes_supervisor_with_it(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="How did a policy change?", output_language="en")
            model_plan = ResearchPlan(
                run_id=uuid4(),
                subquestions=[
                    AtomicSubquestion(
                        question="What changed?",
                        answer_scope="Identify the policy changes and effective dates.",
                    )
                ],
                entities=[ResearchEntity(canonical_name="Policy X", native_script_variants=["政策X"])],
                ranked_languages=[
                    ResearchLanguage(
                        language="zh",
                        priority=1,
                        query_budget=3,
                        expected_unique_value="Official Chinese records.",
                        selection_rationale="The policy was issued in China.",
                        selection_assessment=_selection_assessment(),
                        expected_source_types=["official"],
                    )
                ],
                language_rationale={"zh": "Selected for primary records."},
                query_variants={"zh": ["政策X 变更"]},
                anticipated_conflict_dimensions=["publication date"],
            )
            stub = _PlannerStub(model_plan)
            original_factory = graph_module.create_qwen_chat_model
            graph_module.create_qwen_chat_model = lambda *args, **kwargs: stub
            try:
                await repository.create_run(run)
                result = await graph_module.multilingual_planner(
                    {"research_brief": run.question},
                    {"configurable": {"run_id": str(run.id), "evidence_repository": repository}},
                )
                plan = (await repository.list_research_plans(run.id))[0]
                self.assertEqual(plan.run_id, run.id)
                self.assertEqual(plan.model_id, "qwen3.7-max")
                self.assertEqual(plan.prompt_version, "multilingual_planner_v1")
                self.assertEqual(result.goto, "research_supervisor")
                self.assertEqual(result.update["research_plan"], plan)
                self.assertIn("政策X", result.update["supervisor_messages"]["value"][1].content)
                self.assertIn("ResearchBrief", stub.messages[0].content)
                self.assertIn("marginal information gain", stub.messages[0].content)
                self.assertIn("do not use a fixed default language list", stub.messages[0].content)
            finally:
                graph_module.create_qwen_chat_model = original_factory
                repository.close()
