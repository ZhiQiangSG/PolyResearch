import importlib
import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from polyresearch.models import (
    AtomicSubquestion,
    LanguageSelectionAssessment,
    LanguageExpansionDecision,
    LanguageDecision,
    ResearchEntity,
    ResearchLanguage,
    ResearchPlan,
    ResearchRun,
    TerminologyRecord,
)
from polyresearch.repositories import SqliteEvidenceRepository

graph_module = importlib.import_module("polyresearch.workflows.orchestrator")


def _selection_assessment() -> LanguageSelectionAssessment:
    return LanguageSelectionAssessment(
        place_and_institutional_jurisdiction="The policy was issued in China.",
        primary_actors_and_official_records="The issuing authority publishes in Chinese.",
        scholarly_technical_and_media_ecosystems="Chinese policy analysis supplies local context.",
        diasporic_or_regional_coverage="Not applicable: domestic policy is the focus.",
        primary_source_availability="Official Chinese records are expected to be available.",
        marginal_information_gain="Adds primary records unavailable in English coverage.",
    )


def _fixture_selection_assessment(language: str) -> LanguageSelectionAssessment:
    return LanguageSelectionAssessment(
        place_and_institutional_jurisdiction=f"Fixture jurisdiction supports {language}.",
        primary_actors_and_official_records=f"Relevant records are available in {language}.",
        scholarly_technical_and_media_ecosystems=f"The {language} ecosystem adds context.",
        diasporic_or_regional_coverage="Not applicable for this fixture.",
        primary_source_availability=f"Primary sources are expected in {language}.",
        marginal_information_gain=f"{language} adds material evidence beyond earlier languages.",
    )


def _load_multilingual_fixtures() -> list[dict]:
    fixture_path = Path(__file__).parent / "fixtures" / "multilingual_planner_cases.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


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
    def test_multilingual_planner_fixtures_preserve_aliases_scripts_and_language_value(self):
        for fixture in _load_multilingual_fixtures():
            with self.subTest(fixture=fixture["id"]):
                languages = fixture["languages"]
                plan = ResearchPlan(
                    run_id=uuid4(),
                    subquestions=[
                        AtomicSubquestion(
                            question="What evidence answers the fixture question?",
                            answer_scope="Find source-backed evidence for the fixture entity.",
                        )
                    ],
                    entities=[ResearchEntity.model_validate(fixture["entity"])],
                    terminology=[
                        TerminologyRecord.model_validate(term)
                        for term in fixture["terminology"]
                    ],
                    ranked_languages=[
                        ResearchLanguage(
                            language=language["language"],
                            priority=language["priority"],
                            query_budget=language["query_budget"],
                            expected_unique_value=language["expected_unique_value"],
                            selection_rationale=language["expected_unique_value"],
                            selection_assessment=_fixture_selection_assessment(
                                language["language"]
                            ),
                            expected_source_types=language["expected_source_types"],
                        )
                        for language in languages
                    ],
                    language_decisions=[
                        LanguageDecision(
                            language=language["language"],
                            status="selected",
                            rationale=language["expected_unique_value"],
                        )
                        for language in languages
                    ],
                    language_rationale={
                        language["language"]: language["expected_unique_value"]
                        for language in languages
                    },
                    query_variants={
                        language["language"]: language["queries"] for language in languages
                    },
                )
                expected = fixture["expected"]
                entity = plan.entities[0]
                self.assertTrue(
                    set(expected.get("aliases", [])).issubset(entity.aliases)
                )
                self.assertTrue(
                    set(expected.get("native_scripts", [])).issubset(
                        entity.native_script_variants
                    )
                )
                if "disambiguation_contains" in expected:
                    self.assertIn(expected["disambiguation_contains"], entity.disambiguation)
                if "first_language" in expected:
                    self.assertEqual(plan.ranked_languages[0].language, expected["first_language"])
                if "not_first_language" in expected:
                    self.assertNotEqual(
                        plan.ranked_languages[0].language,
                        expected["not_first_language"],
                    )
                self.assertEqual(
                    [language.priority for language in plan.ranked_languages],
                    sorted(language.priority for language in plan.ranked_languages),
                )
                for language in plan.ranked_languages:
                    self.assertTrue(plan.query_variants[language.language])

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

        with self.assertRaises(ValueError):
            TerminologyRecord(
                original_term="政策",
                original_language="zh",
                normalized_term="policy",
                translated_term="policy",
                translation_equivalence="approximate",
            )

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
                terminology=[
                    TerminologyRecord(
                        original_term="政策X",
                        original_language="zh",
                        normalized_term="Policy X",
                        translated_term="Policy X",
                        translation_equivalence="approximate",
                        translation_note="The Chinese term has a broader administrative scope.",
                    )
                ],
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
                language_decisions=[
                    LanguageDecision(
                        language="zh",
                        status="selected",
                        rationale="Chinese primary records are in scope.",
                    ),
                    LanguageDecision(
                        language="en",
                        status="skipped",
                        rationale="English coverage is not expected to add primary records.",
                    ),
                ],
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
                self.assertIn("translation_note", stub.messages[0].content)
            finally:
                graph_module.create_qwen_chat_model = original_factory
                repository.close()

    async def test_gap_review_records_a_no_expansion_decision_after_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="How did a policy change?", output_language="en")
            plan = ResearchPlan(
                run_id=run.id,
                subquestions=[
                    AtomicSubquestion(
                        question="What changed?", answer_scope="Identify the policy change."
                    )
                ],
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
                language_decisions=[
                    LanguageDecision(
                        language="zh",
                        status="selected",
                        rationale="Chinese primary records are in scope.",
                    )
                ],
                query_variants={"zh": ["政策X 变更"]},
            )
            decision = LanguageExpansionDecision(
                should_add_languages=False,
                rationale="No retrieved gap justifies another language yet.",
                considered_but_skipped=[
                    LanguageDecision(
                        language="ko",
                        status="skipped",
                        rationale="The available gaps are not specific to Korean sources.",
                    )
                ],
            )
            stub = _PlannerStub(decision)
            original_factory = graph_module.create_qwen_chat_model
            graph_module.create_qwen_chat_model = lambda *args, **kwargs: stub
            try:
                await repository.create_run(run)
                await repository.append_research_plans(run.id, [plan])
                result = await graph_module.language_gap_analysis(
                    {"research_brief": run.question, "research_plan": plan},
                    {"configurable": {"run_id": str(run.id), "evidence_repository": repository}},
                )
                plans = await repository.list_research_plans(run.id)
                self.assertEqual(result.goto, "final_report_generation")
                self.assertTrue(result.update["language_gap_reviewed"])
                self.assertEqual(len(plans), 2)
                self.assertFalse(plans[-1].post_retrieval_decision.should_add_languages)
                self.assertIn(
                    LanguageDecision(
                        language="ko",
                        status="skipped",
                        rationale="The available gaps are not specific to Korean sources.",
                    ),
                    plans[-1].language_decisions,
                )
                self.assertIn("InitialRetrievalLedger", stub.messages[0].content)
            finally:
                graph_module.create_qwen_chat_model = original_factory
                repository.close()
