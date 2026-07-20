import sqlite3
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from polyresearch.models import (
    Claim,
    ClaimExtractionDraft,
    ClaimScope,
    EvidencePassage,
    SourceRecord,
    TranslationRecord,
)
from polyresearch.models.graph_state import merge_evidence_by_id, override_reducer
from polyresearch.repositories import SqliteEvidenceRepository


class EvidenceContractTests(unittest.TestCase):
    def test_schema_validation_rejects_uncitable_passages_and_claims(self) -> None:
        source = SourceRecord(canonical_url="https://example.test", title="Example")

        with self.assertRaises(ValidationError):
            EvidencePassage(source_id=source.id, text="", locator="paragraph 1")
        with self.assertRaises(ValidationError):
            Claim(
                statement="A claim without evidence.",
                evidence_passage_ids=[],
                extraction_confidence=0.8,
            )
        with self.assertRaises(ValidationError):
            Claim(
                statement="An overconfident claim.",
                evidence_passage_ids=[source.id],
                extraction_confidence=1.1,
            )

    def test_ids_are_uuids_and_survive_model_round_trip(self) -> None:
        source = SourceRecord(canonical_url="https://example.test", title="Example")
        restored = SourceRecord.model_validate_json(source.model_dump_json())

        self.assertIsInstance(source.id, UUID)
        self.assertEqual(restored.id, source.id)
        self.assertEqual(restored, source)

    def test_original_passages_are_immutable_and_translations_reference_their_hash(self) -> None:
        source = SourceRecord(canonical_url="https://example.test", title="Example")
        passage = EvidencePassage(
            source_id=source.id,
            text="原始证据。",
            locator="paragraph-1",
            original_language="zh",
        )
        translation = TranslationRecord(
            passage_id=passage.id,
            translated_text="Original evidence.",
            target_language="en",
            source_original_text_hash=passage.original_text_hash,
        )

        self.assertEqual(len(passage.original_text_hash), 64)
        self.assertEqual(translation.source_original_text_hash, passage.original_text_hash)
        with self.assertRaises(ValidationError):
            passage.text = "Altered evidence."

    def test_claim_extraction_draft_requires_atomic_scope_modality_and_passage_reference(self) -> None:
        source = SourceRecord(canonical_url="https://example.test", title="Example")
        passage = EvidencePassage(
            source_id=source.id, text="The policy changed.", locator="paragraph-1"
        )
        with self.assertRaises(ValidationError):
            ClaimExtractionDraft(
                atomic_proposition="The policy changed.",
                normalized_statement="The policy changed.",
                extraction_confidence=0.9,
                evidence_passage_ids=[passage.id],
            )
        draft = ClaimExtractionDraft(
            atomic_proposition="The policy changed.",
            normalized_statement="The policy changed.",
            scope=ClaimScope(description="The cited policy document."),
            modality="asserted",
            extraction_confidence=0.9,
            evidence_passage_ids=[passage.id],
        )
        self.assertEqual(draft.evidence_passage_ids, [passage.id])

    def test_evidence_reducer_deduplicates_ids_and_override_reducer_replaces(self) -> None:
        source = SourceRecord(canonical_url="https://example.test", title="Example")
        duplicate_as_dict = source.model_dump(mode="python")
        another_source = SourceRecord(
            canonical_url="https://example.test/second", title="Second example"
        )

        merged = merge_evidence_by_id([source], [duplicate_as_dict, another_source])

        self.assertEqual({item.id if hasattr(item, "id") else item["id"] for item in merged}, {
            source.id,
            another_source.id,
        })
        self.assertEqual(override_reducer(["old"], ["new"]), ["old", "new"])
        self.assertEqual(
            override_reducer(["old"], {"type": "override", "value": ["new"]}),
            ["new"],
        )


class SqliteMigrationTests(unittest.TestCase):
    def test_migrates_a_version_one_database_to_add_provenance_attachments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.db"
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)"
                )
                connection.execute("INSERT INTO schema_migrations(version) VALUES (1)")
                connection.execute(
                    "CREATE TABLE research_runs (id TEXT PRIMARY KEY, payload TEXT, created_at TEXT)"
                )
                for table in SqliteEvidenceRepository._ARTIFACT_TABLES:
                    connection.execute(
                        f"CREATE TABLE {table} (id TEXT PRIMARY KEY, run_id TEXT, payload TEXT, created_at TEXT)"
                    )
                connection.commit()
            finally:
                connection.close()

            repository = SqliteEvidenceRepository(path)
            try:
                migrations = repository._connection.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                ).fetchall()
                attachment_table = repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'provenance_attachments'"
                ).fetchone()

                self.assertEqual([row["version"] for row in migrations], [1, 2])
                self.assertIsNotNone(attachment_table)
            finally:
                repository.close()
