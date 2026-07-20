import unittest

from polyresearch.models import SourceRecord
from polyresearch.retrieval.source_ingestion import detect_language, extract_document, languages_match
from polyresearch.retrieval.search_utils import _chunk_evidence_passages, select_citable_passages


class SourceIngestionTests(unittest.TestCase):
    def test_html_extraction_preserves_original_passages_and_metadata(self) -> None:
        document = extract_document(
            """
            <html lang="zh-CN"><head>
              <title>Fallback title</title>
              <link rel="canonical" href="/official-policy" />
              <meta property="og:title" content="Official policy" />
              <meta property="og:site_name" content="Policy Office" />
              <meta name="author" content="Research Unit" />
              <meta property="article:published_time" content="2026-07-20T10:00:00Z" />
              <meta property="article:modified_time" content="2026-07-20T12:00:00Z" />
            </head><body>
              <h1>政策更新</h1><p>第一段原文。</p><p>第二段原文。</p>
              <script>ignore this content</script>
            </body></html>
            """,
            content_type="text/html",
        )

        self.assertEqual(document.title, "Official policy")
        self.assertEqual(document.publisher, "Policy Office")
        self.assertEqual(document.author, "Research Unit")
        self.assertEqual(document.language, "zh-cn")
        self.assertEqual(document.content_language, "zh")
        self.assertEqual(document.metadata_language, "zh-cn")
        self.assertEqual(document.language_detection_method, "metadata_and_content")
        self.assertEqual(document.canonical_url, "/official-policy")
        self.assertEqual(document.published_at.isoformat(), "2026-07-20T10:00:00+00:00")
        self.assertEqual(document.updated_at.isoformat(), "2026-07-20T12:00:00+00:00")
        self.assertEqual(document.passages, [
            ("政策更新 / paragraph-1", "第一段原文。"),
            ("政策更新 / paragraph-2", "第二段原文。"),
        ])
        self.assertEqual(document.document_structure, [{
            "heading": "政策更新",
            "first_passage_locator": "政策更新 / paragraph-1",
            "last_passage_locator": "政策更新 / paragraph-2",
        }])
        self.assertGreater(document.extraction_quality, 0.8)

    def test_language_detection_uses_metadata_then_script_evidence(self) -> None:
        self.assertEqual(detect_language("English text", "fr-FR"), "fr-fr")
        self.assertEqual(detect_language("这是中文证据。"), "zh")
        self.assertEqual(detect_language("", None), None)
        self.assertTrue(languages_match("zh-cn", "zh"))
        self.assertFalse(languages_match("en", "zh-CN"))
        self.assertIsNone(languages_match(None, "zh"))

    def test_passages_include_stable_heading_and_character_anchors(self) -> None:
        document = extract_document(
            "<html><body><h1>Findings</h1><p>First original sentence.</p>"
            "<p>Second original sentence.</p></body></html>",
            content_type="text/html",
        )
        source = SourceRecord(
            canonical_url="https://example.test/source",
            title="Source",
            language="en",
        )

        passages = _chunk_evidence_passages(
            source,
            document.raw_content,
            document.passages,
            extracted_content=document.content,
        )

        self.assertEqual(passages[0].locator, "Findings / paragraph-1")
        self.assertEqual(passages[0].heading, "Findings")
        self.assertEqual(passages[0].character_start, 0)
        self.assertEqual(passages[0].character_end, len("First original sentence."))
        self.assertEqual(
            passages[1].character_start,
            len("First original sentence.\n\n"),
        )

    def test_passage_selection_returns_original_text_not_a_generated_summary(self) -> None:
        source = SourceRecord(canonical_url="https://example.test/source", title="Policy update")
        passages = _chunk_evidence_passages(
            source, "Unrelated background.\n\nPolicy begins Monday."
        )

        selected = select_citable_passages([source], passages, "When does policy begin?")

        self.assertEqual(selected[0].text, "Policy begins Monday.")
        self.assertEqual(selected[1].text, "Unrelated background.")
