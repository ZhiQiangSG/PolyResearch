import unittest

from polyresearch.source_ingestion import detect_language, extract_document, languages_match


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
        self.assertEqual(document.passages, [
            ("政策更新 / paragraph-1", "第一段原文。"),
            ("政策更新 / paragraph-2", "第二段原文。"),
        ])
        self.assertGreater(document.extraction_quality, 0.8)

    def test_language_detection_uses_metadata_then_script_evidence(self) -> None:
        self.assertEqual(detect_language("English text", "fr-FR"), "fr-fr")
        self.assertEqual(detect_language("这是中文证据。"), "zh")
        self.assertEqual(detect_language("", None), None)
        self.assertTrue(languages_match("zh-cn", "zh"))
        self.assertFalse(languages_match("en", "zh-CN"))
        self.assertIsNone(languages_match(None, "zh"))
