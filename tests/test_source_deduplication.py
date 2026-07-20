import unittest

from polyresearch.deduplication import deduplicate_source_artifacts
from polyresearch.models import EvidencePassage, SourceRecord, SourceVersion


def _artifact(url: str, title: str, content: str):
    source = SourceRecord(canonical_url=url, title=title, content_hash=__import__("hashlib").sha256(content.encode()).hexdigest())
    version = SourceVersion(source_id=source.id, version_number=1, content_hash=source.content_hash, raw_content=content)
    passage = EvidencePassage(source_id=source.id, text=content, locator="paragraph-1")
    return source, version, passage


class SourceDeduplicationTests(unittest.TestCase):
    def test_suppresses_canonical_urls_and_exact_content_hashes(self) -> None:
        first = _artifact("https://one.example/article", "Article", "The original evidence.")
        same_url = _artifact("https://one.example/article", "Other", "Different content.")
        same_content = _artifact("https://two.example/copy", "Copy", "The original evidence.")

        sources, versions, passages = deduplicate_source_artifacts(
            [first[0], same_url[0], same_content[0]],
            [first[1], same_url[1], same_content[1]],
            [first[2], same_url[2], same_content[2]],
        )

        self.assertEqual(sources, [sources[0]])
        self.assertEqual(len(versions), 1)
        self.assertEqual(len(passages), 1)

    def test_retains_but_clusters_near_duplicate_syndication_and_publisher_family(self) -> None:
        first = _artifact(
            "https://news.example/article",
            "Wire report",
            "The agency said the policy will begin on Monday after a national review.",
        )
        copy = _artifact(
            "https://mirror.example/report",
            "Wire report",
            "The agency said the policy will begin on Monday after national review.",
        )

        sources, _, _ = deduplicate_source_artifacts(
            [first[0], copy[0]], [first[1], copy[1]], [first[2], copy[2]]
        )

        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0].publisher_family, "domain:news.example")
        self.assertEqual(sources[1].publisher_family, "domain:mirror.example")
        self.assertEqual(
            sources[1].near_duplicate_cluster_id,
            sources[0].shared_origin_cluster_id,
        )
