"""Fetch and extract source content without treating source text as instructions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urljoin

import aiohttp

from polyresearch.runtime.retry import retry_async


_SPACE = re.compile(r"\s+")


def _clean(value: str) -> str:
    return _SPACE.sub(" ", unescape(value)).strip()


class _DocumentParser(HTMLParser):
    """Small dependency-free HTML extractor for titles, metadata, and passages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.metadata: dict[str, str] = {}
        self.language: str | None = None
        self.canonical_url: str | None = None
        self._ignored_depth = 0
        self._current_tag: str | None = None
        self._parts: list[str] = []
        self.blocks: list[tuple[str, str]] = []
        self._heading: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.casefold(): value for key, value in attrs if value is not None}
        if tag in {"script", "style", "noscript", "template"}:
            self._ignored_depth += 1
            return
        if tag == "html":
            self.language = attrs_dict.get("lang") or self.language
        if tag == "meta":
            key = attrs_dict.get("name") or attrs_dict.get("property")
            content = attrs_dict.get("content")
            if key and content:
                self.metadata[key.casefold()] = content.strip()
        if tag == "link" and attrs_dict.get("rel", "").casefold() == "canonical":
            self.canonical_url = attrs_dict.get("href")
        if tag in {"title", "p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._current_tag = tag
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "template"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if tag != self._current_tag:
            return
        text = _clean(" ".join(self._parts))
        self._current_tag = None
        self._parts = []
        if not text:
            return
        if tag == "title":
            self.title = text
        elif tag.startswith("h"):
            self._heading = text
        else:
            self.blocks.append((self._heading or "document", text))

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and self._current_tag:
            self._parts.append(data)


@dataclass(frozen=True)
class ExtractedDocument:
    """Content and provenance produced by direct fetch or provider fallback."""

    raw_content: str
    content: str
    title: str | None = None
    publisher: str | None = None
    author: str | None = None
    language: str | None = None
    content_language: str | None = None
    metadata_language: str | None = None
    language_detection_method: str | None = None
    canonical_url: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    passages: list[tuple[str, str]] = field(default_factory=list)
    document_structure: list[dict[str, Any]] = field(default_factory=list)
    http_metadata: dict[str, Any] = field(default_factory=dict)
    extraction_method: str = "provider_content"
    extraction_quality: float = 0.0
    extraction_notes: list[str] = field(default_factory=list)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)


def _normalize_language(language: str | None) -> str | None:
    return language.replace("_", "-").casefold() if language else None


def detect_content_language(content: str) -> str | None:
    """Conservatively infer language from the source's visible original text."""
    letters = [char for char in content if char.isalpha()]
    if not letters:
        return None
    if sum("\u4e00" <= char <= "\u9fff" for char in letters) / len(letters) > 0.15:
        return "zh"
    if sum("\u0400" <= char <= "\u04ff" for char in letters) / len(letters) > 0.15:
        return "ru"
    if sum("\u0600" <= char <= "\u06ff" for char in letters) / len(letters) > 0.15:
        return "ar"
    if sum("\u3040" <= char <= "\u30ff" for char in letters) / len(letters) > 0.10:
        return "ja"
    return "en"


def detect_language(content: str, metadata_language: str | None = None) -> str | None:
    """Use declared metadata when present, retaining content detection separately."""
    return _normalize_language(metadata_language) or detect_content_language(content)


def languages_match(detected_language: str | None, planned_language: str | None) -> bool | None:
    """Compare BCP-47-like tags without treating regional variants as mismatches."""
    if not detected_language or not planned_language:
        return None
    return _normalize_language(detected_language).split("-", 1)[0] == _normalize_language(
        planned_language
    ).split("-", 1)[0]


def extract_document(content: str, *, content_type: str | None = None) -> ExtractedDocument:
    """Extract stable original-text passages and metadata from supplied content."""
    is_html = "html" in (content_type or "").casefold() or bool(re.search(r"<html\b|<body\b", content, re.I))
    if not is_html:
        passages = [(f"paragraph-{index}", paragraph.strip()) for index, paragraph in enumerate(re.split(r"\n\s*\n", content), 1) if paragraph.strip()]
        quality = 0.7 if passages else 0.0
        content_language = detect_content_language(content)
        return ExtractedDocument(
            raw_content=content,
            content=content,
            passages=passages,
            document_structure=(
                [{
                    "heading": "document",
                    "first_passage_locator": passages[0][0],
                    "last_passage_locator": passages[-1][0],
                }]
                if passages
                else []
            ),
            language=content_language,
            content_language=content_language,
            language_detection_method="content_script",
            extraction_quality=quality,
            extraction_notes=["plain_text"],
        )

    parser = _DocumentParser()
    parser.feed(content)
    passages = [(f"{heading} / paragraph-{index}", text) for index, (heading, text) in enumerate(parser.blocks, 1)]
    document_structure: list[dict[str, Any]] = []
    for locator, (heading, _) in zip((locator for locator, _ in passages), parser.blocks):
        if document_structure and document_structure[-1]["heading"] == heading:
            document_structure[-1]["last_passage_locator"] = locator
        else:
            document_structure.append(
                {
                    "heading": heading,
                    "first_passage_locator": locator,
                    "last_passage_locator": locator,
                }
            )
    visible_text = "\n\n".join(text for _, text in parser.blocks)
    metadata = parser.metadata
    content_language = detect_content_language(visible_text or content)
    metadata_language = _normalize_language(parser.language)
    detected_language = metadata_language or content_language
    language_notes = ["html", "visible_text" if passages else "no_semantic_blocks"]
    if metadata_language and content_language and not languages_match(metadata_language, content_language):
        language_notes.append("metadata_language_conflicts_with_content")
    quality = min(1.0, 0.25 + (0.45 if passages else 0) + (0.15 if parser.title else 0) + (0.15 if parser.language else 0))
    return ExtractedDocument(
        raw_content=content,
        content=visible_text or content,
        title=metadata.get("og:title") or parser.title or None,
        publisher=metadata.get("og:site_name") or metadata.get("publisher") or None,
        author=metadata.get("author") or metadata.get("article:author") or None,
        language=detected_language,
        content_language=content_language,
        metadata_language=metadata_language,
        language_detection_method=("metadata_and_content" if metadata_language and content_language else "metadata" if metadata_language else "content_script" if content_language else None),
        canonical_url=parser.canonical_url,
        published_at=_parse_datetime(
            metadata.get("article:published_time")
            or metadata.get("datepublished")
            or metadata.get("publishdate")
            or metadata.get("date")
        ),
        updated_at=_parse_datetime(
            metadata.get("article:modified_time")
            or metadata.get("datemodified")
            or metadata.get("last-modified")
        ),
        passages=passages,
        document_structure=document_structure,
        extraction_quality=quality,
        extraction_notes=language_notes,
    )


async def fetch_source_content(url: str, *, timeout_seconds: float = 15.0) -> ExtractedDocument:
    """Fetch a discovered URL and retain HTTP provenance alongside extracted text."""
    async def fetch_once() -> ExtractedDocument:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        headers = {"User-Agent": "PolyResearch/0.1 evidence fetcher"}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url, allow_redirects=True) as response:
                raw_content = await response.text(errors="replace")
                document = extract_document(raw_content, content_type=response.headers.get("Content-Type"))
                chain = [str(item.url) for item in response.history] + [str(response.url)]
                metadata = {
                    "status": response.status, "content_type": response.headers.get("Content-Type"),
                    "content_length": response.headers.get("Content-Length"), "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"), "final_url": str(response.url),
                    "redirect_chain": chain,
                }
                return ExtractedDocument(
                    **{**document.__dict__, "canonical_url": document.canonical_url and urljoin(str(response.url), document.canonical_url), "http_metadata": metadata, "extraction_method": "direct_http_fetch"}
                )
    return await retry_async(fetch_once, attempts=3)  # type: ignore[return-value]
