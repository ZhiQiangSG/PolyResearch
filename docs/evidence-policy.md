# PolyResearch Evidence Policy

## Purpose and scope

This policy governs evidence collected, transformed, verified, retained, and rendered by PolyResearch. It implements the evidence-first operating model in [the implementation plan](../implementation_plan.md): reports are views over a structured evidence base, not standalone model outputs.

It applies to every research run, including user-supplied material, discovered web content, translations, model-generated artifacts, and final Markdown, HTML, and JSON reports. Fetched content is untrusted data and must never be followed as instructions.

## Core rules

1. No substantive factual statement may appear in a final report unless it is linked to one or more verified atomic claims.
2. Every claim must link to exact, original-language evidence passages with a stable source ID and passage locator.
3. A translation is a linked derivative of original evidence, never a substitute for it.
4. A source discovery result or LLM summary is not, by itself, evidence for a factual assertion; the underlying source must be fetched or otherwise preserved as a citable passage.
5. Conflicts, gaps, and uncertainty are findings. They must be recorded and, when material, disclosed rather than averaged away.
6. Corroboration counts independent origins, not the number of URLs.
7. Every automated decision must be reproducible from its typed artifact, inputs, timestamps, provider or model ID, prompt version where applicable, and relevant configuration.

## Evidence artifacts and minimum provenance

PolyResearch uses the typed artifacts defined in the [implementation plan](../implementation_plan.md): research requests and plans, query records, sources, passages, translations, claims, evidence links, verification results, report statements, and report bundles. Artifacts use stable IDs and preserve their links.

At minimum, the system records:

- **Discovery:** query text, query language and locale, planned purpose, provider, result rank, timestamp, fallback or failure, and rationale for provider selection.
- **Source:** canonical URL, redirect information when available, publisher, author, title, publication and update dates, source type, detected language, retrieval time, extraction quality, and content hash.
- **Passage:** immutable original text, source ID, original language, stable locator (such as page, heading, paragraph, or character offsets), and any normalization separately from the original.
- **Translation:** passage ID, translated text, target language, model or translator identity, prompt/version where applicable, timestamp, and translation-confidence assessment.
- **Claim and verification:** normalized and original wording where available; entities, dates, quantities, scope, qualifiers, evidence-link IDs, decision status, confidence factors, conflict reasoning, verifier model/prompt version, and timestamp.
- **Report statement:** rendered text, linked claim and citation IDs, verification status, report version, and QA outcome.

Raw source and passage provenance is immutable once recorded. Corrections or refreshed fetches create new versions that retain the predecessor relationship; they do not overwrite prior evidence.

## Language and translation policy

The output language is the requester's report language. Research languages are selected adaptively for the topic, using the language-planning process in the [implementation plan](../implementation_plan.md), rather than a fixed language list. The run records every selected, skipped, and later-added language with its rationale, expected information gain, budget, and gap-analysis result.

Original-language text is authoritative. Preserve native script, relevant aliases, transliterations, historical names, and culturally, legally, or politically loaded terms alongside any normalized wording. Do not claim that two terms are equivalent if the translation is approximate or contested.

Translations are produced only when needed for analysis or display. They must be labeled as translations, linked to their source passage, and displayed with the original text and language metadata in evidence views. Translation ambiguity reduces verification confidence and may require `partially_supported`, `not_comparable`, or `insufficient_evidence` rather than a stronger result.

If adequate evidence cannot be obtained in the requested output language, PolyResearch may report in the closest supported output language only when the user approves or the product explicitly defines that fallback. The report must state the fallback, its reason, and any resulting limitations. Lack of output-language support never authorizes replacing original evidence with an untraceable translation.

## Source-quality hierarchy

Source quality is assessed for the specific claim and is explainable, revisable, and distinct from claim confidence.

1. **Primary or official sources:** original records, statutes, regulatory filings, datasets, direct institutional statements, court documents, standards, and first-party research. These are preferred for material factual disputes, while still checked for scope and incentives.
2. **Peer-reviewed or methodologically transparent research:** scholarly work and documented analyses with identifiable methods, data, and authorship.
3. **Reputable secondary reporting:** transparent editorial sources that attribute claims, distinguish reporting from commentary, and provide dates and authorship where appropriate.
4. **Commentary and analysis:** expert opinion, advocacy, blogs, and synthesis. These may provide context or leads but require careful claim-specific evaluation.
5. **Unverified material:** anonymous, unattributed, undated, unverifiable, manipulated, or otherwise weak material. It may be retained as a lead or context record but cannot independently support a substantive final-report assertion.

Quality assessment considers source type, provenance, transparency, author/date information, directness, methodological quality, relevance, recency, and applicable conflicts of interest. A high-tier source is not automatically decisive when it addresses a different time, population, definition, or geography.

## Independent corroboration and deduplication

PolyResearch treats evidence as independent only when it has meaningfully distinct origins. The system canonicalizes URLs and deduplicates exact URLs, content hashes, near duplicates, syndication, publisher-family copies, mirrors, and shared-origin reporting.

The following do **not** count as separate corroboration unless they add independently obtained, material evidence:

- wire stories republished by multiple outlets;
- press releases, official statements, or datasets repeated by others without independent reporting;
- translations, mirrors, scraped copies, and archival copies of the same source;
- articles relying on the same named source, study, filing, or undisclosed common reporting;
- multiple pages from the same publisher repeating the same underlying evidence.

Related sources remain useful as discovery and context records, but their relationship is stored so they cannot inflate agreement or confidence. When origin cannot be determined, the system records the uncertainty and applies a conservative independence assessment.

## Claim verification and status vocabulary

Claims are atomic and falsifiable. They preserve relevant entities, dates, quantities, locations, populations, definitions, methods, and qualifiers. Evidence links explicitly state whether a passage supports, contradicts, or contextualizes a claim.

Verification uses only these statuses:

| Status | Meaning | Reporting rule |
| --- | --- | --- |
| `supported` | Sufficient, relevant, independent evidence supports the claim within its stated scope. | State the claim with its material qualifiers and citations. |
| `partially_supported` | Evidence supports some, but not all, of the claim's scope, precision, or implications. | State only the supported portion and describe the limitation. |
| `contradicted` | Relevant evidence directly conflicts after comparability checks. | Do not present the claim as established; show the conflict and evidence. |
| `insufficient_evidence` | Available evidence does not permit a reliable finding. | State that evidence is insufficient; do not infer a conclusion. |
| `outdated` | Evidence was once applicable but is superseded, stale, or no longer reliable for the report's time scope. | Identify the applicable period and seek current evidence. |
| `not_comparable` | Evidence addresses superficially similar propositions but cannot be validly compared. | Keep the findings separate and explain the mismatch. |

Confidence is conservative and explainable from directness, source quality, independence, scope fit, recency, agreement, and translation certainty. A confidence score never overrides an adverse status or the absence of passage-level evidence.

## Conflict analysis

Before classifying evidence as contradictory, PolyResearch checks whether the apparent disagreement arises from:

- different time periods or an intervening change;
- different geographic or institutional scope;
- different definitions, terminology, units, or measurement methods;
- different populations, samples, or inclusion criteria;
- different causal versus descriptive questions;
- a translation or entity-resolution ambiguity; or
- genuinely conflicting observations or assertions.

The analysis records the applicable dimensions and evidence. For a material unresolved conflict, the system seeks the strongest relevant primary or official evidence, documents what remains unresolved, and identifies what evidence could resolve it. Reports must surface material conflicts rather than synthesize them into a confident consensus.

## Discovery, provider routing, and limitations

Provider choice follows the planned language and source intent. Chinese-language queries selected for unique Chinese evidence route to the configured, allowlisted Alibaba Bailian Web Search MCP tool first; other research languages and bridge queries route to Tavily. Failures and fallbacks are retained in `QueryRecord`; the system must never silently claim equivalent coverage after substitution.

### Bailian activation

Set `DASHSCOPE_API_KEY` in the **process environment** to activate the default allowlisted Bailian Web Search configuration for CLI, graph, and API callers. A `.env` file is not loaded automatically; use your deployment launcher or shell to export it. Callers can explicitly supply `bailian_web_search` in the runtime configuration to override the endpoint, timeout, rate limit, or credentials; supplying `None` disables the environment-derived default.

Search result snippets are discovery aids only. The system records inaccessible, blocked, unavailable, paywalled, or low-quality sources and does not imply their content was verified. The final report discloses material missing languages, inaccessible source categories, provider failures, retrieval-date limits, unresolved conflicts, and other uncertainty that could affect its findings.

## Retention, privacy, and redaction

Retention must support auditability while minimizing unnecessary storage. Default periods may be configured by deployment, but the following categories and controls are mandatory:

| Material | Retention rule |
| --- | --- |
| Source records, original passages, hashes, locators, and provenance links | Retain for the life of the research run and for the configured audit period; preserve immutable versions. |
| Translations, claims, verification results, report statements, and report bundles | Retain with their linked source IDs, model/prompt metadata, and versions for the same audit purpose. |
| Raw fetched content and provider/tool responses | Retain as access-controlled provenance attachments only as long as needed for reproducibility, legal obligations, and the configured audit period; minimize copies. |
| User-provided material | Retain only for the requested research purpose and configured audit period; apply stricter access and deletion handling where required. |
| Run logs and telemetry | Retain minimally for operational troubleshooting and reproducibility; exclude secrets and redact personal or sensitive data where feasible. |

API keys, tokens, credentials, and retrieved secrets must never be committed, rendered in reports, or retained in ordinary evidence artifacts. Redaction creates a derived display or working artifact while retaining a controlled record that redaction occurred; it must not silently alter the underlying provenance. Deletion and legal-hold behavior are deployment responsibilities and must preserve audit metadata to the extent permitted.

## Report requirements and acceptance gates

Final reports are generated from verified claim artifacts, never directly from raw search messages. HTML report statements must be clickable to reveal source metadata, original-language passage and locator, labeled translation, language metadata, confidence, verification status and history, and related supporting or conflicting evidence. Markdown reports must include stable citation IDs and be delivered with a JSON provenance bundle.

A report passes QA only when all of the following are true:

- each substantive factual statement links to at least one verified claim and resolvable source/passage citation;
- each claim has original-language, passage-level evidence and typed support, contradiction, or context links;
- translations preserve and link to the original text and record their provenance;
- material conflicts, stale evidence, and non-comparability are surfaced with appropriate status;
- report wording does not exceed the linked verification status or confidence;
- citations are not orphaned, and bibliography entries are used by at least one report statement;
- limitations disclose language coverage, source mix, retrieval date, missing or inaccessible evidence, provider fallbacks, and unresolved uncertainty.

Reports that fail any gate must be flagged or rejected for correction. The system must favor an explicit limitation over an unsupported conclusion.
