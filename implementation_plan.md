# PolyResearch: Multilingual Evidence-Centric Research Operating System

## 1. Vision

Transform PolyResearch from a prompt-and-summary research agent into a multilingual research operating system that:

- discovers material knowledge across the languages most likely to contain unique and important evidence;
- preserves original source material and creates durable, passage-level provenance;
- identifies, verifies, and explains conflicting claims rather than blending them into one narrative;
- uses Qwen's multilingual reasoning to plan, normalize, verify, and write;
- produces transparent reports where every factual statement can be opened to reveal its evidence.

The system must treat reports as views over a structured evidence base, not as the only output of an LLM workflow.

## 2. Current Baseline and Gaps

The existing repository has a useful foundation:

- Qwen is already the configured model transport through Model Studio's OpenAI-compatible API.
- LangGraph coordinates clarification, planning, a supervisor, parallel researchers, compression, and report writing.
- Tavily supplies web search, and generic MCP tool support exists.

However, research is currently passed between nodes as unstructured text. The workflow has no persistent source records, passage locators, claim objects, source-independence model, verification step, or machine-verifiable relationship between final-report text and citations. Tavily result summaries can also replace the precise passages required for trustworthy evidence review.

## 3. Target Operating Model

```mermaid
flowchart LR
  Q["Question in the user's language"] --> P["Qwen: research & language plan"]
  P --> D["Language-aware discovery"]
  D --> L["Source and passage ledger"]
  L --> C["Qwen: atomic claims"]
  C --> V["Verify, corroborate, or contradict"]
  V --> G["Claim provenance graph"]
  G --> R["Citation-backed report"]
  R --> X["Clickable statement evidence"]
```

### Language roles

- **User/output language:** language used for the request and final report.
- **Research languages:** languages selected for discovery based on expected unique evidence.
- **Evidence language:** original language of each retrieved passage.
- **Translation language:** a stored, attributable translation used for analysis or display.

Original-language evidence remains authoritative. Translation is an aid and must never overwrite it.

## 4. Core Data Contract

Create typed domain models and persist them from the first successful retrieval.

| Artifact | Purpose | Minimum fields |
| --- | --- | --- |
| `ResearchRequest` | Defines the run | question, output language, scope, date boundaries |
| `ResearchPlan` | Makes research reproducible | subquestions, language ranking, query variants, target source types |
| `QueryRecord` | Records discovery decisions | query, language, provider, locale, timestamp, rank rationale |
| `SourceRecord` | Represents a fetched source | canonical URL, publisher, title, dates, language, source type, content hash |
| `EvidencePassage` | Preserves citable text | source ID, exact text, locator, original language, normalized text |
| `TranslationRecord` | Keeps translations accountable | passage ID, translated text, target language, Qwen model/prompt version, confidence |
| `Claim` | An atomic proposition | normalized statement, entities, dates, scope, extraction confidence |
| `EvidenceLink` | Connects evidence to a claim | supports/contradicts/contextualizes, strength, rationale, passage ID |
| `VerificationResult` | States the research judgement | status, confidence, conflict explanation, evidence-link IDs |
| `ReportStatement` | Makes report prose auditable | rendered text, claim IDs, citation IDs, verification status |
| `ReportBundle` | Delivers the result | Markdown/HTML, JSON graph, source list, methodology, limitations |

Use stable UUIDs for all artifacts. Record retrieval time, source content hash, tool/provider version, model ID, and prompt version wherever an automated decision occurs.

## 5. Work Breakdown

### Milestone 0 — Product and evidence rules

1. Write `docs/evidence-policy.md`.
2. Define supported languages and output-language fallback behavior.
3. Define the source-quality hierarchy: primary/official, peer-reviewed, reputable secondary, commentary, and unverified material.
4. Define what counts as independent corroboration; copies, wire syndication, and shared-source reporting must not count as separate confirmation.
5. Define verification statuses: `supported`, `partially_supported`, `contradicted`, `insufficient_evidence`, `outdated`, and `not_comparable`.
6. Define what constitutes a contradiction versus a difference in date, geography, population, terminology, or method.
7. Define a retention policy for fetched content, user material, translations, and run logs.
8. Define acceptance gates:
   - factual final-report statements link to one or more verified claims;
   - claims link to passage-level original-language evidence;
   - material conflicts are surfaced;
   - translations retain original text and provenance;
   - limitations disclose missing languages, inaccessible sources, and unresolved uncertainty.

### Milestone 1 — Typed evidence state and persistence

1. Split `src/polyresearch/state.py` into focused Pydantic models under `src/polyresearch/models/`.
2. Replace free-form `notes`, `raw_notes`, and `compressed_research` handoffs with typed collections of sources, passages, claims, and verification results.
3. Add a repository interface under `src/polyresearch/repositories/`.
4. Implement SQLite storage for local development:
   - research runs and plans;
   - query records;
   - sources and source versions;
   - passages and translations;
   - claims, evidence links, verification results;
   - report statements and report bundles.
5. Keep raw tool output as immutable provenance attachments, not as the primary reasoning input.
6. Add migrations and tests for schema validation, IDs, reducers, and durable reload/resume.

**Exit criteria:** a retrieved page can be saved, chunked into stable passages, loaded in a later process, and cited by ID.

### Milestone 2 — Adaptive multilingual planning with Qwen

1. Add a `multilingual_planner` LangGraph node after research-brief creation.
2. Create a strict structured-output model for `ResearchPlan` containing:
   - atomic subquestions;
   - entities, aliases, transliterations, and native-script variants;
   - ranked research languages;
   - explanation of the expected unique value of each language;
   - query variants per language;
   - expected source types and domains;
   - anticipated conflict dimensions.
3. Prompt Qwen to select languages adaptively, not from a fixed list. Language selection should consider:
   - place/country and institutional jurisdiction;
   - language of primary actors and likely official records;
   - topic-specific scholarly, technical, and media ecosystems;
   - diasporic or regional coverage where appropriate;
   - likely primary-source availability;
   - expected marginal information gain over already selected languages.
4. Assign each language a priority and budget. Start with the highest expected information gain, then use evidence gaps to decide whether to add languages.
5. Add a second Qwen decision point after initial retrieval: choose whether more languages are warranted, and document why.
6. Preserve original terminology alongside normalized/translated terms; do not assert equivalence when a translation is approximate.
7. Add fixture-based tests for multilingual aliases, mixed scripts, ambiguous names, and topics whose most valuable sources are not English.

**Exit criteria:** every run records why each research language was selected, skipped, or added later.

### Milestone 3 — Discovery adapters: Tavily and Alibaba Bailian Web Search MCP

1. Keep Tavily as the existing generic web-search provider. Do not remove its current direct integration during the migration.
2. Add **Alibaba Bailian Web Search through MCP** as the only new MCP integration for this phase. Configure it specifically for Chinese-language source discovery.
3. Do not add other MCP providers at this stage.
4. Replace the current single `SearchAPI` choice with a provider-routing abstraction:
   - `TavilySearchProvider` for broad/general discovery;
   - `BailianWebSearchProvider` backed by the configured MCP tool for Chinese discovery;
   - provider selection driven by `ResearchPlan` language and source-type intent.
5. Extend `MCPConfig` or replace it with a narrowly scoped Bailian configuration model:
   - server URL;
   - allowlisted Bailian web-search tool name;
   - authentication configuration;
   - Chinese locale/language defaults;
   - timeout and rate limit.
6. Make the MCP allowlist explicit. Only the configured Bailian search tool may be loaded for this feature.
7. Extend search calls with query language, locale, freshness/date bounds, target source type, query rationale, and run ID.
8. Implement provider fallback rules:
   - use Bailian first for Chinese-language queries selected by the plan;
   - use Tavily for other selected languages and cross-language bridge searches;
   - record failures and fallback decisions in `QueryRecord`;
   - never silently substitute a provider while claiming equivalent coverage.
9. Canonicalize URLs, preserve redirect information, record result rank, and deduplicate exact URLs before fetching.
10. Test provider routing without live credentials using mocked Tavily and MCP results.

**Exit criteria:** a Chinese evidence query selected by Qwen is routed to the allowlisted Bailian MCP tool, while a non-Chinese query uses Tavily; both paths emit the same typed source artifacts.

### Milestone 4 — Source ingestion and evidence ledger

1. Fetch source content after discovery and record retrieval date, HTTP metadata where available, content hash, and extraction quality.
2. Detect source language using content and metadata; compare the result with the planned query language.
3. Extract title, publisher, author, publication/update dates, canonical URL, and document structure.
4. Chunk content into stable citable passages with locators such as heading, paragraph number, page number, or character offsets.
5. Preserve original passage text before any summarization or translation.
6. Produce translations only when analysis or output requires them. Persist translations as `TranslationRecord` objects linked to original passages.
7. Deduplicate at multiple levels:
   - canonical URL;
   - exact content hash;
   - near-duplicate or syndication match;
   - publisher-family/shared-origin clustering.
8. Score initial source quality from source type, publisher transparency, author/date presence, primary-source signals, and relevance. Keep this score explainable and revisable.
9. Replace the current “summarize each result before research” path in `utils.py` with passage selection. LLM summaries can be supplemental but cannot replace evidence passages.
10. Build a ledger inspection command that prints a source, passages, translations, queries, and downstream claims.

**Exit criteria:** every sentence used as evidence can be retrieved in its original language with a stable source and passage locator.

### Milestone 5 — Claim extraction, normalization, and provenance graph

1. Add a Qwen structured-output `claim_extractor` that processes selected evidence passages.
2. Require claim extraction to emit:
   - an atomic, falsifiable proposition;
   - original-language claim wording where available;
   - normalized output-language wording;
   - entities, quantities, dates, locations, and scope;
   - qualifiers and modality;
   - extraction confidence;
   - direct passage references.
3. Add entity resolution that supports aliases, scripts, transliterations, historical names, and uncertain mappings.
4. Normalize dates, currencies, units, and numerical formats while retaining original values and units.
5. Create a claim-clustering step that groups evidence addressing the same proposition.
6. Create a graph layer with nodes for research runs, queries, sources, passages, translations, claims, evidence links, verification decisions, and report statements.
7. Add graph edges such as:
   - `FOUND_BY` query → source;
   - `CONTAINS` source → passage;
   - `TRANSLATED_AS` passage → translation;
   - `ASSERTS` passage → claim;
   - `SUPPORTS` / `CONTRADICTS` evidence → claim;
   - `VERIFIED_BY` claim → verification result;
   - `RENDERED_AS` claim → report statement.
8. Keep a SQLite-compatible graph projection initially; do not require a dedicated graph database for the first implementation.
9. Add graph traversal tests proving that every report statement can reach its evidence passages.

**Exit criteria:** the system can answer “why does this sentence appear?” with a complete path from report statement to query, original source passage, and translation.

### Milestone 6 — Verification and conflict resolution

1. Add a structured Qwen `verifier` node operating on claim clusters, not raw report text.
2. Have it classify each claim as `supported`, `partially_supported`, `contradicted`, `insufficient_evidence`, `outdated`, or `not_comparable`.
3. Require the verifier to identify whether apparent disagreement is caused by:
   - different time periods;
   - different geographic scope;
   - differing definitions or measurement methods;
   - different populations/samples;
   - translation ambiguity;
   - genuinely conflicting evidence.
4. Weight confidence conservatively using directness, source quality, independence, scope fit, recency, agreement, and translation certainty.
5. Prevent syndicated/reposted pages from increasing corroboration count.
6. Add a conflict-resolution loop that seeks the strongest relevant primary or official sources before declaring consensus.
7. Preserve unresolved disagreement as a first-class output. The report must state what conflicts, why it may conflict, and what evidence would resolve it.
8. Add adversarial tests for conflicting Chinese and English sources, source copies, stale claims, incompatible measurements, and ambiguous translations.

**Exit criteria:** material conflicts are represented in the provenance graph and cannot be silently converted into a single unsupported assertion.

### Milestone 7 — Transparent report generation and clickable evidence

1. Replace free-form final report generation with a two-stage process:
   - Qwen builds a structured report outline from verified claims;
   - Qwen writes prose only from approved claim and verification artifacts.
2. Add a `ReportStatement` record for every sentence or displayable factual clause in the final report.
3. Bind every `ReportStatement` to its claim IDs and verification status before rendering.
4. Render report statements as clickable citations/evidence anchors in HTML. On click, show an evidence panel containing:
   - original source title, publisher, URL, publication date, and retrieval date;
   - original-language passage and stable locator;
   - stored translation, labeled as a translation;
   - language of source and translation;
   - source-quality and claim-confidence scores with explanations;
   - support/contradiction status;
   - verification history, including model/prompt versions and timestamps;
   - related corroborating and conflicting evidence.
5. For Markdown output, include stable citation IDs and a companion JSON provenance bundle. Markdown alone cannot offer reliable click-to-evidence behavior.
6. Include report sections for:
   - findings supported by evidence;
   - conflicting or uncertain claims;
   - language coverage and source mix;
   - method, retrieval date, and limitations;
   - complete sources.
7. Add report QA that rejects or flags:
   - factual statements without evidence links;
   - citations without source/passages;
   - bibliography entries not used by any statement;
   - claims reported more strongly than their verification status permits.
8. Export `ReportBundle` in Markdown, HTML, and JSON. Add PDF/DOCX only after the evidence bundle is stable.

**Exit criteria:** clicking any factual report statement opens its provenance view, including original source/language, translation, confidence, and verification history.

### Milestone 8 — LangGraph orchestration, CLI, and operational hardening

1. Reshape the graph into:

   `clarify → brief → multilingual plan → provider-routed discovery → fetch/extract → ledger → claim extraction → verification/conflict loop → report composition → report QA`

2. Keep bounded parallelism, but assign agents typed evidence tasks rather than open-ended prose summaries.
3. Add idempotency and retry boundaries around search, MCP calls, fetches, extraction, structured output, and report rendering.
4. Add CLI commands:
   - `polyresearch research "question" --output-language zh --research-languages auto`
   - `polyresearch inspect <run-id>`
   - `polyresearch evidence <statement-id>`
   - `polyresearch export <run-id> --format markdown,html,json`
   - `polyresearch verify <run-id>`
5. Make the run configuration reproducible: Qwen model IDs, prompts, selected languages, provider routing, query budgets, and retrieval timestamps.
6. Add tracing from query to report statement, with latency, cost, retries, provider failures, and graph artifacts.
7. Apply security controls:
   - MCP tool allowlisting for Bailian;
   - strict secret handling;
   - fetched-page prompt-injection defenses;
   - domain allow/block policy hooks;
   - query/token/rate budgets;
   - retention and redaction controls.

### Milestone 9 — Evaluation and release gates

1. Build a multilingual evaluation corpus with known evidence, conflicts, and expected language choices.
2. Measure:
   - citation coverage and passage-level entailment;
   - unsupported-claim rate;
   - conflict-detection recall;
   - source-independence accuracy;
   - multilingual retrieval lift over an English-only baseline;
   - language-plan quality and marginal-information gain;
   - provenance-graph completeness;
   - latency and cost per research run.
3. Add regression tests for all critical graph contracts.
4. Document local setup, Bailian MCP setup, Qwen configuration, privacy behavior, report interpretation, and known limitations in `README.md`.
5. Release in increments:
   - **A:** source/passage ledger and mechanically linked citations;
   - **B:** adaptive multilingual planning and Bailian Chinese-source discovery;
   - **C:** claim verification and conflict reporting;
   - **D:** clickable HTML provenance graph and full evaluation gates.

## 6. Implementation Priorities

The first implementation slice should be Milestone 1 plus the minimal pieces of Milestones 4 and 7: persistent source/passages, structured report statements, and a report QA gate. This establishes trustworthy citation infrastructure before broader multilingual discovery and conflict reasoning increase the number of sources and claims.

The second slice should add adaptive language planning and the scoped Alibaba Bailian Web Search MCP adapter. This makes Chinese-source discovery intentional, observable, and comparable with Tavily rather than an untracked side path.

## 7. Definition of Done

PolyResearch becomes the intended operating system when a research run can demonstrate all of the following:

1. Qwen selected research languages based on the topic and recorded the rationale.
2. Chinese queries selected for unique Chinese evidence used the allowlisted Bailian Web Search MCP tool; Tavily was retained for applicable general/cross-language discovery.
3. Each substantive report statement is backed by a typed claim and original-language passage(s).
4. Readers can click a statement in the HTML report and inspect source, language, translation, confidence, and verification history.
5. Conflicting claims are visible, explained, and linked to their opposing evidence.
6. The complete report, evidence ledger, and provenance graph can be exported and independently audited.
