# PolyResearch

PolyResearch is a multilingual, evidence-centric research workflow. It plans research languages for a question, retrieves and preserves original-language passages, extracts atomic claims, verifies them, records disagreements, and renders reports with passage-level provenance.

The project treats reports as views over a durable evidence ledger rather than as standalone model output. Every substantive report statement is linked to typed claims and citations; unresolved or non-comparable evidence remains visible instead of being smoothed into consensus.

## What it does

- Selects research languages adaptively for the topic.
- Routes Chinese-source discovery to the allowlisted Alibaba Bailian Web Search MCP tool and other selected languages to Tavily.
- Stores sources, immutable source versions, original-language passages, translations, claims, evidence links, verification attempts, and report bundles in SQLite.
- Detects duplicate, syndicated, and shared-origin sources before counting corroboration.
- Verifies claims against passage-level evidence and records support, contradiction, context, confidence factors, and conflict-resolution attempts.
- Produces Markdown, HTML, JSON provenance bundles, and PDF exports.

See [the evidence policy](docs/evidence-policy.md) for the operating rules and retention model.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Credentials for the services you intend to use:
  - `QWEN_API_KEY` is required for the Qwen planning, extraction, verification, and report-generation models.
  - `TAVILY_API_KEY` is required for Tavily discovery, including Bailian fallback and non-Chinese/bridge searches.
  - `DASHSCOPE_API_KEY` enables the default Bailian Web Search configuration for Chinese discovery.

The same DashScope credential can be exported as both `QWEN_API_KEY` and `DASHSCOPE_API_KEY` when it is authorized for both services.

## Install

```sh
uv sync
```

Credentials must be available in the process environment. A `.env` file is not loaded automatically.

```sh
export QWEN_API_KEY="..."
export TAVILY_API_KEY="..."
export DASHSCOPE_API_KEY="..."  # Enables Bailian; optional but recommended for Chinese sources
```

## Run a research query

```sh
uv run polyresearch "What changed in China's latest AI regulations?" --log-level INFO
```

The default SQLite database is `polyresearch.db` in the current directory. Set `POLYRESEARCH_DB_PATH` to use another location:

```sh
POLYRESEARCH_DB_PATH=/path/to/research.db \
  uv run polyresearch "Compare official AI policy updates in China and the EU"
```

`--log-level` accepts `CRITICAL`, `ERROR`, `WARNING`, `INFO`, or `DEBUG`. It defaults to `WARNING`, or `POLYRESEARCH_LOG_LEVEL` when set.

## Bailian configuration

When `DASHSCOPE_API_KEY` is present, PolyResearch automatically instantiates the allowlisted default Bailian Web Search configuration for CLI, graph, and direct API callers. Chinese discovery selected by the research plan tries Bailian first; a recorded failure falls back to Tavily.

Programmatic callers can override the default through `RunnableConfig`:

```python
config = {
    "configurable": {
        "bailian_web_search": {
            "server_url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
            "timeout_seconds": 30,
            "max_requests_per_second": 10,
            "authentication": {"api_key": "..."},
        }
    }
}
```

Set `"bailian_web_search": None` explicitly to disable the environment-derived default. The MCP loader is intentionally restricted to Bailian's `web_search` tool; it does not expose arbitrary remote tools.

## Inspect and export a run

The CLI works with durable run IDs. Once you have a run ID, inspect the evidence ledger or a report statement's full trace:

```sh
uv run polyresearch --inspect-ledger <RUN_ID>
uv run polyresearch --inspect-ledger <RUN_ID> --source-id <SOURCE_ID>
uv run polyresearch --inspect-trace <RUN_ID> --report-statement-id <STATEMENT_ID>
```

Export the most recent report bundle:

```sh
uv run polyresearch --export <RUN_ID> --format markdown,html,json,pdf --output-dir ./exports
```

Exports contain:

- Markdown with stable passage citation IDs.
- HTML with clickable evidence panels.
- JSON provenance, including citations, verification history, language coverage, and report QA state.
- A readable PDF rendered from the audited Markdown bundle.

## Architecture

```text
question
  -> research brief and adaptive multilingual plan
  -> provider-routed discovery and immutable evidence ingestion
  -> claims, entity/value normalization, and verification attempts
  -> conflict resolution and report QA
  -> Markdown / HTML / JSON provenance / PDF report bundle
```

Key modules:

- `src/polyresearch/workflows/`: LangGraph orchestration, researcher, supervisor, and report generation.
- `src/polyresearch/retrieval/`: Tavily ingestion, Bailian MCP loading, routing, extraction, and source deduplication.
- `src/polyresearch/models/`: Pydantic domain artifacts and workflow state.
- `src/polyresearch/repositories/`: repository interface and SQLite implementation.
- `src/polyresearch/evidence/`: claim clustering, verification confidence, QA, normalization, and provenance graph helpers.

The public workflow entry point is `polyresearch.graph.graph`.

## Development

Run the test suite:

```sh
uv run pytest
```

Run lint checks:

```sh
uv run ruff check src tests
```

Do not commit API keys, raw secrets, or generated local databases. Retrieved web content is treated as untrusted data, never as executable instructions.
