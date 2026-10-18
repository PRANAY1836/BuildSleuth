# BuildSleuth

**AI-assisted CI failure diagnosis grounded in build logs, code changes, and technical documentation.**

> Status: starter project / in development. The current version provides a deterministic baseline and retrieval scaffolding; an LLM-backed explanation layer and live GitHub Actions ingestion are planned extensions, not yet implemented.

## Why BuildSleuth?

CI failures often produce long logs, while the likely cause may be a small code change or a known dependency/configuration issue. BuildSleuth aims to help developers narrow the investigation by surfacing relevant evidence and producing a structured diagnosis.

## Current starter features

- FastAPI service with `/health` and `/diagnose` endpoints.
- Basic log-signal extraction for common Python, dependency, test, and configuration failures.
- Lightweight hybrid retrieval baseline combining lexical token overlap and exact phrase matching across a small evidence corpus.
- Responses include matched evidence snippets and clearly labelled candidate causes; no unsupported certainty claims.
- Unit tests and GitHub Actions CI workflow.
- Synthetic sample incidents for local testing.

## Architecture (starter)

```text
CI log + optional code diff
          |
          v
   Signal extraction
          |
          v
  Evidence corpus retrieval ----> docs / known incidents
          |
          v
 Structured, evidence-linked diagnosis
          |
          v
       FastAPI JSON
```

## Run locally

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/diagnose \
  -H 'Content-Type: application/json' \
  -d '{"repository":"demo/service","run_id":"local-001","log":"ModuleNotFoundError: No module named \'requests\'\\nFAILED tests/test_api.py::test_create_user"}'
```

## API

### `GET /health`
Returns service status and current version.

### `POST /diagnose`
Accepts:

- `repository` (optional): repository name or URL label
- `run_id` (optional): CI run identifier
- `log` (required): build or test output
- `diff` (optional): relevant code-change summary

Returns extracted signals, candidate causes, evidence, and recommended next checks. The current implementation is a baseline and should not be treated as an authoritative root-cause determination.

## Roadmap

1. Add robust parsers for GitHub Actions job logs and test reports.
2. Ingest commit diffs and map changed files to relevant failures.
3. Implement configurable embeddings + vector search, alongside BM25/lexical retrieval.
4. Add a provider-agnostic LLM explanation layer that must cite retrieved evidence.
5. Add evaluation fixtures and metrics for retrieval recall, citation correctness, and diagnosis usefulness.
6. Add GitHub App/webhook integration with least-privilege permissions and repository-scoped access.
7. Add deployment, authentication, rate limits, and privacy controls before public multi-user use.

## Privacy and security notes

Build logs can contain credentials, tokens, internal URLs, and personal data. Do not submit sensitive logs to a hosted instance. Before real-world use, add redaction, retention controls, authentication, access isolation, and explicit user consent for external model providers. Treat logs and repository text as untrusted data.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
