"""Small transparent lexical retrieval baseline; replace/extend for production RAG."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    source: str
    text: str


# Curated starter corpus. Keep evidence specific and version it as it grows.
CORPUS = [
    Document("docs/python-imports.md", "ModuleNotFoundError or ImportError: verify the missing package is declared in requirements.txt or pyproject.toml, installed in the CI environment, and compatible with the active Python version."),
    Document("docs/pytest.md", "Pytest assertion failures: inspect the first failing assertion, actual versus expected values, test fixtures, and any behavioural changes introduced by the commit."),
    Document("docs/dependencies.md", "Dependency resolution failures: inspect lockfile changes, pinned versions, Python/runtime compatibility, package index access, and whether the lockfile is in sync with project metadata."),
    Document("docs/configuration.md", "Missing environment variable or configuration error: compare required variables with CI configuration, confirm secret names are configured, and never print secret values in logs."),
    Document("docs/network.md", "Network or timeout failure: inspect service availability, retry policy, DNS/connectivity, request timeouts, and whether the test depends on a flaky external service."),
    Document("docs/lint.md", "Lint or formatting failure: run the same formatter and linter versions locally, inspect changed lines, and align CI commands with project configuration."),
]

TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_.-]{1,}")
STOPWORDS = {"this", "that", "with", "from", "were", "have", "your", "into", "then", "when", "what", "error", "failed"}


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS}


def retrieve(query: str, limit: int = 3) -> list[tuple[Document, float]]:
    """Rank documents by lexical overlap plus a small exact-phrase bonus."""
    q_tokens = tokens(query)
    if not q_tokens:
        return []
    ranked: list[tuple[Document, float]] = []
    query_lower = query.lower()
    for doc in CORPUS:
        d_tokens = tokens(doc.text)
        overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
        phrase_bonus = 0.15 if any(phrase in query_lower for phrase in _distinct_phrases(doc.text)) else 0.0
        score = min(1.0, overlap + phrase_bonus)
        if score > 0:
            ranked.append((doc, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def _distinct_phrases(text: str) -> list[str]:
    known = ["ModuleNotFoundError", "ImportError", "assertion failure", "lockfile", "environment variable", "timeout", "lint"]
    lower = text.lower()
    return [p for p in known if p.lower() in lower]
