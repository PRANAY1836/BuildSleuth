"""Lexical retrieval over documentation corpus and BM25 line-preserving log chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.models import EvidenceChunk
from app.security import normalize_log_text


@dataclass(frozen=True)
class Document:
    source: str
    text: str


# Curated starter knowledge corpus
CORPUS = [
    Document(
        "docs/python-imports.md",
        "ModuleNotFoundError or ImportError: verify the missing package is declared in requirements.txt or pyproject.toml, installed in the CI environment, and compatible with the active Python version.",
    ),
    Document(
        "docs/pytest.md",
        "Pytest assertion failures: inspect the first failing assertion, actual versus expected values, test fixtures, and any behavioural changes introduced by the commit.",
    ),
    Document(
        "docs/dependencies.md",
        "Dependency resolution failures: inspect lockfile changes, pinned versions, Python/runtime compatibility, package index access, and whether the lockfile is in sync with project metadata.",
    ),
    Document(
        "docs/configuration.md",
        "Missing environment variable or configuration error: compare required variables with CI configuration, confirm secret names are configured, and never print secret values in logs.",
    ),
    Document(
        "docs/network.md",
        "Network or timeout failure: inspect service availability, retry policy, DNS/connectivity, request timeouts, and whether the test depends on a flaky external service.",
    ),
    Document(
        "docs/lint.md",
        "Lint or formatting failure: run the same formatter and linter versions locally, inspect changedlines, and align CI commands with project configuration.",
    ),
]

TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_./:-]{1,}")
STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "were",
    "have",
    "your",
    "into",
    "then",
    "when",
    "what",
    "error",
    "failed",
}
ERROR_SIGNAL_KEYWORDS = {
    "modulenotfounderror",
    "importerror",
    "syntaxerror",
    "assertionerror",
    "keyerror",
    "resolutionimpossible",
    "connectionerror",
    "timeout",
    "traceback",
    "failed",
    "error",
}


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS}


def tokenize_list(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _distinct_phrases(text: str) -> list[str]:
    known = [
        "ModuleNotFoundError",
        "ImportError",
        "assertion failure",
        "lockfile",
        "environment variable",
        "timeout",
        "lint",
    ]
    lower = text.lower()
    return [p for p in known if p.lower() in lower]


def retrieve(query: str, limit: int = 3) -> list[tuple[Document, float]]:
    """Rank starter knowledge documents by lexical overlap plus exact-phrase bonus."""
    q_tokens = tokens(query)
    if not q_tokens:
        return []
    ranked: list[tuple[Document, float]] = []
    query_lower = query.lower()
    for doc in CORPUS:
        d_tokens = tokens(doc.text)
        overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
        phrase_bonus = (
            0.15 if any(phrase in query_lower for phrase in _distinct_phrases(doc.text)) else 0.0
        )
        score = min(1.0, overlap + phrase_bonus)
        if score > 0:
            ranked.append((doc, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def build_chunk_id(source_type: str, line_start: int, line_end: int, text: str) -> str:
    """Create a deterministic SHA-256 chunk ID from coordinates and normalized content."""
    digest = hashlib.sha256(
        f"{source_type}:{line_start}:{line_end}:{text}".encode("utf-8")
    ).hexdigest()[:12]
    return f"chk_{source_type}_{line_start}_{line_end}_{digest}"


def chunk_log_lines(
    raw_log: str,
    window_size: int = 6,
    overlap: int = 2,
    source_type: str = "ci_log",
    source_label: str = "ci_log",
) -> tuple[list[EvidenceChunk], bool]:
    """Normalize log, redact secrets, and split into overlapping line-numbered chunks."""
    normalized_text, timestamps, injection_detected = normalize_log_text(raw_log)
    lines = normalized_text.split("\n")
    if not lines:
        return [], injection_detected

    step = max(1, window_size - overlap)
    chunks: list[EvidenceChunk] = []

    for start_idx in range(0, len(lines), step):
        end_idx = min(len(lines), start_idx + window_size)
        snippet = "\n".join(lines[start_idx:end_idx]).strip()
        if not snippet:
            continue
        line_start = start_idx + 1
        line_end = end_idx
        ts_start = next((t for t in timestamps[start_idx:end_idx] if t is not None), None)
        chunk_id = build_chunk_id(source_type, line_start, line_end, snippet)
        chunks.append(
            EvidenceChunk(
                source_id=chunk_id,
                source_type=source_type,  # type: ignore[arg-type]
                source=f"{source_label}:L{line_start}-L{line_end}",
                text=snippet,
                line_start=line_start,
                line_end=line_end,
                timestamp_start=ts_start,
                score=0.0,
                retrieval_method="lexical_bm25",
            )
        )
        if end_idx == len(lines):
            break

    return chunks, injection_detected


def retrieve_evidence_chunks(
    raw_log: str,
    query: str,
    diff: str | None = None,
    top_k: int = 5,
) -> tuple[list[EvidenceChunk], bool]:
    """Retrieve line-numbered log and diff chunks ranked via BM25 + error signal density."""
    chunks, log_injection = chunk_log_lines(raw_log, source_type="ci_log", source_label="ci_log")
    diff_injection = False
    if diff and diff.strip():
        diff_chunks, diff_injection = chunk_log_lines(
            diff, source_type="commit_diff", source_label="git_diff"
        )
        chunks.extend(diff_chunks)

    for doc, doc_score in retrieve(raw_log + "\n" + (diff or ""), limit=2):
        doc_id = build_chunk_id("doc", 1, 1, doc.text)
        chunks.append(
            EvidenceChunk(
                source_id=doc_id,
                source_type="doc",
                source=doc.source,
                text=doc.text,
                line_start=1,
                line_end=1,
                score=round(doc_score, 4),
                retrieval_method="corpus_lexical",
            )
        )

    if not chunks:
        return [], log_injection or diff_injection

    tokenized_corpus = [tokenize_list(c.text) for c in chunks]
    query_tokens = tokenize_list(query) + list(ERROR_SIGNAL_KEYWORDS)

    if any(tokenized_corpus):
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query_tokens)
    else:
        scores = [0.0] * len(chunks)

    scored: list[EvidenceChunk] = []
    for chunk, bm25_score in zip(chunks, scores):
        tok_set = set(tokenize_list(chunk.text))
        err_hits = len(tok_set & ERROR_SIGNAL_KEYWORDS)
        final_score = round(float(bm25_score) + (err_hits * 1.25) + chunk.score, 4)
        scored.append(chunk.model_copy(update={"score": final_score}))

    scored.sort(key=lambda c: (-c.score, c.line_start))
    return scored[:top_k], (log_injection or diff_injection)