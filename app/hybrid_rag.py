"""Stage 3: Genuine Hybrid RAG combining Dense Embeddings, BM25, RRF, Metadata Filtering & Reranking."""

from __future__ import annotations

import hashlib
import re
import numpy as np
from rank_bm25 import BM25Okapi

from app.models import (
    EvidenceChunk,
    RetrievalFilter,
    RetrievalInspectRequest,
    RetrievalInspectResponse,
)
from app.retrieval import CORPUS, build_chunk_id, chunk_log_lines, tokenize_list

# Semantic concept clusters for CI failure domain projection in local dense embeddings
SEMANTIC_CONCEPTS: tuple[tuple[str, set[str]], ...] = (
    (
        "dependency_missing",
        {
            "modulenotfounderror",
            "importerror",
            "module",
            "named",
            "package",
            "install",
            "pip",
            "requirements",
            "pyproject",
            "node_modules",
            "cannot",
            "find",
            "dependency",
        },
    ),
    (
        "version_conflict",
        {
            "resolutionimpossible",
            "conflict",
            "satisfies",
            "version",
            "incompatible",
            "lockfile",
            "constraint",
            "pinned",
            "requires",
        },
    ),
    (
        "test_assertion",
        {
            "assertionerror",
            "assert",
            "pytest",
            "failed",
            "expected",
            "actual",
            "fixture",
            "unittest",
            "test",
            "traceback",
        },
    ),
    (
        "syntax_parse",
        {
            "syntaxerror",
            "indentationerror",
            "invalid",
            "syntax",
            "parse",
            "unexpected",
            "eof",
            "token",
        },
    ),
    (
        "config_env",
        {
            "keyerror",
            "environment",
            "variable",
            "secret",
            "config",
            "configuration",
            "unset",
            "missing",
            "validationerror",
            "env",
        },
    ),
    (
        "network_timeout",
        {
            "timeout",
            "timed",
            "connectionerror",
            "dns",
            "resolution",
            "refused",
            "unreachable",
            "socket",
            "502",
            "503",
            "504",
            "network",
        },
    ),
    (
        "lint_format",
        {
            "ruff",
            "flake8",
            "black",
            "eslint",
            "prettier",
            "mypy",
            "lint",
            "format",
            "style",
            "unused",
        },
    ),
)

FILE_PATH_RE = re.compile(r"([A-Za-z0-9_./\-]+\.(?:py|ts|tsx|js|json|yml|yaml|toml|txt|md))")


def _infer_chunk_metadata(
    chunk: EvidenceChunk,
    repository: str | None = None,
    commit_sha: str | None = None,
) -> EvidenceChunk:
    """Attach inferred file_path, failure_type, repository, and commit metadata to a chunk."""
    text_lower = chunk.text.lower()
    inferred_failure: str | None = None
    for concept_name, concept_words in SEMANTIC_CONCEPTS:
        if any(w in text_lower for w in list(concept_words)[:4]):
            inferred_failure = concept_name
            break

    file_match = FILE_PATH_RE.search(chunk.text)
    file_path = chunk.file_path or (file_match.group(1) if file_match else None)

    return chunk.model_copy(
        update={
            "repository": repository or chunk.repository,
            "commit_sha": commit_sha or chunk.commit_sha,
            "file_path": file_path,
            "failure_type": inferred_failure or chunk.failure_type,
        }
    )


class DenseEmbedder:
    """Computes L2-normalized dense embeddings for semantic retrieval.

    Uses domain-projected semantic dimensions + signed character/token n-gram feature hashing
    in NumPy so dense cosine similarity runs deterministically in <5ms on CPU without requiring
    external API calls or multi-GB GPU downloads during CI.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.backend_name = "numpy_semantic_ngram_256d"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        num_concepts = len(SEMANTIC_CONCEPTS)

        for row_idx, text in enumerate(texts):
            toks = tokenize_list(text)
            tok_set = set(toks)

            # 1. Explicit semantic concept dimensions (dims 0 .. num_concepts-1)
            for c_idx, (_name, words) in enumerate(SEMANTIC_CONCEPTS):
                overlap = len(tok_set & words)
                if overlap > 0:
                    matrix[row_idx, c_idx] = float(overlap) * 2.5

            # 2. Signed token + character trigram projections across remaining dimensions
            for tok in toks:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
                bucket = num_concepts + (h % (self.dim - num_concepts))
                sign = 1.0 if (h & 1) == 0 else -1.0
                matrix[row_idx, bucket] += sign * 1.0

                # Subword character 3-grams for resilient partial identifier matching
                if len(tok) >= 4:
                    for i in range(len(tok) - 2):
                        trigram = tok[i : i + 3]
                        th = int(hashlib.md5(trigram.encode("utf-8")).hexdigest()[:8], 16)
                        tbucket = num_concepts + (th % (self.dim - num_concepts))
                        tsign = 1.0 if (th & 1) == 0 else -1.0
                        matrix[row_idx, tbucket] += tsign * 0.35

        # L2 normalize rows for exact cosine similarity via dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return matrix / norms


def apply_metadata_filters(
    chunks: list[EvidenceChunk], filters: RetrievalFilter | None
) -> list[EvidenceChunk]:
    """Filter candidate chunks by repository, commit_sha, source_type, file_path, or failure_type."""
    if filters is None:
        return chunks

    filtered: list[EvidenceChunk] = []
    for chunk in chunks:
        if filters.repository and chunk.repository != filters.repository:
            continue
        if filters.commit_sha and chunk.commit_sha != filters.commit_sha:
            continue
        if filters.source_types and chunk.source_type not in filters.source_types:
            continue
        if filters.file_path_prefix and (
            not chunk.file_path or not chunk.file_path.startswith(filters.file_path_prefix)
        ):
            continue
        if filters.failure_type and chunk.failure_type != filters.failure_type:
            continue
        filtered.append(chunk)
    return filtered


def reciprocal_rank_fusion(
    dense_ranked: list[EvidenceChunk],
    bm25_ranked: list[EvidenceChunk],
    rrf_k: int = 60,
) -> list[EvidenceChunk]:
    """Fuse independently ranked Dense and BM25 lists using Reciprocal Rank Fusion (RRF)."""
    by_id: dict[str, EvidenceChunk] = {}
    dense_ranks: dict[str, int] = {}
    dense_scores: dict[str, float] = {}
    bm25_ranks: dict[str, int] = {}
    bm25_scores: dict[str, float] = {}

    for rank_idx, chunk in enumerate(dense_ranked, start=1):
        by_id[chunk.source_id] = chunk
        dense_ranks[chunk.source_id] = rank_idx
        dense_scores[chunk.source_id] = chunk.dense_score

    for rank_idx, chunk in enumerate(bm25_ranked, start=1):
        by_id[chunk.source_id] = chunk
        bm25_ranks[chunk.source_id] = rank_idx
        bm25_scores[chunk.source_id] = chunk.bm25_score

    fused_list: list[EvidenceChunk] = []
    for cid, base_chunk in by_id.items():
        d_rank = dense_ranks.get(cid, len(by_id) + 100)
        b_rank = bm25_ranks.get(cid, len(by_id) + 100)
        rrf_val = (1.0 / (rrf_k + d_rank)) + (1.0 / (rrf_k + b_rank))
        fused_list.append(
            base_chunk.model_copy(
                update={
                    "dense_score": round(dense_scores.get(cid, 0.0), 4),
                    "dense_rank": dense_ranks.get(cid),
                    "bm25_score": round(bm25_scores.get(cid, 0.0), 4),
                    "bm25_rank": bm25_ranks.get(cid),
                    "rrf_score": round(rrf_val, 6),
                    "score": round(rrf_val, 6),
                    "retrieval_method": "hybrid_rrf",
                }
            )
        )

    fused_list.sort(key=lambda c: (-c.rrf_score, -c.bm25_score, c.line_start))
    return [
        c.model_copy(update={"rrf_rank": idx}) for idx, c in enumerate(fused_list, start=1)
    ]


def cross_interaction_rerank(
    query: str,
    candidates: list[EvidenceChunk],
) -> list[EvidenceChunk]:
    """Rerank top RRF candidates using fine-grained query-document interaction signals."""
    query_tokens = set(tokenize_list(query))
    high_signal_patterns = re.compile(
        r"(ModuleNotFoundError|ImportError|SyntaxError|AssertionError|KeyError|"
        r"ResolutionImpossible|ConnectionError|TimeoutError|FAILED|Traceback|exit code [1-9])",
        re.IGNORECASE,
    )

    reranked: list[EvidenceChunk] = []
    for chunk in candidates:
        chunk_tokens = set(tokenize_list(chunk.text))
        overlap_count = len(query_tokens & chunk_tokens)
        overlap_ratio = overlap_count / max(1, len(query_tokens))
        signal_matches = len(high_signal_patterns.findall(chunk.text))
        has_file_ref = 1.0 if chunk.file_path else 0.0

        # Combine RRF base score with interaction precision score
        interaction_score = (
            (chunk.rrf_score * 25.0)
            + (overlap_ratio * 0.45)
            + (min(signal_matches, 4) * 0.22)
            + (has_file_ref * 0.08)
        )
        explanation = (
            f"RRF=#{chunk.rrf_rank} (dense=#{chunk.dense_rank} [{chunk.dense_score:.3f}], "
            f"bm25=#{chunk.bm25_rank} [{chunk.bm25_score:.2f}]), "
            f"error_signals={signal_matches}, query_overlap={overlap_count}"
        )
        reranked.append(
            chunk.model_copy(
                update={
                    "rerank_score": round(interaction_score, 4),
                    "score": round(interaction_score, 4),
                    "retrieval_method": "hybrid_rrf_rerank",
                    "explanation": explanation,
                }
            )
        )

    reranked.sort(key=lambda c: (-c.rerank_score, c.line_start))
    return reranked


def run_hybrid_retrieval(
    request: RetrievalInspectRequest,
) -> tuple[RetrievalInspectResponse, bool]:
    """Execute full Hybrid RAG pipeline (Ingest -> Filter -> Dense + BM25 -> RRF -> Rerank)."""
    chunks, log_inj = chunk_log_lines(
        request.log_text, source_type="ci_log", source_label="ci_log"
    )
    diff_inj = False
    if request.diff and request.diff.strip():
        diff_chunks, diff_inj = chunk_log_lines(
            request.diff, source_type="commit_diff", source_label="git_diff"
        )
        chunks.extend(diff_chunks)

    # Include curated documentation corpus
    for doc in CORPUS:
        doc_id = build_chunk_id("doc", 1, 1, doc.text)
        chunks.append(
            EvidenceChunk(
                source_id=doc_id,
                source_type="doc",
                source=doc.source,
                text=doc.text,
                line_start=1,
                line_end=1,
                file_path=doc.source,
                retrieval_method="corpus",
            )
        )

    enriched_chunks = [
        _infer_chunk_metadata(c, repository=request.repository, commit_sha=request.commit_sha)
        for c in chunks
    ]
    total_indexed = len(enriched_chunks)
    active_chunks = apply_metadata_filters(enriched_chunks, request.filters)

    embedder = DenseEmbedder()
    if not active_chunks:
        empty_resp = RetrievalInspectResponse(
            query=request.query,
            total_chunks_indexed=total_indexed,
            filtered_chunks_count=0,
            rrf_k=request.rrf_k,
            reranker_applied=request.enable_reranker,
            embedding_backend=embedder.backend_name,
            dense_top=[],
            bm25_top=[],
            rrf_fused=[],
            final_reranked=[],
        )
        return empty_resp, (log_inj or diff_inj)

    # 1. Dense Semantic Search (Cosine Similarity over L2-normalized vectors)
    augmented_query = f"{request.query} error failed exception traceback root cause"
    doc_embeddings = embedder.encode([c.text for c in active_chunks])
    query_embedding = embedder.encode([augmented_query])[0]
    cosine_scores = np.dot(doc_embeddings, query_embedding)

    dense_scored = [
        c.model_copy(
            update={
                "dense_score": round(float(score), 4),
                "score": round(float(score), 4),
                "retrieval_method": "dense_semantic",
            }
        )
        for c, score in zip(active_chunks, cosine_scores)
    ]
    dense_scored.sort(key=lambda c: (-c.dense_score, c.line_start))
    dense_ranked = [
        c.model_copy(update={"dense_rank": idx}) for idx, c in enumerate(dense_scored, start=1)
    ]

    # 2. Sparse Lexical Search (BM25Okapi)
    tokenized_corpus = [tokenize_list(c.text) for c in active_chunks]
    query_tokens = tokenize_list(augmented_query)
    if any(tokenized_corpus) and query_tokens:
        bm25 = BM25Okapi(tokenized_corpus)
        raw_bm25 = bm25.get_scores(query_tokens)
    else:
        raw_bm25 = [0.0] * len(active_chunks)

    bm25_scored = [
        c.model_copy(
            update={
                "bm25_score": round(float(bscore), 4),
                "score": round(float(bscore), 4),
                "retrieval_method": "sparse_bm25",
            }
        )
        for c, bscore in zip(active_chunks, raw_bm25)
    ]
    bm25_scored.sort(key=lambda c: (-c.bm25_score, c.line_start))
    bm25_ranked = [
        c.model_copy(update={"bm25_rank": idx}) for idx, c in enumerate(bm25_scored, start=1)
    ]

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_fused = reciprocal_rank_fusion(dense_ranked, bm25_ranked, rrf_k=request.rrf_k)

    # 4. Cross-Encoder / Interaction Reranking
    if request.enable_reranker:
        final_results = cross_interaction_rerank(augmented_query, rrf_fused)
    else:
        final_results = rrf_fused

    k = request.top_k
    return (
        RetrievalInspectResponse(
            query=request.query,
            total_chunks_indexed=total_indexed,
            filtered_chunks_count=len(active_chunks),
            rrf_k=request.rrf_k,
            reranker_applied=request.enable_reranker,
            embedding_backend=embedder.backend_name,
            dense_top=dense_ranked[:k],
            bm25_top=bm25_ranked[:k],
            rrf_fused=rrf_fused[:k],
            final_reranked=final_results[:k],
        ),
        (log_inj or diff_inj),
    )