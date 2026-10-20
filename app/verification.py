"""Stage 4: Deterministic citation existence, line-range, and claim support verification."""

from __future__ import annotations

from app.models import CitationItem, CitationVerificationReport, EvidenceChunk


def verify_citations(
    citations: list[CitationItem],
    evidence_chunks: list[EvidenceChunk],
) -> CitationVerificationReport:
    """Validate that every cited source_id, line range, and quoted anchor is backed by retrieved evidence."""
    if not citations:
        return CitationVerificationReport(
            total_citations=0,
            verified_citations=0,
            citation_precision=1.0,
            all_verified=True,
        )

    chunk_map: dict[str, EvidenceChunk] = {c.source_id: c for c in evidence_chunks}
    verified_count = 0
    invalid_ids: list[str] = []
    out_of_bounds: list[str] = []
    unsupported_claims: list[str] = []

    for item in citations:
        chunk = chunk_map.get(item.source_id)
        if chunk is None:
            invalid_ids.append(item.source_id)
            unsupported_claims.append(item.claim)
            continue

        # Check line range bounds
        if item.line_start < chunk.line_start or item.line_end > chunk.line_end:
            out_of_bounds.append(
                f"{item.source_id}: cited L{item.line_start}-L{item.line_end} outside chunk L{chunk.line_start}-L{chunk.line_end}"
            )
            unsupported_claims.append(item.claim)
            continue

        # Check quoted anchor text exists inside chunk if provided
        if item.quoted_Anchor and item.quoted_Anchor.strip():
            anchor_norm = item.quoted_Anchor.strip().lower()
            if anchor_norm not in chunk.text.lower():
                unsupported_claims.append(
                    f"{item.claim} (anchor '{item.quoted_Anchor}' not found in {item.source_id})"
                )
                continue

        verified_count += 1

    total = len(citations)
    precision = round(verified_count / total, 4) if total > 0 else 1.0
    all_ok = (
        len(invalid_ids) == 0
        and len(out_of_bounds) == 0
        and len(unsupported_claims) == 0
    )

    return CitationVerificationReport(
        total_citations=total,
        verified_citations=verified_count,
        citation_precision=precision,
        invalid_source_ids=invalid_ids,
        out_of_bounds_lines=out_of_bounds,
        unsupported_claims=unsupported_claims,
        all_verified=all_ok,
    )