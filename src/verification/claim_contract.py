"""Helpers for producing a stable client-facing claim contract."""

from __future__ import annotations

from typing import Any


SUPPORTED_SUPPORT_LEVELS = {"VERIFIED", "SUPPORTED"}
POSSIBLE_SUPPORT_LEVELS = {"POSSIBLE_SUPPORT"}
UNSUPPORTED_EXPLANATION = "Claim verification results are unavailable for this claim."


def support_level_from_verdict(verdict: Any) -> str:
    normalized = str(verdict or "").strip().upper()
    if normalized in SUPPORTED_SUPPORT_LEVELS:
        return "supported"
    if normalized in POSSIBLE_SUPPORT_LEVELS:
        return "possibly_supported"
    return "unsupported"


def build_claim_citation_links(
    claims: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    citation_by_chunk_id = {
        str(citation.get("id") or citation.get("chunk_id")): citation
        for citation in citations
        if isinstance(citation, dict) and (citation.get("id") or citation.get("chunk_id"))
    }
    links: list[dict[str, Any]] = []

    for claim in claims:
        verification = claim.get("verification")
        if not isinstance(verification, dict):
            continue

        link_specs = [
            (
                "supporting",
                verification.get("best_supporting_chunk") or verification.get("best_chunk"),
                verification.get("best_supporting_score", verification.get("final_score")),
            ),
            (
                "contradicting",
                verification.get("best_contradicting_chunk"),
                verification.get("best_contradiction_score"),
            ),
        ]
        for relationship, chunk, score in link_specs:
            if not isinstance(chunk, dict):
                continue

            chunk_id = chunk.get("id") or chunk.get("chunk_id")
            if not chunk_id:
                continue

            citation = citation_by_chunk_id.get(str(chunk_id), chunk)
            links.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "relationship": relationship,
                    "score": _coerce_score(score),
                    "chunk_id": chunk_id,
                    "doc_id": citation.get("doc_id"),
                    "source_label": (
                        citation.get("citation")
                        or citation.get("source_file")
                        or citation.get("source_label")
                        or citation.get("doc_id")
                    ),
                    "citation": citation,
                }
            )
    return links


def attach_claim_citation_links(
    claims: list[dict[str, Any]],
    claim_citation_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    links_by_claim_id: dict[str, list[dict[str, Any]]] = {}
    for link in claim_citation_links:
        claim_id = link.get("claim_id")
        if isinstance(claim_id, str) and claim_id:
            links_by_claim_id.setdefault(claim_id, []).append(link)

    enriched_claims: list[dict[str, Any]] = []
    for claim in claims:
        enriched_claim = dict(claim)
        claim_id = enriched_claim.get("claim_id")
        linked = links_by_claim_id.get(claim_id, []) if isinstance(claim_id, str) else []
        enriched_claim["linked_citations"] = _sort_evidence_links(linked)
        enriched_claims.append(enriched_claim)
    return enriched_claims


def normalize_claims_for_frontend(
    claims: list[dict[str, Any]],
    *,
    citations: list[dict[str, Any]] | None = None,
    claim_citation_links: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_claims = [dict(claim) for claim in claims if isinstance(claim, dict)]
    if claim_citation_links is None:
        resolved_links = build_claim_citation_links(normalized_claims, citations or [])
    else:
        resolved_links = [dict(link) for link in claim_citation_links if isinstance(link, dict)]

    claims_with_links = attach_claim_citation_links(normalized_claims, resolved_links)
    for claim in claims_with_links:
        claim["annotation"] = build_claim_annotation(claim)

    return claims_with_links, resolved_links


def build_claim_annotation(claim: dict[str, Any]) -> dict[str, Any]:
    verification = claim.get("verification")
    if not isinstance(verification, dict):
        verification = {}

    support_level = support_level_from_verdict(verification.get("verdict"))
    explanation = verification.get("verdict_explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = UNSUPPORTED_EXPLANATION

    evidence = claim.get("linked_citations")
    if not isinstance(evidence, list):
        evidence = []

    return {
        "support_level": support_level,
        "explanation": explanation,
        "response_span": _build_response_span(claim),
        "evidence": [
            {
                "relationship": link.get("relationship"),
                "score": _coerce_score(link.get("score")),
                "chunk_id": link.get("chunk_id"),
                "doc_id": link.get("doc_id"),
                "source_label": link.get("source_label"),
                "citation": _coerce_citation_payload(link.get("citation")),
            }
            for link in _sort_evidence_links(evidence)
        ],
    }


def _build_response_span(claim: dict[str, Any]) -> dict[str, Any] | None:
    span = claim.get("span")
    if not isinstance(span, dict):
        return None

    start_char = span.get("start_char")
    end_char = span.get("end_char")
    if not isinstance(start_char, int) or not isinstance(end_char, int):
        return None

    return {
        "doc_id": span.get("doc_id"),
        "para_id": span.get("para_id"),
        "sent_id": span.get("sent_id"),
        "start_char": start_char,
        "end_char": end_char,
        "text": claim.get("text"),
    }


def _sort_evidence_links(links: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(link) for link in links if isinstance(link, dict)],
        key=_evidence_sort_key,
    )


def _evidence_sort_key(link: dict[str, Any]) -> tuple[int, float, str]:
    relationship = str(link.get("relationship") or "")
    relationship_priority = 0 if relationship == "supporting" else 1
    score = _coerce_score(link.get("score")) or 0.0
    source_label = str(link.get("source_label") or "")
    return (relationship_priority, -score, source_label)


def _coerce_citation_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_score(score: Any) -> float | None:
    if isinstance(score, (int, float)):
        return float(score)
    return None
