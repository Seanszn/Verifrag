"""Helpers for inspecting claim-level evidence in the client app."""

from __future__ import annotations

from typing import Any


def extract_claim_evaluations(interaction_detail: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(interaction_detail, dict):
        return []
    claims = interaction_detail.get("claims")
    if not isinstance(claims, list):
        return []
    return [claim for claim in claims if isinstance(claim, dict)]


def find_cases_for_claim(claim: dict[str, Any]) -> dict[str, Any]:
    verification = claim.get("verification") if isinstance(claim, dict) else None
    if not isinstance(verification, dict):
        verification = {}

    annotation = claim.get("annotation") if isinstance(claim, dict) else None
    if not isinstance(annotation, dict):
        annotation = {}

    linked_citations = claim.get("linked_citations") if isinstance(claim, dict) else None
    if not isinstance(linked_citations, list):
        linked_citations = annotation.get("evidence")
    if isinstance(linked_citations, list):
        supporting_case = _select_linked_case(linked_citations, relationship="supporting")
        contradicting_case = _select_linked_case(linked_citations, relationship="contradicting")
    else:
        supporting_case = None
        contradicting_case = None

    if supporting_case is None:
        supporting_case = _build_case_reference(
            verification.get("best_supporting_chunk") or verification.get("best_chunk"),
            relation="support",
            score=verification.get("best_supporting_score", verification.get("final_score")),
        )
    if contradicting_case is None:
        contradicting_case = _build_case_reference(
            verification.get("best_contradicting_chunk"),
            relation="contradiction",
            score=verification.get("best_contradiction_score"),
        )

    return {
        "claim_id": claim.get("claim_id"),
        "claim_text": claim.get("text") or claim.get("claim_text"),
        "verdict": verification.get("verdict"),
        "support_level": annotation.get("support_level"),
        "support_explanation": annotation.get("explanation"),
        "supporting_case": supporting_case,
        "contradicting_case": contradicting_case,
    }


def find_cases_for_interaction(interaction_detail: dict[str, Any]) -> dict[str, Any]:
    interaction = interaction_detail.get("interaction") if isinstance(interaction_detail, dict) else {}
    claim_summaries = [find_cases_for_claim(claim) for claim in extract_claim_evaluations(interaction_detail)]

    supporting_cases = [
        summary["supporting_case"]
        for summary in claim_summaries
        if isinstance(summary.get("supporting_case"), dict)
    ]
    contradicting_cases = [
        summary["contradicting_case"]
        for summary in claim_summaries
        if isinstance(summary.get("contradicting_case"), dict)
    ]

    return {
        "interaction_id": interaction.get("id"),
        "claims": claim_summaries,
        "top_supporting_cases": _dedupe_and_sort_cases(supporting_cases),
        "top_contradicting_cases": _dedupe_and_sort_cases(contradicting_cases),
    }


def _build_case_reference(
    chunk: dict[str, Any] | None,
    *,
    relation: str,
    score: Any,
) -> dict[str, Any] | None:
    if not isinstance(chunk, dict):
        return None

    numeric_score = _coerce_score(score)
    source_label = (
        chunk.get("citation")
        or chunk.get("case_name")
        or chunk.get("source_file")
        or chunk.get("doc_id")
        or chunk.get("chunk_id")
        or chunk.get("id")
    )
    text_preview = chunk.get("text_preview") or chunk.get("text")

    return {
        "relation": relation,
        "doc_id": chunk.get("doc_id"),
        "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
        "case_name": chunk.get("case_name"),
        "citation": chunk.get("citation"),
        "court": chunk.get("court"),
        "court_level": chunk.get("court_level"),
        "doc_type": chunk.get("doc_type"),
        "source_label": source_label,
        "score": numeric_score,
        "text_preview": text_preview,
    }


def _select_linked_case(
    linked_citations: list[dict[str, Any]],
    *,
    relationship: str,
) -> dict[str, Any] | None:
    matching_links = [
        link for link in linked_citations if link.get("relationship") == relationship
    ]
    if not matching_links:
        return None

    best_link = max(matching_links, key=_case_sort_key)
    citation = best_link.get("citation")
    relation = "support" if relationship == "supporting" else "contradiction"
    return _build_case_reference(
        citation if isinstance(citation, dict) else None,
        relation=relation,
        score=best_link.get("score"),
    )


def _dedupe_and_sort_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for case in cases:
        key = (case.get("relation"), case.get("doc_id"), case.get("chunk_id"))
        current = best_by_key.get(key)
        if current is None or _case_sort_key(case) > _case_sort_key(current):
            best_by_key[key] = case
    return sorted(best_by_key.values(), key=_case_sort_key, reverse=True)


def _case_sort_key(case: dict[str, Any]) -> tuple[float, str]:
    return (_coerce_score(case.get("score")) or 0.0, str(case.get("source_label") or ""))


def _coerce_score(score: Any) -> float | None:
    if isinstance(score, (int, float)):
        return float(score)
    return None
