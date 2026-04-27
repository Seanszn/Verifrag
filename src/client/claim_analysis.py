"""Helpers for inspecting claim-level evidence in the client app."""

from __future__ import annotations

from typing import Any

from src.verification.claim_contract import support_level_from_verdict


SUPPORT_LEVEL_ORDER = ("supported", "possibly_supported", "unsupported")


def extract_claim_evaluations(interaction_detail: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(interaction_detail, dict):
        return []
    claims = interaction_detail.get("claims")
    if not isinstance(claims, list):
        return []
    return [claim for claim in claims if isinstance(claim, dict)]


def group_evidence_case_references_by_support_level(
    claims: list[dict[str, Any]] | None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[tuple[str, str, str, str], dict[str, Any]]] = {
        level: {} for level in SUPPORT_LEVEL_ORDER
    }

    for claim in claims or []:
        if not isinstance(claim, dict):
            continue
        support_level = _claim_support_level(claim)
        if support_level not in grouped:
            continue

        claim_text = _string_or_none(claim.get("text") or claim.get("claim_text"))
        for evidence in _claim_evidence_items(claim):
            case_ref = _case_reference_from_evidence(
                evidence,
                support_level=support_level,
                claim_text=claim_text,
            )
            if case_ref is None:
                continue

            key = _case_reference_dedupe_key(case_ref)
            existing = grouped[support_level].get(key)
            if existing is None:
                grouped[support_level][key] = case_ref
            else:
                _merge_case_reference(existing, case_ref)

    return {
        level: sorted(grouped[level].values(), key=_case_reference_sort_key)
        for level in SUPPORT_LEVEL_ORDER
    }


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


def _claim_support_level(claim: dict[str, Any]) -> str:
    annotation = claim.get("annotation")
    if isinstance(annotation, dict):
        support_level = annotation.get("support_level")
        if support_level in SUPPORT_LEVEL_ORDER:
            return str(support_level)

    verification = claim.get("verification")
    if isinstance(verification, dict):
        return support_level_from_verdict(verification.get("verdict"))
    return "unsupported"


def _claim_evidence_items(claim: dict[str, Any]) -> list[dict[str, Any]]:
    annotation = claim.get("annotation")
    evidence = annotation.get("evidence") if isinstance(annotation, dict) else None
    if not isinstance(evidence, list):
        evidence = claim.get("linked_citations")
    if not isinstance(evidence, list):
        return []
    return [item for item in evidence if isinstance(item, dict)]


def _case_reference_from_evidence(
    evidence: dict[str, Any],
    *,
    support_level: str,
    claim_text: str | None,
) -> dict[str, Any] | None:
    citation_value = evidence.get("citation")
    citation = citation_value if isinstance(citation_value, dict) else {}
    reporter_citation = _first_string(
        citation.get("citation"),
        citation_value if isinstance(citation_value, str) else None,
        evidence.get("reporter_citation"),
        evidence.get("case_citation"),
    )
    doc_id = _first_string(evidence.get("doc_id"), citation.get("doc_id"))
    chunk_id = _first_string(
        evidence.get("chunk_id"),
        citation.get("chunk_id"),
        citation.get("id"),
    )
    case_name = _first_string(citation.get("case_name"), evidence.get("case_name"))
    source_label = _first_string(
        evidence.get("source_label"),
        citation.get("source_label"),
        citation.get("source_file"),
        reporter_citation,
        case_name,
        doc_id,
        chunk_id,
    )

    if not any((case_name, reporter_citation, source_label, doc_id, chunk_id)):
        return None

    relationship = _string_or_none(evidence.get("relationship"))
    score = _coerce_score(evidence.get("score"))
    claim_texts = [claim_text] if claim_text else []
    relationships = [relationship] if relationship else []
    evidence_quote = _first_string(
        evidence.get("evidence_quote"),
        citation.get("evidence_quote"),
        citation.get("text"),
        citation.get("text_preview"),
    )
    evidence_quotes = [evidence_quote] if evidence_quote else []

    return {
        "support_level": support_level,
        "case_name": case_name,
        "reporter_citation": reporter_citation,
        "source_label": source_label,
        "relationship": relationship,
        "relationships": relationships,
        "score": score,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "claim_text": claim_text,
        "claim_texts": claim_texts,
        "evidence_quote": evidence_quote,
        "evidence_quotes": evidence_quotes,
    }


def _case_reference_dedupe_key(case_ref: dict[str, Any]) -> tuple[str, str, str, str]:
    source_key = (
        case_ref.get("source_label")
        or case_ref.get("reporter_citation")
        or case_ref.get("case_name")
        or ""
    )
    return (
        _case_reference_key_part(case_ref.get("support_level")),
        _case_reference_key_part(case_ref.get("doc_id")),
        _case_reference_key_part(case_ref.get("chunk_id")),
        _case_reference_key_part(source_key),
    )


def _merge_case_reference(current: dict[str, Any], incoming: dict[str, Any]) -> None:
    for field in ("case_name", "reporter_citation", "source_label", "doc_id", "chunk_id"):
        if not current.get(field) and incoming.get(field):
            current[field] = incoming[field]

    for relationship in incoming.get("relationships", []):
        _append_unique(current.setdefault("relationships", []), relationship)
    if not current.get("relationship") and incoming.get("relationship"):
        current["relationship"] = incoming["relationship"]

    for claim_text in incoming.get("claim_texts", []):
        _append_unique(current.setdefault("claim_texts", []), claim_text)
    if not current.get("claim_text") and incoming.get("claim_text"):
        current["claim_text"] = incoming["claim_text"]

    for evidence_quote in incoming.get("evidence_quotes", []):
        _append_unique(current.setdefault("evidence_quotes", []), evidence_quote)
    if not current.get("evidence_quote") and incoming.get("evidence_quote"):
        current["evidence_quote"] = incoming["evidence_quote"]

    incoming_score = _coerce_score(incoming.get("score"))
    current_score = _coerce_score(current.get("score"))
    if incoming_score is not None and (current_score is None or incoming_score > current_score):
        current["score"] = incoming_score


def _case_reference_sort_key(case_ref: dict[str, Any]) -> tuple[float, str, str, str]:
    score = _coerce_score(case_ref.get("score"))
    score_key = -score if score is not None else 1.0
    label = str(case_ref.get("case_name") or case_ref.get("source_label") or "")
    citation = str(case_ref.get("reporter_citation") or "")
    claim_text = str(case_ref.get("claim_text") or "")
    return (score_key, label, citation, claim_text)


def _case_reference_key_part(value: Any) -> str:
    return str(value or "").strip().casefold()


def _first_string(*values: Any) -> str | None:
    for value in values:
        text = _string_or_none(value)
        if text:
            return text
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _append_unique(values: list[Any], value: Any) -> None:
    if value and value not in values:
        values.append(value)


def _coerce_score(score: Any) -> float | None:
    if isinstance(score, (int, float)):
        return float(score)
    return None
