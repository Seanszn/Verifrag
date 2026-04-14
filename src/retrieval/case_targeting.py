"""Helpers for extracting and enforcing target-case retrieval scope."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence, TypeVar

from src.ingestion.document import LegalChunk


_QUERY_CASE_PATTERNS = (
    re.compile(
        r"\bIn\s+(?P<case>[A-Z][A-Za-z0-9&.,'â€™()\-:/ ]+?\s+v\.\s+[A-Z][A-Za-z0-9&.,'â€™()\-:/ ]+?)(?=,|\?|$)",
    ),
    re.compile(
        r"\bWhat did\s+(?P<case>[A-Z][A-Za-z0-9&.,'â€™()\-:/ ]+?\s+v\.\s+[A-Z][A-Za-z0-9&.,'â€™()\-:/ ]+?)\s+(?:hold|decide)\b",
    ),
    re.compile(
        r"(?P<case>[A-Z][A-Za-z0-9&.,'â€™()\-:/ ]+?\s+v\.\s+[A-Z][A-Za-z0-9&.,'â€™()\-:/ ]+?)(?=,|\?|$)",
    ),
)
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_WHITESPACE_RE = re.compile(r"\s+")
_REVISION_SUFFIX_RE = re.compile(r"\s+revisions?\s*:?\s*\d{1,2}/\d{1,2}/\d{2,4}\b", re.IGNORECASE)

_ABBREVIATIONS = (
    (re.compile(r"\bcorporation\b"), "corp"),
    (re.compile(r"\bcorp\b"), "corp"),
    (re.compile(r"\bincorporated\b"), "inc"),
    (re.compile(r"\binc\b"), "inc"),
    (re.compile(r"\bcompany\b"), "co"),
    (re.compile(r"\bco\b"), "co"),
    (re.compile(r"\blimited\b"), "ltd"),
    (re.compile(r"\bltd\b"), "ltd"),
)
_CASE_SEPARATOR_RE = re.compile(r"\bv\b")
_PARTY_TOKEN_RE = re.compile(r"[a-z0-9]+")
_IGNORABLE_PARTY_SUFFIXES = {"corp", "inc", "co", "ltd", "llc", "pllc", "pc", "lp", "llp"}

T = TypeVar("T")


def extract_target_case_name(query: str) -> str | None:
    """Best-effort extraction of a target case name from a user query."""
    if not query:
        return None

    normalized_query = " ".join(query.split())
    for pattern in _QUERY_CASE_PATTERNS:
        match = pattern.search(normalized_query)
        if match is not None:
            return match.group("case").strip(" ,?")
    return None


def normalize_case_name(case_name: str | None, *, strip_revision_suffix: bool = True) -> str:
    """Normalize case names for conservative equality checks."""
    if not case_name:
        return ""

    normalized = " ".join(str(case_name).split()).lower().replace("â€™", "'")
    if strip_revision_suffix:
        normalized = _REVISION_SUFFIX_RE.sub("", normalized)
    normalized = _PUNCT_RE.sub(" ", normalized)
    for pattern, replacement in _ABBREVIATIONS:
        normalized = pattern.sub(replacement, normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def canonical_case_key(case_name: str | None) -> str:
    """Collapse a case name to a revision-insensitive canonical identity."""
    normalized = normalize_case_name(case_name)
    if not normalized:
        return ""

    parties = _split_case_parties(normalized)
    if parties is None:
        return normalized

    canonical_parties: list[str] = []
    for party in parties:
        tokens = _normalized_party_tokens(party)
        if not tokens:
            return normalized
        canonical_parties.append(" ".join(tokens))
    return " v ".join(canonical_parties)


def case_match_rank(target_case: str | None, candidate_case: str | None) -> int:
    """Return a stable match priority for two case names."""
    target_raw = normalize_case_name(target_case, strip_revision_suffix=False)
    candidate_raw = normalize_case_name(candidate_case, strip_revision_suffix=False)
    if target_raw and target_raw == candidate_raw:
        return 2

    target_canonical = canonical_case_key(target_case)
    candidate_canonical = canonical_case_key(candidate_case)
    if target_canonical and target_canonical == candidate_canonical:
        return 1
    return 0


def case_names_match(target_case: str | None, candidate_case: str | None) -> bool:
    """Return True when two case names refer to the same target case."""
    return case_match_rank(target_case, candidate_case) > 0


def canonical_doc_family_key(
    *,
    case_name: str | None,
    citation: str | None = None,
    date_decided=None,
    court_level: str | None = None,
    doc_id: str | None = None,
) -> str:
    """Build a revision-insensitive document-family key."""
    case_key = canonical_case_key(case_name)
    if case_key:
        parts = [f"case:{case_key}"]
        normalized_date = _normalize_date_value(date_decided)
        if normalized_date:
            parts.append(f"date:{normalized_date}")
        return "|".join(parts)

    parts: list[str] = []
    normalized_citation = _normalize_free_text(citation)
    if normalized_citation:
        parts.append(f"citation:{normalized_citation}")

    normalized_date = _normalize_date_value(date_decided)
    if normalized_date:
        parts.append(f"date:{normalized_date}")

    normalized_court = _normalize_free_text(court_level)
    if normalized_court:
        parts.append(f"court:{normalized_court}")

    if parts:
        return "|".join(parts)
    if doc_id:
        return f"doc:{doc_id}"
    return ""


def canonical_chunk_key(
    *,
    case_name: str | None,
    chunk_index,
    text: str | None,
    citation: str | None = None,
    date_decided=None,
    court_level: str | None = None,
    doc_id: str | None = None,
) -> str:
    """Build a stable chunk key that collapses duplicate revised documents."""
    family_key = canonical_doc_family_key(
        case_name=case_name,
        citation=citation,
        date_decided=date_decided,
        court_level=court_level,
        doc_id=doc_id,
    )

    try:
        chunk_position = str(int(chunk_index))
    except (TypeError, ValueError):
        chunk_position = "0"

    fingerprint = _text_fingerprint(text)
    key_parts = [
        family_key or f"doc:{doc_id or ''}",
        f"chunk:{chunk_position}",
    ]
    if fingerprint:
        key_parts.append(f"text:{fingerprint}")
    return "|".join(key_parts)


def filter_chunks_to_target_case(
    query: str,
    chunks: Sequence[LegalChunk],
    *,
    limit: int | None = None,
) -> list[LegalChunk]:
    """Restrict retrieval results to the target case when a confident match exists."""
    ordered = list(chunks)
    target_case = extract_target_case_name(query)
    if not target_case:
        return ordered[:limit] if limit is not None else ordered

    ranked_matches = [
        (case_match_rank(target_case, getattr(chunk, "case_name", None)), index, chunk)
        for index, chunk in enumerate(ordered)
    ]
    matches = [
        chunk
        for match_rank, _, chunk in sorted(
            ranked_matches,
            key=lambda item: (-item[0], item[1]),
        )
        if match_rank > 0
    ]
    if not matches:
        return []
    return matches[:limit] if limit is not None else matches


def filter_search_hits_to_target_case(
    query: str,
    hits: Iterable[T],
    *,
    metadata_case_name,
    limit: int | None = None,
) -> list[T]:
    """Restrict raw search hits to the target case when the query names one."""
    ordered = list(hits)
    target_case = extract_target_case_name(query)
    if not target_case:
        return ordered[:limit] if limit is not None else ordered

    ranked_matches = [
        (case_match_rank(target_case, metadata_case_name(hit)), index, hit)
        for index, hit in enumerate(ordered)
    ]
    matches = [
        hit
        for match_rank, _, hit in sorted(
            ranked_matches,
            key=lambda item: (-item[0], item[1]),
        )
        if match_rank > 0
    ]
    if not matches:
        return []
    return matches[:limit] if limit is not None else matches


def dedupe_chunks_by_canonical_identity(
    chunks: Sequence[LegalChunk],
    *,
    limit: int | None = None,
) -> list[LegalChunk]:
    """Collapse duplicate revised chunks while preserving distinct chunk positions."""
    deduped: list[LegalChunk] = []
    seen_keys: set[str] = set()

    for chunk in chunks:
        key = canonical_chunk_key(
            case_name=getattr(chunk, "case_name", None),
            chunk_index=getattr(chunk, "chunk_index", 0),
            text=getattr(chunk, "text", ""),
            citation=getattr(chunk, "citation", None),
            date_decided=getattr(chunk, "date_decided", None),
            court_level=getattr(chunk, "court_level", None),
            doc_id=getattr(chunk, "doc_id", None),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(chunk)
        if limit is not None and len(deduped) >= limit:
            break

    return deduped


def dedupe_search_hits_by_canonical_identity(
    hits: Iterable[T],
    *,
    metadata_doc_id,
    metadata_chunk_index,
    metadata_text,
    metadata_case_name=lambda item: None,
    metadata_citation=lambda item: None,
    metadata_date_decided=lambda item: None,
    metadata_court_level=lambda item: None,
    limit: int | None = None,
) -> list[T]:
    """Collapse duplicate revised search hits while preserving the best-ranked variant."""
    deduped: list[T] = []
    seen_keys: set[str] = set()

    for hit in hits:
        key = canonical_chunk_key(
            case_name=metadata_case_name(hit),
            chunk_index=metadata_chunk_index(hit),
            text=metadata_text(hit),
            citation=metadata_citation(hit),
            date_decided=metadata_date_decided(hit),
            court_level=metadata_court_level(hit),
            doc_id=metadata_doc_id(hit),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(hit)
        if limit is not None and len(deduped) >= limit:
            break

    return deduped


def _split_case_parties(case_name: str) -> tuple[str, str] | None:
    parts = [part.strip() for part in _CASE_SEPARATOR_RE.split(case_name, maxsplit=1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _party_names_match(target_party: str, candidate_party: str) -> bool:
    if target_party == candidate_party:
        return True

    target_tokens = _normalized_party_tokens(target_party)
    candidate_tokens = _normalized_party_tokens(candidate_party)
    if not target_tokens or not candidate_tokens:
        return False
    return target_tokens == candidate_tokens


def _normalized_party_tokens(party_name: str) -> tuple[str, ...]:
    tokens = _PARTY_TOKEN_RE.findall(party_name)
    while tokens and tokens[-1] in _IGNORABLE_PARTY_SUFFIXES:
        tokens.pop()
    return tuple(tokens)


def _normalize_free_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = _PUNCT_RE.sub(" ", str(value).lower())
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _normalize_date_value(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _text_fingerprint(text: str | None) -> str:
    normalized = _normalize_free_text(text)
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
