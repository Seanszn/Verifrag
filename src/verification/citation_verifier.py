"""Semantic and citation-aware verification utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any
import re

import numpy as np
from sentence_transformers import SentenceTransformer, util

from src.ingestion.document import LegalChunk
from src.verification.claim_decomposer import Claim


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_CASE_CITATION_RE = re.compile(
    r"\b(?P<volume>\d+)\s+"
    r"(?P<reporter>U\.?\s*S\.?|S\.?\s*Ct\.?|F\.?\s*3d|F\.?\s*2d|F\.?\s*Supp\.?\s*\d*)"
    r"\s+(?P<page>\d+)\b",
    re.IGNORECASE,
)
_USC_CITATION_RE = re.compile(
    r"\b(?P<title>\d+)\s+U\.?\s*S\.?\s*C\.?\s*(?:§+\s*)?(?P<section>[\w\-.]+)\b",
    re.IGNORECASE,
)
_CASE_NAME_RE = re.compile(
    r"(?P<left>[A-Z][\w'&.-]+(?:\s+[A-Z][\w'&.-]+)*)\s+v\.?\s+"
    r"(?P<right>[A-Z][\w'&.-]+(?:\s+[A-Z][\w'&.-]+)*)"
)


@dataclass(frozen=True)
class ParsedCitation:
    """Structured citation extracted from claim text."""

    citation_type: str
    raw_text: str
    normalized_text: str
    case_name: str | None = None
    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    title: str | None = None
    section: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load and cache the sentence-transformer used for semantic matching."""
    return SentenceTransformer(model_name)


def verify_claim_against_text(
    claim: Claim,
    source_text: str,
    *,
    model: SentenceTransformer | None = None,
    chunk_size: int = 200,
) -> dict[str, Any]:
    """Verify a single claim against source text by nearest semantic chunk."""
    chunks = chunk_text(source_text, chunk_size=chunk_size)
    if not chunks:
        return _error_result(claim, "No source text")

    encoder = model or get_model()
    claim_embedding = encoder.encode(claim.text, convert_to_tensor=True)
    chunk_embeddings = encoder.encode(chunks, convert_to_tensor=True)

    scores = util.cos_sim(claim_embedding, chunk_embeddings)[0]
    score_values = _scores_to_array(scores)
    best_idx = int(np.argmax(score_values))
    best_chunk = chunks[best_idx]
    best_score = float(score_values[best_idx])

    return {
        "claim_id": claim.claim_id,
        "text": claim.text,
        "verdict": classify_score(best_score, claim),
        "confidence": round(best_score, 3),
        "evidence": best_chunk[:300],
        "source": claim.source,
        "claim_type": claim.claim_type,
        "certainty": claim.certainty,
    }


def verify_claims(
    claims: list[Claim],
    source_text: str,
    *,
    model: SentenceTransformer | None = None,
    chunk_size: int = 200,
) -> list[dict[str, Any]]:
    """Verify a batch of claims against a shared block of source text."""
    return [
        verify_claim_against_text(
            claim,
            source_text,
            model=model,
            chunk_size=chunk_size,
        )
        for claim in claims
    ]


def extract_citations(text: str) -> list[ParsedCitation]:
    """Extract case and statute citations from free-form text."""
    if not text or not text.strip():
        return []

    matches: list[tuple[int, ParsedCitation]] = []
    seen_spans: set[tuple[int, int]] = set()

    for match in _CASE_CITATION_RE.finditer(text):
        span = match.span()
        if span in seen_spans:
            continue
        seen_spans.add(span)
        reporter = _canonical_case_reporter(match.group("reporter"))
        raw_text = f"{match.group('volume')} {reporter} {match.group('page')}"
        case_name = _find_case_name_near_match(text, match.start())
        parsed = ParsedCitation(
            citation_type="case",
            raw_text=raw_text,
            normalized_text=_normalize_citation_key(raw_text),
            case_name=case_name,
            volume=match.group("volume"),
            reporter=reporter,
            page=match.group("page"),
        )
        matches.append((match.start(), parsed))

    for match in _USC_CITATION_RE.finditer(text):
        span = match.span()
        if span in seen_spans:
            continue
        seen_spans.add(span)
        raw_text = f"{match.group('title')} U.S.C. § {match.group('section')}"
        parsed = ParsedCitation(
            citation_type="statute",
            raw_text=raw_text,
            normalized_text=_normalize_citation_key(raw_text),
            title=match.group("title"),
            section=match.group("section"),
        )
        matches.append((match.start(), parsed))

    matches.sort(key=lambda item: item[0])
    return [parsed for _, parsed in matches]


def extract_claimed_proposition(text: str, citations: list[ParsedCitation] | None = None) -> str:
    """Remove citation scaffolding so support scoring focuses on the proposition."""
    proposition = text or ""
    parsed_citations = citations if citations is not None else extract_citations(text)

    for citation in parsed_citations:
        if citation.case_name:
            proposition = proposition.replace(citation.case_name, " ")
        proposition = proposition.replace(citation.raw_text, " ")

    proposition = re.sub(r"\(\d{4}\)", " ", proposition)
    proposition = re.sub(r"\b(?:see|under|cf\.?|accord|citing)\b", " ", proposition, flags=re.IGNORECASE)
    proposition = proposition.replace(",", " ")
    proposition = re.sub(r"\s+", " ", proposition).strip(" .;:-,")
    return proposition


def verify_citation_claim(
    claim: Claim,
    source_chunks: list[LegalChunk],
    *,
    model: SentenceTransformer | None = None,
) -> dict[str, Any]:
    """Verify that a claim cites the correct authority and states a supported proposition."""
    citations = extract_citations(claim.text)
    proposition = extract_claimed_proposition(claim.text, citations)

    base_result = {
        "claim_id": claim.claim_id,
        "text": claim.text,
        "claim_type": claim.claim_type,
        "source": claim.source,
        "certainty": claim.certainty,
        "parsed_citations": [citation.to_dict() for citation in citations],
        "claimed_proposition": proposition,
        "citation_exists": False,
        "citation_format_valid": bool(citations),
        "authority_found": False,
        "matched_citation": None,
        "matched_doc_id": None,
        "matched_chunk_id": None,
        "supporting_document": None,
        "candidate_chunk_ids": [],
        "confidence": 0.0,
        "evidence": "",
    }

    if not citations:
        return {
            **base_result,
            "verdict": "CITATION_ERROR",
            "explanation": "No legal citation found in claim text.",
        }

    primary_citation = citations[0]
    candidate_chunks = _candidate_chunks_for_citation(primary_citation, source_chunks)
    candidate_ids = [chunk.id for chunk in candidate_chunks]

    if not candidate_chunks:
        return {
            **base_result,
            "candidate_chunk_ids": candidate_ids,
            "verdict": "CITATION_ERROR",
            "explanation": f"No retrieved source matches citation {primary_citation.raw_text}.",
        }

    if not proposition:
        matched_chunk = candidate_chunks[0]
        return {
            **base_result,
            "citation_exists": True,
            "authority_found": True,
            "matched_citation": matched_chunk.citation,
            "matched_doc_id": matched_chunk.doc_id,
            "matched_chunk_id": matched_chunk.id,
            "supporting_document": _serialize_supporting_chunk(matched_chunk),
            "candidate_chunk_ids": candidate_ids,
            "confidence": 1.0,
            "evidence": matched_chunk.text[:300],
            "verdict": "VERIFIED",
            "explanation": "Citation matched retrieved authority.",
        }

    encoder = model or get_model()
    proposition_embedding = encoder.encode(proposition, convert_to_tensor=True)
    chunk_embeddings = encoder.encode([chunk.text for chunk in candidate_chunks], convert_to_tensor=True)
    scores = util.cos_sim(proposition_embedding, chunk_embeddings)[0]
    score_values = _scores_to_array(scores)
    best_idx = int(np.argmax(score_values))
    best_chunk = candidate_chunks[best_idx]
    best_score = float(score_values[best_idx])
    verdict, explanation = _classify_citation_support(best_score, primary_citation.raw_text)

    return {
        **base_result,
        "citation_exists": True,
        "authority_found": True,
        "matched_citation": best_chunk.citation,
        "matched_doc_id": best_chunk.doc_id,
        "matched_chunk_id": best_chunk.id,
        "supporting_document": _serialize_supporting_chunk(best_chunk),
        "candidate_chunk_ids": candidate_ids,
        "confidence": round(best_score, 3),
        "evidence": best_chunk.text[:300],
        "verdict": verdict,
        "explanation": explanation,
    }


def classify_score(score: float, claim: Claim) -> str:
    """Map a similarity score to a citation-verification verdict."""
    base_high = 0.75
    base_mid = 0.55

    if claim.certainty == "alleged":
        base_high -= 0.10
    elif claim.certainty == "found":
        base_high += 0.05

    if claim.claim_type == "holding":
        base_high += 0.05
    elif claim.claim_type == "fact":
        base_mid -= 0.05

    if score >= base_high:
        return "SUPPORTED"
    if score >= base_mid:
        return "PARTIAL"
    return "NOT SUPPORTED"


def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    """Split text into roughly fixed-size word chunks."""
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def _scores_to_array(scores: Any) -> np.ndarray:
    if hasattr(scores, "detach"):
        return scores.detach().cpu().numpy()
    return np.asarray(scores, dtype=float)


def _candidate_chunks_for_citation(
    citation: ParsedCitation,
    source_chunks: list[LegalChunk],
) -> list[LegalChunk]:
    citation_key = citation.normalized_text
    matches: list[LegalChunk] = []

    for chunk in source_chunks:
        metadata_key = _normalize_citation_key(chunk.citation or "")
        text_key = _normalize_citation_key(chunk.text)
        if metadata_key == citation_key or citation_key in text_key:
            matches.append(chunk)

    return matches


def _classify_citation_support(score: float, citation_text: str) -> tuple[str, str]:
    if score >= 0.75:
        return "VERIFIED", f"Citation {citation_text} matched the retrieved authority and supports the proposition."
    if score >= 0.55:
        return "WEAK", f"Citation {citation_text} matched the authority, but support for the proposition is limited."
    return "UNSUPPORTED", f"Citation {citation_text} matched the authority, but the retrieved text does not support the proposition."


def _serialize_supporting_chunk(chunk: LegalChunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.id,
        "citation": chunk.citation,
        "text_preview": chunk.text[:300],
    }


def _canonical_case_reporter(reporter: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]", "", reporter).upper()
    mapping = {
        "US": "U.S.",
        "SCT": "S. Ct.",
        "F3D": "F.3d",
        "F2D": "F.2d",
        "FSUPP": "F. Supp.",
        "FSUPP2D": "F. Supp. 2d",
        "FSUPP3D": "F. Supp. 3d",
    }
    return mapping.get(compact, re.sub(r"\s+", " ", reporter).strip())


def _find_case_name_near_match(text: str, citation_start: int) -> str | None:
    window_start = max(0, citation_start - 120)
    prefix = text[window_start:citation_start]
    matches = list(_CASE_NAME_RE.finditer(prefix))
    if not matches:
        return None
    return matches[-1].group(0).strip()


def _normalize_citation_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", (text or "")).upper()


def _error_result(claim: Claim, reason: str) -> dict[str, Any]:
    return {
        "claim_id": claim.claim_id,
        "text": claim.text,
        "verdict": "ERROR",
        "confidence": 0.0,
        "evidence": "",
        "reason": reason,
        "source": claim.source,
        "claim_type": claim.claim_type,
        "certainty": claim.certainty,
    }
