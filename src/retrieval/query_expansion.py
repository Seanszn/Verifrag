"""Conservative query expansion for broad legal research retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable, Iterable, Sequence


ExpansionMode = str
LLMQueryExpander = Callable[[str], Sequence[str]]

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)
_CASE_NAME_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z.&'-]+)\s+v\.\s+(?:[A-Z][A-Za-z.&'-]+)\b"
    r"|\b(?:In re|Ex parte|Matter of)\s+[A-Z][A-Za-z0-9.&' -]+\b"
)
_CITATION_RE = re.compile(
    r"\b\d+\s+(?:U\.?\s*S\.?|S\.?\s*Ct\.?|F\.?\s?\d?d|F\.?\s?Supp\.?\s?\d?d|"
    r"U\.?\s*S\.?\s*C\.?|C\.?\s*F\.?\s*R\.?)\s+\d+\b",
    re.IGNORECASE,
)
_AUTHORITY_SEEKING_RE = re.compile(
    r"\b(?:find|identify|what|which|any|examples?|authorit(?:y|ies)|cases?|precedent|"
    r"doctrine|rule|standard|test|factors?|relevant|involving|address(?:es|ing)?|"
    r"discuss(?:es|ing)?|deals?\s+with|appl(?:y|ies|ication))\b",
    re.IGNORECASE,
)

_DRIFTY_TERM_RE = re.compile(r"\b(?:v\.|versus|\d+\s+u\.?\s*s\.?|\d+\s+f\.?\s?\d?d)\b", re.IGNORECASE)

_RULE_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("standing",), ("article iii", "injury in fact", "causation", "redressability")),
    (("certiorari", "cert"), ("petition", "writ", "denied", "granted", "supreme court review")),
    (("injunction", "enjoin"), ("preliminary injunction", "irreparable harm", "equitable relief")),
    (("dismissal", "dismiss"), ("motion to dismiss", "failure to state a claim", "rule 12")),
    (("summary judgment",), ("genuine dispute", "material fact", "rule 56")),
    (("remand",), ("vacate", "reverse", "further proceedings", "jurisdiction")),
    (("damages",), ("remedy", "compensatory damages", "punitive damages", "relief")),
    (("penalty", "penalties"), ("sanction", "fine", "civil penalty", "consequence")),
    (("exclusion", "suppress", "suppression"), ("exclusionary rule", "motion to suppress", "fourth amendment")),
    (("sanction", "sanctions"), ("rule 11", "discipline", "penalty", "bad faith")),
    (("search", "seizure"), ("fourth amendment", "warrant", "probable cause", "reasonable suspicion")),
    (("due process",), ("notice", "hearing", "procedural due process", "substantive due process")),
    (("equal protection",), ("scrutiny", "classification", "fourteenth amendment")),
    (("first amendment", "speech"), ("free speech", "content based", "public forum", "strict scrutiny")),
    (("administrative", "agency"), ("arbitrary and capricious", "administrative procedure act", "deference")),
)

_AUTHORITY_TERMS = (
    "case",
    "authority",
    "precedent",
    "holding",
    "standard",
)

_CORPUS_FACET_KEYS = ("court_level", "court", "doc_type", "title", "section")
_ALLOWED_DOC_TYPES = {"case", "statute", "regulation"}


@dataclass(frozen=True)
class QueryExpansionPlan:
    original_query: str
    mode: ExpansionMode
    variants: tuple[str, ...]
    terms: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    status: str = "not_applied"
    drift_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "mode": self.mode,
            "original_query": self.original_query,
            "variants": list(self.variants),
            "terms": list(self.terms),
            "sources": list(self.sources),
            "drift_warnings": list(self.drift_warnings),
        }


@dataclass
class QueryExpansionConfig:
    mode: ExpansionMode = "hybrid"
    max_variants: int = 5
    max_terms: int = 16
    enable_for_broad_queries_only: bool = True
    llm_expander: LLMQueryExpander | None = None
    corpus_facet_keys: tuple[str, ...] = field(default_factory=lambda: _CORPUS_FACET_KEYS)


def build_query_expansion_plan(
    query: str,
    *,
    config: QueryExpansionConfig | None = None,
    corpus_metadata: Sequence[dict] | None = None,
) -> QueryExpansionPlan:
    cfg = config or QueryExpansionConfig()
    original = " ".join(str(query or "").split())
    mode = (cfg.mode or "raw").strip().lower()
    if not original:
        return QueryExpansionPlan(original, mode, (original,), status="not_applied:blank_query")
    if mode in {"raw", "none", "off", "disabled"}:
        return QueryExpansionPlan(original, mode, (original,), status="not_applied:mode_disabled")
    if _contains_named_authority(original):
        return QueryExpansionPlan(original, mode, (original,), status="not_applied:named_authority_query")
    if cfg.enable_for_broad_queries_only and not _looks_like_broad_legal_research_query(original):
        return QueryExpansionPlan(original, mode, (original,), status="not_applied:not_broad_research_query")

    sources: list[str] = []
    terms: list[str] = []
    if mode in {"rule", "hybrid", "reranked"}:
        rule_terms = _rule_terms(original)
        if rule_terms:
            terms.extend(rule_terms)
            sources.append("rule")
    if mode in {"corpus-aware", "hybrid", "reranked"}:
        corpus_terms = _corpus_terms(corpus_metadata or (), cfg.corpus_facet_keys)
        if corpus_terms:
            terms.extend(corpus_terms)
            sources.append("corpus-aware")
    if mode in {"llm", "hybrid", "reranked"} and cfg.llm_expander is not None:
        llm_terms = _llm_terms(original, cfg.llm_expander)
        if llm_terms:
            terms.extend(llm_terms)
            sources.append("llm")

    deduped_terms = _dedupe_terms(terms, original)[: max(0, int(cfg.max_terms))]
    variants = _build_variants(original, deduped_terms, max_variants=cfg.max_variants)
    warnings = tuple(term for term in deduped_terms if _DRIFTY_TERM_RE.search(term))
    if not deduped_terms or len(variants) == 1:
        return QueryExpansionPlan(
            original,
            mode,
            tuple(variants),
            terms=tuple(deduped_terms),
            sources=tuple(sources),
            status="not_applied:no_safe_expansions",
            drift_warnings=warnings,
        )
    return QueryExpansionPlan(
        original,
        mode,
        tuple(variants),
        terms=tuple(deduped_terms),
        sources=tuple(dict.fromkeys(sources)),
        status="applied",
        drift_warnings=warnings,
    )


def _contains_named_authority(query: str) -> bool:
    return bool(_CASE_NAME_RE.search(query) or _CITATION_RE.search(query))


def _looks_like_broad_legal_research_query(query: str) -> bool:
    lowered = query.lower()
    if len(_tokens(lowered)) < 3:
        return False
    return bool(_AUTHORITY_SEEKING_RE.search(lowered))


def _rule_terms(query: str) -> list[str]:
    lowered = query.lower()
    terms: list[str] = list(_AUTHORITY_TERMS)
    for triggers, expansions in _RULE_EXPANSIONS:
        if any(trigger in lowered for trigger in triggers):
            terms.extend(expansions)
    return terms


def _corpus_terms(metadata_rows: Sequence[dict], facet_keys: Iterable[str]) -> list[str]:
    terms: list[str] = []
    for metadata in metadata_rows[:8]:
        for key in facet_keys:
            value = metadata.get(key)
            if value is None:
                continue
            normalized = " ".join(str(value).replace("_", " ").split()).lower()
            if not normalized:
                continue
            if key == "doc_type" and normalized not in _ALLOWED_DOC_TYPES:
                continue
            if key in {"title", "section"}:
                terms.append(f"{key} {normalized}")
            else:
                terms.append(normalized)
    return terms


def _llm_terms(query: str, expander: LLMQueryExpander) -> list[str]:
    safe_terms: list[str] = []
    query_tokens = set(_tokens(query))
    for raw in expander(query) or ():
        candidate = " ".join(str(raw or "").split())
        if not candidate or _contains_named_authority(candidate):
            continue
        if len(candidate.split()) > 8:
            continue
        candidate_tokens = set(_tokens(candidate))
        if not candidate_tokens or candidate_tokens == query_tokens:
            continue
        safe_terms.append(candidate.lower())
    return safe_terms


def _dedupe_terms(terms: Sequence[str], query: str) -> list[str]:
    query_lower = query.lower()
    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        normalized = " ".join(str(term or "").split()).lower()
        if not normalized or normalized in seen or normalized in query_lower:
            continue
        if _contains_named_authority(normalized):
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _build_variants(original: str, terms: Sequence[str], *, max_variants: int) -> list[str]:
    variants = [original]
    cap = max(1, int(max_variants))
    if cap == 1 or not terms:
        return variants

    batch_size = max(3, min(5, len(terms)))
    for start in range(0, len(terms), batch_size):
        if len(variants) >= cap:
            break
        batch = terms[start : start + batch_size]
        variant = f"{original} {' '.join(batch)}"
        if variant not in variants:
            variants.append(variant)
    return variants


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
