"""
Algorithm 2: Deterministic claim decomposition for legal text.

This implementation is intentionally non-LLM and lightweight:
- sentence splitting
- clause splitting
- attribution detection
- typed claim construction
- source span tracing
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_PROTECTED_PERIOD = "\x00"
_LEGAL_ABBREVIATION_RE = re.compile(
    r"\b(?:U\.\s*S\.\s*C|U\.\s*S|S\.\s*Ct|F\.\s*(?:3d|2d|Supp)|"
    r"No|Nos|App|Id|e\.g|i\.e|v)\.",
    re.IGNORECASE,
)
_CLAIM_CITATION_PREFIX_RE = re.compile(r"^(?:\[\d+\]\s*)+")
_CLAIM_MARKDOWN_HEADING_RE = re.compile(
    r"^\*{0,2}(?:short answer|analysis|limits)\s*:?\*{0,2}\s*:?\s*",
    re.IGNORECASE,
)
_CLAIM_BULLET_PREFIX_RE = re.compile(r"^(?:[-*]\s+)+")
_CLAIM_CONTEXT_CITATION_RE = re.compile(r"\s*\[\d+(?:\s*[-,]\s*\d+)*\]")
_CLAIM_MARKDOWN_EMPHASIS_RE = re.compile(r"\*{1,2}([^*]+?)\*{1,2}")
_HEDGE_RE = re.compile(
    r"\b(may|might|could|possibly|likely|arguably|appears|suggests)\b",
    re.IGNORECASE,
)
_HOLDING_RE = re.compile(
    r"\b(held|hold|ruled|found|concluded|determined|vacated|affirmed|reversed)\b",
    re.IGNORECASE,
)
_STATUTE_RE = re.compile(r"\b\d+\s+U\.?\s*S\.?\s*C\.?\b", re.IGNORECASE)
_CITATION_RE = re.compile(r"\b\d+\s+U\.?\s*S\.?\s+\d+\b")
_ATTRIBUTION_RE = re.compile(
    r"^(?P<speaker>[A-Z][\w .,'-]{1,120}?)\s+"
    r"(?P<verb>alleges|alleged|argues|argued|claims|claimed|contends|"
    r"contended|asserts|asserted|states|stated|testifies|testified|said)"
    r"\s+that\s+(?P<content>.+)$",
    re.IGNORECASE,
)
_AND_SPLIT_RE = re.compile(r"\s+and\s+", re.IGNORECASE)
_COMMON_VERB_LEMMAS = {
    "accept",
    "affirm",
    "allege",
    "appeal",
    "argue",
    "arrest",
    "assert",
    "award",
    "claim",
    "conclude",
    "contend",
    "deny",
    "determine",
    "dismiss",
    "enter",
    "file",
    "find",
    "grant",
    "hold",
    "leave",
    "order",
    "receive",
    "remand",
    "request",
    "return",
    "reverse",
    "rule",
    "say",
    "seek",
    "state",
    "submit",
    "testify",
    "vacate",
}


@dataclass(frozen=True)
class SpanRef:
    """Original location of a claim in source text."""

    doc_id: str
    para_id: int
    sent_id: int
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Claim:
    """Atomic claim ready for downstream NLI verification."""

    claim_id: str
    text: str
    claim_type: str
    source: str
    certainty: str
    doc_section: str
    span: SpanRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "claim_type": self.claim_type,
            "source": self.source,
            "certainty": self.certainty,
            "doc_section": self.doc_section,
            "span": {
                "doc_id": self.span.doc_id,
                "para_id": self.span.para_id,
                "sent_id": self.span.sent_id,
                "start_char": self.span.start_char,
                "end_char": self.span.end_char,
            },
        }


def load_document(path: str) -> dict[str, Any]:
    """
    Load a document from txt/json/jsonl.

    For JSONL, returns the first record in the file.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return {"id": file_path.stem, "full_text": file_path.read_text(encoding="utf-8")}
    if suffix == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON document must be an object.")
        return data
    if suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("JSONL record must be an object.")
                return data
        raise ValueError("JSONL file is empty.")

    raise ValueError(f"Unsupported file type: {suffix}")


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving sentence punctuation."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def assign_rhetorical_roles(document: Mapping[str, Any]) -> str:
    """
    Lightweight section tagging.
    """
    text = normalize_text(str(document.get("full_text", "")))
    lower = text[:500].lower()
    if "held:" in lower or "we hold" in lower or "holding" in lower:
        return "holding"
    if "facts" in lower:
        return "facts"
    return "body"


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like spans using punctuation boundaries."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    protected = _protect_sentence_abbreviations(normalized)
    return [
        _restore_sentence_abbreviations(m.group(0).strip())
        for m in _SENTENCE_RE.finditer(protected)
        if m.group(0).strip()
    ]


def split_clauses(sentence: str) -> list[str]:
    """
    Split sentence into simpler clauses.
    """
    sentence = sentence.strip()
    if not sentence:
        return []

    clauses: list[str] = []
    for segment in re.split(r"\s*;\s*", sentence):
        segment = segment.strip()
        if not segment:
            continue
        clauses.extend(_split_conjoined_predicates(segment))
    return clauses


def _split_conjoined_predicates(clause: str) -> list[str]:
    trimmed = clause.strip().rstrip(".")
    parts = _AND_SPLIT_RE.split(trimmed, maxsplit=1)
    if len(parts) != 2:
        return [_ensure_terminal_period(clause)]

    first_part, second_part = (part.strip() for part in parts)
    if not first_part or not second_part:
        return [_ensure_terminal_period(clause)]

    subject_predicate = _split_subject_predicate(first_part)
    if subject_predicate is None or not _starts_with_verb(second_part):
        return [_ensure_terminal_period(clause)]

    subject, predicate = subject_predicate
    first = f"{subject} {predicate}"
    second = f"{subject} {second_part}"
    return [_ensure_terminal_period(first), _ensure_terminal_period(second)]


def _split_subject_predicate(text: str) -> Optional[tuple[str, str]]:
    tokens = text.split()
    if len(tokens) < 3:
        return None

    for index in range(len(tokens) - 2, 0, -1):
        if _looks_like_verb(tokens[index]):
            subject = " ".join(tokens[:index]).strip()
            predicate = " ".join(tokens[index:]).strip()
            if subject and predicate:
                return subject, predicate
    return None


def _starts_with_verb(text: str) -> bool:
    tokens = text.split()
    if not tokens:
        return False
    return _looks_like_verb(tokens[0])


def _looks_like_verb(token: str) -> bool:
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", token).lower()
    if len(cleaned) < 2:
        return False
    if cleaned in _COMMON_VERB_LEMMAS:
        return True
    if cleaned.endswith("ed") or cleaned.endswith("ing"):
        return len(cleaned) > 3
    if cleaned.endswith("es") and cleaned[:-2] in _COMMON_VERB_LEMMAS:
        return True
    if cleaned.endswith("s") and cleaned[:-1] in _COMMON_VERB_LEMMAS:
        return True
    return False


def detect_attribution(clause: str) -> Optional[dict[str, str]]:
    """
    Detect attributions like:
    "Plaintiff alleges that Officer Smith entered the home..."
    """
    cleaned = clause.strip().rstrip(".")
    match = _ATTRIBUTION_RE.match(cleaned)
    if not match:
        return None
    return {
        "speaker": match.group("speaker").strip(),
        "verb": match.group("verb").strip().lower(),
        "content": _ensure_terminal_period(match.group("content").strip()),
    }


def make_claims_from_clause(
    clause: str,
    *,
    doc_id: str,
    para_id: int,
    sent_id: int,
    start_char: int,
    end_char: int,
    doc_section: str,
) -> list[Claim]:
    """
    Build one or more claims from a clause.
    """
    clause_text = _ensure_terminal_period(_clean_claim_text(clause))
    if not clause_text or clause_text == ".":
        return []

    attribution = detect_attribution(clause_text)
    span = SpanRef(
        doc_id=doc_id,
        para_id=para_id,
        sent_id=sent_id,
        start_char=start_char,
        end_char=end_char,
    )

    if attribution:
        certainty = _certainty_from_verb(attribution["verb"])
        claim_a = Claim(
            claim_id=_claim_id(doc_id, sent_id, start_char, end_char, clause_text),
            text=clause_text,
            claim_type="attribution",
            source=attribution["speaker"],
            certainty=certainty,
            doc_section=doc_section,
            span=span,
        )
        embedded_text = attribution["content"]
        claim_b = Claim(
            claim_id=_claim_id(doc_id, sent_id, start_char, end_char, embedded_text),
            text=embedded_text,
            claim_type=_classify_claim_type(embedded_text),
            source=attribution["speaker"],
            certainty=certainty,
            doc_section=doc_section,
            span=span,
        )
        return [claim_a, claim_b]

    return [
        Claim(
            claim_id=_claim_id(doc_id, sent_id, start_char, end_char, clause_text),
            text=clause_text,
            claim_type=_classify_claim_type(clause_text),
            source="court",
            certainty=_classify_certainty(clause_text),
            doc_section=doc_section,
            span=span,
        )
    ]


def decompose_document(document: Mapping[str, Any] | str) -> list[Claim]:
    """
    Decompose a document into deduplicated atomic claims.

    Accepts:
    - plain text string
    - mapping with corpus-builder style fields (expects "full_text")
    """
    if isinstance(document, str):
        doc = {"id": "input_doc", "full_text": document}
    else:
        doc = dict(document)

    doc_id = str(doc.get("id") or "unknown_doc")
    text = normalize_text(str(doc.get("full_text", "")))
    if not text:
        return []

    doc_section = assign_rhetorical_roles(doc)
    claims: list[Claim] = []

    sentence_index = 0
    protected_text = _protect_sentence_abbreviations(text)
    for sent_match in _SENTENCE_RE.finditer(protected_text):
        sentence = _restore_sentence_abbreviations(sent_match.group(0).strip())
        if not sentence:
            continue
        sentence_index += 1
        sent_start = sent_match.start()
        sent_end = sent_match.end()

        for clause in split_clauses(sentence):
            clause_clean = clause.strip()
            if not clause_clean:
                continue
            rel_idx = sentence.find(clause_clean.rstrip("."))
            clause_start = sent_start + max(rel_idx, 0)
            clause_end = min(sent_end, clause_start + len(clause_clean))
            claims.extend(
                make_claims_from_clause(
                    clause_clean,
                    doc_id=doc_id,
                    para_id=0,
                    sent_id=sentence_index,
                    start_char=clause_start,
                    end_char=clause_end,
                    doc_section=doc_section,
                )
            )

    return _dedupe_claims(claims)


def claims_to_jsonl(claims: list[Claim], output_path: str) -> None:
    """Write claims to JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for claim in claims:
            handle.write(json.dumps(claim.to_dict(), ensure_ascii=False) + "\n")


def _classify_claim_type(text: str) -> str:
    if _STATUTE_RE.search(text):
        return "statutory"
    if _CITATION_RE.search(text):
        return "citation"
    if _HOLDING_RE.search(text):
        return "holding"
    return "fact"


def _protect_sentence_abbreviations(text: str) -> str:
    return _LEGAL_ABBREVIATION_RE.sub(
        lambda match: match.group(0).replace(".", _PROTECTED_PERIOD),
        text,
    )


def _restore_sentence_abbreviations(text: str) -> str:
    return text.replace(_PROTECTED_PERIOD, ".")


def _clean_claim_text(text: str) -> str:
    cleaned = text.strip()
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = _CLAIM_CITATION_PREFIX_RE.sub("", cleaned).strip()
        cleaned = _CLAIM_MARKDOWN_HEADING_RE.sub("", cleaned).strip()
        cleaned = _CLAIM_BULLET_PREFIX_RE.sub("", cleaned).strip()
        cleaned = _CLAIM_CONTEXT_CITATION_RE.sub("", cleaned).strip()
        cleaned = _CLAIM_MARKDOWN_EMPHASIS_RE.sub(r"\1", cleaned).strip()
    return cleaned


def _classify_certainty(text: str) -> str:
    if _HEDGE_RE.search(text):
        return "hedged"
    return "found"


def _certainty_from_verb(verb: str) -> str:
    if verb in {"alleges", "alleged", "claims", "claimed", "contends", "contended"}:
        return "alleged"
    if verb in {"argues", "argued"}:
        return "argued"
    if verb in {"testifies", "testified"}:
        return "testified"
    return "stated"


def _ensure_terminal_period(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[-1] in ".!?":
        return text
    return f"{text}."


def _claim_id(doc_id: str, sent_id: int, start_char: int, end_char: int, text: str) -> str:
    base = f"{doc_id}|{sent_id}|{start_char}|{end_char}|{text.lower()}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"clm_{digest}"


def _dedupe_claims(claims: list[Claim]) -> list[Claim]:
    seen: set[str] = set()
    deduped: list[Claim] = []
    for claim in claims:
        key = _WHITESPACE_RE.sub(" ", claim.text.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped
