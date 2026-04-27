"""
Legal prompt templates for direct analysis and retrieval-grounded generation.
"""

from __future__ import annotations

import re
from typing import Any, Sequence


GENERAL_LEGAL_SYSTEM_PROMPT = """You are a legal analysis assistant.

Provide careful, neutral, jurisdiction-sensitive analysis. Distinguish clearly
between facts, procedural posture, issues, holdings, and reasoning when those
distinctions matter. Do not invent authorities, quotes, procedural history, or
record facts. If the user's question is underspecified, identify the missing
jurisdiction, time period, or factual assumptions that affect the answer.

Do not present legal information as a substitute for advice from a licensed
attorney. Avoid absolute language unless the result is clear from the materials
provided by the user.
"""


GENERAL_LEGAL_USER_TEMPLATE = """User question:
{query}

Answering instructions:
- Use any recent conversation history only to resolve follow-up references or
  maintain continuity. Do not treat prior turns as legal authority.
- Give a direct answer first.
- Explain the controlling legal rule or likely rule.
- Separate what is known from what depends on jurisdiction or additional facts.
- If the question asks for litigation risk or likely outcome, explain the key
  variables instead of overstating certainty.
"""


RAG_LEGAL_SYSTEM_PROMPT = """You are LegalVerifiRAG, a retrieval-grounded legal
analysis assistant.

You must answer using only the retrieved legal authorities and facts provided in
the context section. Treat the context as the record for this response. If the
context does not support a proposition, say that the available materials are
insufficient rather than filling the gap from general knowledge.

Requirements:
- Ground every material legal proposition in the provided context.
- When the context contains user-uploaded document facts, treat those facts as
  the client's record or work product, not as precedential legal authority.
- If the user asks about an uploaded document, answer from the uploaded-document
  facts first and use retrieved authorities only for legal comparison or
  background.
- Do not invent citations, quotations, holdings, procedural facts, or dates.
- Prefer narrow statements that can be verified from the retrieved text.
- Treat "Allowed answer fact" entries, canonical answer fact entries, and
  numbered context chunks as the only facts you may use.
- If an allowed answer fact directly answers the question, restate that fact
  plainly instead of adding broader legal analysis.
- Distinguish binding authority from persuasive authority when the context
  makes that possible.
- Respect any structured case posture section as a binding constraint derived
  from the retrieved text.
- If authorities conflict or are incomplete, say so explicitly.
- Keep the answer structured so downstream claim verification can evaluate it.
- Follow the response-format instructions for whether bullets are allowed.
- Do not use Markdown headings, bold text, tables, or bracketed context
  citations in the answer.
- Write each substantive point as a complete declarative sentence.
"""


RAG_LEGAL_USER_TEMPLATE = """User question:
{query}

{case_posture_section}Retrieved context:
{context}

Answering instructions:
- Use any recent conversation history only to interpret follow-up references.
  The retrieved context remains the only authority for the answer.
{response_format_instructions}
- Do not include prompt labels such as "requested format", "short answer",
  "analysis", or "allowed answer fact" in the answer.
- Do not refer to numbered context entries, retrieved context entries, or
  allowed answer facts by label or number.
- Do not quote partial citation fragments or page-header fragments.
- Then provide only analysis that is directly supported by allowed answer
  facts in the retrieved context.
{named_case_precision_instructions}- Only say "Insufficient support in retrieved authorities" if you cannot state
  even one direct proposition from the retrieved text that answers the
  question.
- Do not use Markdown headings such as "Short Answer", "Analysis", or "Limits".
- Do not use bold text, tables, or square-bracket citations.
"""


_CONCISE_RESPONSE_FORMAT_INSTRUCTIONS = """Response format:
- Start with one direct plain-text answer sentence.
- Keep the answer to 1 to 3 complete plain-text sentences.
- Do not use bullets or numbered lists."""


_DETAILED_RESPONSE_FORMAT_INSTRUCTIONS = """Response format:
- Start with one direct plain-text answer sentence.
- Then provide 2 to 4 short paragraphs or a compact hyphen-bullet list when the
  user's question asks for more detail, practical takeaways, comparison, or
  bullets.
- Use only the allowed answer facts in the retrieved context to build those
  paragraphs or bullets.
- If the allowed answer facts do not support a richer explanation, give a
  shorter answer rather than adding general background.
- If using bullets, use simple hyphen bullets only.
- Each bullet must be a complete declarative sentence containing one
  supportable legal proposition.
- Do not add introductory labels or section headings."""


_NAMED_CASE_QUERY_RE = re.compile(
    r"^\s*In\s+(.+?),\s*(?:what|who|when|where|why|how|could|can|did|does|do|is|are|was|were|should)\b",
    re.IGNORECASE,
)


def format_case_posture(case_posture: dict[str, Any] | None) -> str:
    """Render inferred case posture into a compact prompt section."""
    if not case_posture:
        return ""

    def _line(label: str, key: str) -> str | None:
        value = case_posture.get(key)
        if value in (None, "", []):
            return None
        if isinstance(value, bool):
            rendered = "yes" if value else "no"
        elif isinstance(value, list):
            rendered = ", ".join(str(item) for item in value if item)
        else:
            rendered = str(value).replace("_", " ")
        return f"{label}: {rendered}"

    lines = [
        "Structured case posture derived from the retrieved context:",
    ]
    for label, key in [
        ("Target case", "target_case"),
        ("Decision type", "decision_type"),
        ("Court action", "court_action"),
        ("Opinion role", "opinion_role"),
        ("Opinion author", "author"),
        ("Separate opinion", "is_separate_opinion"),
    ]:
        line = _line(label, key)
        if line:
            lines.append(line)

    decision_type = str(case_posture.get("decision_type") or "")
    opinion_role = str(case_posture.get("opinion_role") or "")
    if decision_type == "cert_denial":
        lines.append("Constraint: state that the Court denied certiorari before describing any separate opinion.")
    if opinion_role.startswith("dissent") or opinion_role.startswith("concurrence"):
        lines.append("Constraint: do not describe the separate opinion as the Court's holding.")

    return "\n".join(lines) + "\n\n"


def _named_case_precision_instructions(query: str, case_posture: dict[str, Any] | None) -> str:
    target_case = ""
    if isinstance(case_posture, dict):
        target_case = str(case_posture.get("target_case") or "").strip()

    is_named_case_query = bool(target_case) or _NAMED_CASE_QUERY_RE.match(query.strip())
    if not is_named_case_query:
        return ""

    return (
        "- For named-case queries, answer only with propositions explicitly supported by the retrieved text.\n"
        "- Do not infer procedural posture from a caption or from the phrase \"certiorari to\" alone.\n"
        "- Do not say \"denied certiorari\" unless the retrieved text explicitly says the petition for certiorari was denied.\n"
        "- If the retrieved text contains a syllabus holding, a \"Held:\" sentence, or an opinion sentence stating what the Court did, prefer that language over broader paraphrase.\n"
        "- Do not use generic appellate boilerplate such as \"the lower court judgment remains in effect\" or \"the Court expressed no view on the merits\" unless that proposition is explicitly stated in the retrieved text.\n"
        "- For merits cases, state the holding first. For cert-denial cases, state the denial first.\n"
    )


def format_retrieved_context(chunks: Sequence[str]) -> str:
    """Render retrieved chunks into a numbered prompt block."""
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    if not cleaned_chunks:
        return "[No retrieved context provided]"
    return "\n\n".join(_format_context_item(idx, chunk) for idx, chunk in enumerate(cleaned_chunks, start=1))


def _format_context_item(idx: int, chunk: str) -> str:
    if "Allowed answer fact:" in chunk or "Evidence type:" in chunk:
        return f"Context item {idx}:\n{chunk}"
    return f"Context item {idx}:\nEvidence type: allowed answer fact\nAllowed answer fact: {chunk}"


def format_conversation_history(messages: Sequence[dict[str, Any]] | None) -> str:
    """Render recent conversation turns into a compact prompt block."""
    if not messages:
        return ""

    formatted_messages: list[str] = []
    for message in messages:
        role = "Assistant" if message.get("role") == "assistant" else "User"
        content = " ".join(str(message.get("content") or "").split())
        if not content:
            continue
        formatted_messages.append(f"{role}: {content}")

    if not formatted_messages:
        return ""
    return "\n".join(formatted_messages)


def _conversation_history_section(messages: Sequence[dict[str, Any]] | None) -> str:
    history = format_conversation_history(messages)
    if not history:
        return ""
    return f"Recent conversation history:\n{history}\n\n"


def build_general_legal_prompt(
    query: str,
    conversation_history: Sequence[dict[str, Any]] | None = None,
) -> str:
    """Build a direct legal-analysis prompt without retrieval context."""
    history_section = _conversation_history_section(conversation_history)
    return (
        f"{GENERAL_LEGAL_SYSTEM_PROMPT.strip()}\n\n"
        f"{history_section}"
        f"{GENERAL_LEGAL_USER_TEMPLATE.format(query=query.strip()).strip()}"
    )


def build_rag_legal_prompt(
    query: str,
    context: Sequence[str] | str,
    conversation_history: Sequence[dict[str, Any]] | None = None,
    case_posture: dict[str, Any] | None = None,
    response_depth: str = "concise",
) -> str:
    """Build a legal RAG prompt from a user query and retrieved authorities."""
    if isinstance(context, str):
        formatted_context = context.strip() or "[No retrieved context provided]"
    else:
        formatted_context = format_retrieved_context(context)
    history_section = _conversation_history_section(conversation_history)
    case_posture_section = format_case_posture(case_posture)
    named_case_precision_instructions = _named_case_precision_instructions(query, case_posture)
    response_format_instructions = _response_format_instructions(response_depth)

    return (
        f"{RAG_LEGAL_SYSTEM_PROMPT.strip()}\n\n"
        f"{history_section}"
        f"{RAG_LEGAL_USER_TEMPLATE.format(query=query.strip(), context=formatted_context, case_posture_section=case_posture_section, named_case_precision_instructions=named_case_precision_instructions, response_format_instructions=response_format_instructions).strip()}"
    )


def _response_format_instructions(response_depth: str) -> str:
    if str(response_depth or "").strip().lower() == "detailed":
        return _DETAILED_RESPONSE_FORMAT_INSTRUCTIONS
    return _CONCISE_RESPONSE_FORMAT_INSTRUCTIONS


def get_prompt_templates() -> dict[str, str]:
    """Expose the base templates for callers that want raw prompt strings."""
    return {
        "general_legal_system": GENERAL_LEGAL_SYSTEM_PROMPT,
        "general_legal_user": GENERAL_LEGAL_USER_TEMPLATE,
        "rag_legal_system": RAG_LEGAL_SYSTEM_PROMPT,
        "rag_legal_user": RAG_LEGAL_USER_TEMPLATE,
    }
