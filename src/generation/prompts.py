"""
Legal prompt templates for direct analysis and retrieval-grounded generation.
"""

from __future__ import annotations

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
- Do not invent citations, quotations, holdings, procedural facts, or dates.
- Prefer narrow statements that can be verified from the retrieved text.
- Distinguish binding authority from persuasive authority when the context
  makes that possible.
- If authorities conflict or are incomplete, say so explicitly.
- Keep the answer structured so downstream claim verification can evaluate it.
- Use plain text only: no Markdown headings, bullets, bold text, tables, or
  bracketed context citations in the answer.
- Write each substantive point as a complete declarative sentence.
"""


RAG_LEGAL_USER_TEMPLATE = """User question:
{query}

Retrieved context:
{context}

Answering instructions:
- Use any recent conversation history only to interpret follow-up references.
  The retrieved context remains the only authority for the answer.
- Start with one direct plain-text answer sentence.
- Then provide analysis based only on the retrieved context, using complete
  plain-text sentences.
- If the context is insufficient, say "Insufficient support in retrieved
  authorities" and explain what is missing.
- Do not use Markdown headings such as "Short Answer", "Analysis", or "Limits".
- Do not use bullets, numbered lists, bold text, or square-bracket citations.
"""


def format_retrieved_context(chunks: Sequence[str]) -> str:
    """Render retrieved chunks into a numbered prompt block."""
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    if not cleaned_chunks:
        return "[No retrieved context provided]"
    return "\n\n".join(
        f"[{idx}] {chunk}" for idx, chunk in enumerate(cleaned_chunks, start=1)
    )


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
) -> str:
    """Build a legal RAG prompt from a user query and retrieved authorities."""
    if isinstance(context, str):
        formatted_context = context.strip() or "[No retrieved context provided]"
    else:
        formatted_context = format_retrieved_context(context)
    history_section = _conversation_history_section(conversation_history)

    return (
        f"{RAG_LEGAL_SYSTEM_PROMPT.strip()}\n\n"
        f"{history_section}"
        f"{RAG_LEGAL_USER_TEMPLATE.format(query=query.strip(), context=formatted_context).strip()}"
    )


def get_prompt_templates() -> dict[str, str]:
    """Expose the base templates for callers that want raw prompt strings."""
    return {
        "general_legal_system": GENERAL_LEGAL_SYSTEM_PROMPT,
        "general_legal_user": GENERAL_LEGAL_USER_TEMPLATE,
        "rag_legal_system": RAG_LEGAL_SYSTEM_PROMPT,
        "rag_legal_user": RAG_LEGAL_USER_TEMPLATE,
    }
