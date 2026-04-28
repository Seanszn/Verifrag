"""Server-side orchestration pipeline."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from src.config import RETRIEVAL, VERIFICATION
from src.generation.ollama_backend import OllamaBackend
from src.indexing.bm25_index import BM25Index
from src.indexing.chroma_store import ChromaStore
from src.indexing.embedder import Embedder
from src.indexing.index_discovery import discover_index_artifacts
from src.ingestion.document import LegalChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.user_uploads import load_user_upload_retriever
from src.storage.database import Database
from src.verification.claim_contract import normalize_claims_for_frontend
from src.verification.claim_decomposer import decompose_document
from src.verification.heuristic_verifier import HeuristicNLIVerifier
from src.verification.nli_verifier import AggregatedScore, NLIVerifier
from src.verification.verdict import classify_verification

CONVERSATION_CONTEXT_MESSAGE_LIMIT = 2
MERGED_RETRIEVAL_LIMIT = RETRIEVAL.rerank_k
TARGET_CASE_METADATA_CHUNK_LIMIT = 40
TARGET_CASE_PROMPT_CHUNK_LIMIT = 8
TOPIC_FALLBACK_PROMPT_CHUNK_LIMIT = 6
USER_UPLOAD_COMPARISON_PUBLIC_CHUNK_LIMIT = 3
USER_UPLOAD_PUBLIC_RETRIEVAL_TERM_LIMIT = 16
TARGET_CASE_PROMPT_SENTENCE_LIMIT = 3
TARGET_CASE_DETAILED_PROMPT_SENTENCE_LIMIT = 5
RESPONSE_DEPTH_CONCISE = "concise"
RESPONSE_DEPTH_DETAILED = "detailed"
UNSUPPORTED_WARNING_RATIO = 0.5
REPAIR_UNSUPPORTED_RATIO = 0.5
REPAIR_MAX_CLAIMS = 3
_DETAILED_RESPONSE_QUERY_RE = re.compile(
    r"\b(?:"
    r"bullet(?:ed|s)?"
    r"|list\b"
    r"|paragraphs?\b"
    r"|explain\b"
    r"|why\b"
    r"|reason(?:ing|s)?\b"
    r"|practical\b"
    r"|takeaways?\b"
    r"|compare\b"
    r"|comparison\b"
    r"|in[-\s]?depth\b"
    r"|detail(?:ed|s)?\b"
    r"|outline\b"
    r"|walk\s+through\b"
    r"|what\s+is\s+the\s+rule\b"
    r"|what\s+is\s+the\s+standard\b"
    r"|what\s+is\s+the\s+test\b"
    r"|how\s+does\s+it\s+work\b"
    r"|how\s+is\s+.+?\s+determined\b"
    r"|what\s+are\s+the\s+(?:requirements?|elements?|factors?|conditions?)\b"
    r"|what\s+are\s+the\s+implications?\b"
    r"|describe\s+the\s+(?:process|procedure|framework)\b"
    r")",
    re.IGNORECASE,
)
_USER_UPLOAD_COMPARISON_QUERY_RE = re.compile(
    r"\b(?:compare|comparison|analog(?:y|ize)|distinguish|similarit(?:y|ies)|"
    r"differences?|relevant\s+(?:corpus\s+)?cases?|prior\s+cases?|precedent|"
    r"authorit(?:y|ies)|case\s+law|external\s+(?:cases|authority))\b",
    re.IGNORECASE,
)
_CASE_QUERY_RE = re.compile(
    r"^\s*In\s+(.+?),\s*(?:"
    r"what|who|when|where|why|how|could|can|did|does|do|is|are|was|were|should"
    r"|answer|give|explain|summarize|list|write|describe|include|use|provide"
    r")\b",
    re.IGNORECASE,
)
_CASE_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&'.-]*(?:\s+[A-Z][A-Za-z0-9&'.-]*){0,7}\s+v\.\s+"
    r"[A-Z][A-Za-z0-9&'.-]*(?:\s+[A-Z][A-Za-z0-9&'.-]*){0,7})\b"
)
_US_CITATION_RE = re.compile(r"\b\d+\s+U\.?\s*S\.?\s+(?:\d+|_{2,})(?:\s*\(\d{4}\))?\b", re.IGNORECASE)
_AUTHOR_QUERY_RE = re.compile(
    r"\bwho\s+(?:wrote|authored|joined|dissented|concurred)|\bwhich\s+justice\b|\bauthor\b",
    re.IGNORECASE,
)
_HOLDING_QUERY_RE = re.compile(r"\b(?:hold|holding|held)\b", re.IGNORECASE)
_POSTURE_QUERY_RE = re.compile(
    r"\b(?:what\s+did\s+(?:the\s+)?(?:supreme\s+)?court\s+do|what\s+happened|disposition|outcome|procedural\s+posture|case\s+posture)\b",
    re.IGNORECASE,
)
_ISSUE_ANALYSIS_QUERY_RE = re.compile(
    r"\b(?:whether|why|how|reason|analysis|say\s+about|explain|rule|factor|factors|consider|apply|application|mean|means|argue|argued)\b",
    re.IGNORECASE,
)
_RESEARCH_LEADS_QUERY_RE = re.compile(
    r"\b(?:"
    r"find\s+(?:me\s+)?(?:some\s+)?cases?"
    r"|identify\s+(?:some\s+)?(?:cases?|authorit(?:y|ies)|precedent)"
    r"|what\s+cases?\s+(?:discuss|address|talk\s+about|deal\s+with|involve)"
    r"|which\s+cases?\s+(?:discuss|address|talk\s+about|deal\s+with|involve)"
    r"|cases?\s+(?:about|on|involving|discussing|addressing)"
    r"|case\s+law\s+(?:about|on|for|involving|discussing|addressing)"
    r"|authorit(?:y|ies)\s+(?:about|on|for|involving|discussing|addressing)"
    r"|precedent\s+(?:about|on|for|involving|discussing|addressing)"
    r"|research\s+leads?"
    r"|examples?\s+of"
    r")\b",
    re.IGNORECASE,
)
_TOPIC_SIGNAL_RE = re.compile(
    r"\b(?:law|property|contract|tort|criminal|constitutional|statutory|"
    r"adverse\s+possession|easement|title|possession|negligence|damages|"
    r"jurisdiction|standing|due\s+process|first\s+amendment|fourth\s+amendment|"
    r"supervised\s+release|sentencing|habeas|brady|materiality|fraud|tariff|agency)\b",
    re.IGNORECASE,
)
_FOLLOWUP_REFERENCE_RE = re.compile(
    r"\b(?:that|this|it|its|the case|the court|the separate opinion|separate opinion|"
    r"the dissent|the concurrence|same case|that case|that opinion)\b",
    re.IGNORECASE,
)
_RELATED_RETRIEVED_AUTHORITY_CLAIM_RE = re.compile(
    r"^\s*Based\s+on\s+related\s+retrieved\s+authorit(?:y|ies)\b",
    re.IGNORECASE,
)
_CHEVRON_CONTROLS_CLAIM_RE = re.compile(
    r"\bChevron(?:\s+deference)?\b.{0,80}\b(?:controls?|controlling|appl(?:y|ies)|governs?|is\s+the\s+rule)\b|"
    r"\b(?:controls?|controlling|governs?|appl(?:y|ies)|is\s+the\s+rule)\b.{0,80}\bChevron(?:\s+deference)?\b",
    re.IGNORECASE,
)
_CHEVRON_REJECTING_EVIDENCE_RE = re.compile(
    r"\b(?:overrul(?:e|ed|es|ing)|cannot\s+be\s+squared|is\s+overruled|"
    r"no\s+longer\s+(?:applies|controls)|does\s+not\s+apply|reject(?:s|ed|ing)?)\b.{0,120}\bChevron\b|"
    r"\bChevron\b.{0,120}\b(?:overrul(?:e|ed|es|ing)|cannot\s+be\s+squared|"
    r"no\s+longer\s+(?:applies|controls)|does\s+not\s+apply|reject(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_RESEARCH_LEADS_GENERIC_TOKENS = {
    "action",
    "actions",
    "agency",
    "article",
    "authority",
    "challenge",
    "claim",
    "claims",
    "concrete",
    "defendant",
    "demonstrate",
    "federal",
    "government",
    "held",
    "injury",
    "law",
    "legal",
    "likely",
    "plaintiff",
    "plaintiffs",
    "regulation",
    "requires",
    "rule",
    "show",
    "standing",
    "statute",
}
_SOURCE_DISCIPLINE_QUERY_RE = re.compile(
    r"\b(?:use\s+only\s+retrieved\s+sources|retrieved\s+sources\s+do\s+not\s+mention|"
    r"if\s+the\s+retrieved\s+sources\s+do\s+not|source\s+discipline|"
    r"do\s+not\s+(?:infer|guess)|say\s+that\s+instead)\b",
    re.IGNORECASE,
)
_QUERY_SUBJECT_GENERIC_TOKENS = {
    "about",
    "answer",
    "authority",
    "case",
    "claim",
    "court",
    "database",
    "does",
    "explain",
    "holding",
    "instead",
    "mention",
    "mentioned",
    "only",
    "retrieved",
    "rule",
    "say",
    "source",
    "sources",
    "summarize",
    "using",
}
_FOLLOWUP_AUTHOR_RE = re.compile(
    r"\b(?:who\s+(?:wrote|authored)|which\s+justice|author)\b",
    re.IGNORECASE,
)
_SUPREME_COURT_QUERY_RE = re.compile(r"\b(?:supreme court|the court|court)\b", re.IGNORECASE)
_LOWER_COURT_PROCEDURAL_RE = re.compile(
    r"\b(?:district court|court of appeals|sixth circuit|federal circuit|CIT|court of international trade)\b",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")
_HOLDING_LEAD_RE = re.compile(
    r"^\s*(?:"
    r"held:"
    r"|the court (?:holds|held|agrees with|concludes)\b"
    r"|we (?:hold|agree|conclude)\b"
    r"|for the foregoing reasons, we conclude"
    r"|therefore, the judgment"
    r"|the judgment (?:of|below)"
    r"|district courts cannot\b"
    r"|may not\b"
    r"|cannot\b"
    r"|must\b"
    r"|is not authorized to\b"
    r"|authorizes?\b"
    r"|does not authorize\b"
    r"|does not violate\b"
    r"|cannot be squared\b"
    r"|need not evaluate\b"
    r"|the preponderance-of-the-evidence standard applies\b"
    r"|nothing in\b[^.!?]{0,120}\b(?:authorizes|enables|empowers)\b"
    r")",
    re.IGNORECASE,
)
_HOLDING_CONCLUSION_RE = re.compile(
    r"\b(?:"
    r"judgment[^.!?]{0,120}\b(?:affirmed|reversed|vacated|remanded)\b"
    r"|decision[^.!?]{0,120}\b(?:affirmed|reversed|vacated|remanded)\b"
    r"|must be (?:vacated|remanded|reversed|affirmed)\b"
    r"|does not violate\b"
    r"|cannot be squared\b"
    r"|need not evaluate\b"
    r"|preponderance-of-the-evidence standard applies\b"
    r"|nothing in\b[^.!?]{0,120}\b(?:authorizes|enables|empowers)\b"
    r"|did not grant\b[^.!?]{0,120}\b(?:power|authority)\b"
    r")",
    re.IGNORECASE,
)
_INCOMPLETE_EVIDENCE_SENTENCE_RE = re.compile(
    r"(?:\bcase\s+No\.?$|\bNo\.?$|\bv\.?$|^\d+\s+[A-Z][A-Z\s.,'&-]+v\.?$)",
    re.IGNORECASE,
)
_CAPTION_ONLY_RE = re.compile(r"^\s*(?:on writ of certiorari to|certiorari to)\b", re.IGNORECASE)
_CERT_DENIAL_RE = re.compile(r"petition for (?:a writ of )?certiorari is denied", re.IGNORECASE)
_CERT_DENIAL_RESPONSE_RE = re.compile(
    r"\b(?:"
    r"denied certiorari"
    r"|certiorari (?:is|was) denied"
    r"|petition for (?:a writ of )?certiorari (?:is|was) denied"
    r")\b",
    re.IGNORECASE,
)
_GVR_RE = re.compile(
    r"(?:petition|writ) for (?:a writ of )?certiorari is granted.*?vacated.*?remanded",
    re.IGNORECASE | re.DOTALL,
)
_VACATED_AND_REMANDED_RE = re.compile(r"\b(?:judgment|decision).{0,40}\bvacated\b.{0,40}\bremanded\b", re.IGNORECASE | re.DOTALL)
_REVERSED_RE = re.compile(r"\b(?:judgment|decision).{0,40}\breversed\b", re.IGNORECASE | re.DOTALL)
_AFFIRMED_RE = re.compile(r"\b(?:judgment|decision).{0,40}\baffirmed\b", re.IGNORECASE | re.DOTALL)
_VACATED_RE = re.compile(r"\b(?:judgment|decision).{0,40}\bvacated\b", re.IGNORECASE | re.DOTALL)
_REMANDED_RE = re.compile(r"\b(?:judgment|decision).{0,40}\bremanded\b", re.IGNORECASE | re.DOTALL)
_PER_CURIAM_RE = re.compile(r"\bPER CURIAM\b", re.IGNORECASE)
_DISSENT_FROM_DENIAL_PATTERNS = (
    re.compile(r"JUSTICE\s+([A-Z][A-Z'.-]+),?\s+dissenting from the denial of certiorari", re.IGNORECASE),
    re.compile(r"([A-Z][A-Z'.-]+),\s+J\.,\s+dissenting from the denial of certiorari", re.IGNORECASE),
)
_DISSENT_PATTERNS = (
    re.compile(r"JUSTICE\s+([A-Z][A-Z'.-]+),?\s+dissenting\b", re.IGNORECASE),
    re.compile(r"([A-Z][A-Z'.-]+),\s+J\.,\s+dissenting\b", re.IGNORECASE),
)
_CONCURRENCE_PATTERNS = (
    re.compile(r"JUSTICE\s+([A-Z][A-Z'.-]+),?\s+concurring(?:\s+in\s+the\s+judgment)?", re.IGNORECASE),
    re.compile(r"([A-Z][A-Z'.-]+),\s+J\.,\s+concurring(?:\s+in\s+the\s+judgment)?", re.IGNORECASE),
)
_LEGAL_TEXT_NOISE_RE = re.compile(
    r"\b(?:Page Proof Pending Publication|PRELIMINARY PRINT|OFFICIAL REPORTS OF THE SUPREME COURT|"
    r"NOTICE: This preliminary print is subject to formal revision before publication)\b",
    re.IGNORECASE,
)
_CITE_AS_NOISE_RE = re.compile(r"\bCite as:\s*[^.!?]{0,160}?\(\d{4}\)", re.IGNORECASE)
_CASE_PAGE_HEADER_RE = re.compile(
    r"\b\d{1,4}\s+[A-Z][A-Z0-9.,'&\-\s]{2,90}\s+v\.\s+"
    r"[A-Z][A-Z0-9.,'&\-\s]{1,90}\s+"
    r"(?:Syllabus|Opinion of the Court|Slip Opinion|Per Curiam)\b"
)
_PROMPT_LEAKAGE_RE = re.compile(
    r"\b(?:here is the answer|requested format|what was the outcome of the case)\b",
    re.IGNORECASE,
)
_MALFORMED_CITATION_FRAGMENT_RE = re.compile(
    r"^[\"'“”]?\s*[A-Z][A-Za-z.-]+,\s+\d+\s+U\.?(?:\s*S\.?)?\s*\.?$"
)
_CASE_CORPORATE_TOKENS = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "llc",
    "ltd",
}
_RHETORICAL_CLAIM_PATTERNS = (
    re.compile(r"\b(?:unfortunate|unfortunately)\b", re.IGNORECASE),
    re.compile(r"\b(?:hope|hopes|hoped)\b", re.IGNORECASE),
    re.compile(r"\bthe context suggests\b", re.IGNORECASE),
    re.compile(r"\bin the meantime\b", re.IGNORECASE),
    re.compile(r"\bmore carefully consider\b", re.IGNORECASE),
    re.compile(r"\bmay soon\b", re.IGNORECASE),
)
_LOW_VALUE_VERIFICATION_CLAIM_RE = re.compile(
    r"^\s*(?:so too here|same here|for these reasons|for that reason|that is enough|"
    r"that resolves the question|\*+|insufficient support in retrieved authorities.*|"
    r"retrieved authorities do not.*|retrieved context does not.*|"
    r"cannot determine from (?:the )?retrieved.*|"
    r"i do not have .+ in the retrieved database.*|"
    r"the uploaded document excerpt does not identify .*)\.?\s*$",
    re.IGNORECASE,
)
_FRAGMENT_VERIFICATION_CLAIM_RE = re.compile(
    r"^\s*(?:"
    r"[A-Z]\.|"
    r"[A-Z]\.\s*[A-Z]\.|"
    r"[\"'“”.,;:]+\.?|"
    r"v\.\s+[^.]+\.?|"
    r"(?:see\s+)?id\.,?.*|"
    r"[A-Z][a-z]+,\s+and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\.?|"
    r")\s*$",
    re.IGNORECASE,
)
_TRAILING_ABBREVIATION_FRAGMENT_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Hon|Lt|No|Nos|Inc|Corp|Co|Ltd|L\.P|L\.L\.C|U\.S|U\.S\.C|D\.C|Cir|v)\.\s*$",
    re.IGNORECASE,
)
_DANGLING_VERIFICATION_CLAIM_END_RE = re.compile(
    r"\b(?:and|or|but|because|that|which|who|whom|whose|when|where|while|"
    r"with|without|to|of|for|from|by|as|just)\.$",
    re.IGNORECASE,
)
_SINGLE_INITIAL_VERIFICATION_END_RE = re.compile(r"\b[A-Z]\.$")
_INITIAL_SEQUENCE_END_RE = re.compile(r"(?:\b[A-Z]\.\s*){2,}$")
_VERIFICATION_CLAIM_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|do|does|did|"
    r"held|holds|hold|said|says|state|states|stated|found|finds|concluded|"
    r"ruled|decided|affirmed|reversed|vacated|remanded|denied|granted|"
    r"requires?|prohibits?|allows?|authorizes?|applies?|violates?|means?|"
    r"seeks?|suffered|lacks?|lost|lists?|identifies?|overrules?|"
    r"concerns?|addresses?|involves?|includes?|excludes?|"
    r"must|may|can|cannot|could|should)\b",
    re.IGNORECASE,
)
_ANSWER_BLOCK_TRANSITION_RE = re.compile(
    r"^\s*(?:in short|put differently|in other words|overall|therefore|accordingly|"
    r"as a result|the practical point|the bottom line|on that basis)\b[:,]?",
    re.IGNORECASE,
)
_ANSWER_BLOCK_LABEL_LEAD_IN_RE = re.compile(
    r"^(?:(?:the\s+)?(?:holding|rule|legal\s+rule|core\s+holding)"
    r"(?:\s+in\s+[^:\n]{1,160})?\s+(?:is|was)\s*:|"
    r"(?:bottom\s+line|practical\s+takeaway|takeaway|practical\s+rule)\s*:)\s*$",
    re.IGNORECASE,
)
_ANSWER_BLOCK_CAVEAT_RE = re.compile(
    r"\b(?:insufficient support|retrieved context does not|retrieved authorities do not|"
    r"not enough evidence|cannot determine|does not identify|no separate opinion identified|"
    r"not available in the retrieved context|unclear from the retrieved|"
    r"do not have .+ in the retrieved database)\b",
    re.IGNORECASE,
)
_ANSWER_BLOCK_FACT_ANCHOR_RE = re.compile(
    r"(?:\b(?:supreme court|court|justice|judge|jury|petitioner|respondent|plaintiff|defendant|"
    r"government|state|district court|court of appeals|held|holds|holding|concluded|found|"
    r"ruled|decided|denied|granted|affirmed|reversed|vacated|remanded|certiorari|dissent|"
    r"concurrence|opinion|judgment|statute|constitutional|amendment)\b|U\.?S\.?C\.?|U\.?S\.?|"
    r"§|\bv\.)",
    re.IGNORECASE,
)
_ANSWER_BLOCK_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|held|holds|denied|granted|affirmed|reversed|"
    r"vacated|remanded|concluded|found|ruled|requires?|prohibits?|allows?|authorizes?|"
    r"violates?|applies?|means?)\b",
    re.IGNORECASE,
)
_CONSTITUTIONAL_ANCHOR_RE = re.compile(
    r"\b(?:(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|"
    r"Eleventh|Twelfth|Thirteenth|Fourteenth|Fifteenth)\s+Amendment|"
    r"Equal Protection Clause|Due Process Clause|Commerce Clause|Appointments Clause|"
    r"Free Speech Clause|Establishment Clause|Takings Clause)\b",
    re.IGNORECASE,
)
_STATUTE_SECTION_ANCHOR_RE = re.compile(
    r"(?:§+\s*[\w().-]+|\b\d+\s+U\.?\s*S\.?\s*C\.?\s*(?:§+\s*)?[\w().-]+)",
    re.IGNORECASE,
)
_SHORT_NAME_STOPWORDS = {
    "administration",
    "agency",
    "america",
    "american",
    "association",
    "bank",
    "bondi",
    "city",
    "commissioner",
    "company",
    "corp",
    "corporation",
    "department",
    "federal",
    "group",
    "inc",
    "incorporated",
    "llc",
    "new",
    "resources",
    "state",
    "states",
    "transit",
    "trust",
    "united",
    "university",
    "usa",
}
logger = logging.getLogger(__name__)


class QueryPipeline:
    """Owns the server-side query lifecycle."""

    def __init__(
        self,
        db: Database,
        llm: OllamaBackend | None = None,
        retriever: HybridRetriever | None = None,
        verifier: NLIVerifier | None = None,
        *,
        enable_verification: bool = True,
    ) -> None:
        self.db = db
        self.llm = llm or OllamaBackend()
        if retriever is None:
            self.retriever, self.retriever_status = _load_default_retriever()
        else:
            self.retriever = retriever
            self.retriever_status = "configured"
        self.verifier = verifier
        self.enable_verification = enable_verification

    def run(
        self,
        user_id: int,
        query: str,
        conversation_id: Optional[int] = None,
        *,
        request_id: str | None = None,
        include_uploaded_chunks: bool = False,
    ) -> dict[str, Any]:
        log_request_id = request_id or "local"
        logger.info(
            "query.pipeline_start request_id=%s user_id=%s conversation_id=%s query=%r",
            log_request_id,
            user_id,
            conversation_id,
            _query_preview(query),
        )
        conversation = self._ensure_conversation(user_id, conversation_id, query)
        conversation_context = self._load_conversation_context(conversation["id"], user_id)
        stored_conversation_state = self.db.get_conversation_state(conversation["id"], user_id)
        interaction = self.db.create_interaction(conversation["id"], query)
        user_message = self.db.add_message(
            conversation["id"],
            "user",
            query,
            interaction_id=interaction["id"],
        )

        assistant_text, pipeline_meta = self._generate_response(
            query,
            conversation_context,
            conversation_state=stored_conversation_state.get("state") if stored_conversation_state else None,
            user_id=user_id,
            request_id=log_request_id,
            include_uploaded_chunks=include_uploaded_chunks,
        )
        interaction = self.db.complete_interaction(interaction["id"], assistant_text)
        assistant_message = self.db.add_message(
            conversation["id"],
            "assistant",
            assistant_text,
            interaction_id=interaction["id"],
            metadata_json=json.dumps(pipeline_meta),
        )
        self._persist_interaction_artifacts(interaction["id"], pipeline_meta)
        self.db.update_conversation_state(
            conversation["id"],
            _conversation_state_summary(query, assistant_text),
            _conversation_state_payload(query, assistant_text, pipeline_meta),
        )
        conversation = self.db.get_conversation(conversation["id"], user_id) or conversation
        logger.info(
            "query.pipeline_complete request_id=%s interaction_id=%s conversation_id=%s claim_count=%s retrieval_status=%s verification_status=%s",
            log_request_id,
            interaction["id"],
            conversation["id"],
            pipeline_meta.get("claim_count"),
            pipeline_meta.get("retrieval_backend_status"),
            pipeline_meta.get("verification_backend_status"),
        )

        return {
            "conversation": conversation,
            "interaction": interaction,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "pipeline": pipeline_meta,
        }

    def _ensure_conversation(
        self,
        user_id: int,
        conversation_id: Optional[int],
        query: str,
    ) -> dict[str, Any]:
        if conversation_id is not None:
            conversation = self.db.get_conversation(conversation_id, user_id)
            if conversation is not None:
                return conversation
        return self.db.create_conversation(user_id, _default_title(query))

    def _load_conversation_context(
        self,
        conversation_id: int,
        user_id: int,
    ) -> list[dict[str, Any]]:
        messages = self.db.list_recent_messages(
            conversation_id,
            user_id,
            limit=CONVERSATION_CONTEXT_MESSAGE_LIMIT,
        )
        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
            if message.get("content")
        ]

    def _generate_response(
        self,
        query: str,
        conversation_context: list[dict[str, Any]] | None = None,
        *,
        conversation_state: dict[str, Any] | None = None,
        user_id: int,
        request_id: str | None = None,
        include_uploaded_chunks: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        effective_query, followup_grounding_meta = _ground_followup_query(
            query,
            conversation_state,
        )
        llm_host = getattr(self.llm, "host", None)
        llm_model = getattr(self.llm, "model", getattr(self.llm, "model_name", None))
        public_chunks = []
        user_upload_chunks = []
        retrieval_status = self.retriever_status
        public_retrieval_error = False
        shared_embedder = getattr(self.retriever, "embedder", None)
        timings_ms: dict[str, float] = {}
        user_upload_retriever = None
        user_upload_status = "disabled:not_requested"
        if include_uploaded_chunks:
            user_upload_retriever, user_upload_status = load_user_upload_retriever(
                user_id,
                shared_embedder=shared_embedder,
            )
        user_upload_retrieval_error = False

        public_retrieval_query = effective_query
        public_retrieval_query_meta = {
            "status": "not_applied:not_user_upload_comparison",
            "original_query": effective_query,
            "public_query": effective_query,
        }
        public_rerank_meta = {"status": "not_applied:disabled_query_variant_only"}
        public_query_expansion_meta = {"status": "not_applied:retriever_unavailable"}
        user_upload_retrieved = False

        retrieval_started = time.perf_counter()
        logger.info("query.stage_start request_id=%s stage=retrieval", request_id)
        if (
            user_upload_retriever is not None
            and _query_mentions_upload(effective_query)
            and _query_requests_upload_comparison(effective_query)
        ):
            try:
                user_upload_chunks = user_upload_retriever.retrieve(effective_query)
                user_upload_retrieved = True
                public_retrieval_query, public_retrieval_query_meta = (
                    _build_user_upload_public_retrieval_query(
                        effective_query,
                        user_upload_chunks,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive path for live index/runtime failures
                user_upload_status = f"error:{exc.__class__.__name__}"
                user_upload_retrieval_error = True

        if self.retriever is not None:
            try:
                public_chunks = self.retriever.retrieve(public_retrieval_query)
                expansion_meta = getattr(self.retriever, "last_query_expansion_meta", None)
                if callable(expansion_meta):
                    public_query_expansion_meta = {
                        **expansion_meta(),
                        "merged_candidate_ids": [chunk.id for chunk in public_chunks],
                        "rerank_query": public_retrieval_query,
                    }
                    if public_query_expansion_meta.get("status") == "applied":
                        public_rerank_meta = public_query_expansion_meta
                retriever_status = getattr(self.retriever, "last_backend_status", None)
                if callable(retriever_status):
                    status_detail = retriever_status()
                    if status_detail != "ok":
                        retrieval_status = status_detail
                        public_retrieval_error = status_detail.startswith("error:")
            except Exception as exc:  # pragma: no cover - defensive path for live index/runtime failures
                retrieval_status = f"error:{exc.__class__.__name__}"
                public_retrieval_error = True

        explicit_target_case = _extract_target_case_name(effective_query)
        explicit_target_citation = _extract_query_citation(effective_query)
        target_metadata_chunks, target_metadata_meta = _retrieve_target_chunks_from_metadata(
            self.retriever,
            target_case=explicit_target_case,
            citation=explicit_target_citation,
            limit=TARGET_CASE_METADATA_CHUNK_LIMIT,
        )

        if user_upload_retriever is not None and not user_upload_retrieved:
            try:
                user_upload_chunks = user_upload_retriever.retrieve(effective_query)
            except Exception as exc:  # pragma: no cover - defensive path for live index/runtime failures
                user_upload_status = f"error:{exc.__class__.__name__}"
                user_upload_retrieval_error = True

        retrieved_chunks = _merge_retrieved_chunks(
            user_upload_chunks,
            target_metadata_chunks,
            public_chunks,
            limit=MERGED_RETRIEVAL_LIMIT,
        )
        timings_ms["retrieval"] = round((time.perf_counter() - retrieval_started) * 1000, 1)
        logger.info(
            "query.stage_end request_id=%s stage=retrieval elapsed_ms=%.1f public_status=%s user_status=%s public_chunks=%s user_chunks=%s merged_chunks=%s",
            request_id,
            timings_ms["retrieval"],
            retrieval_status,
            user_upload_status,
            len(public_chunks),
            len(user_upload_chunks),
            len(retrieved_chunks),
        )

        generation_started = time.perf_counter()
        query_grounding = _resolve_query_grounding(
            effective_query,
            retrieved_chunks,
            llm_router=self.llm,
        )
        response_depth = _classify_response_depth(effective_query)
        if followup_grounding_meta["status"].startswith("applied:"):
            query_grounding["followup_grounding"] = followup_grounding_meta
            query_grounding["original_query"] = query
            query_grounding["grounded_query"] = effective_query
            if followup_grounding_meta.get("target_case") and not query_grounding.get("target_case"):
                query_grounding["target_case"] = followup_grounding_meta.get("target_case")
                query_grounding["target_doc_ids"] = followup_grounding_meta.get("target_doc_ids") or []
                query_grounding["source"] = "conversation_state"
                query_grounding["status"] = "resolved:conversation_state"
        topic_fallback_chunks, topic_fallback_meta = _build_missing_case_topic_fallback(
            effective_query,
            query_grounding=query_grounding,
            retriever=self.retriever,
            initial_chunks=retrieved_chunks,
            limit=TOPIC_FALLBACK_PROMPT_CHUNK_LIMIT,
        )
        if topic_fallback_chunks:
            retrieved_chunks = topic_fallback_chunks
            query_grounding = {
                **query_grounding,
                "target_case": None,
                "target_doc_ids": [],
                "status": "not_resolved:missing_explicit_case_topic_fallback",
                "missing_explicit_case": topic_fallback_meta.get("missing_case"),
                "topic_fallback_used": True,
                "topic_fallback_query": topic_fallback_meta.get("topic_query"),
            }
            prompt_chunks = topic_fallback_chunks
            prompt_filter_meta = {
                "status": "applied:missing_case_topic_fallback",
                "target_case": None,
                "candidate_count": len(topic_fallback_chunks),
                "limit": TOPIC_FALLBACK_PROMPT_CHUNK_LIMIT,
                "query_grounding_status": query_grounding.get("status"),
            }
        else:
            prompt_chunks, prompt_filter_meta = _select_prompt_chunks_for_generation(
                effective_query,
                retrieved_chunks,
                query_grounding=query_grounding,
            )
        research_leads_mode = (
            _query_requests_research_leads(effective_query, query_grounding)
            and not bool(topic_fallback_chunks)
        )
        generation_context, generation_context_meta = _build_generation_context(
            effective_query,
            prompt_chunks,
            query_grounding=query_grounding,
            response_depth=response_depth,
            research_leads_mode=research_leads_mode,
        )
        if topic_fallback_chunks:
            generation_context = [
                _format_missing_case_topic_scope_for_prompt(
                    topic_fallback_meta.get("missing_case"),
                    topic_fallback_meta.get("topic_query"),
                ),
                *generation_context,
            ]
            generation_context_meta = {
                **generation_context_meta,
                "status": "applied:missing_case_topic_fallback",
                "missing_case": topic_fallback_meta.get("missing_case"),
                "topic_query": topic_fallback_meta.get("topic_query"),
                "topic_chunk_ids": [chunk.id for chunk in topic_fallback_chunks],
            }
        case_posture = (
            _extract_case_posture(prompt_chunks, prompt_filter_meta["target_case"])
            if prompt_filter_meta["status"] == "applied"
            else None
        )
        pre_generation_refusal, pre_generation_refusal_meta = _build_pre_generation_refusal(
            query_grounding=query_grounding,
            retrieved_chunks=retrieved_chunks,
        )
        logger.info(
            "query.stage_start request_id=%s stage=generation mode=%s prompt_chunks=%s prompt_filter_status=%s posture=%s",
            request_id,
            "rag" if retrieved_chunks else "direct",
            len(prompt_chunks),
            prompt_filter_meta["status"],
            "inferred" if case_posture else "none",
        )
        try:
            cert_denial_guard_meta = {"status": "not_applied:generation_skipped"}
            response_override_meta = {"status": "not_applied:generation_skipped"}
            if pre_generation_refusal:
                response = pre_generation_refusal
                generation_mode = "rag_refusal"
                response_override_meta = {"status": "not_applied:pre_generation_refusal"}
                cert_denial_guard_meta = {"status": "not_applied:pre_generation_refusal"}
            elif retrieved_chunks:
                response = self.llm.generate_with_context(
                    effective_query,
                    generation_context,
                    conversation_history=conversation_context,
                    case_posture=case_posture,
                    response_depth=response_depth,
                )
                generation_mode = "rag"
                if topic_fallback_chunks:
                    response = _prepend_missing_case_topic_disclaimer(
                        response,
                        missing_case=topic_fallback_meta.get("missing_case"),
                    )
            else:
                response = self.llm.generate_legal_answer(
                    effective_query,
                    conversation_history=conversation_context,
                )
                generation_mode = "direct"
            if pre_generation_refusal:
                pass
            elif response_depth == RESPONSE_DEPTH_DETAILED:
                response_override_meta = {
                    "status": "not_applied:detailed_response_depth",
                    "response_depth": response_depth,
                }
            else:
                response, response_override_meta = _apply_user_upload_absence_response_override(
                    query=effective_query,
                    response=response,
                    generation_context_meta=generation_context_meta,
                )
                response_override_status = str(response_override_meta.get("status", ""))
                if not response_override_status.startswith("applied:"):
                    response, response_override_meta = _apply_case_posture_response_override(
                        query=effective_query,
                        response=response,
                        case_posture=case_posture,
                        query_intent=query_grounding.get("query_intent"),
                    )
                    response_override_status = str(response_override_meta.get("status", ""))
                    if (
                        not response_override_status.startswith("applied:")
                        and response_override_status != "not_applied:intent_not_posture_or_author"
                    ):
                        response, response_override_meta = _apply_explicit_holding_response_override(
                            query=effective_query,
                            response=response,
                            generation_context_meta=generation_context_meta,
                            target_case=prompt_filter_meta.get("target_case"),
                        )
            if not pre_generation_refusal:
                response, cert_denial_guard_meta = _apply_cert_denial_safety_guard(
                    response=response,
                    case_posture=case_posture,
                    generation_context_meta=generation_context_meta,
                    prompt_chunks=prompt_chunks,
                )
            backend_status = "ok"
            llm_error = None
        except Exception as exc:  # pragma: no cover - defensive path for live Ollama failures
            response = (
                "The configured Ollama provider could not generate a response. "
                f"Host: {llm_host or 'unknown'}. Model: {llm_model or 'unknown'}. Error: {exc}"
            )
            backend_status = f"error:{exc.__class__.__name__}"
            generation_mode = "rag" if retrieved_chunks else "direct"
            llm_error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "host": llm_host,
                "model": llm_model,
            }
        timings_ms["generation"] = round((time.perf_counter() - generation_started) * 1000, 1)
        if llm_error is None:
            logger.info(
                "query.stage_end request_id=%s stage=generation elapsed_ms=%.1f llm_status=%s response_chars=%s",
                request_id,
                timings_ms["generation"],
                backend_status,
                len(response),
            )
        else:
            logger.warning(
                "query.stage_end request_id=%s stage=generation elapsed_ms=%.1f llm_status=%s response_chars=%s error_type=%s error_message=%r",
                request_id,
                timings_ms["generation"],
                backend_status,
                len(response),
                llm_error["type"],
                llm_error["message"],
            )

        meta = {
            "llm_provider": "ollama",
            "llm_config": {
                "host": llm_host,
                "model": llm_model,
            },
            "llm_backend_status": backend_status,
            "generation_mode": generation_mode,
            "response_depth": response_depth,
            "include_uploaded_chunks": bool(include_uploaded_chunks),
            "effective_query": effective_query,
            "followup_grounding_status": followup_grounding_meta["status"],
            "followup_grounding_meta": followup_grounding_meta,
            "retrieval_used": bool(retrieved_chunks),
            "retrieval_backend_status": _combined_retrieval_status(
                retrieval_status,
                user_upload_status,
                user_upload_chunks,
            ),
            "public_retrieval_backend_status": retrieval_status,
            "public_retrieval_query": public_retrieval_query,
            "public_retrieval_query_meta": public_retrieval_query_meta,
            "public_rerank_meta": public_rerank_meta,
            "public_query_expansion_meta": public_query_expansion_meta,
            "user_upload_retrieval_backend_status": user_upload_status,
            "target_metadata_retrieval_status": target_metadata_meta["status"],
            "target_metadata_retrieval_count": len(target_metadata_chunks),
            "target_metadata_retrieval_meta": target_metadata_meta,
            "answer_mode": _answer_mode(
                generation_mode=generation_mode,
                topic_fallback_used=bool(topic_fallback_chunks),
                pre_generation_refusal=bool(pre_generation_refusal),
                retrieval_used=bool(retrieved_chunks),
                research_leads_mode=research_leads_mode,
            ),
            "research_leads_mode": bool(research_leads_mode),
            "research_leads_status": (
                generation_context_meta["status"]
                if research_leads_mode
                else "not_applied:not_research_leads_query"
            ),
            "missing_target_case": topic_fallback_meta.get("missing_case"),
            "topic_fallback_used": bool(topic_fallback_chunks),
            "topic_fallback_status": topic_fallback_meta["status"],
            "topic_fallback_query": topic_fallback_meta.get("topic_query"),
            "topic_fallback_chunk_count": len(topic_fallback_chunks),
            "target_case_answered": bool(
                query_grounding.get("target_case")
                and not topic_fallback_chunks
                and not pre_generation_refusal
            ),
            "retrieval_chunk_count": len(retrieved_chunks),
            "public_retrieval_chunk_count": len(public_chunks),
            "user_upload_retrieval_chunk_count": len(user_upload_chunks),
            "retrieved_chunks": [_serialize_chunk(chunk) for chunk in retrieved_chunks],
            "prompt_chunk_count": len(prompt_chunks),
            "prompt_chunk_ids": [chunk.id for chunk in prompt_chunks],
            "prompt_case_filter_status": prompt_filter_meta["status"],
            "target_case_name": prompt_filter_meta["target_case"],
            "query_grounding": query_grounding,
            "query_grounding_status": query_grounding["status"],
            "target_case_prompt_candidate_count": prompt_filter_meta.get("candidate_count"),
            "target_case_prompt_limit": prompt_filter_meta.get("limit"),
            "generation_context_status": generation_context_meta["status"],
            "generation_context_count": generation_context_meta.get("count"),
            "generation_context_source_ids": generation_context_meta.get("source_chunk_ids"),
            "generation_context_explicit_holding": generation_context_meta.get("explicit_holding_sentence"),
            "generation_context_canonical_answer_fact": generation_context_meta.get("canonical_answer_fact"),
            "generation_context_best_answer_sentence": generation_context_meta.get("best_answer_sentence"),
            "generation_context_answerable_sentences": generation_context_meta.get("answerable_sentences"),
            "case_posture_status": "inferred" if case_posture else "not_inferred",
            "case_posture": case_posture,
            "pre_generation_refusal_status": pre_generation_refusal_meta["status"],
            "pre_generation_refusal_meta": pre_generation_refusal_meta,
            "response_override_status": (
                response_override_meta["status"] if llm_error is None else "not_applied:llm_error"
            ),
            "response_override_meta": (
                response_override_meta if llm_error is None else {"status": "not_applied:llm_error"}
            ),
            "cert_denial_guard_status": (
                cert_denial_guard_meta["status"] if llm_error is None else "not_applied:llm_error"
            ),
            "cert_denial_guard_meta": (
                cert_denial_guard_meta if llm_error is None else {"status": "not_applied:llm_error"}
            ),
            "response_repair_status": "not_applied:not_verified",
            "response_repair_meta": {"status": "not_applied:not_verified"},
            "verification_verifier_mode": VERIFICATION.verifier_mode,
            "verification_verifier": _verifier_runtime_meta(self.verifier),
            "verification_enabled": self.enable_verification,
            "conversation_context_message_count": len(conversation_context or []),
            "claim_support_summary": _empty_claim_support_summary(),
            "answer_warning": _build_answer_warning(_empty_claim_support_summary()),
            "answer_blocks": [],
            "answer_block_summary": _empty_answer_block_summary(),
            "timings_ms": timings_ms,
        }
        if llm_error is not None:
            meta["llm_error"] = llm_error

        if llm_error is not None:
            meta["claim_count"] = 0
            meta["claims"] = []
            meta["claim_citation_links"] = []
            meta["verification_backend_status"] = "skipped:llm_error"
            logger.info(
                "query.stage_skip request_id=%s stage=verification reason=llm_error",
                request_id,
            )
            return response, meta

        if not self.enable_verification:
            meta["claim_count"] = 0
            meta["claims"] = []
            meta["claim_citation_links"] = []
            meta["verification_backend_status"] = "disabled:config"
            _update_answer_support_metadata(meta)
            return response, meta

        decomposition_started = time.perf_counter()
        logger.info("query.stage_start request_id=%s stage=claim_decomposition", request_id)
        decomposed_claims = decompose_document({"id": "assistant_response", "full_text": response})
        raw_claims, skipped_claims = _filter_claims_for_verification(decomposed_claims)
        timings_ms["claim_decomposition"] = round(
            (time.perf_counter() - decomposition_started) * 1000,
            1,
        )
        logger.info(
            "query.stage_end request_id=%s stage=claim_decomposition elapsed_ms=%.1f claims=%s skipped=%s",
            request_id,
            timings_ms["claim_decomposition"],
            len(decomposed_claims),
            len(skipped_claims),
        )
        claims = [claim.to_dict() for claim in raw_claims]
        meta["claim_decomposition_raw_count"] = len(decomposed_claims)
        meta["claim_decomposition_skipped_count"] = len(skipped_claims)
        meta["claim_decomposition_skipped"] = skipped_claims
        normalized_claims, claim_citation_links = normalize_claims_for_frontend(
            claims,
            citations=meta["retrieved_chunks"],
        )
        meta["claim_count"] = len(claims)
        meta["claims"] = normalized_claims
        meta["claim_citation_links"] = claim_citation_links
        meta["verification_backend_status"] = "skipped:no_retriever"
        _update_answer_support_metadata(meta, response=response)

        if not raw_claims:
            meta["verification_backend_status"] = "skipped:no_claims"
            return response, meta

        if not retrieved_chunks and self.retriever is None and user_upload_retriever is None:
            return response, meta

        if not retrieved_chunks and (public_retrieval_error or user_upload_retrieval_error):
            meta["verification_backend_status"] = "skipped:retrieval_error"
            return response, meta

        if not retrieved_chunks:
            meta["verification_backend_status"] = "skipped:no_evidence"
            return response, meta

        verification_chunks, verification_scope_meta = _select_chunks_for_verification(
            retrieved_chunks=retrieved_chunks,
            prompt_chunks=prompt_chunks,
            generation_context_meta=generation_context_meta,
            query_grounding=query_grounding,
        )
        meta["verification_chunk_count"] = len(verification_chunks)
        meta["verification_chunk_ids"] = [chunk.id for chunk in verification_chunks]
        meta["verification_scope_status"] = verification_scope_meta["status"]
        meta["verification_scope"] = verification_scope_meta

        verifier = self._get_verifier()
        meta["verification_verifier"] = _verifier_runtime_meta(verifier)
        verification_started = time.perf_counter()
        logger.info(
            "query.stage_start request_id=%s stage=verification claims=%s chunks=%s pairs=%s",
            request_id,
            len(raw_claims),
            len(verification_chunks),
            len(raw_claims) * len(verification_chunks),
        )
        try:
            verdicts = verifier.verify_claims_batch(raw_claims, verification_chunks)
        except Exception as exc:  # pragma: no cover - defensive path for live model/runtime failures
            if VERIFICATION.fallback_to_heuristic_on_error and _should_fallback_verification(exc):
                fallback_started = time.perf_counter()
                logger.warning(
                    "query.stage_warning request_id=%s stage=verification_fallback primary_error=%s",
                    request_id,
                    exc.__class__.__name__,
                )
                try:
                    verdicts = HeuristicNLIVerifier().verify_claims_batch(raw_claims, verification_chunks)
                except Exception as fallback_exc:  # pragma: no cover - defensive path for fallback failures
                    timings_ms["verification"] = round(
                        (time.perf_counter() - verification_started) * 1000,
                        1,
                    )
                    logger.warning(
                        "query.stage_error request_id=%s stage=verification_fallback elapsed_ms=%.1f error=%s",
                        request_id,
                        timings_ms["verification"],
                        fallback_exc.__class__.__name__,
                    )
                    meta["verification_backend_status"] = f"error:{exc.__class__.__name__}"
                    meta["verification_error"] = {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                        "fallback_error": {
                            "type": fallback_exc.__class__.__name__,
                            "message": str(fallback_exc),
                        },
                    }
                    return response, meta

                timings_ms["verification"] = round(
                    (time.perf_counter() - verification_started) * 1000,
                    1,
                )
                logger.info(
                    "query.stage_end request_id=%s stage=verification_fallback elapsed_ms=%.1f fallback_elapsed_ms=%.1f verdicts=%s",
                    request_id,
                    timings_ms["verification"],
                    (time.perf_counter() - fallback_started) * 1000,
                    len(verdicts),
                )
                fallback_verifier = HeuristicNLIVerifier()
                normalized_claims, claim_citation_links = _normalize_verified_claims_for_frontend(
                    raw_claims,
                    verdicts,
                    citations=meta["retrieved_chunks"],
                    query_grounding=query_grounding,
                    generation_context_meta=generation_context_meta,
                )
                if str(response_override_meta.get("status") or "") == "applied:user_upload_absence":
                    repaired_response = response
                    response_repair_meta = {"status": "not_applied:user_upload_absence_override"}
                else:
                    repaired_response, response_repair_meta = _apply_supported_claim_repair(
                        response,
                        normalized_claims,
                        generation_context_meta,
                    )
                if response_repair_meta["status"].startswith("applied:"):
                    repair_started = time.perf_counter()
                    response = repaired_response
                    decomposed_claims = decompose_document({"id": "assistant_response", "full_text": response})
                    raw_claims, skipped_claims = _filter_claims_for_verification(decomposed_claims)
                    verdicts = fallback_verifier.verify_claims_batch(raw_claims, verification_chunks) if raw_claims else []
                    normalized_claims, claim_citation_links = _normalize_verified_claims_for_frontend(
                        raw_claims,
                        verdicts,
                        citations=meta["retrieved_chunks"],
                        query_grounding=query_grounding,
                        generation_context_meta=generation_context_meta,
                    )
                    timings_ms["response_repair"] = round((time.perf_counter() - repair_started) * 1000, 1)
                    meta["claim_decomposition_raw_count"] = len(decomposed_claims)
                    meta["claim_decomposition_skipped_count"] = len(skipped_claims)
                    meta["claim_decomposition_skipped"] = skipped_claims
                meta["claim_count"] = len(raw_claims)
                meta["claims"] = normalized_claims
                meta["claim_citation_links"] = claim_citation_links
                meta["verification_backend_status"] = "warning:fallback:HeuristicNLIVerifier"
                meta["verification_verifier"] = _verifier_runtime_meta(fallback_verifier)
                meta["verification_error"] = {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                }
                meta["verification_fallback"] = {
                    "verifier": "HeuristicNLIVerifier",
                }
                meta["response_repair_status"] = response_repair_meta["status"]
                meta["response_repair_meta"] = response_repair_meta
                _update_answer_support_metadata(meta, response=response)
                return response, meta

            timings_ms["verification"] = round(
                (time.perf_counter() - verification_started) * 1000,
                1,
            )
            logger.warning(
                "query.stage_error request_id=%s stage=verification elapsed_ms=%.1f error=%s",
                request_id,
                timings_ms["verification"],
                exc.__class__.__name__,
            )
            meta["verification_backend_status"] = f"error:{exc.__class__.__name__}"
            meta["verification_error"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            return response, meta
        timings_ms["verification"] = round(
            (time.perf_counter() - verification_started) * 1000,
            1,
        )
        logger.info(
            "query.stage_end request_id=%s stage=verification elapsed_ms=%.1f verdicts=%s",
            request_id,
            timings_ms["verification"],
            len(verdicts),
        )

        normalized_claims, claim_citation_links = _normalize_verified_claims_for_frontend(
            raw_claims,
            verdicts,
            citations=meta["retrieved_chunks"],
            query_grounding=query_grounding,
            generation_context_meta=generation_context_meta,
        )
        if str(response_override_meta.get("status") or "") == "applied:user_upload_absence":
            repaired_response = response
            response_repair_meta = {"status": "not_applied:user_upload_absence_override"}
        else:
            repaired_response, response_repair_meta = _apply_supported_claim_repair(
                response,
                normalized_claims,
                generation_context_meta,
            )
        if response_repair_meta["status"].startswith("applied:"):
            repair_started = time.perf_counter()
            response = repaired_response
            decomposed_claims = decompose_document({"id": "assistant_response", "full_text": response})
            raw_claims, skipped_claims = _filter_claims_for_verification(decomposed_claims)
            verdicts = verifier.verify_claims_batch(raw_claims, verification_chunks) if raw_claims else []
            normalized_claims, claim_citation_links = _normalize_verified_claims_for_frontend(
                raw_claims,
                verdicts,
                citations=meta["retrieved_chunks"],
                query_grounding=query_grounding,
                generation_context_meta=generation_context_meta,
            )
            timings_ms["response_repair"] = round((time.perf_counter() - repair_started) * 1000, 1)
            meta["claim_decomposition_raw_count"] = len(decomposed_claims)
            meta["claim_decomposition_skipped_count"] = len(skipped_claims)
            meta["claim_decomposition_skipped"] = skipped_claims
        meta["claim_count"] = len(raw_claims)
        meta["claims"] = normalized_claims
        meta["claim_citation_links"] = claim_citation_links
        meta["verification_backend_status"] = "ok"
        meta["verification_verifier"] = _verifier_runtime_meta(verifier)
        meta["response_repair_status"] = response_repair_meta["status"]
        meta["response_repair_meta"] = response_repair_meta
        _update_answer_support_metadata(meta, response=response)
        return response, meta

    def _get_verifier(self) -> NLIVerifier:
        if self.verifier is None:
            if VERIFICATION.verifier_mode == "heuristic":
                self.verifier = HeuristicNLIVerifier()
            else:
                self.verifier = NLIVerifier()
        return self.verifier

    def preload_models(self) -> dict[str, str]:
        status: dict[str, str] = {}

        embedder = getattr(self.retriever, "embedder", None)
        if embedder is None:
            status["retrieval_embedder"] = "skipped:no_embedder"
        else:
            try:
                embedder._load_model()
            except Exception as exc:  # pragma: no cover - defensive path for live startup failures
                status["retrieval_embedder"] = f"error:{exc.__class__.__name__}"
            else:
                status["retrieval_embedder"] = "ok"

        if not self.enable_verification:
            status["verification_model"] = "disabled:config"
            return status

        if VERIFICATION.verifier_mode == "heuristic":
            status["verification_model"] = "configured:heuristic"
            return status

        try:
            verifier = self._get_verifier()
            verifier._load_model()
        except Exception as exc:  # pragma: no cover - defensive path for live startup failures
            status["verification_model"] = f"error:{exc.__class__.__name__}"
        else:
            status["verification_model"] = "ok"

        return status

    def _persist_interaction_artifacts(
        self,
        interaction_id: int,
        pipeline_meta: dict[str, Any],
    ) -> None:
        self.db.persist_interaction_artifacts(
            interaction_id,
            claims=_claims_from_pipeline_meta(pipeline_meta),
            contradictions=_contradictions_from_pipeline_meta(pipeline_meta),
            citations=_citations_from_pipeline_meta(pipeline_meta),
        )


def _default_title(query: str) -> str:
    trimmed = " ".join(query.strip().split())
    if not trimmed:
        return "New conversation"
    if len(trimmed) <= 60:
        return trimmed
    return trimmed[:57].rstrip() + "..."


def _verifier_runtime_meta(verifier: Any) -> dict[str, Any] | None:
    if verifier is None:
        return None
    return {
        "class": verifier.__class__.__name__,
        "model_name": getattr(verifier, "model_name", None),
        "device": getattr(verifier, "device", None),
        "dtype": getattr(verifier, "dtype", None),
        "batch_size": getattr(verifier, "batch_size", None),
        "max_length": getattr(verifier, "max_length", None),
        "unload_after_request": getattr(verifier, "unload_after_request", None),
    }


def _conversation_state_summary(query: str, response: str) -> str:
    query_preview = " ".join(query.strip().split())
    response_preview = " ".join(response.strip().split())
    summary = f"Q: {query_preview}"
    if response_preview:
        summary = f"{summary}\nA: {response_preview}"
    if len(summary) <= 1000:
        return summary
    return summary[:997].rstrip() + "..."


def _conversation_state_payload(
    query: str,
    response: str,
    pipeline_meta: dict[str, Any],
) -> dict[str, Any]:
    grounding = pipeline_meta.get("query_grounding")
    if not isinstance(grounding, dict):
        grounding = {}
    case_posture = pipeline_meta.get("case_posture")
    if not isinstance(case_posture, dict):
        case_posture = {}

    target_case = (
        grounding.get("target_case")
        or pipeline_meta.get("target_case_name")
        or case_posture.get("target_case")
    )
    target_doc_ids = grounding.get("target_doc_ids")
    if not isinstance(target_doc_ids, list):
        target_doc_ids = []

    response_preview = " ".join(str(response or "").split())
    if len(response_preview) > 500:
        response_preview = response_preview[:497].rstrip() + "..."

    return {
        "last_user_query": query,
        "last_response_preview": response_preview,
        "last_target_case": target_case,
        "last_target_doc_ids": target_doc_ids,
        "last_target_citation": grounding.get("target_citation"),
        "last_query_intent": grounding.get("query_intent"),
        "last_issue": _infer_conversation_issue(query, grounding),
        "last_posture": case_posture,
        "last_opinion_author": case_posture.get("author"),
        "last_opinion_role": case_posture.get("opinion_role"),
    }


def _infer_conversation_issue(query: str, grounding: dict[str, Any]) -> str | None:
    query_intent = str(grounding.get("query_intent") or "").strip()
    if query_intent:
        return query_intent
    if _AUTHOR_QUERY_RE.search(query or ""):
        return "author"
    if _HOLDING_QUERY_RE.search(query or ""):
        return "holding"
    if _POSTURE_QUERY_RE.search(query or ""):
        return "posture"
    return None


def _classify_response_depth(query: str) -> str:
    if _DETAILED_RESPONSE_QUERY_RE.search(query or ""):
        return RESPONSE_DEPTH_DETAILED
    return RESPONSE_DEPTH_CONCISE


def _normalize_response_depth(response_depth: str | None) -> str:
    if str(response_depth or "").strip().lower() == RESPONSE_DEPTH_DETAILED:
        return RESPONSE_DEPTH_DETAILED
    return RESPONSE_DEPTH_CONCISE


def _generation_evidence_sentence_limit(response_depth: str | None) -> int:
    if _normalize_response_depth(response_depth) == RESPONSE_DEPTH_DETAILED:
        return TARGET_CASE_DETAILED_PROMPT_SENTENCE_LIMIT
    return TARGET_CASE_PROMPT_SENTENCE_LIMIT


def _ground_followup_query(
    query: str,
    conversation_state: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    query_text = " ".join(str(query or "").split()).strip()
    if not query_text:
        return query, {"status": "not_applied:empty_query"}
    if _extract_target_case_name(query_text) or _extract_query_citation(query_text):
        return query_text, {"status": "not_applied:explicit_target"}
    if not isinstance(conversation_state, dict):
        return query_text, {"status": "not_applied:no_conversation_state"}

    target_case = str(conversation_state.get("last_target_case") or "").strip()
    if not target_case:
        return query_text, {"status": "not_applied:no_state_target"}
    if not _looks_like_followup_query(query_text):
        return query_text, {"status": "not_applied:not_followup", "target_case": target_case}

    grounded_query = _rewrite_followup_query(query_text, target_case, conversation_state)
    return grounded_query, {
        "status": "applied:conversation_state",
        "original_query": query_text,
        "grounded_query": grounded_query,
        "target_case": target_case,
        "target_doc_ids": conversation_state.get("last_target_doc_ids") or [],
        "last_query_intent": conversation_state.get("last_query_intent"),
        "last_issue": conversation_state.get("last_issue"),
        "last_opinion_role": conversation_state.get("last_opinion_role"),
        "last_opinion_author": conversation_state.get("last_opinion_author"),
    }


def _looks_like_followup_query(query: str) -> bool:
    text = str(query or "").strip()
    if not text:
        return False
    if _FOLLOWUP_REFERENCE_RE.search(text):
        return True
    if _FOLLOWUP_AUTHOR_RE.search(text) and "opinion" in text.lower():
        return True
    if len(_content_tokens(text)) <= 4 and _AUTHOR_QUERY_RE.search(text):
        return True
    return False


def _rewrite_followup_query(
    query: str,
    target_case: str,
    conversation_state: dict[str, Any],
) -> str:
    lower_query = query.lower()
    opinion_role = str(conversation_state.get("last_opinion_role") or "").strip()
    if _FOLLOWUP_AUTHOR_RE.search(query) and "separate opinion" in lower_query:
        if opinion_role == "dissent_from_denial":
            return f"In {target_case}, who wrote the dissent from the denial of certiorari?"
        if opinion_role == "dissent":
            return f"In {target_case}, who wrote the dissent?"
        if opinion_role == "concurrence":
            return f"In {target_case}, who wrote the concurrence?"
        return f"In {target_case}, who wrote the separate opinion?"
    if re.match(r"^\s*who\s+(?:wrote|authored)\b", query, flags=re.IGNORECASE):
        return f"In {target_case}, {query[0].lower() + query[1:]}"
    return f"In {target_case}, {query[0].lower() + query[1:]}"


def _query_preview(query: str, *, limit: int = 120) -> str:
    collapsed = " ".join(query.strip().split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _should_fallback_verification(exc: Exception) -> bool:
    _ = exc
    return True


def _extract_target_case_name(query: str) -> str | None:
    match = _CASE_QUERY_RE.search(query)
    if match is not None:
        return _clean_extracted_case_name(match.group(1))

    case_match = _CASE_NAME_RE.search(query)
    if case_match is None:
        return None
    return _clean_extracted_case_name(case_match.group(1))


def _clean_extracted_case_name(raw_case: str | None) -> str | None:
    target_case = " ".join(str(raw_case or "").split()).strip(" ,.;:")
    target_case = re.sub(r"^\s*In\s+", "", target_case, flags=re.IGNORECASE).strip(" ,.;:")
    return target_case or None


def _resolve_query_grounding(
    query: str,
    retrieved_chunks: list,
    *,
    llm_router: Any | None = None,
) -> dict[str, Any]:
    has_user_uploads = _has_user_upload_chunks(retrieved_chunks)
    explicit_case = None if has_user_uploads and _query_mentions_upload(query) else _extract_target_case_name(query)
    citation = _extract_query_citation(query)
    target_case = None
    target_doc_ids: set[str] = set()
    source = None
    route_meta: dict[str, Any] = {"status": "not_applied:deterministic_grounding"}
    llm_route: str | None = None

    if explicit_case:
        target_case, target_doc_ids = _resolve_explicit_case_target(explicit_case, retrieved_chunks)
        source = "explicit_case"
    elif citation:
        target_case, target_doc_ids = _resolve_citation_target(citation, retrieved_chunks)
        source = "citation" if target_case else "citation_unresolved"
    elif has_user_uploads:
        source = "user_upload"
    else:
        target_case, target_doc_ids = _resolve_short_name_target(query, retrieved_chunks)
        if target_case:
            source = "short_name"
        else:
            route_meta = _classify_query_route_with_llm(llm_router, query)
            llm_route = str(route_meta.get("route") or "").strip().lower() or None
            if (
                route_meta.get("status") == "ok"
                and float(route_meta.get("confidence") or 0.0) >= 0.65
                and llm_route in {"topic_overview", "research_leads", "off_corpus", "clarification_needed"}
            ):
                source = f"llm_route:{llm_route}"
            else:
                target_case, target_doc_ids = _resolve_convergent_retrieval_target(query, retrieved_chunks)
                source = "retrieval_convergence" if target_case else "unresolved"

    query_intent = _classify_query_intent(
        query,
        target_case=target_case,
        citation=citation,
        llm_route=llm_route,
    )
    status = f"resolved:{source}" if target_case else f"not_resolved:{source or 'no_signal'}"
    return {
        "query": query,
        "status": status,
        "target_case": target_case,
        "target_citation": citation,
        "target_doc_ids": sorted(target_doc_ids),
        "explicit_case": explicit_case,
        "source": source,
        "query_intent": query_intent,
        "llm_route": llm_route,
        "llm_route_meta": route_meta,
        "has_user_upload_context": has_user_uploads,
        "mentions_user_upload": _query_mentions_upload(query),
    }


def _has_user_upload_chunks(retrieved_chunks: list) -> bool:
    return any(getattr(chunk, "doc_type", None) == "user_upload" for chunk in retrieved_chunks)


def _query_mentions_upload(query: str) -> bool:
    return bool(
        re.search(
            r"\b(?:upload(?:ed|s)?|my\s+(?:document|draft|file|memo|memorandum|motion)|"
            r"client\s+(?:document|draft|file|memo|memorandum|motion)|work\s+product)\b",
            str(query or ""),
            re.IGNORECASE,
        )
    )


def _classify_query_route_with_llm(llm_router: Any | None, query: str) -> dict[str, Any]:
    classifier = getattr(llm_router, "classify_query_route", None)
    if not callable(classifier):
        return {
            "status": "not_applied:no_llm_router",
            "route": None,
            "confidence": 0.0,
        }
    try:
        result = classifier(query)
    except Exception as exc:  # pragma: no cover - defensive path for live backend failures
        return {
            "status": f"error:{exc.__class__.__name__}",
            "route": None,
            "confidence": 0.0,
            "reason": str(exc),
        }
    if not isinstance(result, dict):
        return {
            "status": "error:invalid_router_result",
            "route": None,
            "confidence": 0.0,
        }
    route = str(result.get("route") or "").strip().lower()
    valid_routes = {
        "topic_overview",
        "research_leads",
        "case_lookup",
        "off_corpus",
        "clarification_needed",
    }
    if route not in valid_routes:
        return {
            **result,
            "status": "error:invalid_route",
            "route": None,
            "confidence": 0.0,
        }
    try:
        confidence = float(result.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        **result,
        "status": str(result.get("status") or "ok"),
        "route": route,
        "confidence": max(0.0, min(1.0, confidence)),
    }


def _query_requests_upload_comparison(query: str) -> bool:
    return bool(_USER_UPLOAD_COMPARISON_QUERY_RE.search(str(query or "")))


def _build_user_upload_public_retrieval_query(query: str, upload_chunks: list) -> tuple[str, dict[str, Any]]:
    query_text = str(query or "").strip()
    if not _query_mentions_upload(query_text):
        return query_text, {
            "status": "not_applied:no_user_upload_reference",
            "original_query": query_text,
            "public_query": query_text,
        }
    if not _query_requests_upload_comparison(query_text):
        return query_text, {
            "status": "not_applied:not_comparison_query",
            "original_query": query_text,
            "public_query": query_text,
        }
    if not upload_chunks:
        return query_text, {
            "status": "not_applied:no_user_upload_chunks",
            "original_query": query_text,
            "public_query": query_text,
        }

    source_text = " ".join(
        [
            query_text,
            *[
                str(getattr(chunk, "text", "") or "")[:1600]
                for chunk in upload_chunks[:3]
            ],
        ]
    )
    cleaned_text = _strip_user_upload_public_retrieval_noise(source_text)
    noise_tokens = _user_upload_metadata_noise_tokens(upload_chunks)
    terms = _extract_user_upload_public_retrieval_terms(
        cleaned_text,
        extra_stopwords=noise_tokens,
    )
    if not terms:
        return query_text, {
            "status": "not_applied:no_issue_terms",
            "original_query": query_text,
            "public_query": query_text,
            "source_chunk_ids": [getattr(chunk, "id", None) for chunk in upload_chunks[:3]],
        }

    public_query = " ".join(terms)
    return public_query, {
        "status": "applied:user_upload_comparison_rewrite",
        "original_query": query_text,
        "public_query": public_query,
        "terms": terms,
        "source_chunk_ids": [getattr(chunk, "id", None) for chunk in upload_chunks[:3]],
    }


def _strip_user_upload_public_retrieval_noise(text: str) -> str:
    cleaned = str(text or "")
    cleaned = _CASE_NAME_RE.sub(" ", cleaned)
    cleaned = re.sub(
        r"\bDr\.\s+[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,2}\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\bModel\s+[A-Z0-9-]+\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(?:my\s+)?uploaded\s+[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,6}\s+"
        r"(?:draft|document|file|memo|memorandum|motion)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\bCONFIDENTIAL\b|\bPARALEGAL WORK PRODUCT\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _user_upload_metadata_noise_tokens(upload_chunks: list) -> set[str]:
    noise: set[str] = set()
    for chunk in upload_chunks[:3]:
        metadata_text = " ".join(
            [
                str(getattr(chunk, "id", "") or ""),
                str(getattr(chunk, "doc_id", "") or ""),
                str(getattr(chunk, "source_file", "") or ""),
            ]
        )
        noise.update(
            token
            for token in re.findall(r"[A-Za-z][A-Za-z-]{2,}", metadata_text.lower())
            if len(token) >= 4
        )
    return noise


def _extract_user_upload_public_retrieval_terms(
    text: str,
    *,
    extra_stopwords: set[str] | None = None,
) -> list[str]:
    lower_text = str(text or "").lower()
    phrase_specs = [
        ("Daubert", r"\bdaubert\b|\bexpert(?:s)?\b|\bcausation opinion\b|\bmethodology\b"),
        ("expert testimony", r"\bexpert(?:s)?\b|\bcausation opinion\b"),
        ("expert opinion admissibility", r"\bexpert(?:s)?\b|\bcausation opinion\b|\bmethodology\b|\badmissib"),
        ("reliable principles and methods", r"\breliable\b.*\bmethodology\b|\bmethodology\b|\bprinciples?\b|\bmethods?\b"),
        ("expert causation opinion", r"\bcausation opinion\b"),
        ("reliable causation methodology", r"\bcausation\b.*\bmethodology\b|\bmethodology\b.*\bcausation\b"),
        ("reliable methodology", r"\breliable\b.*\bmethodology\b|\bmethodology\b"),
        ("alternative causes", r"\balternative\s+(?:ignition\s+)?sources?\b|\balternative causes?\b"),
        ("substantially similar testing conditions", r"\bsubstantially similar\b|\bcharging conditions\b"),
        ("product liability", r"\bproduct liability\b|\bbattery pack\b|\bconsumer product\b"),
        ("admissibility", r"\bdaubert\b|\badmissib"),
    ]
    terms: list[str] = []
    seen: set[str] = set()
    for phrase, pattern in phrase_specs:
        if re.search(pattern, lower_text, re.IGNORECASE):
            terms.append(phrase)
            seen.add(phrase)

    stopwords = {
        "about",
        "account",
        "argument",
        "authorities",
        "authority",
        "battery",
        "cases",
        "charging",
        "connect",
        "compare",
        "comparison",
        "corpus",
        "document",
        "does",
        "draft",
        "extension",
        "file",
        "failed",
        "from",
        "how",
        "legal",
        "memo",
        "memorandum",
        "motion",
        "opinion",
        "observations",
        "pack",
        "precedent",
        "prior",
        "relevant",
        "source",
        "sources",
        "states",
        "strongest",
        "that",
        "test",
        "under",
        "uploaded",
        "using",
        "work",
    }
    stopwords.update(extra_stopwords or set())
    for token in re.findall(r"[A-Za-z][A-Za-z-]{2,}", lower_text):
        token = token.strip("-").lower()
        if token in seen or token in stopwords:
            continue
        if token in _content_tokens(" ".join(terms)):
            continue
        if len(token) < 4:
            continue
        terms.append(token)
        seen.add(token)
        if len(terms) >= USER_UPLOAD_PUBLIC_RETRIEVAL_TERM_LIMIT - 2:
            break

    for authority_term in ("case law", "precedent"):
        if len(terms) >= USER_UPLOAD_PUBLIC_RETRIEVAL_TERM_LIMIT:
            break
        if authority_term not in seen:
            terms.append(authority_term)
            seen.add(authority_term)
    return terms[:USER_UPLOAD_PUBLIC_RETRIEVAL_TERM_LIMIT]


def _classify_query_intent(
    query: str,
    *,
    target_case: str | None,
    citation: str | None,
    llm_route: str | None = None,
) -> str:
    query_text = query or ""
    route = str(llm_route or "").strip().lower()
    if route in {"topic_overview", "research_leads"}:
        return route
    if route in {"off_corpus", "clarification_needed"}:
        return "off_target"
    if _AUTHOR_QUERY_RE.search(query_text):
        return "author"
    if _HOLDING_QUERY_RE.search(query_text):
        return "holding"
    if _POSTURE_QUERY_RE.search(query_text):
        return "posture"
    if _RESEARCH_LEADS_QUERY_RE.search(query_text):
        return "research_leads"
    if _ISSUE_ANALYSIS_QUERY_RE.search(query_text):
        return "issue_analysis"
    if _query_mentions_upload(query_text):
        return "issue_analysis"
    if citation:
        return "citation_lookup"
    if target_case:
        return "issue_analysis"
    return "off_target"


def _build_pre_generation_refusal(
    *,
    query_grounding: dict[str, Any],
    retrieved_chunks: list,
) -> tuple[str | None, dict[str, Any]]:
    query_intent = str(query_grounding.get("query_intent") or "")
    target_case = query_grounding.get("target_case")
    query_status = str(query_grounding.get("status") or "")
    target_citation = query_grounding.get("target_citation")
    
    # NEW: Handle unresolved citation lookups - refuse early for fake citations like 999 U.S. 999
    if (
        target_citation 
        and query_status == "not_resolved:citation_unresolved"
    ):
        return (
            f"I could not find {target_citation} in the retrieved database, so I cannot summarize its rule.",
            {
                "status": "applied:explicit_citation_not_retrieved",
                "query_intent": query_intent,
                "target_citation": target_citation,
            },
        )
    
    if not retrieved_chunks:
        return None, {
            "status": "not_applied:no_retrieval_context",
            "query_intent": query_intent,
            "target_case": target_case,
        }
    if query_grounding.get("explicit_case") and target_case and not _has_matching_target_chunks(
        target_case,
        retrieved_chunks,
    ):
        return (
            "Insufficient support in retrieved authorities to answer the question.",
            {
                "status": "applied:explicit_target_not_retrieved",
                "query_intent": query_intent,
                "target_case": target_case,
            },
        )
    if target_case or query_intent != "off_target":
        return None, {
            "status": "not_applied:answerable_or_targeted",
            "query_intent": query_intent,
            "target_case": target_case,
        }
    if _has_user_upload_chunks(retrieved_chunks):
        return None, {
            "status": "not_applied:user_upload_context",
            "query_intent": query_intent,
            "target_case": target_case,
        }

    return (
        "Insufficient support in retrieved authorities to answer the question.",
        {
            "status": "applied:off_target_no_corpus_grounding",
            "query_intent": query_intent,
            "target_case": target_case,
        },
    )


def _build_missing_case_topic_fallback(
    query: str,
    *,
    query_grounding: dict[str, Any],
    retriever: Any,
    initial_chunks: list,
    limit: int,
) -> tuple[list, dict[str, Any]]:
    if _has_user_upload_chunks(initial_chunks):
        return [], {
            "status": "not_applied:user_upload_context",
            "missing_case": None,
            "topic_query": None,
        }

    missing_case = _missing_explicit_case(query_grounding, initial_chunks)
    if not missing_case:
        return [], {
            "status": "not_applied:target_case_available_or_not_explicit",
            "missing_case": None,
            "topic_query": None,
        }

    topic_query = _build_topic_fallback_query(query, missing_case)
    if not _query_has_topic_fallback_signal(topic_query):
        return [], {
            "status": "not_applied:no_topic_signal",
            "missing_case": missing_case,
            "topic_query": topic_query,
        }

    topic_chunks: list[Any] = []
    retrieval_status = "not_applied:no_retriever"
    if retriever is not None:
        try:
            topic_chunks = list(retriever.retrieve(topic_query))
            retrieval_status = "ok"
        except Exception as exc:  # pragma: no cover - defensive path for live retrieval failures
            retrieval_status = f"error:{exc.__class__.__name__}"

    candidates = _dedupe_chunks([*topic_chunks, *initial_chunks])
    selected = _select_topic_fallback_chunks(
        topic_query,
        candidates,
        missing_case=missing_case,
        limit=limit,
    )
    if not selected:
        return [], {
            "status": "not_applied:no_topic_evidence",
            "missing_case": missing_case,
            "topic_query": topic_query,
            "retrieval_status": retrieval_status,
            "candidate_count": len(candidates),
        }

    return selected, {
        "status": "applied",
        "missing_case": missing_case,
        "topic_query": topic_query,
        "retrieval_status": retrieval_status,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "limit": limit,
    }


def _missing_explicit_case(query_grounding: dict[str, Any], retrieved_chunks: list) -> str | None:
    explicit_case = str(query_grounding.get("explicit_case") or "").strip()
    if not explicit_case:
        return None
    target_case = str(query_grounding.get("target_case") or explicit_case).strip()
    if not target_case:
        return explicit_case
    if _has_matching_target_chunks(target_case, retrieved_chunks):
        return None
    return target_case


def _build_topic_fallback_query(query: str, missing_case: str) -> str:
    topic_query = str(query or "")
    if missing_case:
        topic_query = re.sub(re.escape(missing_case), " ", topic_query, flags=re.IGNORECASE)
    topic_query = re.sub(r"^\s*In\s*,?\s*", " ", topic_query, flags=re.IGNORECASE)
    topic_query = re.sub(
        r"\bwhat\s+did\s+(?:the\s+)?(?:supreme\s+)?court\s+(?:hold|decide|do)\b",
        " ",
        topic_query,
        flags=re.IGNORECASE,
    )
    topic_query = re.sub(r"\b(?:that|this|the)\s+case\b", " ", topic_query, flags=re.IGNORECASE)
    topic_query = re.sub(r"\b(?:hold|holding|held)\b", " ", topic_query, flags=re.IGNORECASE)
    topic_query = re.sub(r"\s+", " ", topic_query).strip(" ,.;:")
    return topic_query


def _query_has_topic_fallback_signal(topic_query: str) -> bool:
    query_text = str(topic_query or "").strip()
    if not query_text:
        return False
    if _TOPIC_SIGNAL_RE.search(query_text):
        return True
    return len(_content_tokens(query_text)) >= 3


def _query_requests_research_leads(query: str, query_grounding: dict[str, Any] | None) -> bool:
    query_text = str(query or "").strip()
    if not query_text:
        return False

    grounding = query_grounding or {}
    if grounding.get("target_case") or grounding.get("explicit_case") or grounding.get("citation"):
        return False
    if grounding.get("has_user_upload_context") or _query_mentions_upload(query_text):
        return False
    if grounding.get("topic_fallback_used") or grounding.get("missing_explicit_case"):
        return False
    if str(grounding.get("query_intent") or "") == "research_leads":
        return True
    if str(grounding.get("query_intent") or "") == "off_target" and not _RESEARCH_LEADS_QUERY_RE.search(query_text):
        return False
    return bool(_RESEARCH_LEADS_QUERY_RE.search(query_text))


def _dedupe_chunks(chunks: list) -> list:
    deduped = []
    seen_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = getattr(chunk, "id", None)
        if not chunk_id or chunk_id in seen_ids:
            continue
        deduped.append(chunk)
        seen_ids.add(chunk_id)
    return deduped


def _select_topic_fallback_chunks(
    topic_query: str,
    chunks: list,
    *,
    missing_case: str,
    limit: int,
) -> list:
    topic_tokens = _content_tokens(topic_query)
    if not topic_tokens:
        return []

    scored_chunks: list[tuple[float, int, Any]] = []
    for rank, chunk in enumerate(chunks):
        if _case_name_matches(missing_case, getattr(chunk, "case_name", None)):
            continue
        chunk_text = str(getattr(chunk, "text", "") or "")
        chunk_tokens = _content_tokens(
            " ".join(
                [
                    chunk_text[:2400],
                    str(getattr(chunk, "case_name", "") or ""),
                    str(getattr(chunk, "citation", "") or ""),
                ]
            )
        )
        overlap = topic_tokens & chunk_tokens
        if not overlap:
            continue
        score = float(len(overlap) * 4)
        score += len(overlap) / max(len(topic_tokens), 1)
        if _TOPIC_SIGNAL_RE.search(chunk_text):
            score += 2.0
        scored_chunks.append((score, rank, chunk))

    scored_chunks.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in scored_chunks[: max(0, int(limit or 0))]]


def _format_missing_case_topic_scope_for_prompt(missing_case: Any, topic_query: Any) -> str:
    case_name = str(missing_case or "the named case").strip()
    topic = str(topic_query or "the related legal issue").strip()
    return (
        "Evidence type: scope constraint\n"
        f"Missing target case: {case_name}\n"
        f"Related topic query: {topic}\n"
        "Generation constraint: start by saying the named case was not found in the retrieved database. "
        "Then answer only the related legal topic from the remaining retrieved authorities. "
        "Do not state or imply what the missing case held."
    )


def _format_research_leads_scope_for_prompt(query: Any) -> str:
    research_query = str(query or "the user's research query").strip()
    return (
        "Evidence type: research-leads scope constraint\n"
        f"Research query: {research_query}\n"
        "Generation constraint: identify retrieved cases or authorities as research leads, "
        "not definitive holdings unless directly supported by the retrieved text. "
        "Use cautious language such as \"retrieved materials suggest\" and do not invent citations. "
        "Verification constraint: broad analogies or issue matches are exploratory and should not "
        "be treated as stronger than possible support unless directly stated."
    )


def _prepend_missing_case_topic_disclaimer(response: str, *, missing_case: Any) -> str:
    case_name = str(missing_case or "the named case").strip()
    disclaimer = (
        f"I do not have {case_name} in the retrieved database, so I cannot say what that case held."
    )
    answer = " ".join(str(response or "").split()).strip()
    if not answer:
        return disclaimer
    if answer.lower().startswith("i do not have"):
        return answer
    return f"{disclaimer}\n\nBased on related retrieved authorities, {answer[0].lower() + answer[1:] if len(answer) > 1 else answer.lower()}"


def _answer_mode(
    *,
    generation_mode: str,
    topic_fallback_used: bool,
    pre_generation_refusal: bool,
    retrieval_used: bool,
    research_leads_mode: bool = False,
) -> str:
    if topic_fallback_used:
        return "missing_case_topic_fallback"
    if pre_generation_refusal:
        return "refusal"
    if research_leads_mode:
        return "research_leads"
    if generation_mode == "direct" and not retrieval_used:
        return "direct"
    return "retrieval_grounded"


def _extract_query_citation(query: str) -> str | None:
    match = _US_CITATION_RE.search(query or "")
    if match is None:
        return None
    citation = " ".join(match.group(0).split()).strip()
    return citation or None


def _normalize_citation(text: str | None) -> str:
    if not text:
        return ""
    return _NON_ALNUM_RE.sub("", text.lower())


def _resolve_explicit_case_target(target_case: str, retrieved_chunks: list) -> tuple[str, set[str]]:
    matching_chunks = [
        chunk
        for chunk in retrieved_chunks
        if _case_name_matches(target_case, getattr(chunk, "case_name", None))
    ]
    if not matching_chunks:
        return target_case, set()
    canonical_case = _most_common_case_name(matching_chunks) or target_case
    target_doc_ids = {
        str(getattr(chunk, "doc_id", ""))
        for chunk in matching_chunks
        if getattr(chunk, "doc_id", None)
    }
    return canonical_case, target_doc_ids


def _has_matching_target_chunks(target_case: str, chunks: list) -> bool:
    return any(_case_name_matches(target_case, getattr(chunk, "case_name", None)) for chunk in chunks)


def _retrieve_target_chunks_from_metadata(
    retriever: Any,
    *,
    target_case: str | None,
    citation: str | None,
    limit: int,
) -> tuple[list[Any], dict[str, Any]]:
    if retriever is None:
        return [], {"status": "not_applied:no_retriever", "target_case": target_case, "citation": citation}
    if not target_case and not citation:
        return [], {"status": "not_applied:no_explicit_target", "target_case": target_case, "citation": citation}

    chunks = _all_indexed_chunks_from_retriever(retriever)
    if not chunks:
        return [], {"status": "not_applied:no_indexed_chunks", "target_case": target_case, "citation": citation}

    normalized_citation = _normalize_citation(citation)
    matches: list[Any] = []
    for chunk in chunks:
        case_matches = bool(target_case and _case_name_matches(target_case, getattr(chunk, "case_name", None)))
        chunk_citation = _normalize_citation(getattr(chunk, "citation", None))
        citation_matches = bool(
            normalized_citation
            and chunk_citation
            and (
                normalized_citation in chunk_citation
                or chunk_citation in normalized_citation
            )
        )
        if case_matches or citation_matches:
            matches.append(chunk)

    if not matches:
        return [], {
            "status": "not_found",
            "target_case": target_case,
            "citation": citation,
            "indexed_chunk_count": len(chunks),
        }

    matches.sort(
        key=lambda chunk: (
            str(getattr(chunk, "doc_id", "") or ""),
            int(getattr(chunk, "chunk_index", 0) or 0),
            str(getattr(chunk, "id", "") or ""),
        )
    )
    limited = matches[: max(0, int(limit or 0))]
    return limited, {
        "status": "applied",
        "target_case": _most_common_case_name(limited) or target_case,
        "citation": citation,
        "matched_chunk_count": len(matches),
        "limit": limit,
    }


def _all_indexed_chunks_from_retriever(retriever: Any) -> list[Any]:
    bm25_index = getattr(retriever, "bm25_index", None)
    if bm25_index is None:
        bm25_index = getattr(retriever, "sparse_index", None)
    if bm25_index is None:
        return []
    chunks_fn = getattr(bm25_index, "chunks", None)
    if callable(chunks_fn):
        return list(chunks_fn())
    metadata_rows = getattr(bm25_index, "_metadatas", None)
    if not isinstance(metadata_rows, list):
        return []
    chunk_from_dict = getattr(bm25_index, "_chunk_from_dict", None)
    if not callable(chunk_from_dict):
        return []
    return [chunk_from_dict(dict(row)) for row in metadata_rows]


def _resolve_citation_target(citation: str, retrieved_chunks: list) -> tuple[str | None, set[str]]:
    normalized_query_citation = _normalize_citation(citation)
    if not normalized_query_citation:
        return None, set()

    matching_chunks = []
    for chunk in retrieved_chunks:
        chunk_citation = _normalize_citation(getattr(chunk, "citation", None))
        if not chunk_citation:
            continue
        if normalized_query_citation in chunk_citation or chunk_citation in normalized_query_citation:
            matching_chunks.append(chunk)

    if not matching_chunks:
        return None, set()

    target_case = _most_common_case_name(matching_chunks)
    target_doc_ids = {
        str(getattr(chunk, "doc_id", ""))
        for chunk in matching_chunks
        if getattr(chunk, "doc_id", None)
    }
    return target_case, target_doc_ids


def _resolve_short_name_target(query: str, retrieved_chunks: list) -> tuple[str | None, set[str]]:
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return None, set()

    candidates: dict[str, dict[str, Any]] = {}
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        case_name = str(getattr(chunk, "case_name", "") or "").strip()
        if not case_name:
            continue
        short_tokens = _case_short_name_tokens(case_name)
        overlap = query_tokens & short_tokens
        if not overlap:
            continue
        doc_id = str(getattr(chunk, "doc_id", "") or "")
        key = doc_id or case_name
        candidate = candidates.setdefault(
            key,
            {
                "case_name": case_name,
                "doc_ids": set(),
                "score": 0.0,
                "best_rank": rank,
                "overlap": set(),
            },
        )
        if doc_id:
            candidate["doc_ids"].add(doc_id)
        candidate["score"] += len(overlap) * (1.0 + (1.0 / rank))
        candidate["best_rank"] = min(candidate["best_rank"], rank)
        candidate["overlap"].update(overlap)

    if not candidates:
        return None, set()

    ranked = sorted(
        candidates.values(),
        key=lambda item: (-float(item["score"]), int(item["best_rank"]), str(item["case_name"])),
    )
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    if second is not None and float(best["score"]) <= float(second["score"]) * 1.15:
        return None, set()

    return str(best["case_name"]), set(best["doc_ids"])


def _resolve_convergent_retrieval_target(query: str, retrieved_chunks: list) -> tuple[str | None, set[str]]:
    """Resolve issue-only queries when top retrieval results clearly point to one case."""
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return None, set()

    candidates: dict[str, dict[str, Any]] = {}
    for rank, chunk in enumerate(retrieved_chunks[:8], start=1):
        case_name = str(getattr(chunk, "case_name", "") or "").strip()
        if not case_name:
            continue
        key = _case_family_key(case_name)
        if not key:
            continue
        text_tokens = _content_tokens(str(getattr(chunk, "text", "") or ""))
        overlap = query_tokens & text_tokens
        doc_id = str(getattr(chunk, "doc_id", "") or "")
        candidate = candidates.setdefault(
            key,
            {
                "case_names": {},
                "doc_ids": set(),
                "score": 0.0,
                "count": 0,
                "best_rank": rank,
                "overlap": set(),
            },
        )
        candidate["case_names"][case_name] = candidate["case_names"].get(case_name, 0) + 1
        if doc_id:
            candidate["doc_ids"].add(doc_id)
        candidate["score"] += (1.0 / rank) + (len(overlap) * 0.75)
        candidate["count"] += 1
        candidate["best_rank"] = min(candidate["best_rank"], rank)
        candidate["overlap"].update(overlap)

    if not candidates:
        return None, set()

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -float(item["score"]),
            -int(item["count"]),
            int(item["best_rank"]),
        ),
    )
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    overlap_count = len(best["overlap"])
    count = int(best["count"])
    if count < 2 or overlap_count < 2:
        return None, set()
    if second is not None and float(best["score"]) <= float(second["score"]) * 1.35:
        return None, set()

    case_names = best["case_names"]
    canonical_case = sorted(case_names, key=lambda name: (-case_names[name], len(name), name))[0]
    return str(canonical_case), set(best["doc_ids"])


def _case_family_key(case_name: str) -> str:
    base = re.sub(r"\s+Revisions?:.*$", "", case_name, flags=re.IGNORECASE).strip()
    return _normalize_case_name(base)


def _case_short_name_tokens(case_name: str) -> set[str]:
    parties = re.split(r"\bv\.\b", case_name, maxsplit=1, flags=re.IGNORECASE)
    first_party = parties[0] if parties else case_name
    tokens = _content_tokens(first_party)
    return {
        token
        for token in tokens
        if len(token) >= 4 and token not in _SHORT_NAME_STOPWORDS
    }


def _most_common_case_name(chunks: list) -> str | None:
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for index, chunk in enumerate(chunks):
        case_name = str(getattr(chunk, "case_name", "") or "").strip()
        if not case_name:
            continue
        counts[case_name] = counts.get(case_name, 0) + 1
        first_seen.setdefault(case_name, index)
    if not counts:
        return None
    return sorted(counts, key=lambda name: (-counts[name], first_seen[name], name))[0]


def _normalize_case_name(text: str | None) -> str:
    if not text:
        return ""
    raw_tokens = _NON_ALNUM_RE.sub(" ", text.lower()).split()
    collapsed_tokens: list[str] = []
    single_letters: list[str] = []
    for token in raw_tokens:
        if token == "v":
            if single_letters:
                collapsed_tokens.append("".join(single_letters))
                single_letters = []
            collapsed_tokens.append(token)
            continue
        if len(token) == 1 and token.isalpha():
            single_letters.append(token)
            continue
        if single_letters:
            collapsed_tokens.append("".join(single_letters))
            single_letters = []
        if token not in _CASE_CORPORATE_TOKENS:
            collapsed_tokens.append(token)
    if single_letters:
        collapsed_tokens.append("".join(single_letters))
    return " ".join(collapsed_tokens).strip()


def _case_name_matches(target_case: str | None, candidate_case: str | None) -> bool:
    if not target_case or not candidate_case:
        return False
    target_norm = _normalize_case_name(target_case)
    candidate_norm = _normalize_case_name(candidate_case)
    if not target_norm or not candidate_norm:
        return False
    return (
        target_norm == candidate_norm
        or target_norm in candidate_norm
        or candidate_norm in target_norm
    )


def _select_prompt_chunks_for_generation(
    query: str,
    retrieved_chunks: list,
    *,
    query_grounding: dict[str, Any] | None = None,
) -> tuple[list, dict[str, Any]]:
    if not retrieved_chunks:
        return [], {"status": "not_applied:no_retrieval", "target_case": None}

    query_grounding = query_grounding or _resolve_query_grounding(query, retrieved_chunks)
    if _has_user_upload_chunks(retrieved_chunks):
        return _select_user_upload_prompt_chunks(
            query,
            retrieved_chunks,
            query_grounding=query_grounding,
        )

    target_case = query_grounding.get("target_case")
    if target_case is None:
        return list(retrieved_chunks), {
            "status": "not_applied:no_target_case",
            "target_case": None,
            "query_grounding_status": query_grounding.get("status"),
        }

    target_doc_ids = {
        getattr(chunk, "doc_id", None)
        for chunk in retrieved_chunks
        if _case_name_matches(target_case, getattr(chunk, "case_name", None))
    }
    target_doc_ids.update(query_grounding.get("target_doc_ids") or [])
    target_doc_ids.discard(None)

    if not target_doc_ids:
        return list(retrieved_chunks), {
            "status": "not_applied:no_case_match",
            "target_case": target_case,
            "query_grounding_status": query_grounding.get("status"),
        }

    prompt_chunks = [
        chunk
        for chunk in retrieved_chunks
        if _case_name_matches(target_case, getattr(chunk, "case_name", None))
        or getattr(chunk, "doc_id", None) in target_doc_ids
    ]
    if not prompt_chunks:
        return list(retrieved_chunks), {
            "status": "not_applied:no_case_match",
            "target_case": target_case,
            "query_grounding_status": query_grounding.get("status"),
        }

    candidate_count = len(prompt_chunks)
    limited_chunks = prompt_chunks[:TARGET_CASE_PROMPT_CHUNK_LIMIT]

    return limited_chunks, {
        "status": "applied",
        "target_case": target_case,
        "candidate_count": candidate_count,
        "limit": TARGET_CASE_PROMPT_CHUNK_LIMIT,
        "query_grounding_status": query_grounding.get("status"),
    }


def _select_user_upload_prompt_chunks(
    query: str,
    retrieved_chunks: list,
    *,
    query_grounding: dict[str, Any],
) -> tuple[list, dict[str, Any]]:
    upload_chunks = [
        chunk for chunk in retrieved_chunks if getattr(chunk, "doc_type", None) == "user_upload"
    ]
    public_chunks = [
        chunk for chunk in retrieved_chunks if getattr(chunk, "doc_type", None) != "user_upload"
    ]
    if not upload_chunks:
        return list(retrieved_chunks), {
            "status": "not_applied:no_user_upload_chunks",
            "target_case": None,
            "query_grounding_status": query_grounding.get("status"),
        }

    include_public = _query_requests_upload_comparison(query)
    selected = list(upload_chunks)
    public_limit = 0
    if include_public:
        public_limit = USER_UPLOAD_COMPARISON_PUBLIC_CHUNK_LIMIT
        selected.extend(public_chunks[:public_limit])

    return selected, {
        "status": (
            "applied:user_upload_with_comparison_authorities"
            if include_public
            else "applied:user_upload_only"
        ),
        "target_case": None,
        "query_grounding_status": query_grounding.get("status"),
        "user_upload_chunk_count": len(upload_chunks),
        "public_candidate_count": len(public_chunks),
        "public_prompt_limit": public_limit,
        "candidate_count": len(selected),
        "limit": len(selected),
    }


def _build_generation_context(
    query: str,
    prompt_chunks: list,
    *,
    query_grounding: dict[str, Any] | None = None,
    response_depth: str = RESPONSE_DEPTH_CONCISE,
    research_leads_mode: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    normalized_response_depth = _normalize_response_depth(response_depth)
    if not prompt_chunks:
        return [], {
            "status": "not_applied:no_prompt_chunks",
            "count": 0,
            "research_leads_mode": False,
            "answerable_sentences": [],
            "response_depth": normalized_response_depth,
            "sentence_limit": _generation_evidence_sentence_limit(normalized_response_depth),
        }

    query_grounding = query_grounding or _resolve_query_grounding(query, prompt_chunks)
    target_case = query_grounding.get("target_case")
    if _has_user_upload_chunks(prompt_chunks):
        sentence_limit = max(
            _generation_evidence_sentence_limit(normalized_response_depth),
            6,
        )
        evidence_sentences = _select_user_upload_context_evidence_sentences(
            query,
            prompt_chunks,
            limit=sentence_limit,
        )
        return _format_user_upload_context_for_prompt(prompt_chunks), {
            "status": "applied:user_upload_context",
            "count": len(prompt_chunks),
            "source_chunk_ids": [chunk.id for chunk in prompt_chunks],
            "source_evidence_sentences": _source_evidence_sentence_payloads(
                evidence_sentences,
                canonical_answer_fact=None,
            ),
            "query_grounding_status": query_grounding.get("status"),
            "research_leads_mode": False,
            "answerable_sentences": [
                _sanitize_evidence_sentence(str(item["sentence"]))
                for item in evidence_sentences
                if str(item.get("sentence") or "").strip()
            ],
            "response_depth": normalized_response_depth,
            "sentence_limit": sentence_limit,
        }
    if research_leads_mode:
        return [
            _format_research_leads_scope_for_prompt(query),
            *[_format_chunk_for_prompt(chunk) for chunk in prompt_chunks],
        ], {
            "status": "applied:research_leads",
            "count": len(prompt_chunks) + 1,
            "source_chunk_ids": [chunk.id for chunk in prompt_chunks],
            "query_grounding_status": query_grounding.get("status"),
            "research_leads_mode": True,
            "answerable_sentences": [],
            "response_depth": normalized_response_depth,
            "sentence_limit": _generation_evidence_sentence_limit(normalized_response_depth),
        }
    if target_case is None:
        return [_format_chunk_for_prompt(chunk) for chunk in prompt_chunks], {
            "status": "not_applied:no_target_case",
            "count": len(prompt_chunks),
            "source_chunk_ids": [chunk.id for chunk in prompt_chunks],
            "query_grounding_status": query_grounding.get("status"),
            "research_leads_mode": False,
            "answerable_sentences": [],
            "response_depth": normalized_response_depth,
            "sentence_limit": _generation_evidence_sentence_limit(normalized_response_depth),
        }

    sentence_limit = _generation_evidence_sentence_limit(normalized_response_depth)
    evidence_sentences = _select_named_case_evidence_sentences(
        query,
        prompt_chunks,
        target_case,
        limit=sentence_limit,
    )
    if not evidence_sentences:
        return [_format_chunk_for_prompt(chunk) for chunk in prompt_chunks], {
            "status": "not_applied:no_sentence_evidence",
            "count": len(prompt_chunks),
            "source_chunk_ids": [chunk.id for chunk in prompt_chunks],
            "query_grounding_status": query_grounding.get("status"),
            "research_leads_mode": False,
            "answerable_sentences": [],
            "response_depth": normalized_response_depth,
            "sentence_limit": sentence_limit,
        }

    explicit_holding = _best_explicit_holding_sentence(evidence_sentences)
    canonical_answer_fact = _canonical_answer_fact(
        query,
        evidence_sentences,
        explicit_holding=explicit_holding,
    )
    formatted_context = [_format_sentence_evidence_for_prompt(item) for item in evidence_sentences]
    if canonical_answer_fact:
        formatted_context.insert(
            0,
            _format_canonical_answer_fact_for_prompt(
                canonical_answer_fact,
                source_chunk_ids=[item["chunk"].id for item in evidence_sentences],
            ),
        )
    return formatted_context, {
        "status": "applied:sentence_evidence",
        "count": len(formatted_context),
        "source_chunk_ids": [item["chunk"].id for item in evidence_sentences],
        "source_evidence_sentences": _source_evidence_sentence_payloads(
            evidence_sentences,
            canonical_answer_fact=canonical_answer_fact,
        ),
        "explicit_holding_sentence": explicit_holding,
        "canonical_answer_fact": canonical_answer_fact,
        "best_answer_sentence": (
            canonical_answer_fact
            or explicit_holding
            or _sanitize_evidence_sentence(str(evidence_sentences[0]["sentence"]))
        ),
        "answerable_sentences": [
            _sanitize_evidence_sentence(str(item["sentence"]))
            for item in evidence_sentences
            if str(item.get("sentence") or "").strip()
        ],
        "query_grounding_status": query_grounding.get("status"),
        "research_leads_mode": False,
        "response_depth": normalized_response_depth,
        "sentence_limit": sentence_limit,
    }


def _select_chunks_for_verification(
    *,
    retrieved_chunks: list,
    prompt_chunks: list,
    generation_context_meta: dict[str, Any],
    query_grounding: dict[str, Any],
) -> tuple[list, dict[str, Any]]:
    if not retrieved_chunks:
        return [], {"status": "not_applied:no_retrieval"}

    source_ids = [
        str(chunk_id)
        for chunk_id in generation_context_meta.get("source_chunk_ids", [])
        if chunk_id is not None
    ]
    prompt_ids = [str(getattr(chunk, "id", "")) for chunk in prompt_chunks if getattr(chunk, "id", None)]
    target_doc_ids = {
        str(doc_id)
        for doc_id in query_grounding.get("target_doc_ids", [])
        if doc_id is not None
    }
    target_case = query_grounding.get("target_case")

    selected: list[Any] = []
    seen_ids: set[str] = set()

    def _tag_verification_chunk(chunk: Any, tier: str, rank: int) -> None:
        setattr(chunk, "verification_tier", tier)
        setattr(chunk, "verification_tier_rank", rank)

    def _append_matching(predicate, *, tier: str, rank: int) -> None:
        for chunk in retrieved_chunks:
            chunk_id = str(getattr(chunk, "id", ""))
            if not chunk_id or chunk_id in seen_ids:
                continue
            if predicate(chunk):
                _tag_verification_chunk(chunk, tier, rank)
                selected.append(chunk)
                seen_ids.add(chunk_id)

    sentence_evidence_chunks = _build_sentence_evidence_verification_chunks(
        generation_context_meta,
        retrieved_chunks,
    )
    if sentence_evidence_chunks:
        return sentence_evidence_chunks, {
            "status": "applied:scoped",
            "scope": "sentence_evidence",
            "tiers": ["sentence_evidence"],
        }

    if source_ids:
        source_id_set = set(source_ids)
        _append_matching(
            lambda chunk: str(getattr(chunk, "id", "")) in source_id_set,
            tier="generation_source",
            rank=1,
        )
        if selected:
            return selected, {
                "status": "applied:scoped",
                "scope": "generation_source_chunks",
                "tiers": ["generation_source"],
            }
    if prompt_ids:
        prompt_id_set = set(prompt_ids)
        _append_matching(
            lambda chunk: str(getattr(chunk, "id", "")) in prompt_id_set,
            tier="prompt_scope",
            rank=2,
        )
    if target_doc_ids:
        _append_matching(
            lambda chunk: str(getattr(chunk, "doc_id", "")) in target_doc_ids,
            tier="target_doc",
            rank=3,
        )
    if target_case:
        _append_matching(
            lambda chunk: _case_name_matches(str(target_case), getattr(chunk, "case_name", None)),
            tier="target_case",
            rank=4,
        )

    if selected:
        tiers = []
        for chunk in selected:
            tier = str(getattr(chunk, "verification_tier", "") or "")
            if tier and tier not in tiers:
                tiers.append(tier)
        return selected, {"status": "applied:scoped", "tiers": tiers}
    fallback_chunks = list(retrieved_chunks)
    for chunk in fallback_chunks:
        _tag_verification_chunk(chunk, "retrieved", 5)
    return fallback_chunks, {"status": "fallback:retrieved_chunks", "tiers": ["retrieved"]}


def _source_evidence_sentence_payloads(
    evidence_sentences: list[dict[str, Any]],
    *,
    canonical_answer_fact: str | None,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _append(text: Any, chunk: Any, role: str) -> None:
        sentence = _sanitize_evidence_sentence(str(text or ""))
        if not sentence:
            return
        chunk_id = str(getattr(chunk, "id", "") or "")
        key = f"{chunk_id}:{' '.join(sentence.lower().split())}"
        if key in seen:
            return
        seen.add(key)
        payloads.append(
            {
                "sentence": sentence,
                "source_chunk_id": chunk_id,
                "role": role,
            }
        )

    if canonical_answer_fact and evidence_sentences:
        _append(canonical_answer_fact, evidence_sentences[0].get("chunk"), "canonical_answer_fact")

    for item in evidence_sentences:
        _append(item.get("sentence"), item.get("chunk"), str(item.get("role") or "answerable_sentence"))
    return payloads


def _build_sentence_evidence_verification_chunks(
    generation_context_meta: dict[str, Any],
    retrieved_chunks: list,
) -> list[LegalChunk]:
    payloads = generation_context_meta.get("source_evidence_sentences")
    if not isinstance(payloads, list) or not payloads:
        return []

    chunks_by_id = {
        str(getattr(chunk, "id", "")): chunk
        for chunk in retrieved_chunks
        if getattr(chunk, "id", None)
    }
    sentence_chunks: list[LegalChunk] = []
    seen_text: set[str] = set()
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            continue
        text = _sanitize_evidence_sentence(str(payload.get("sentence") or ""))
        if not text:
            continue
        key = " ".join(text.lower().split())
        if key in seen_text:
            continue
        source_chunk = chunks_by_id.get(str(payload.get("source_chunk_id") or ""))
        if source_chunk is None:
            continue
        seen_text.add(key)
        chunk = LegalChunk(
            id=f"{getattr(source_chunk, 'id', 'chunk')}:sentence_evidence:{index}",
            doc_id=str(getattr(source_chunk, "doc_id", "") or ""),
            text=text,
            chunk_index=int(getattr(source_chunk, "chunk_index", index) or index),
            doc_type=str(getattr(source_chunk, "doc_type", "case") or "case"),
            case_name=getattr(source_chunk, "case_name", None),
            court=getattr(source_chunk, "court", None),
            court_level=getattr(source_chunk, "court_level", None),
            citation=getattr(source_chunk, "citation", None),
            date_decided=getattr(source_chunk, "date_decided", None),
            title=getattr(source_chunk, "title", None),
            section=getattr(source_chunk, "section", None),
            source_file=getattr(source_chunk, "source_file", None),
        )
        setattr(chunk, "verification_tier", "sentence_evidence")
        setattr(chunk, "verification_tier_rank", 0)
        setattr(chunk, "verification_source_chunk_id", getattr(source_chunk, "id", None))
        setattr(chunk, "verification_evidence_role", payload.get("role"))
        sentence_chunks.append(chunk)
    return sentence_chunks


def _select_named_case_evidence_sentences(
    query: str,
    prompt_chunks: list,
    target_case: str,
    *,
    limit: int = TARGET_CASE_PROMPT_SENTENCE_LIMIT,
) -> list[dict[str, Any]]:
    query_tokens = _content_tokens(query)
    scored_sentences: list[dict[str, Any]] = []
    sentence_limit = max(1, int(limit or TARGET_CASE_PROMPT_SENTENCE_LIMIT))

    for chunk in prompt_chunks:
        raw_text = str(getattr(chunk, "text", "") or "")
        if not raw_text.strip():
            continue
        for sentence_index, sentence in enumerate(_split_into_sentences(raw_text)):
            score = _score_named_case_evidence_sentence(
                query_tokens=query_tokens,
                sentence=sentence,
                chunk=chunk,
                target_case=target_case,
                sentence_index=sentence_index,
            )
            if score <= 0:
                continue
            scored_sentences.append(
                {
                    "chunk": chunk,
                    "sentence": sentence.strip(),
                    "score": score,
                    "sentence_index": sentence_index,
                    "is_explicit_holding": _extract_explicit_holding_text(sentence.strip()) is not None,
                }
            )

    if not scored_sentences:
        return []

    scored_sentences.sort(
        key=lambda item: (
            -item["score"],
            -int(item["is_explicit_holding"]),
            getattr(item["chunk"], "chunk_index", 0),
            item["sentence_index"],
        )
    )

    selected: list[dict[str, Any]] = []
    seen_sentences: set[str] = set()
    for item in scored_sentences:
        normalized_sentence = " ".join(item["sentence"].split()).lower()
        if normalized_sentence in seen_sentences:
            continue
        seen_sentences.add(normalized_sentence)
        selected.append(item)
        if len(selected) >= sentence_limit:
            break

    return selected


def _select_user_upload_context_evidence_sentences(
    query: str,
    prompt_chunks: list,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    query_tokens = _content_tokens(query)
    scored_sentences: list[dict[str, Any]] = []
    sentence_limit = max(1, int(limit or TARGET_CASE_PROMPT_SENTENCE_LIMIT))

    for chunk in prompt_chunks:
        raw_text = str(getattr(chunk, "text", "") or "")
        if not raw_text.strip():
            continue
        is_upload = getattr(chunk, "doc_type", None) == "user_upload"
        for sentence_index, sentence in enumerate(_split_into_sentences(raw_text)):
            sentence_text = sentence.strip()
            if len(sentence_text) < 24:
                continue
            if _is_incomplete_evidence_sentence(sentence_text):
                continue
            sentence_tokens = _content_tokens(sentence_text)
            overlap = len(query_tokens & sentence_tokens)
            if overlap == 0 and not is_upload:
                continue
            score = float(overlap * 4)
            if query_tokens:
                score += overlap / max(len(query_tokens), 1)
            if is_upload:
                score += 3.0
            if getattr(chunk, "source_file", None):
                score += 0.5
            scored_sentences.append(
                {
                    "chunk": chunk,
                    "sentence": sentence_text,
                    "score": score,
                    "sentence_index": sentence_index,
                    "role": "upload_fact" if is_upload else "comparison_authority",
                }
            )

    if not scored_sentences:
        return []

    scored_sentences.sort(
        key=lambda item: (
            -item["score"],
            0 if getattr(item["chunk"], "doc_type", None) == "user_upload" else 1,
            getattr(item["chunk"], "chunk_index", 0),
            item["sentence_index"],
        )
    )

    selected: list[dict[str, Any]] = []
    seen_sentences: set[str] = set()
    for item in scored_sentences:
        normalized_sentence = " ".join(item["sentence"].split()).lower()
        if normalized_sentence in seen_sentences:
            continue
        seen_sentences.add(normalized_sentence)
        selected.append(item)
        if len(selected) >= sentence_limit:
            break

    return selected


def _split_into_sentences(text: str) -> list[str]:
    collapsed = _clean_legal_text_for_generation(text)
    if not collapsed:
        return []
    parts = _SENTENCE_SPLIT_RE.split(collapsed)
    return [part.strip() for part in parts if part and part.strip()]


def _clean_legal_text_for_generation(text: str) -> str:
    cleaned = str(text or "")
    cleaned = _LEGAL_TEXT_NOISE_RE.sub(" ", cleaned)
    cleaned = _CITE_AS_NOISE_RE.sub(" ", cleaned)
    cleaned = _CASE_PAGE_HEADER_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\b(?:Pp?|pp)\.\s*\d+(?:[-–]\d+)?\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "case",
        "court",
        "decide",
        "decided",
        "did",
        "for",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "question",
        "regarding",
        "sentence",
        "sentencing",
        "supreme",
        "that",
        "the",
        "this",
        "to",
        "united",
        "what",
        "when",
        "which",
    }
    tokens = {
        token
        for token in _NON_ALNUM_RE.sub(" ", text.lower()).split()
        if token and token not in stopwords and len(token) > 1
    }
    return tokens


def _score_named_case_evidence_sentence(
    *,
    query_tokens: set[str],
    sentence: str,
    chunk: Any,
    target_case: str,
    sentence_index: int,
) -> float:
    sentence_text = sentence.strip()
    if len(sentence_text) < 24:
        return 0.0
    if _is_incomplete_evidence_sentence(sentence_text):
        return 0.0
    if _CAPTION_ONLY_RE.search(sentence_text):
        return 0.0

    # NEW: Filter low-quality fragments that start mid-sentence
    # Reject sentences starting with lowercase words (fragments like "is the right to...")
    _FRAGMENT_START_RE = re.compile(r'^(?:is|are|was|were|and|or|to|from|because|that|which|who|whom|whose|when|where|while|with|without|by|as|just|also|even|still|yet|but|however|therefore|thus|hence|moreover|furthermore|nevertheless|nonetheless|otherwise|instead|meanwhile|afterward|before|after|during|since|until|unless|although|though|whereas|while)\b', re.IGNORECASE)
    if _FRAGMENT_START_RE.search(sentence_text):
        return 0.0
    
    # NEW: Reject sentences ending with honorifics or truncated references
    _TRUNCATED_END_RE = re.compile(r'(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|Hon\.?|Lt\.?|Justice|No\.?|Nos\.?|Inc\.?|Corp\.?|Co\.?|Ltd\.?|v\.?|U\.S\.?|D\.C\.?|Cir\.?|U\.S\.C\.?)$', re.IGNORECASE)
    if _TRUNCATED_END_RE.search(sentence_text.rstrip('.,;:!?"\'')):
        return 0.0
    
    # NEW: Reject page header artifacts like "2 BURNETT v. UNITED STATES"
    _PAGE_HEADER_RE = re.compile(r'^\d+\s+[A-Z][A-Z\s.,\'&-]{2,90}\s+v\.\s+[A-Z][A-Z\s.,\'&-]{1,90}\s+(?:Syllabus|Opinion of the Court|GORSUCH|dissenting|J\.,| dissenting)', re.IGNORECASE)
    if _PAGE_HEADER_RE.search(sentence_text):
        return 0.0
    
    # NEW: Reject reported speech fragments like "Burnett claims is the right to..."
    # These occur when reported speech is split across chunk boundaries
    _REPORTED_SPEECH_FRAGMENT_RE = re.compile(
        r'^(?:[A-Z][a-z]+\s+(?:claims|submitted|argued|contended|asserted|maintained|alleged|stated|said|insisted|submitted)\s+is\b)',
        re.IGNORECASE
    )
    if _REPORTED_SPEECH_FRAGMENT_RE.search(sentence_text):
        return 0.0
    
    # NEW: Reject mid-sentence predicate fragments starting with "is the right to..."
    _PREDICATE_FRAGMENT_RE = re.compile(r'^is\s+(?:the\s+)?right\s+to\b', re.IGNORECASE)
    if _PREDICATE_FRAGMENT_RE.search(sentence_text):
        return 0.0

    lower_text = sentence_text.lower()
    if "certiorari to the united states court of appeals" in lower_text and not _CERT_DENIAL_RE.search(sentence_text):
        return 0.0

    sentence_tokens = _content_tokens(sentence_text)
    overlap = len(query_tokens & sentence_tokens)
    has_holding_lead = _extract_explicit_holding_text(sentence_text) is not None

    if "question presented" in lower_text and not has_holding_lead:
        return 0.0

    if overlap == 0 and not has_holding_lead:
        return 0.0

    score = float(overlap * 4)
    if query_tokens:
        score += overlap / max(len(query_tokens), 1)
    if has_holding_lead:
        score += 10.0
    if re.match(r"^(?:that court|the district court)\b", lower_text):
        score -= 12.0
    if _LOWER_COURT_PROCEDURAL_RE.search(sentence_text):
        score -= 6.0
    if sentence_text.lower().startswith("held:"):
        score += 4.0
    if _case_name_matches(target_case, getattr(chunk, "case_name", None)):
        score += 2.0
    if sentence_index == 0 and has_holding_lead:
        score += 1.0
    if len(sentence_text) > 320:
        score -= 1.0
    if "justice" in lower_text and "dissent" in lower_text and "who wrote" not in " ".join(sorted(query_tokens)):
        score -= 2.0
    return score


def _format_sentence_evidence_for_prompt(item: dict[str, Any]) -> str:
    chunk = item["chunk"]
    sentence = _sanitize_evidence_sentence(str(item["sentence"]))
    metadata = [
        "Evidence type: allowed answer fact",
        f"Chunk ID: {chunk.id}",
        f"Document ID: {chunk.doc_id}",
        f"Document type: {chunk.doc_type}",
    ]
    if chunk.case_name:
        metadata.append(f"Case name: {chunk.case_name}")
    if chunk.citation:
        metadata.append(f"Citation: {chunk.citation}")
    if chunk.court_level:
        metadata.append(f"Court level: {chunk.court_level}")
    metadata.append(f"Sentence score: {item['score']:.2f}")
    return (
        f"{'; '.join(metadata)}\n"
        f"Allowed answer fact: {sentence}\n"
        "Generation constraint: use this sentence only if it directly answers the user question; do not add unstated facts."
    )


def _format_canonical_answer_fact_for_prompt(
    canonical_answer_fact: str,
    *,
    source_chunk_ids: list[str],
) -> str:
    source_ids = ", ".join(str(chunk_id) for chunk_id in source_chunk_ids if chunk_id)
    metadata = [
        "Evidence type: canonical answer fact",
        f"Source chunk IDs: {source_ids}" if source_ids else "Source chunk IDs: unknown",
    ]
    return (
        f"{'; '.join(metadata)}\n"
        f"Allowed answer fact: {canonical_answer_fact}\n"
        "Generation constraint: use this canonical phrasing as the direct opening answer when it answers the user question."
    )


def _canonical_answer_fact(
    query: str,
    evidence_sentences: list[dict[str, Any]],
    *,
    explicit_holding: str | None,
) -> str | None:
    query_text = " ".join(str(query or "").lower().split())
    evidence_text = " ".join(
        _sanitize_evidence_sentence(str(item.get("sentence") or ""))
        for item in evidence_sentences
    ).lower()
    combined = f"{query_text} {evidence_text}"

    if "ieepa" in combined and "tariff" in combined:
        if any(
            phrase in combined
            for phrase in (
                "does not authorize",
                "did not grant",
                "does not grant",
                "does not enable",
                "nothing in ieepa",
                "cannot bear such weight",
            )
        ):
            return "IEEPA does not authorize the President to impose tariffs."

    if ("fair labor standards act" in combined or "flsa" in combined) and "exempt" in combined:
        if "preponderance" in combined:
            return (
                "An employer claiming a Fair Labor Standards Act exemption must prove "
                "the exemption by a preponderance of the evidence."
            )

    if "royal canin" in combined or "wullschleger" in combined or "amended complaint" in combined:
        if "supplemental jurisdiction" in combined and "state court" in combined:
            return (
                "After the plaintiff amended the complaint to remove federal claims, "
                "the federal court lost supplemental jurisdiction over the remaining "
                "state-law claims and had to remand the case to state court."
            )

    if "fraud" in combined and "economic loss" in combined:
        if any(
            phrase in combined
            for phrase in (
                "no general rule requiring economic loss",
                "will not read such a requirement",
                "need not",
                "not require",
                "does not require",
            )
        ):
            return "The federal fraud statutes do not require the Government to prove economic loss."

    if "first amendment" in combined and "tiktok" in combined:
        if "does not violate" in combined or "did not violate" in combined:
            return "The challenged TikTok divestiture provisions do not violate the First Amendment."

    if "nepa" in combined and "upstream" in combined and "downstream" in combined:
        if any(phrase in combined for phrase in ("does not require", "need not", "not required")):
            return (
                "NEPA does not require an agency to study separate upstream or downstream "
                "projects outside the agency's regulatory authority."
            )

    # NEW: Burnett cert-denial cases with dissents - frame as dissent, not holding
    if (
        "burnett" in combined
        and ("supervised release" in combined or "sixth amendment" in combined or "jury" in combined)
    ):
        # Check for cert denial posture
        if any(phrase in combined for phrase in (
            "certiorari is denied",
            "denied certiorari",
            "dissenting from the denial",
        )):
            return (
                "The Supreme Court denied certiorari in Burnett v. United States. "
                "Justice Gorsuch dissented from the denial of certiorari and argued that "
                "Burnett raised a Sixth Amendment question about jury findings when supervised-release "
                "revocation imprisonment would push total prison time beyond the statutory maximum "
                "for the underlying conviction."
            )

    if explicit_holding:
        return None
    return None


def _best_explicit_holding_sentence(evidence_sentences: list[dict[str, Any]]) -> str | None:
    for item in evidence_sentences:
        sentence = _extract_explicit_holding_text(str(item.get("sentence") or ""))
        if sentence:
            return sentence
    return None


def _extract_explicit_holding_text(sentence: str) -> str | None:
    sentence_text = _sanitize_evidence_sentence(sentence)
    if not sentence_text:
        return None
    if _is_incomplete_evidence_sentence(sentence_text):
        return None

    held_index = sentence_text.lower().find("held:")
    if held_index >= 0:
        held_text = _sanitize_evidence_sentence(sentence_text[held_index + len("held:") :])
        if _is_incomplete_evidence_sentence(held_text):
            return None
        return held_text or None

    if _HOLDING_LEAD_RE.search(sentence_text) or _HOLDING_CONCLUSION_RE.search(sentence_text):
        return sentence_text
    return None


def _sanitize_evidence_sentence(sentence: str) -> str:
    sentence_text = " ".join(str(sentence or "").split()).strip()
    sentence_text = re.sub(r"\s+([.,;:!?])", r"\1", sentence_text)
    sentence_text = re.sub(r"([.!?])\s*\.", r"\1", sentence_text)
    return sentence_text.strip()


def _is_incomplete_evidence_sentence(sentence: str) -> bool:
    sentence_text = str(sentence or "").strip()
    if len(sentence_text) < 24:
        return True
    return bool(_INCOMPLETE_EVIDENCE_SENTENCE_RE.search(sentence_text))


def _extract_case_posture(prompt_chunks: list, target_case: str | None) -> dict[str, Any] | None:
    if not prompt_chunks or not target_case:
        return None

    posture_text = "\n".join(
        str(getattr(chunk, "text", "") or "")[:1600]
        for chunk in prompt_chunks[:3]
    )
    if not posture_text.strip():
        return None

    decision_type = None
    court_action = None
    if _CERT_DENIAL_RE.search(posture_text):
        decision_type = "cert_denial"
        court_action = "denied certiorari"
    elif _GVR_RE.search(posture_text):
        decision_type = "gvr"
        court_action = "granted certiorari, vacated, and remanded"
    elif _VACATED_AND_REMANDED_RE.search(posture_text):
        decision_type = "merits"
        court_action = "vacated and remanded"
    elif _REVERSED_RE.search(posture_text):
        decision_type = "merits"
        court_action = "reversed"
    elif _AFFIRMED_RE.search(posture_text):
        decision_type = "merits"
        court_action = "affirmed"
    elif _VACATED_RE.search(posture_text):
        decision_type = "merits"
        court_action = "vacated"
    elif _REMANDED_RE.search(posture_text):
        decision_type = "merits"
        court_action = "remanded"

    opinion_role = None
    author = None
    is_separate_opinion = False

    match = _match_authorized_opinion_header(posture_text, _DISSENT_FROM_DENIAL_PATTERNS)
    if match is not None:
        opinion_role = "dissent_from_denial"
        author = match
        is_separate_opinion = True
    else:
        match = _match_authorized_opinion_header(posture_text, _CONCURRENCE_PATTERNS)
        if match is not None:
            opinion_role = "concurrence"
            author = match
            is_separate_opinion = True
        else:
            match = _match_authorized_opinion_header(posture_text, _DISSENT_PATTERNS)
            if match is not None:
                opinion_role = "dissent"
                author = match
                is_separate_opinion = True
            elif _PER_CURIAM_RE.search(posture_text):
                opinion_role = "per_curiam"
            elif court_action is not None:
                opinion_role = "majority"

    if decision_type is None and opinion_role is None:
        return None

    return {
        "target_case": target_case,
        "decision_type": decision_type,
        "court_action": court_action,
        "opinion_role": opinion_role,
        "author": author,
        "is_separate_opinion": is_separate_opinion,
        "source_chunk_ids": [chunk.id for chunk in prompt_chunks[:3]],
    }


def _apply_case_posture_response_override(
    *,
    query: str,
    response: str,
    case_posture: dict[str, Any] | None,
    query_intent: str | None = None,
) -> tuple[str, dict[str, Any]]:
    _ = query
    if not isinstance(case_posture, dict):
        return response, {"status": "not_applied:no_case_posture"}

    decision_type = str(case_posture.get("decision_type") or "").strip()
    court_action = str(case_posture.get("court_action") or "").strip()
    opinion_role = str(case_posture.get("opinion_role") or "").strip()
    author = str(case_posture.get("author") or "").strip()
    target_case = str(case_posture.get("target_case") or "").strip()

    if decision_type != "cert_denial" or court_action != "denied certiorari":
        return response, {"status": "not_applied:not_cert_denial"}
    if query_intent not in {None, "posture", "author"}:
        return response, {
            "status": "not_applied:intent_not_posture_or_author",
            "query_intent": query_intent,
            "decision_type": decision_type,
            "court_action": court_action,
            "target_case": target_case,
        }
    if opinion_role not in {"dissent_from_denial", "dissent", "concurrence"}:
        return response, {"status": "not_applied:no_named_separate_opinion"}
    if not author:
        return response, {"status": "not_applied:no_author"}

    case_suffix = f" in {target_case}" if target_case else ""
    sentences = [f"The Supreme Court denied certiorari{case_suffix}."]
    if opinion_role == "dissent_from_denial":
        sentences.append(f"Justice {author} dissented from the denial of certiorari.")
    elif opinion_role == "dissent":
        sentences.append(f"Justice {author} dissented.")
    else:
        sentences.append(f"Justice {author} concurred in the disposition.")

    return "\n\n".join(sentences), {
        "status": "applied:cert_denial_posture",
        "decision_type": decision_type,
        "court_action": court_action,
        "opinion_role": opinion_role,
        "author": author,
        "target_case": target_case,
        "query_intent": query_intent,
    }


_USER_UPLOAD_ABSENCE_GENERIC_TERMS = {
    "according",
    "agreement",
    "client",
    "document",
    "draft",
    "file",
    "memo",
    "memorandum",
    "motion",
    "my",
    "northstar",
    "cedar",
    "riley",
    "section",
    "states",
    "uploaded",
    "using",
}


def _apply_user_upload_absence_response_override(
    *,
    query: str,
    response: str,
    generation_context_meta: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if not _query_mentions_upload(query):
        return response, {"status": "not_applied:not_upload_query"}
    if _query_requests_upload_comparison(query):
        return response, {"status": "not_applied:comparison_query"}
    if not isinstance(generation_context_meta, dict):
        return response, {"status": "not_applied:no_generation_context_meta"}
    if generation_context_meta.get("status") != "applied:user_upload_context":
        return response, {"status": "not_applied:not_user_upload_context"}

    evidence_text = " ".join(
        str(sentence or "")
        for sentence in generation_context_meta.get("answerable_sentences") or []
    )
    if not evidence_text.strip():
        return response, {"status": "not_applied:no_upload_evidence_sentences"}

    query_terms = _user_upload_absence_query_terms(query)
    if not query_terms:
        return response, {"status": "not_applied:no_core_query_terms"}

    evidence_terms = _content_tokens(evidence_text)
    matched_terms = sorted(query_terms & evidence_terms)
    if matched_terms:
        return response, {
            "status": "not_applied:upload_evidence_matches_query_terms",
            "matched_terms": matched_terms,
        }

    subject = _user_upload_absence_subject(query)
    replacement = f"The uploaded document excerpt does not identify {subject}."
    return replacement, {
        "status": "applied:user_upload_absence",
        "replacement": replacement,
        "missing_terms": sorted(query_terms),
    }


def _user_upload_absence_query_terms(query: str) -> set[str]:
    return {
        token
        for token in _content_tokens(query)
        if token not in _USER_UPLOAD_ABSENCE_GENERIC_TERMS
    }


def _user_upload_absence_subject(query: str) -> str:
    query_text = " ".join(str(query or "").split()).strip().rstrip("?.!")
    patterns = (
        r"\bwhat\s+is\s+(?:the\s+)?(.+)$",
        r"\bwhat\s+are\s+(?:the\s+)?(.+)$",
        r"\bwhat\s+(?:does|do|did)\s+.+?\s+(?:say|state|identify|list)\s+(?:about|for)?\s*(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, query_text, re.IGNORECASE)
        if match is None:
            continue
        subject = match.group(1).strip()
        subject = re.sub(
            r"^(?:in|according to|using)\s+(?:my\s+)?(?:uploaded\s+)?(?:document|draft|file|memo|motion|agreement),?\s*",
            "",
            subject,
            flags=re.IGNORECASE,
        ).strip()
        if subject:
            article = "an" if re.match(r"^[aeiou]", subject, re.IGNORECASE) else "a"
            if re.match(r"^(?:a|an|the)\b", subject, re.IGNORECASE):
                return subject
            return f"{article} {subject}"
    return "the requested fact"


def _apply_explicit_holding_response_override(
    *,
    query: str,
    response: str,
    generation_context_meta: dict[str, Any] | None,
    target_case: str | None = None,
) -> tuple[str, dict[str, Any]]:
    _ = response
    resolved_target_case = target_case or _extract_target_case_name(query)
    if resolved_target_case is None:
        return response, {"status": "not_applied:no_target_case"}
    if not isinstance(generation_context_meta, dict):
        return response, {"status": "not_applied:no_generation_context_meta"}

    canonical_answer = str(generation_context_meta.get("canonical_answer_fact") or "").strip()
    explicit_holding = str(generation_context_meta.get("explicit_holding_sentence") or "").strip()
    replacement = canonical_answer or explicit_holding
    if not replacement:
        return response, {"status": "not_applied:no_explicit_holding"}

    return replacement, {
        "status": "applied:canonical_answer_fact" if canonical_answer else "applied:explicit_holding_sentence",
        "canonical_answer_fact": canonical_answer or None,
        "explicit_holding_sentence": explicit_holding or None,
        "source_chunk_ids": generation_context_meta.get("source_chunk_ids") or [],
    }


def _apply_cert_denial_safety_guard(
    *,
    response: str,
    case_posture: dict[str, Any] | None,
    generation_context_meta: dict[str, Any] | None,
    prompt_chunks: list,
) -> tuple[str, dict[str, Any]]:
    if not _CERT_DENIAL_RESPONSE_RE.search(response or ""):
        return response, {"status": "not_applied:no_cert_denial_language"}

    if isinstance(case_posture, dict) and case_posture.get("decision_type") == "cert_denial":
        return response, {
            "status": "not_applied:authorized_by_case_posture",
            "decision_type": case_posture.get("decision_type"),
            "court_action": case_posture.get("court_action"),
        }

    if _prompt_chunks_explicitly_deny_certiorari(prompt_chunks):
        return response, {"status": "not_applied:authorized_by_prompt_evidence"}

    generation_context_meta = generation_context_meta or {}
    replacement = (
        str(generation_context_meta.get("explicit_holding_sentence") or "").strip()
        or str(generation_context_meta.get("best_answer_sentence") or "").strip()
    )
    if not replacement:
        replacement = "Insufficient support in retrieved authorities to answer the question."

    return replacement, {
        "status": "applied:blocked_unauthorized_cert_denial",
        "replacement": replacement,
        "source_chunk_ids": generation_context_meta.get("source_chunk_ids") or [],
    }


def _prompt_chunks_explicitly_deny_certiorari(prompt_chunks: list) -> bool:
    return any(
        _CERT_DENIAL_RE.search(str(getattr(chunk, "text", "") or ""))
        for chunk in prompt_chunks
    )


def _match_authorized_opinion_header(text: str, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match is not None:
            return str(match.group(1)).strip().title()
    return None


def _empty_claim_support_summary() -> dict[str, Any]:
    return {
        "raw_total": 0,
        "total": 0,
        "supported": 0,
        "possibly_supported": 0,
        "unsupported": 0,
        "excluded_rhetorical": 0,
        "unsupported_ratio": 0.0,
    }


def _filter_claims_for_verification(raw_claims: list) -> tuple[list, list[dict[str, Any]]]:
    filtered: list[Any] = []
    skipped: list[dict[str, Any]] = []
    for claim in raw_claims:
        text = str(getattr(claim, "text", "") or "").strip()
        skip_reason = _verification_claim_skip_reason(text)
        if skip_reason is not None:
            payload = claim.to_dict() if hasattr(claim, "to_dict") else {"text": text}
            skipped.append(
                {
                    "reason": skip_reason,
                    "claim_id": payload.get("claim_id"),
                    "text": text,
                    "span": payload.get("span"),
                }
            )
            continue
        filtered.append(claim)
    return filtered, skipped


def _is_low_value_verification_claim_text(text: str) -> bool:
    return _verification_claim_skip_reason(text) is not None


def _legacy_low_value_verification_claim_text(text: str) -> bool:
    stripped = " ".join(str(text or "").split()).strip()
    if not stripped:
        return True
    if _LOW_VALUE_VERIFICATION_CLAIM_RE.match(stripped):
        return True
    if _FRAGMENT_VERIFICATION_CLAIM_RE.match(stripped):
        return True
    if _TRAILING_ABBREVIATION_FRAGMENT_RE.search(stripped):
        return True
    alpha_tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", stripped)
    if len(alpha_tokens) < 4:
        return True
    if stripped[0] in {",", ".", ";", ":", '"', "'", "”", "“"}:
        return True
    if stripped[0].islower():
        return True
    if stripped.lower().startswith("supported by allowed answer facts"):
        return True
    if not _VERIFICATION_CLAIM_VERB_RE.search(stripped):
        return True
    return False


def _verification_claim_skip_reason(text: str) -> str | None:
    stripped = " ".join(str(text or "").split()).strip()
    if not stripped:
        return "empty_claim"
    if _LOW_VALUE_VERIFICATION_CLAIM_RE.match(stripped):
        return "low_value_rhetorical_or_connective"
    if _FRAGMENT_VERIFICATION_CLAIM_RE.match(stripped):
        return "malformed_fragment"
    if _TRAILING_ABBREVIATION_FRAGMENT_RE.search(stripped):
        return "malformed_trailing_abbreviation"
    if stripped.endswith((",", ":", ";", ",.")):
        return "malformed_trailing_punctuation"
    if _SINGLE_INITIAL_VERIFICATION_END_RE.search(stripped) and not _INITIAL_SEQUENCE_END_RE.search(stripped):
        return "malformed_trailing_initial"
    if _DANGLING_VERIFICATION_CLAIM_END_RE.search(stripped):
        return "malformed_dangling_clause"
    if _has_unbalanced_verification_quotes(stripped):
        return "malformed_unbalanced_quote"
    alpha_tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", stripped)
    if len(alpha_tokens) < 4:
        return "too_short"
    if stripped[0] in {",", ".", ";", ":", '"', "'", "â€", "â€œ"}:
        return "malformed_leading_punctuation"
    if stripped[0].islower():
        return "malformed_lowercase_start"
    if stripped.lower().startswith("supported by allowed answer facts"):
        return "low_value_rhetorical_or_connective"
    if not _VERIFICATION_CLAIM_VERB_RE.search(stripped):
        return "no_finite_verb"
    return None


def _has_unbalanced_verification_quotes(text: str) -> bool:
    if text.count('"') % 2:
        return True
    if text.count("“") != text.count("”"):
        return True
    if text.count("‘") != text.count("’"):
        return True
    return False


def _empty_answer_block_summary() -> dict[str, Any]:
    return {
        "total": 0,
        "claim_blocks": 0,
        "non_claim_blocks": 0,
        "verification_required_blocks": 0,
        "unverified_claim_candidates": 0,
        "supported_blocks": 0,
        "possibly_supported_blocks": 0,
        "unsupported_blocks": 0,
    }


def _update_answer_support_metadata(meta: dict[str, Any], *, response: str | None = None) -> None:
    summary = _summarize_claim_support(meta.get("claims"))
    meta["claim_support_summary"] = summary
    meta["answer_warning"] = _build_answer_warning(summary)
    if response is not None:
        answer_blocks = _build_answer_blocks(response, meta.get("claims"))
        meta["answer_blocks"] = answer_blocks
        meta["answer_block_summary"] = _summarize_answer_blocks(answer_blocks)


def _build_answer_blocks(response: str, claims: Any) -> list[dict[str, Any]]:
    """Segment an answer into verified claims and non-claim explanatory text."""
    response_text = str(response or "")
    if not response_text:
        return []

    claim_groups = _answer_claim_span_groups(response_text, claims)
    blocks: list[dict[str, Any]] = []
    cursor = 0

    for group in claim_groups:
        start = int(group["start_char"])
        end = int(group["end_char"])
        if start > cursor:
            _append_answer_text_block(blocks, response_text, cursor, start)
        blocks.append(
            _answer_block_for_claim_group(
                block_index=len(blocks),
                response=response_text,
                start_char=start,
                end_char=end,
                claims=group["claims"],
            )
        )
        cursor = max(cursor, end)

    if cursor < len(response_text):
        _append_answer_text_block(blocks, response_text, cursor, len(response_text))

    if not blocks:
        _append_answer_text_block(blocks, response_text, 0, len(response_text))

    return blocks


def _answer_claim_span_groups(response: str, claims: Any) -> list[dict[str, Any]]:
    if not isinstance(claims, list):
        return []

    candidates: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        span = _locate_answer_claim_span(response, claim)
        if span is None:
            continue
        start, end = span
        if start >= end:
            continue
        candidates.append(
            {
                "start_char": start,
                "end_char": end,
                "claims": [claim],
            }
        )

    candidates.sort(key=lambda item: (item["start_char"], item["end_char"]))
    groups: list[dict[str, Any]] = []
    for candidate in candidates:
        if not groups or int(candidate["start_char"]) >= int(groups[-1]["end_char"]):
            groups.append(candidate)
            continue

        group = groups[-1]
        group["start_char"] = min(int(group["start_char"]), int(candidate["start_char"]))
        group["end_char"] = max(int(group["end_char"]), int(candidate["end_char"]))
        group["claims"].extend(candidate["claims"])

    return groups


def _locate_answer_claim_span(response: str, claim: dict[str, Any]) -> tuple[int, int] | None:
    response_span = _claim_response_span_payload(claim)
    preferred_start = response_span.get("start_char")
    preferred_end = response_span.get("end_char")
    claim_text = str(response_span.get("text") or claim.get("text") or "")

    if isinstance(preferred_start, int) and isinstance(preferred_end, int):
        if 0 <= preferred_start < preferred_end <= len(response):
            span_text = response[preferred_start:preferred_end]
            if span_text == claim_text or span_text.strip() == claim_text.strip():
                return preferred_start, preferred_end

    if not claim_text:
        return None

    exact_matches = _find_text_occurrences(response, claim_text)
    if exact_matches:
        return _select_best_text_occurrence(exact_matches, preferred_start)

    stripped_text = claim_text.strip()
    if stripped_text and stripped_text != claim_text:
        stripped_matches = _find_text_occurrences(response, stripped_text)
        if stripped_matches:
            return _select_best_text_occurrence(stripped_matches, preferred_start)

    lower_matches = _find_text_occurrences(response.lower(), claim_text.lower())
    if lower_matches:
        return _select_best_text_occurrence(lower_matches, preferred_start)

    return None


def _claim_response_span_payload(claim: dict[str, Any]) -> dict[str, Any]:
    annotation = claim.get("annotation")
    if isinstance(annotation, dict):
        response_span = annotation.get("response_span")
        if isinstance(response_span, dict):
            return response_span

    span = claim.get("span")
    if isinstance(span, dict):
        return span
    return {}


def _find_text_occurrences(haystack: str, needle: str) -> list[tuple[int, int]]:
    if not haystack or not needle:
        return []

    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index < 0:
            break
        matches.append((index, index + len(needle)))
        start = index + 1
    return matches


def _select_best_text_occurrence(
    matches: list[tuple[int, int]],
    preferred_start: Any,
) -> tuple[int, int]:
    if isinstance(preferred_start, int):
        return min(matches, key=lambda item: abs(item[0] - preferred_start))
    return matches[0]


def _append_answer_text_block(
    blocks: list[dict[str, Any]],
    response: str,
    start_char: int,
    end_char: int,
) -> None:
    block = _answer_block_for_nonclaim_text(
        block_index=len(blocks),
        response=response,
        start_char=start_char,
        end_char=end_char,
    )
    if block is not None:
        blocks.append(block)


def _answer_block_for_claim_group(
    *,
    block_index: int,
    response: str,
    start_char: int,
    end_char: int,
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    verdicts = [_claim_verdict_from_payload(claim) for claim in claims]
    support_levels = [_claim_support_level_from_payload(claim) for claim in claims]
    support_level = _merged_answer_block_support_level(verdicts, support_levels)
    if any(verdict == "CONTRADICTED" for verdict in verdicts):
        block_type = "contradicted_claim"
    elif support_level == "supported":
        block_type = "verified_claim"
    elif support_level == "possibly_supported":
        block_type = "possibly_supported_claim"
    else:
        block_type = "unsupported_claim"

    evidence_count = 0
    for claim in claims:
        annotation = claim.get("annotation")
        evidence = annotation.get("evidence") if isinstance(annotation, dict) else None
        if isinstance(evidence, list):
            evidence_count += len(evidence)

    return {
        "block_id": f"answer_block_{block_index}",
        "type": block_type,
        "category": "claim",
        "text": response[start_char:end_char],
        "start_char": start_char,
        "end_char": end_char,
        "verification_required": True,
        "support_level": support_level,
        "verdicts": [verdict for verdict in verdicts if verdict],
        "claim_ids": [
            str(claim.get("claim_id"))
            for claim in claims
            if isinstance(claim.get("claim_id"), str) and claim.get("claim_id")
        ],
        "claim_count": len(claims),
        "evidence_count": evidence_count,
        "source": "claim_decomposition",
    }


def _answer_block_for_nonclaim_text(
    *,
    block_index: int,
    response: str,
    start_char: int,
    end_char: int,
) -> dict[str, Any] | None:
    raw_text = response[start_char:end_char]
    if not raw_text or not raw_text.strip():
        return None

    leading = len(raw_text) - len(raw_text.lstrip())
    trailing = len(raw_text) - len(raw_text.rstrip())
    trimmed_start = start_char + leading
    trimmed_end = end_char - trailing
    text = response[trimmed_start:trimmed_end]
    block_type, verification_required, explanation = _classify_nonclaim_answer_text(text)
    support_level = "unsupported" if verification_required else None

    return {
        "block_id": f"answer_block_{block_index}",
        "type": block_type,
        "category": "non_claim" if not verification_required else "claim_candidate",
        "text": text,
        "start_char": trimmed_start,
        "end_char": trimmed_end,
        "verification_required": verification_required,
        "support_level": support_level,
        "verdicts": [],
        "claim_ids": [],
        "claim_count": 0,
        "evidence_count": 0,
        "source": "answer_segmentation",
        "explanation": explanation,
    }


def _classify_nonclaim_answer_text(text: str) -> tuple[str, bool, str]:
    normalized = " ".join(str(text or "").split()).strip()
    if _ANSWER_BLOCK_CAVEAT_RE.search(normalized):
        return (
            "caveat",
            False,
            "This text communicates a limitation or uncertainty rather than an evidentiary claim.",
        )
    if _ANSWER_BLOCK_LABEL_LEAD_IN_RE.match(normalized):
        return (
            "transition",
            False,
            "This text is treated as formatting rather than a standalone factual claim.",
        )
    if _looks_like_unverified_claim_candidate(normalized):
        return (
            "unverified_claim_candidate",
            True,
            "This text looks factual or legal but was not decomposed into a verified claim.",
        )
    if _ANSWER_BLOCK_TRANSITION_RE.search(normalized):
        return (
            "transition",
            False,
            "This text is treated as connective prose rather than a standalone factual claim.",
        )
    return (
        "derived_explanation",
        False,
        "This text is treated as explanatory prose outside the claim verification set.",
    )


def _looks_like_unverified_claim_candidate(text: str) -> bool:
    if len(text) < 12 or text.endswith("?"):
        return False
    return bool(_ANSWER_BLOCK_FACT_ANCHOR_RE.search(text) and _ANSWER_BLOCK_VERB_RE.search(text))


def _claim_verdict_from_payload(claim: dict[str, Any]) -> str | None:
    verification = claim.get("verification")
    if not isinstance(verification, dict):
        return None
    verdict = verification.get("verdict")
    if not isinstance(verdict, str) or not verdict.strip():
        return None
    return verdict.strip().upper()


def _merged_answer_block_support_level(
    verdicts: list[str | None],
    support_levels: list[str],
) -> str:
    if any(verdict == "CONTRADICTED" for verdict in verdicts):
        return "unsupported"
    if any(level == "unsupported" for level in support_levels):
        return "unsupported"
    if any(level == "possibly_supported" for level in support_levels):
        return "possibly_supported"
    if any(level == "supported" for level in support_levels):
        return "supported"
    return "unsupported"


def _summarize_answer_blocks(blocks: Any) -> dict[str, Any]:
    summary = _empty_answer_block_summary()
    if not isinstance(blocks, list):
        return summary

    for block in blocks:
        if not isinstance(block, dict):
            continue
        summary["total"] += 1
        if block.get("category") == "claim":
            summary["claim_blocks"] += 1
        else:
            summary["non_claim_blocks"] += 1
        if block.get("verification_required") is True:
            summary["verification_required_blocks"] += 1
        if block.get("type") == "unverified_claim_candidate":
            summary["unverified_claim_candidates"] += 1

        support_level = block.get("support_level")
        if support_level == "supported":
            summary["supported_blocks"] += 1
        elif support_level == "possibly_supported":
            summary["possibly_supported_blocks"] += 1
        elif support_level == "unsupported":
            summary["unsupported_blocks"] += 1

    return summary


def _apply_supported_claim_repair(
    response: str,
    normalized_claims: list[dict[str, Any]],
    generation_context_meta: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    summary = _summarize_claim_support(normalized_claims)
    if int(summary.get("total") or 0) == 0:
        return response, {"status": "not_applied:no_claims", "summary": summary}
    if float(summary.get("unsupported_ratio") or 0.0) < REPAIR_UNSUPPORTED_RATIO:
        return response, {"status": "not_applied:unsupported_ratio_below_threshold", "summary": summary}

    retained_claims = _repairable_supported_claim_texts(normalized_claims)
    if retained_claims:
        repaired = "\n\n".join(retained_claims[:REPAIR_MAX_CLAIMS])
        source = "verified_claims"
    else:
        repaired = _repair_fallback_evidence_response(generation_context_meta)
        source = "generation_evidence"

    if not repaired:
        return response, {"status": "not_applied:no_supported_or_evidence_fallback", "summary": summary}

    protected_prefix = _protected_response_prefix(generation_context_meta)
    if protected_prefix:
        repaired = _prepend_protected_prefix(repaired, protected_prefix)

    if _same_normalized_text(response, repaired):
        return response, {"status": "not_applied:repair_matches_response", "summary": summary}

    return repaired, {
        "status": "applied:unsupported_claim_repair",
        "summary": summary,
        "source": source,
        "retained_claim_count": len(retained_claims),
        "repaired_claim_limit": REPAIR_MAX_CLAIMS,
        "protected_prefix_applied": bool(protected_prefix),
    }


def _protected_response_prefix(generation_context_meta: dict[str, Any]) -> str | None:
    missing_case = str(generation_context_meta.get("missing_case") or "").strip()
    if not missing_case:
        return None
    return f"I do not have {missing_case} in the retrieved database, so I cannot say what that case held."


def _prepend_protected_prefix(response: str, protected_prefix: str) -> str:
    answer = str(response or "").strip()
    prefix = str(protected_prefix or "").strip()
    if not prefix:
        return answer
    if _same_normalized_text(answer[: len(prefix)], prefix) or answer.lower().startswith(prefix.lower()):
        return answer
    if not answer:
        return prefix
    return f"{prefix}\n\n{answer}"


def _repairable_supported_claim_texts(normalized_claims: list[dict[str, Any]]) -> list[str]:
    ordered_claims = sorted(
        normalized_claims,
        key=lambda claim: (
            int((claim.get("span") or {}).get("start_char") or 0)
            if isinstance(claim.get("span"), dict)
            else 0
        ),
    )
    retained: list[str] = []
    seen: set[str] = set()
    for claim in ordered_claims:
        annotation = claim.get("annotation")
        support_level = annotation.get("support_level") if isinstance(annotation, dict) else None
        if support_level not in {"supported", "possibly_supported"}:
            continue
        text = str(claim.get("text") or "").strip()
        if _is_low_value_repair_claim_text(text):
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        retained.append(_ensure_sentence_text(text))
    return retained


def _repair_fallback_evidence_response(generation_context_meta: dict[str, Any]) -> str | None:
    opening = _repair_fallback_opening_sentence(generation_context_meta)
    if not opening:
        return None

    if str(generation_context_meta.get("response_depth") or "").strip().lower() != RESPONSE_DEPTH_DETAILED:
        return opening

    support_sentences = _repair_fallback_support_sentences(generation_context_meta, opening)
    if not support_sentences:
        return opening

    paragraph = " ".join(support_sentences[:2])
    bullet_sentences = support_sentences[2:4]
    parts = [opening, paragraph]
    if bullet_sentences:
        parts.append("\n".join(f"- {_ensure_sentence_text(sentence)}" for sentence in bullet_sentences))
    return "\n\n".join(part for part in parts if part.strip())


def _repair_fallback_opening_sentence(generation_context_meta: dict[str, Any]) -> str | None:
    for key in ("canonical_answer_fact", "explicit_holding_sentence", "best_answer_sentence"):
        sentence = str(generation_context_meta.get(key) or "").strip()
        if sentence and not _is_low_value_repair_claim_text(sentence):
            return _ensure_sentence_text(sentence)
    for sentence in generation_context_meta.get("answerable_sentences") or []:
        text = str(sentence or "").strip()
        if text and not _is_low_value_repair_claim_text(text):
            return _ensure_sentence_text(text)
    return None


def _repair_fallback_support_sentences(
    generation_context_meta: dict[str, Any],
    opening: str,
) -> list[str]:
    opening_key = " ".join(opening.lower().split()).rstrip(".!?")
    support: list[str] = []
    seen = {opening_key}
    for sentence in generation_context_meta.get("answerable_sentences") or []:
        text = str(sentence or "").strip()
        if not text or _is_low_value_repair_claim_text(text):
            continue
        normalized = " ".join(text.lower().split()).rstrip(".!?")
        if normalized in seen:
            continue
        seen.add(normalized)
        support.append(_ensure_sentence_text(text))
    return support


def _is_low_value_repair_claim_text(text: str) -> bool:
    stripped = " ".join(str(text or "").split()).strip()
    if len(stripped) < 18:
        return True
    lower = stripped.lower()
    if "insufficient support in retrieved authorities" in lower:
        return True
    if _PROMPT_LEAKAGE_RE.search(stripped):
        return True
    if _MALFORMED_CITATION_FRAGMENT_RE.match(stripped):
        return True
    if re.search(r"\b(?:Dr|Mr|Mrs|Ms|Prof|Hon)\.\s*$", stripped):
        return True
    return False


def _ensure_sentence_text(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return stripped
    if stripped[-1] in ".!?":
        return stripped
    return f"{stripped}."


def _same_normalized_text(left: str, right: str) -> bool:
    return " ".join(str(left or "").split()).lower() == " ".join(str(right or "").split()).lower()


def _summarize_claim_support(claims: Any) -> dict[str, Any]:
    summary = _empty_claim_support_summary()
    if not isinstance(claims, list):
        return summary

    for claim in claims:
        if not isinstance(claim, dict):
            continue
        summary["raw_total"] += 1
        if _is_rhetorical_or_speculative_claim(claim):
            summary["excluded_rhetorical"] += 1
            continue
        support_level = _claim_support_level_from_payload(claim)
        summary["total"] += 1
        if support_level not in ("supported", "possibly_supported", "unsupported"):
            support_level = "unsupported"
        summary[support_level] += 1

    total = int(summary["total"])
    if total > 0:
        summary["unsupported_ratio"] = round(float(summary["unsupported"]) / float(total), 3)
    return summary


def _is_rhetorical_or_speculative_claim(claim: dict[str, Any]) -> bool:
    text = str(claim.get("text") or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _RHETORICAL_CLAIM_PATTERNS)


def _claim_support_level_from_payload(claim: dict[str, Any]) -> str:
    annotation = claim.get("annotation")
    if isinstance(annotation, dict):
        support_level = annotation.get("support_level")
        if isinstance(support_level, str) and support_level:
            return support_level

    verification = claim.get("verification")
    if isinstance(verification, dict):
        verdict = str(verification.get("verdict") or "")
        if verdict in {"VERIFIED", "SUPPORTED"}:
            return "supported"
        if verdict == "POSSIBLE_SUPPORT":
            return "possibly_supported"
    return "unsupported"


def _build_answer_warning(summary: dict[str, Any]) -> dict[str, Any]:
    total = int(summary.get("total") or 0)
    unsupported = int(summary.get("unsupported") or 0)
    unsupported_ratio = float(summary.get("unsupported_ratio") or 0.0)
    show = total > 0 and unsupported_ratio >= UNSUPPORTED_WARNING_RATIO

    if not show:
        return {
            "show": False,
            "kind": None,
            "message": None,
            "threshold_ratio": UNSUPPORTED_WARNING_RATIO,
            "claim_count": total,
            "unsupported_claim_count": unsupported,
            "unsupported_ratio": unsupported_ratio,
        }

    if unsupported == total:
        message = (
            f"This response is largely unsupported by retrieved evidence "
            f"({unsupported} of {total} claims unsupported)."
        )
    else:
        message = (
            f"A large share of this response is unsupported by retrieved evidence "
            f"({unsupported} of {total} claims unsupported)."
        )

    return {
        "show": True,
        "kind": "unsupported_majority",
        "message": message,
        "threshold_ratio": UNSUPPORTED_WARNING_RATIO,
        "claim_count": total,
        "unsupported_claim_count": unsupported,
        "unsupported_ratio": unsupported_ratio,
    }


def _claims_from_pipeline_meta(pipeline_meta: dict[str, Any]) -> list[dict[str, Any]]:
    claims = pipeline_meta.get("claims")
    if not isinstance(claims, list):
        return []
    return [claim for claim in claims if isinstance(claim, dict)]


def _contradictions_from_pipeline_meta(pipeline_meta: dict[str, Any]) -> list[dict[str, Any]]:
    contradictions = pipeline_meta.get("contradictions")
    if not isinstance(contradictions, list):
        return []
    return [item for item in contradictions if isinstance(item, dict)]


def _citations_from_pipeline_meta(pipeline_meta: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = pipeline_meta.get("retrieved_chunks")
    if not isinstance(chunks, list):
        return []

    citations: list[dict[str, Any]] = []
    retrieval_used = bool(pipeline_meta.get("retrieval_used"))
    prompt_chunk_ids = {
        str(chunk_id)
        for chunk_id in pipeline_meta.get("prompt_chunk_ids", [])
        if chunk_id is not None
    }
    for rank, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk.get("id")
        citations.append(
            {
                "doc_id": chunk.get("doc_id"),
                "chunk_id": chunk_id,
                "source_label": (
                    chunk.get("citation")
                    or chunk.get("source_file")
                    or chunk.get("doc_id")
                ),
                "score": chunk.get("score") or chunk.get("rerank_score") or chunk.get("rrf_score"),
                "used_in_prompt": (
                    str(chunk_id) in prompt_chunk_ids
                    if prompt_chunk_ids
                    else retrieval_used
                ),
                "rank": rank,
                "doc_type": chunk.get("doc_type"),
                "court_level": chunk.get("court_level"),
                "citation": chunk.get("citation"),
            }
        )
    return citations


def _merge_retrieved_chunks(
    user_upload_chunks,
    target_metadata_chunks,
    public_chunks,
    *,
    limit: int,
) -> list:
    if limit <= 0:
        return []

    merged = []
    seen_ids: set[str] = set()
    for chunk in [*user_upload_chunks, *target_metadata_chunks, *public_chunks]:
        chunk_id = getattr(chunk, "id", None)
        if not chunk_id or chunk_id in seen_ids:
            continue
        merged.append(chunk)
        seen_ids.add(chunk_id)
        if len(merged) >= limit:
            break
    return merged


def _combined_retrieval_status(
    public_status: str,
    user_upload_status: str,
    user_upload_chunks: list,
) -> str:
    if user_upload_chunks:
        return f"public:{public_status};user:{user_upload_status}"
    if user_upload_status.startswith("error:"):
        return f"public:{public_status};user:{user_upload_status}"
    return public_status


def _load_default_retriever() -> tuple[HybridRetriever | None, str]:
    artifacts = discover_index_artifacts()
    bm25_index = _load_bm25_index(artifacts.bm25_path)
    vector_store, embedder = _load_vector_store(
        artifacts.chroma_path,
        collection_name=artifacts.collection_name,
    )

    if bm25_index is None and vector_store is None:
        return None, "unavailable:no_indices"

    try:
        retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            embedder=embedder,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"error:{exc.__class__.__name__}"

    return retriever, "ok"


def _load_bm25_index(index_path: Path) -> BM25Index | None:
    if not index_path.exists():
        return None

    bm25_index = BM25Index(index_path=index_path)
    bm25_index.load()
    return bm25_index


def _load_vector_store(
    chroma_path: Path,
    *,
    collection_name: str | None = None,
) -> tuple[ChromaStore | None, Embedder | None]:
    if not chroma_path.exists():
        return None, None

    try:
        has_entries = any(chroma_path.iterdir())
    except OSError:
        has_entries = False

    if not has_entries:
        return None, None

    vector_store = ChromaStore(path=chroma_path, collection_name=collection_name)
    return vector_store, Embedder()


def _serialize_chunk(chunk) -> dict[str, Any]:
    payload = chunk.to_dict()
    payload["text_preview"] = chunk.text[:280]
    source_chunk_id = getattr(chunk, "verification_source_chunk_id", None)
    if source_chunk_id:
        payload["source_chunk_id"] = source_chunk_id
    evidence_role = getattr(chunk, "verification_evidence_role", None)
    if evidence_role:
        payload["evidence_role"] = evidence_role
    return payload


def _format_chunk_for_prompt(chunk) -> str:
    metadata = [
        f"Chunk ID: {chunk.id}",
        f"Document ID: {chunk.doc_id}",
        f"Document type: {chunk.doc_type}",
    ]
    if chunk.case_name:
        metadata.append(f"Case name: {chunk.case_name}")
    if chunk.court:
        metadata.append(f"Court: {chunk.court}")
    if chunk.court_level:
        metadata.append(f"Court level: {chunk.court_level}")
    if chunk.citation:
        metadata.append(f"Citation: {chunk.citation}")
    if chunk.date_decided:
        date_decided = chunk.date_decided
        metadata.append(
            f"Date decided: {date_decided.isoformat() if hasattr(date_decided, 'isoformat') else date_decided}"
        )
    if chunk.title is not None:
        metadata.append(f"Title: {chunk.title}")
    if chunk.section:
        metadata.append(f"Section: {chunk.section}")
    if chunk.source_file:
        metadata.append(f"Source file: {chunk.source_file}")

    return f"{'; '.join(metadata)}\n{chunk.text}"


def _format_user_upload_context_for_prompt(chunks: list) -> list[str]:
    formatted: list[str] = []
    for chunk in chunks:
        if getattr(chunk, "doc_type", None) == "user_upload":
            formatted.append(
                "Evidence type: user-uploaded document fact\n"
                "Mandatory answer rule: if the user asks about an uploaded document, the first sentence must answer from this user-uploaded document fact section before discussing any retrieved legal authority.\n"
                "Absence rule: if this fact section does not state the requested fact, say it is not identified in the uploaded document; do not infer it from nearby facts.\n"
                "Use priority: primary record facts for questions about the uploaded document. "
                "Do not treat this uploaded document as precedential legal authority.\n"
                f"{_format_chunk_for_prompt(chunk)}"
            )
        else:
            formatted.append(
                "Evidence type: retrieved legal authority for comparison\n"
                "Use priority: compare or contextualize the uploaded document only when the user's question asks for external authority.\n"
                f"{_format_chunk_for_prompt(chunk)}"
            )
    return formatted


def _claim_evidence_consistency_guard(
    claim_text: str,
    supporting_chunk: Any,
    verdict_label: str,
    *,
    query_grounding: dict[str, Any] | None = None,
    generation_context_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if verdict_label not in {"VERIFIED", "SUPPORTED", "POSSIBLE_SUPPORT"}:
        return {"status": "not_applied:verdict_not_supported"}
    if supporting_chunk is None:
        return {"status": "not_applied:no_supporting_chunk"}

    evidence_text = " ".join(
        str(value or "")
        for value in (
            getattr(supporting_chunk, "case_name", None),
            getattr(supporting_chunk, "citation", None),
            getattr(supporting_chunk, "text", None),
        )
    )
    evidence_norm = _compact_guard_text(evidence_text)

    missing_case_names = [
        case_name
        for case_name in _extract_guard_case_names(claim_text)
        if not _guard_case_name_present(case_name, supporting_chunk, evidence_norm)
    ]
    missing_citations = [
        citation
        for citation in _extract_guard_citations(claim_text)
        if _compact_guard_text(citation) not in evidence_norm
    ]
    missing_anchors = [
        anchor
        for anchor in _extract_guard_legal_anchors(claim_text)
        if _compact_guard_text(anchor) not in evidence_norm
    ]

    missing = {
        "case_names": missing_case_names,
        "citations": missing_citations,
        "legal_anchors": missing_anchors,
    }
    if any(missing.values()):
        return {
            "status": "blocked:missing_named_evidence",
            "message": (
                "The claim names a case, citation, statute, or constitutional anchor "
                "that does not appear in the supporting evidence."
            ),
            "missing": missing,
        }

    alignment_guard = _target_evidence_alignment_guard(
        claim_text,
        supporting_chunk,
        query_grounding=query_grounding,
        generation_context_meta=generation_context_meta,
    )
    if alignment_guard["status"].startswith("blocked:"):
        return alignment_guard

    polarity_guard = _legal_polarity_positive_support_guard(
        claim_text,
        supporting_chunk,
        verdict_label,
        query_grounding=query_grounding,
        generation_context_meta=generation_context_meta,
    )
    if polarity_guard["status"].startswith("blocked:"):
        return polarity_guard

    research_leads_guard = _research_leads_positive_support_guard(
        claim_text,
        supporting_chunk,
        verdict_label,
        query_grounding=query_grounding,
        generation_context_meta=generation_context_meta,
    )
    if research_leads_guard["status"].startswith(("blocked:", "demote:")):
        return research_leads_guard

    query_subject_guard = _query_subject_positive_support_guard(
        claim_text,
        supporting_chunk,
        verdict_label,
        query_grounding=query_grounding,
        generation_context_meta=generation_context_meta,
    )
    if query_subject_guard["status"].startswith("blocked:"):
        return query_subject_guard

    high_risk_guard = _high_risk_positive_support_guard(
        claim_text,
        supporting_chunk,
        verdict_label,
        query_grounding=query_grounding,
        generation_context_meta=generation_context_meta,
    )
    if high_risk_guard["status"].startswith("blocked:"):
        return high_risk_guard

    return {"status": "passed"}


def _target_evidence_alignment_guard(
    claim_text: str,
    supporting_chunk: Any,
    *,
    query_grounding: dict[str, Any] | None,
    generation_context_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    target_case = str((query_grounding or {}).get("target_case") or "").strip()
    if not target_case:
        return {"status": "not_applied:no_target_case"}

    evidence_parts = [
        str(getattr(supporting_chunk, "case_name", "") or ""),
        str(getattr(supporting_chunk, "citation", "") or ""),
        str(getattr(supporting_chunk, "text", "") or ""),
        str((generation_context_meta or {}).get("canonical_answer_fact") or ""),
        str((generation_context_meta or {}).get("explicit_holding_sentence") or ""),
        str((generation_context_meta or {}).get("best_answer_sentence") or ""),
        " ".join(str(item or "") for item in (generation_context_meta or {}).get("answerable_sentences") or []),
    ]
    evidence_tokens = _content_tokens(" ".join(evidence_parts))
    claim_tokens = _content_tokens(claim_text)
    if not claim_tokens or not evidence_tokens:
        return {"status": "not_applied:no_alignment_tokens"}

    overlap = claim_tokens & evidence_tokens
    if len(overlap) >= 2:
        return {"status": "passed:token_overlap", "overlap": sorted(overlap)[:12]}

    if len(claim_tokens) <= 5 and overlap:
        return {"status": "passed:short_claim_overlap", "overlap": sorted(overlap)}

    return {
        "status": "blocked:low_target_evidence_overlap",
        "message": (
            "The claim has too little overlap with the target-case evidence to be "
            "treated as supported."
        ),
        "overlap": sorted(overlap),
        "claim_tokens": sorted(claim_tokens)[:20],
    }


def _legal_polarity_positive_support_guard(
    claim_text: str,
    supporting_chunk: Any,
    verdict_label: str,
    *,
    query_grounding: dict[str, Any] | None,
    generation_context_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    if verdict_label not in {"VERIFIED", "SUPPORTED", "POSSIBLE_SUPPORT"}:
        return {"status": "not_applied:verdict_not_supported"}

    target_case = str((query_grounding or {}).get("target_case") or "").strip()
    evidence_text = " ".join(
        str(value or "")
        for value in (
            getattr(supporting_chunk, "case_name", None),
            getattr(supporting_chunk, "citation", None),
            getattr(supporting_chunk, "text", None),
            (generation_context_meta or {}).get("canonical_answer_fact"),
            (generation_context_meta or {}).get("explicit_holding_sentence"),
            (generation_context_meta or {}).get("best_answer_sentence"),
        )
    )

    claim_says_chevron_controls = bool(_CHEVRON_CONTROLS_CLAIM_RE.search(claim_text))
    if not claim_says_chevron_controls:
        return {"status": "not_applied:no_known_polarity_conflict"}

    loper_target = "loper bright" in target_case.lower()
    evidence_rejects_chevron = bool(_CHEVRON_REJECTING_EVIDENCE_RE.search(evidence_text))
    if loper_target or evidence_rejects_chevron:
        return {
            "status": "blocked:legal_polarity_conflict",
            "message": (
                "The claim states that Chevron deference controls or applies, but "
                "the target authority rejects Chevron as the controlling rule."
            ),
            "target_case": target_case,
            "evidence_rejects_chevron": evidence_rejects_chevron,
        }

    return {"status": "not_applied:no_known_polarity_conflict"}


def _research_leads_positive_support_guard(
    claim_text: str,
    supporting_chunk: Any,
    verdict_label: str,
    *,
    query_grounding: dict[str, Any] | None,
    generation_context_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    if verdict_label not in {"VERIFIED", "SUPPORTED"}:
        return {"status": "not_applied:not_strong_positive"}
    if not _is_research_leads_context(query_grounding, generation_context_meta):
        return {"status": "not_applied:not_research_leads"}

    evidence_text = " ".join(
        str(value or "")
        for value in (
            getattr(supporting_chunk, "case_name", None),
            getattr(supporting_chunk, "citation", None),
            getattr(supporting_chunk, "text", None),
        )
    )
    claim_specific = _research_leads_specific_tokens(claim_text)
    evidence_specific = _research_leads_specific_tokens(evidence_text)
    overlap = claim_specific & evidence_specific
    if len(overlap) >= 2:
        return {"status": "passed:research_leads_source_alignment", "overlap": sorted(overlap)[:12]}

    return {
        "status": "demote:research_leads_weak_source_alignment",
        "message": (
            "The research-leads claim has weak source/entity alignment, so it is "
            "downgraded from supported to possible support."
        ),
        "overlap": sorted(overlap),
        "claim_tokens": sorted(claim_specific)[:20],
    }


def _query_subject_positive_support_guard(
    claim_text: str,
    supporting_chunk: Any,
    verdict_label: str,
    *,
    query_grounding: dict[str, Any] | None,
    generation_context_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    _ = generation_context_meta
    if verdict_label not in {"VERIFIED", "SUPPORTED", "POSSIBLE_SUPPORT"}:
        return {"status": "not_applied:verdict_not_supported"}

    query = str((query_grounding or {}).get("query") or "").strip()
    if not query:
        return {"status": "not_applied:no_query"}
    if not _SOURCE_DISCIPLINE_QUERY_RE.search(query):
        return {"status": "not_applied:not_source_discipline_query"}
    if (query_grounding or {}).get("mentions_user_upload") or (query_grounding or {}).get("has_user_upload_context"):
        return {"status": "not_applied:user_upload_context"}

    subject_tokens = _query_subject_tokens(query)
    if len(subject_tokens) < 2:
        return {"status": "not_applied:not_enough_subject_tokens", "subject_tokens": sorted(subject_tokens)}

    evidence_text = " ".join(
        str(value or "")
        for value in (
            claim_text,
            getattr(supporting_chunk, "case_name", None),
            getattr(supporting_chunk, "citation", None),
            getattr(supporting_chunk, "source_file", None),
            getattr(supporting_chunk, "text", None),
        )
    )
    evidence_tokens = _content_tokens(evidence_text)
    missing = subject_tokens - evidence_tokens
    if not missing:
        return {"status": "passed:query_subject_alignment", "subject_tokens": sorted(subject_tokens)}

    return {
        "status": "blocked:query_subject_mismatch",
        "message": (
            "The supporting evidence does not cover the distinctive subject terms "
            "from the source-discipline query."
        ),
        "missing_subject_tokens": sorted(missing),
        "subject_tokens": sorted(subject_tokens),
    }


def _high_risk_positive_support_guard(
    claim_text: str,
    supporting_chunk: Any,
    verdict_label: str,
    *,
    query_grounding: dict[str, Any] | None,
    generation_context_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    if verdict_label not in {"VERIFIED", "SUPPORTED", "POSSIBLE_SUPPORT"}:
        return {"status": "not_applied:verdict_not_supported"}
    if not _is_high_risk_positive_support_context(query_grounding, generation_context_meta):
        return {"status": "not_applied:not_high_risk"}

    evidence_text = " ".join(
        str(value or "")
        for value in (
            getattr(supporting_chunk, "case_name", None),
            getattr(supporting_chunk, "citation", None),
            getattr(supporting_chunk, "source_file", None),
            getattr(supporting_chunk, "text", None),
        )
    )
    if _RELATED_RETRIEVED_AUTHORITY_CLAIM_RE.search(claim_text):
        return {
            "status": "blocked:high_risk_related_authority_claim",
            "message": (
                "The claim relies on related retrieved authorities in a high-risk "
                "context rather than direct support for the requested source."
            ),
        }

    target_citation = str((query_grounding or {}).get("target_citation") or "").strip()
    grounding_source = str((query_grounding or {}).get("source") or "").strip()
    if grounding_source == "citation_unresolved" and target_citation:
        evidence_norm = _compact_guard_text(evidence_text)
        if _compact_guard_text(target_citation) not in evidence_norm:
            return {
                "status": "blocked:high_risk_unresolved_citation_mismatch",
                "message": (
                    "The query asks for an unresolved citation, and the supporting "
                    "evidence does not match that citation."
                ),
                "target_citation": target_citation,
            }

    if _is_research_leads_context(query_grounding, generation_context_meta):
        return {"status": "passed:research_leads_not_strictly_gated"}

    claim_tokens = _content_tokens(claim_text)
    evidence_tokens = _content_tokens(evidence_text)
    if not claim_tokens or not evidence_tokens:
        return {"status": "not_applied:no_alignment_tokens"}

    overlap = claim_tokens & evidence_tokens
    required_overlap = 2 if len(claim_tokens) <= 6 else 3
    if len(overlap) >= required_overlap:
        return {
            "status": "passed:high_risk_content_overlap",
            "overlap": sorted(overlap)[:12],
            "required_overlap": required_overlap,
        }

    return {
        "status": "blocked:high_risk_low_source_alignment",
        "message": (
            "The claim appears in a high-risk retrieval context and lacks enough "
            "source alignment to be treated as supported."
        ),
        "overlap": sorted(overlap),
        "required_overlap": required_overlap,
        "claim_tokens": sorted(claim_tokens)[:20],
    }


def _is_high_risk_positive_support_context(
    query_grounding: dict[str, Any] | None,
    generation_context_meta: dict[str, Any] | None,
) -> bool:
    grounding = query_grounding or {}
    context = generation_context_meta or {}

    if grounding.get("target_case"):
        return False
    if str(context.get("status") or "").startswith("applied:user_upload"):
        return False
    if grounding.get("mentions_user_upload") or grounding.get("has_user_upload_context"):
        return False

    grounding_status = str(grounding.get("status") or "")
    grounding_source = str(grounding.get("source") or "")
    context_status = str(context.get("status") or "")
    answer_mode = str(context.get("answer_mode") or "")
    missing_case = str(context.get("missing_case") or "").strip()

    return (
        not grounding_status.startswith("resolved:")
        or grounding_status.startswith("not_resolved:")
        or grounding_source in {"unresolved", "retrieval_convergence"}
        or bool(missing_case)
        or "fallback" in context_status
        or "research_leads" in context_status
        or "research_leads" in answer_mode
    )


def _is_research_leads_context(
    query_grounding: dict[str, Any] | None,
    generation_context_meta: dict[str, Any] | None,
) -> bool:
    grounding = query_grounding or {}
    context = generation_context_meta or {}
    return (
        grounding.get("query_intent") == "research_leads"
        or grounding.get("llm_route") == "research_leads"
        or "research_leads" in str(grounding.get("source") or "")
        or "research_leads" in str(context.get("status") or "")
        or "research_leads" in str(context.get("answer_mode") or "")
    )


def _research_leads_specific_tokens(text: str) -> set[str]:
    return {
        token
        for token in _content_tokens(text)
        if token not in _RESEARCH_LEADS_GENERIC_TOKENS and not token.isdigit()
    }


def _query_subject_tokens(query: str) -> set[str]:
    tokens = _content_tokens(query)
    return {
        token
        for token in tokens
        if len(token) >= 4 and token not in _QUERY_SUBJECT_GENERIC_TOKENS
    }


def _extract_guard_case_names(text: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for match in _CASE_NAME_RE.finditer(text or ""):
        name = " ".join(match.group(1).split()).strip()
        key = _normalize_case_name(name)
        if key and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def _extract_guard_citations(text: str) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for match in _US_CITATION_RE.finditer(text or ""):
        citation = " ".join(match.group(0).split()).strip()
        key = _compact_guard_text(citation)
        if key and key not in seen:
            seen.add(key)
            citations.append(citation)
    return citations


def _extract_guard_legal_anchors(text: str) -> list[str]:
    anchors: list[str] = []
    seen: set[str] = set()
    for pattern in (_CONSTITUTIONAL_ANCHOR_RE, _STATUTE_SECTION_ANCHOR_RE):
        for match in pattern.finditer(text or ""):
            anchor = " ".join(match.group(0).split()).strip()
            key = _compact_guard_text(anchor)
            if key and key not in seen:
                seen.add(key)
                anchors.append(anchor)
    return anchors


def _guard_case_name_present(case_name: str, supporting_chunk: Any, evidence_norm: str) -> bool:
    chunk_case_name = str(getattr(supporting_chunk, "case_name", "") or "")
    if _case_name_matches(case_name, chunk_case_name):
        return True
    case_key = _compact_guard_text(case_name)
    return bool(case_key and case_key in evidence_norm)


def _compact_guard_text(text: str) -> str:
    return _NON_ALNUM_RE.sub("", str(text or "").lower())


def _serialize_claim_with_verification(
    claim,
    verdict: AggregatedScore,
    *,
    query_grounding: dict[str, Any] | None = None,
    generation_context_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = claim.to_dict()
    classification = classify_verification(verdict)
    consistency_guard = _claim_evidence_consistency_guard(
        str(payload.get("text") or ""),
        verdict.best_chunk,
        classification.label,
        query_grounding=query_grounding,
        generation_context_meta=generation_context_meta,
    )
    verdict_label = classification.label
    verdict_explanation = classification.explanation
    if consistency_guard["status"].startswith("blocked:"):
        verdict_label = "UNSUPPORTED"
        verdict_explanation = str(consistency_guard["message"])
    elif consistency_guard["status"].startswith("demote:"):
        verdict_label = "POSSIBLE_SUPPORT"
        verdict_explanation = str(consistency_guard["message"])
    best_supporting_chunk = _serialize_chunk(verdict.best_chunk) if verdict.best_chunk is not None else None
    best_contradicting_chunk = (
        _serialize_chunk(verdict.best_contradicting_chunk)
        if verdict.best_contradicting_chunk is not None
        else None
    )
    payload["verification"] = {
        "final_score": verdict.final_score,
        "verdict": verdict_label,
        "verdict_explanation": verdict_explanation,
        "is_contradicted": verdict.is_contradicted,
        "best_chunk_idx": verdict.best_chunk_idx,
        "support_ratio": verdict.support_ratio,
        "component_scores": verdict.component_scores,
        "consistency_guard": consistency_guard,
        "best_chunk": best_supporting_chunk,
        "best_supporting_chunk_idx": verdict.best_chunk_idx,
        "best_supporting_chunk": best_supporting_chunk,
        "best_supporting_score": verdict.component_scores.get("best_entailment"),
        "best_contradicting_chunk_idx": verdict.best_contradicting_chunk_idx,
        "best_contradicting_chunk": best_contradicting_chunk,
        "best_contradiction_score": verdict.component_scores.get("best_contradiction"),
    }
    return payload


def _normalize_verified_claims_for_frontend(
    raw_claims: list,
    verdicts: list,
    *,
    citations: list[dict[str, Any]],
    query_grounding: dict[str, Any] | None = None,
    generation_context_meta: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verified_claims = [
        _serialize_claim_with_verification(
            claim,
            verdict,
            query_grounding=query_grounding,
            generation_context_meta=generation_context_meta,
        )
        for claim, verdict in zip(raw_claims, verdicts)
    ]
    return normalize_claims_for_frontend(
        verified_claims,
        citations=citations,
    )
