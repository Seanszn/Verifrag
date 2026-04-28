from __future__ import annotations

from src import pipeline as pipeline_module
from src.ingestion.document import LegalChunk


class _RouteLLM:
    def __init__(self, route: str, confidence: float = 0.9):
        self.route = route
        self.confidence = confidence
        self.queries = []

    def classify_query_route(self, query: str):
        self.queries.append(query)
        return {
            "status": "ok",
            "route": self.route,
            "confidence": self.confidence,
            "reason": "test route",
        }


def test_extract_target_case_name_handles_rich_command_prompt():
    assert (
        pipeline_module._extract_target_case_name(
            "In Burnett v. United States, give a bottom line and then explain the posture."
        )
        == "Burnett v. United States"
    )
    assert (
        pipeline_module._extract_target_case_name(
            "In Ames v. Ohio Department of Youth Services, explain the Court's Title VII rule."
        )
        == "Ames v. Ohio Department of Youth Services"
    )


def test_upload_query_does_not_treat_uploaded_caption_as_missing_public_case():
    query = (
        "In the uploaded Riley v. Northstar draft, what is the strongest motion "
        "argument about Dr. Lena Marquez?"
    )
    chunks = [
        LegalChunk(
            id="upload_1:0",
            doc_id="upload_1",
            text="Riley v. Northstar draft says Dr. Lena Marquez failed to test the H-17 battery.",
            chunk_index=0,
            doc_type="user_upload",
            case_name="daubert product liability motion",
            source_file="motion.txt",
        )
    ]

    grounding = pipeline_module._resolve_query_grounding(query, chunks)
    fallback_chunks, fallback_meta = pipeline_module._build_missing_case_topic_fallback(
        query,
        query_grounding=grounding,
        retriever=None,
        initial_chunks=chunks,
        limit=6,
    )

    assert grounding["explicit_case"] is None
    assert grounding["source"] == "user_upload"
    assert grounding["has_user_upload_context"] is True
    assert fallback_chunks == []
    assert fallback_meta["status"] == "not_applied:user_upload_context"


def test_user_upload_generation_context_keeps_corpus_authorities_for_comparison():
    query = "Compare my uploaded Northstar draft to relevant expert testimony cases."
    upload_chunk = LegalChunk(
        id="upload_1:0",
        doc_id="upload_1",
        text="Northstar argues Dr. Marquez failed to test the H-17 battery.",
        chunk_index=0,
        doc_type="user_upload",
        case_name="daubert product liability motion",
        source_file="motion.txt",
    )
    authority_chunk = LegalChunk(
        id="case_1:0",
        doc_id="case_1",
        text="The Court discussed risks from expert testimony about mental state.",
        chunk_index=0,
        doc_type="case",
        case_name="Diaz v. United States",
        citation="602 U.S. 526",
        court_level="scotus",
    )

    grounding = pipeline_module._resolve_query_grounding(query, [upload_chunk, authority_chunk])
    context, meta = pipeline_module._build_generation_context(
        query,
        [upload_chunk, authority_chunk],
        query_grounding=grounding,
        response_depth="detailed",
    )

    assert meta["status"] == "applied:user_upload_context"
    assert len(context) == 2
    assert "Evidence type: user-uploaded document fact" in context[0]
    assert "Absence rule:" in context[0]
    assert "Source file: motion.txt" in context[0]
    assert "Evidence type: retrieved legal authority for comparison" in context[1]
    assert "Diaz v. United States" in context[1]


def test_user_upload_generation_context_provides_sentence_evidence_for_verification():
    query = "Using my uploaded draft, what did Marquez fail to test?"
    upload_chunk = LegalChunk(
        id="upload_1:0",
        doc_id="upload_1",
        text=(
            "Riley v. Northstar draft concerns expert causation. "
            "Marquez did not personally test the H-17 battery pack. "
            "Northstar argues her thermal model was not peer reviewed."
        ),
        chunk_index=0,
        doc_type="user_upload",
        case_name="daubert product liability motion",
        source_file="motion.txt",
    )

    grounding = pipeline_module._resolve_query_grounding(query, [upload_chunk])
    _context, generation_meta = pipeline_module._build_generation_context(
        query,
        [upload_chunk],
        query_grounding=grounding,
    )
    chunks, verification_meta = pipeline_module._select_chunks_for_verification(
        retrieved_chunks=[upload_chunk],
        prompt_chunks=[upload_chunk],
        generation_context_meta=generation_meta,
        query_grounding=grounding,
    )

    assert generation_meta["status"] == "applied:user_upload_context"
    assert any(
        "Marquez did not personally test the H-17 battery pack." == sentence
        for sentence in generation_meta["answerable_sentences"]
    )
    assert generation_meta["source_evidence_sentences"]
    assert verification_meta == {
        "status": "applied:scoped",
        "scope": "sentence_evidence",
        "tiers": ["sentence_evidence"],
    }
    assert any("H-17 battery pack" in chunk.text for chunk in chunks)
    assert all(chunk.doc_type == "user_upload" for chunk in chunks)
    assert all(chunk.source_file == "motion.txt" for chunk in chunks)


def test_user_upload_prompt_selection_excludes_public_chunks_without_comparison_signal():
    query = "Using my uploaded draft, what is the strongest causation argument?"
    upload_chunk = LegalChunk(
        id="upload_1:0",
        doc_id="upload_1",
        text="Northstar argues Dr. Marquez failed to test the H-17 battery.",
        chunk_index=0,
        doc_type="user_upload",
        source_file="motion.txt",
    )
    public_chunk = LegalChunk(
        id="case_1:0",
        doc_id="case_1",
        text="Unrelated public authority.",
        chunk_index=0,
        doc_type="case",
        case_name="Alexander v. South Carolina State Conference of the NAACP",
    )

    grounding = pipeline_module._resolve_query_grounding(query, [upload_chunk, public_chunk])
    prompt_chunks, meta = pipeline_module._select_prompt_chunks_for_generation(
        query,
        [upload_chunk, public_chunk],
        query_grounding=grounding,
    )

    assert grounding["query_intent"] == "issue_analysis"
    assert meta["status"] == "applied:user_upload_only"
    assert [chunk.id for chunk in prompt_chunks] == ["upload_1:0"]


def test_user_upload_prompt_selection_caps_public_chunks_for_comparison():
    query = "Compare my uploaded draft to relevant corpus cases about expert reliability."
    upload_chunk = LegalChunk(
        id="upload_1:0",
        doc_id="upload_1",
        text="Northstar argues Dr. Marquez failed to test the H-17 battery.",
        chunk_index=0,
        doc_type="user_upload",
        source_file="motion.txt",
    )
    public_chunks = [
        LegalChunk(
            id=f"case_{index}:0",
            doc_id=f"case_{index}",
            text=f"Public expert authority {index}.",
            chunk_index=0,
            doc_type="case",
            case_name=f"Example {index} v. United States",
        )
        for index in range(5)
    ]

    grounding = pipeline_module._resolve_query_grounding(query, [upload_chunk, *public_chunks])
    prompt_chunks, meta = pipeline_module._select_prompt_chunks_for_generation(
        query,
        [upload_chunk, *public_chunks],
        query_grounding=grounding,
    )

    assert meta["status"] == "applied:user_upload_with_comparison_authorities"
    assert meta["public_prompt_limit"] == pipeline_module.USER_UPLOAD_COMPARISON_PUBLIC_CHUNK_LIMIT
    assert [chunk.id for chunk in prompt_chunks] == [
        "upload_1:0",
        "case_0:0",
        "case_1:0",
        "case_2:0",
    ]


def test_user_upload_public_retrieval_query_rewrites_to_issue_terms():
    query = (
        "Compare my uploaded Northstar Model H-17 draft to relevant corpus cases or precedent. "
        "How does the argument about Dr. Lena Marquez's causation methodology compare to prior cases?"
    )
    upload_chunk = LegalChunk(
        id="upload_1:0",
        doc_id="upload_1",
        text=(
            "Franklin County Superior Court. Riley v. Northstar Home Robotics, Inc. "
            "The uploaded draft states that Dr. Lena Marquez did not test the Northstar Model H-17 "
            "battery pack under substantially similar charging conditions. "
            "The strongest motion argument is that Dr. Marquez failed to connect her observations "
            "to a reliable causation methodology and failed to account for alternative ignition sources."
        ),
        chunk_index=0,
        doc_type="user_upload",
        source_file="northstar_upload_probe.txt",
    )

    public_query, meta = pipeline_module._build_user_upload_public_retrieval_query(
        query,
        [upload_chunk],
    )

    assert meta["status"] == "applied:user_upload_comparison_rewrite"
    assert public_query != query
    assert "Daubert" in public_query
    assert "expert opinion admissibility" in public_query
    assert "reliable principles and methods" in public_query
    assert "reliable causation methodology" in public_query
    assert "alternative causes" in public_query
    assert "case law" in public_query
    assert "Riley" not in public_query
    assert "Northstar" not in public_query
    assert "Marquez" not in public_query
    assert "H-17" not in public_query


def test_user_upload_public_retrieval_rewrite_improves_lexical_match_to_issue_authority():
    query = (
        "Compare my uploaded Northstar Model H-17 draft to relevant corpus cases or precedent. "
        "How does Dr. Lena Marquez's causation methodology compare to prior cases?"
    )
    upload_chunk = LegalChunk(
        id="upload_1:0",
        doc_id="upload_1",
        text=(
            "Riley v. Northstar Home Robotics, Inc. Dr. Lena Marquez failed to connect her "
            "observations to a reliable causation methodology and failed to account for "
            "alternative ignition sources."
        ),
        chunk_index=0,
        doc_type="user_upload",
        source_file="northstar_upload_probe.txt",
    )
    issue_authority = LegalChunk(
        id="case_issue:0",
        doc_id="case_issue",
        text=(
            "Expert testimony may be excluded when a causation opinion lacks reliable "
            "methodology and fails to address alternative causes."
        ),
        chunk_index=0,
        doc_type="case",
        case_name="Example Expert Reliability Case",
    )
    party_noise = LegalChunk(
        id="case_noise:0",
        doc_id="case_noise",
        text="Northstar and Riley are mentioned in an unrelated corporate filing.",
        chunk_index=0,
        doc_type="case",
        case_name="Northstar Holdings v. Riley",
    )

    rewritten_query, meta = pipeline_module._build_user_upload_public_retrieval_query(
        query,
        [upload_chunk],
    )
    query_tokens = pipeline_module._content_tokens(rewritten_query)

    def overlap(chunk: LegalChunk) -> int:
        return len(query_tokens & pipeline_module._content_tokens(chunk.text))

    assert meta["status"] == "applied:user_upload_comparison_rewrite"
    assert overlap(issue_authority) > overlap(party_noise)


def test_build_generation_context_prefers_explicit_holding_sentence():
    query = (
        "In Esteras v. United States, what did the Supreme Court decide about which "
        "sentencing factors a court may consider when revoking supervised release?"
    )
    prompt_chunks = [
        LegalChunk(
            id="esteras_chunk_0",
            doc_id="esteras_doc",
            text=(
                "On writ of certiorari to the United States Court of Appeals for the Sixth Circuit. "
                "The Court agrees with Esteras. "
                "District courts cannot consider section 3553(a)(2)(A) when revoking supervised release."
            ),
            chunk_index=0,
            doc_type="case",
            case_name="Esteras v. United States",
            court="scotus",
            court_level="scotus",
        )
    ]

    context, meta = pipeline_module._build_generation_context(query, prompt_chunks)

    assert meta["status"] == "applied:sentence_evidence"
    assert meta["explicit_holding_sentence"] == (
        "District courts cannot consider section 3553(a)(2)(A) when revoking supervised release."
    )
    assert context
    assert "District courts cannot consider section 3553(a)(2)(A) when revoking supervised release." in context[0]
    assert all("On writ of certiorari to the United States Court of Appeals" not in item for item in context)


def test_build_generation_context_extracts_held_sentence_with_page_noise():
    query = (
        "In Royal Canin U.S.A., Inc. v. Wullschleger, what happened to federal "
        "jurisdiction after the plaintiff amended the complaint to remove federal claims?"
    )
    prompt_chunks = [
        LegalChunk(
            id="royal_chunk_0",
            doc_id="royal_doc",
            text=(
                "PRELIMINARY PRINT Volume 604 U. S. Part 1 Pages 22-44. "
                "22 ROYAL CANIN U. S. A. v. WULLSCHLEGER Syllabus "
                "Held: When a plaintiff amends her complaint to delete the federal-law claims "
                "that enabled removal to federal court, leaving only state-law claims behind, "
                "the federal court loses supplemental jurisdiction over the state claims, and "
                "the case must be remanded to state court."
            ),
            chunk_index=0,
            doc_type="case",
            case_name="Royal Canin U. S. A. v. Wullschleger",
            citation="604 U.S. 22",
            court_level="scotus",
        )
    ]

    grounding = pipeline_module._resolve_query_grounding(query, prompt_chunks)
    context, meta = pipeline_module._build_generation_context(
        query,
        prompt_chunks,
        query_grounding=grounding,
    )

    assert grounding["target_case"] == "Royal Canin U. S. A. v. Wullschleger"
    assert meta["explicit_holding_sentence"].startswith(
        "When a plaintiff amends her complaint to delete the federal-law claims"
    )
    assert "Allowed answer fact:" in context[0]
    assert "PRELIMINARY PRINT" not in context[0]


def test_build_generation_context_expands_sentence_limit_for_detailed_answers():
    query = (
        "In Example v. United States, explain the holding, reasoning, practical rule, "
        "and give bullets."
    )
    text = (
        "The Court held that the statute does not authorize the agency action. "
        "The Court explained that the statutory text controls the analysis. "
        "The Court rejected the Government's broader reading of the statute. "
        "The Court said district courts must apply the narrower statutory rule. "
        "The practical result is that agencies need clear statutory authority. "
        "The judgment of the court of appeals was affirmed."
    )
    prompt_chunks = [
        LegalChunk(
            id="example_chunk_0",
            doc_id="example_doc",
            text=text,
            chunk_index=0,
            doc_type="case",
            case_name="Example v. United States",
            court_level="scotus",
        )
    ]
    grounding = pipeline_module._resolve_query_grounding(query, prompt_chunks)

    _, concise_meta = pipeline_module._build_generation_context(
        query,
        prompt_chunks,
        query_grounding=grounding,
    )
    detailed_context, detailed_meta = pipeline_module._build_generation_context(
        query,
        prompt_chunks,
        query_grounding=grounding,
        response_depth="detailed",
    )

    assert concise_meta["sentence_limit"] == pipeline_module.TARGET_CASE_PROMPT_SENTENCE_LIMIT
    assert len(concise_meta["answerable_sentences"]) == 3
    assert detailed_meta["response_depth"] == "detailed"
    assert detailed_meta["sentence_limit"] == pipeline_module.TARGET_CASE_DETAILED_PROMPT_SENTENCE_LIMIT
    assert len(detailed_meta["answerable_sentences"]) > len(concise_meta["answerable_sentences"])
    assert len(detailed_context) == len(detailed_meta["answerable_sentences"])


def test_explicit_holding_override_replaces_garbage_response():
    query = (
        "In Esteras v. United States, what did the Supreme Court decide about which "
        "sentencing factors a court may consider when revoking supervised release?"
    )
    response, meta = pipeline_module._apply_explicit_holding_response_override(
        query=query,
        response="The Court denied certiorari.",
        generation_context_meta={
            "explicit_holding_sentence": (
                "District courts cannot consider section 3553(a)(2)(A) when revoking supervised release."
            ),
            "source_chunk_ids": ["esteras_chunk_0"],
        },
    )

    assert response == "District courts cannot consider section 3553(a)(2)(A) when revoking supervised release."
    assert meta["status"] == "applied:explicit_holding_sentence"
    assert meta["source_chunk_ids"] == ["esteras_chunk_0"]


def test_user_upload_absence_override_blocks_unsupported_fact_inference():
    response, meta = pipeline_module._apply_user_upload_absence_response_override(
        query="In my uploaded Cedar agreement, what is the arbitration seat?",
        response="The arbitration seat is listed in Section 9.",
        generation_context_meta={
            "status": "applied:user_upload_context",
            "answerable_sentences": [
                "Section 9.2 Force Majeure.",
                "A party invoking force majeure must give written notice within fourteen calendar days after the event begins.",
                "The agreement lists natural disasters, government embargoes, labor strikes, and port closures as qualifying events.",
            ],
        },
    )

    assert response == "The uploaded document excerpt does not identify an arbitration seat."
    assert meta["status"] == "applied:user_upload_absence"
    assert meta["missing_terms"] == ["arbitration", "seat"]


def test_user_upload_absence_override_allows_supported_upload_fact():
    response, meta = pipeline_module._apply_user_upload_absence_response_override(
        query="According to my uploaded Cedar agreement, how many days does a party have to give force majeure notice?",
        response="A party has fourteen calendar days to give notice.",
        generation_context_meta={
            "status": "applied:user_upload_context",
            "answerable_sentences": [
                "A party invoking force majeure must give written notice within fourteen calendar days after the event begins.",
            ],
        },
    )

    assert response == "A party has fourteen calendar days to give notice."
    assert meta["status"] == "not_applied:upload_evidence_matches_query_terms"
    assert {"days", "force", "majeure", "notice"} & set(meta["matched_terms"])


def test_cert_denial_guard_blocks_unauthorized_merits_boilerplate():
    response, meta = pipeline_module._apply_cert_denial_safety_guard(
        response="The Court denied certiorari.",
        case_posture={"decision_type": "merits", "court_action": "affirmed"},
        generation_context_meta={
            "explicit_holding_sentence": "The challenged provisions do not violate petitioners' First Amendment rights.",
            "source_chunk_ids": ["tiktok_chunk_0"],
        },
        prompt_chunks=[],
    )

    assert response == "The challenged provisions do not violate petitioners' First Amendment rights."
    assert meta["status"] == "applied:blocked_unauthorized_cert_denial"


def test_cert_denial_guard_allows_real_cert_denial_posture():
    response, meta = pipeline_module._apply_cert_denial_safety_guard(
        response="The Supreme Court denied certiorari in Burnett v. United States.",
        case_posture={"decision_type": "cert_denial", "court_action": "denied certiorari"},
        generation_context_meta={},
        prompt_chunks=[],
    )

    assert response == "The Supreme Court denied certiorari in Burnett v. United States."
    assert meta["status"] == "not_applied:authorized_by_case_posture"


def test_query_grounding_resolves_non_what_did_named_case():
    chunks = [
        LegalChunk(
            id="burnett_chunk_0",
            doc_id="burnett_doc",
            text="The petition for a writ of certiorari is denied. JUSTICE GORSUCH, dissenting.",
            chunk_index=0,
            doc_type="case",
            case_name="Burnett v. United States",
            court_level="scotus",
        ),
        LegalChunk(
            id="other_chunk_0",
            doc_id="other_doc",
            text="Other case text.",
            chunk_index=0,
            doc_type="case",
            case_name="Other v. United States",
            court_level="scotus",
        ),
    ]

    grounding = pipeline_module._resolve_query_grounding(
        "In Burnett v. United States, who wrote the dissent from the denial of certiorari?",
        chunks,
    )
    prompt_chunks, meta = pipeline_module._select_prompt_chunks_for_generation(
        "In Burnett v. United States, who wrote the dissent from the denial of certiorari?",
        chunks,
        query_grounding=grounding,
    )

    assert grounding["status"] == "resolved:explicit_case"
    assert grounding["target_case"] == "Burnett v. United States"
    assert grounding["query_intent"] == "author"
    assert meta["status"] == "applied"
    assert [chunk.id for chunk in prompt_chunks] == ["burnett_chunk_0"]


def test_query_grounding_resolves_citation_to_case_metadata():
    chunks = [
        LegalChunk(
            id="esteras_chunk_0",
            doc_id="esteras_doc",
            text="District courts cannot consider section 3553(a)(2)(A) when revoking supervised release.",
            chunk_index=0,
            doc_type="case",
            case_name="Esteras v. United States",
            citation="606 U. S. 185 (2025)",
            court_level="scotus",
        ),
        LegalChunk(
            id="other_chunk_0",
            doc_id="other_doc",
            text="Other case text.",
            chunk_index=0,
            doc_type="case",
            case_name="Other v. United States",
            citation="605 U. S. 101 (2025)",
            court_level="scotus",
        ),
    ]

    grounding = pipeline_module._resolve_query_grounding(
        "What did 606 U.S. 185 hold about supervised release revocation factors?",
        chunks,
    )
    prompt_chunks, meta = pipeline_module._select_prompt_chunks_for_generation(
        "What did 606 U.S. 185 hold about supervised release revocation factors?",
        chunks,
        query_grounding=grounding,
    )

    assert grounding["status"] == "resolved:citation"
    assert grounding["target_case"] == "Esteras v. United States"
    assert grounding["target_citation"] == "606 U.S. 185"
    assert grounding["query_intent"] == "holding"
    assert meta["status"] == "applied"
    assert [chunk.id for chunk in prompt_chunks] == ["esteras_chunk_0"]


def test_query_grounding_resolves_short_case_name_from_retrieved_metadata():
    chunks = [
        LegalChunk(
            id="burnett_chunk_0",
            doc_id="burnett_doc",
            text="Mr. Burnett argued that supervised release could exceed the statutory maximum.",
            chunk_index=0,
            doc_type="case",
            case_name="Burnett v. United States",
            court_level="scotus",
        ),
        LegalChunk(
            id="esteras_chunk_0",
            doc_id="esteras_doc",
            text="Esteras concerns supervised release revocation factors.",
            chunk_index=0,
            doc_type="case",
            case_name="Esteras v. United States",
            court_level="scotus",
        ),
    ]

    grounding = pipeline_module._resolve_query_grounding(
        "What did Burnett say about supervised release and the statutory maximum?",
        chunks,
    )
    prompt_chunks, meta = pipeline_module._select_prompt_chunks_for_generation(
        "What did Burnett say about supervised release and the statutory maximum?",
        chunks,
        query_grounding=grounding,
    )

    assert grounding["status"] == "resolved:short_name"
    assert grounding["target_case"] == "Burnett v. United States"
    assert grounding["query_intent"] == "issue_analysis"
    assert meta["status"] == "applied"
    assert [chunk.id for chunk in prompt_chunks] == ["burnett_chunk_0"]


def test_query_grounding_resolves_issue_only_when_retrieval_converges():
    chunks = [
        LegalChunk(
            id="learning_chunk_0",
            doc_id="learning_doc",
            text="Petitioners alleged that IEEPA does not authorize tariffs.",
            chunk_index=0,
            doc_type="case",
            case_name="Learning Resources, Inc. v. Trump",
            court_level="scotus",
        ),
        LegalChunk(
            id="learning_chunk_1",
            doc_id="learning_doc",
            text="The Government relied on IEEPA to impose tariffs on imports.",
            chunk_index=1,
            doc_type="case",
            case_name="Learning Resources, Inc. v. Trump",
            court_level="scotus",
        ),
        LegalChunk(
            id="learning_revision_chunk",
            doc_id="learning_revision_doc",
            text="The opinion discusses whether IEEPA authorizes duties or tariffs.",
            chunk_index=2,
            doc_type="case",
            case_name="Learning Resources, Inc. v. Trump Revisions: 2/23/26",
            court_level="scotus",
        ),
        LegalChunk(
            id="other_chunk",
            doc_id="other_doc",
            text="Other case text.",
            chunk_index=0,
            doc_type="case",
            case_name="Other v. United States",
            court_level="scotus",
        ),
    ]

    grounding = pipeline_module._resolve_query_grounding(
        "What did the Supreme Court say about whether IEEPA lets the President impose tariffs?",
        chunks,
    )

    assert grounding["status"] == "resolved:retrieval_convergence"
    assert grounding["target_case"] == "Learning Resources, Inc. v. Trump"


def test_research_leads_mode_detects_general_case_finding_query():
    query = "Find cases about agency civil penalties and jury trial rights."
    chunks = [
        LegalChunk(
            id="agency_penalty_chunk",
            doc_id="agency_penalty_doc",
            text="The Court discussed civil penalties imposed by an agency.",
            chunk_index=0,
            doc_type="case",
            case_name="Securities and Exchange Commission v. Jarkesy",
            court_level="scotus",
        ),
        LegalChunk(
            id="jury_right_chunk",
            doc_id="jury_right_doc",
            text="The Court addressed when the Seventh Amendment preserves a jury trial.",
            chunk_index=0,
            doc_type="case",
            case_name="Tull v. United States",
            court_level="scotus",
        ),
    ]

    grounding = pipeline_module._resolve_query_grounding(query, chunks)

    assert grounding["target_case"] is None
    assert grounding["query_intent"] == "research_leads"
    assert pipeline_module._query_requests_research_leads(query, grounding) is True


def test_research_leads_mode_ignores_named_case_queries():
    query = "Find cases like Miranda v. Arizona about custodial interrogation warnings."
    chunks = [
        LegalChunk(
            id="diaz_chunk",
            doc_id="diaz_doc",
            text="The opinion referenced warnings before custodial interrogation.",
            chunk_index=0,
            doc_type="case",
            case_name="Diaz v. United States",
            court_level="scotus",
        )
    ]

    grounding = pipeline_module._resolve_query_grounding(query, chunks)

    assert grounding["explicit_case"] == "Miranda v. Arizona"
    assert pipeline_module._query_requests_research_leads(query, grounding) is False


def test_generation_context_adds_research_leads_scope_constraint():
    query = "Identify authorities on whether administrative penalties require a jury."
    chunks = [
        LegalChunk(
            id="agency_penalty_chunk",
            doc_id="agency_penalty_doc",
            text="The Court discussed civil penalties imposed by an agency.",
            chunk_index=0,
            doc_type="case",
            case_name="Securities and Exchange Commission v. Jarkesy",
            court_level="scotus",
        ),
        LegalChunk(
            id="jury_right_chunk",
            doc_id="jury_right_doc",
            text="The Court addressed when the Seventh Amendment preserves a jury trial.",
            chunk_index=0,
            doc_type="case",
            case_name="Tull v. United States",
            court_level="scotus",
        ),
    ]

    grounding = pipeline_module._resolve_query_grounding(query, chunks)
    context, meta = pipeline_module._build_generation_context(
        query,
        chunks,
        query_grounding=grounding,
        research_leads_mode=True,
    )

    assert context[0].startswith("Evidence type: research-leads scope constraint")
    assert "not definitive holdings unless directly supported" in context[0]
    assert meta["status"] == "applied:research_leads"
    assert meta["research_leads_mode"] is True
    assert meta["source_chunk_ids"] == ["agency_penalty_chunk", "jury_right_chunk"]


def test_answer_mode_reports_research_leads():
    assert (
        pipeline_module._answer_mode(
            generation_mode="rag",
            topic_fallback_used=False,
            pre_generation_refusal=False,
            retrieval_used=True,
            research_leads_mode=True,
        )
        == "research_leads"
    )


def test_followup_grounding_rewrites_separate_opinion_author_query():
    grounded_query, meta = pipeline_module._ground_followup_query(
        "Who wrote the separate opinion?",
        {
            "last_target_case": "Burnett v. United States",
            "last_target_doc_ids": ["burnett_doc"],
            "last_opinion_role": "dissent_from_denial",
            "last_opinion_author": "Gorsuch",
        },
    )

    assert grounded_query == (
        "In Burnett v. United States, who wrote the dissent from the denial of certiorari?"
    )
    assert meta["status"] == "applied:conversation_state"
    assert meta["target_case"] == "Burnett v. United States"


def test_generation_context_prefers_ieepa_conclusion_over_question_presented():
    query = "In Learning Resources, Inc. v. Trump, what did the Supreme Court say about whether IEEPA authorizes tariffs?"
    chunks = [
        LegalChunk(
            id="learning_question",
            doc_id="learning_doc",
            text=(
                "The question presented is whether the International Emergency Economic "
                "Powers Act authorizes the President to impose tariffs."
            ),
            chunk_index=0,
            doc_type="case",
            case_name="Learning Resources, Inc. v. Trump",
            court_level="scotus",
        ),
        LegalChunk(
            id="learning_answer",
            doc_id="learning_doc",
            text="Nothing in IEEPA's text, nor anything in its context, enables the President to unilaterally impose tariffs.",
            chunk_index=1,
            doc_type="case",
            case_name="Learning Resources, Inc. v. Trump",
            court_level="scotus",
        ),
    ]

    grounding = pipeline_module._resolve_query_grounding(query, chunks)
    context, meta = pipeline_module._build_generation_context(
        query,
        chunks,
        query_grounding=grounding,
    )

    assert meta["explicit_holding_sentence"] == (
        "Nothing in IEEPA's text, nor anything in its context, enables the President to unilaterally impose tariffs."
    )
    assert "question presented" not in context[0].lower()


def test_generation_context_adds_canonical_ieepa_answer_fact():
    query = "In Learning Resources, Inc. v. Trump, what did the Supreme Court say about whether IEEPA authorizes tariffs?"
    chunks = [
        LegalChunk(
            id="learning_lower_court",
            doc_id="learning_doc",
            text=(
                "The District Court granted the plaintiffs' motion for a preliminary injunction, "
                "concluding that IEEPA did not grant the President the power to impose tariffs."
            ),
            chunk_index=0,
            doc_type="case",
            case_name="Learning Resources, Inc. v. Trump",
            court_level="scotus",
        )
    ]

    grounding = pipeline_module._resolve_query_grounding(query, chunks)
    context, meta = pipeline_module._build_generation_context(
        query,
        chunks,
        query_grounding=grounding,
    )

    assert meta["canonical_answer_fact"] == "IEEPA does not authorize the President to impose tariffs."
    assert context[0].startswith("Evidence type: canonical answer fact")
    assert "Allowed answer fact: IEEPA does not authorize the President to impose tariffs." in context[0]


def test_supported_claim_repair_removes_unsupported_filler():
    repaired, meta = pipeline_module._apply_supported_claim_repair(
        (
            "The Court held that the preponderance standard applies. "
            "The Supreme Court denied certiorari on the petition for review."
        ),
        [
            {
                "text": "The Court held that the preponderance standard applies.",
                "span": {"start_char": 0},
                "annotation": {"support_level": "supported"},
            },
            {
                "text": "The Supreme Court denied certiorari on the petition for review.",
                "span": {"start_char": 55},
                "annotation": {"support_level": "unsupported"},
            },
        ],
        {"explicit_holding_sentence": "The preponderance standard applies."},
    )

    assert repaired == "The Court held that the preponderance standard applies."
    assert meta["status"] == "applied:unsupported_claim_repair"


def test_detailed_repair_uses_canonical_fact_and_supporting_evidence():
    repaired, meta = pipeline_module._apply_supported_claim_repair(
        "The answer drifted into unrelated copyright law.",
        [
            {
                "text": "The answer drifted into unrelated copyright law.",
                "annotation": {"support_level": "unsupported"},
            }
        ],
        {
            "response_depth": "detailed",
            "canonical_answer_fact": (
                "The federal fraud statutes do not require the Government to prove economic loss."
            ),
            "answerable_sentences": [
                "The common law did not establish a general rule requiring economic loss in all fraud cases.",
                "The Court would not read such a requirement into Section 1343.",
            ],
        },
    )

    assert meta["status"] == "applied:unsupported_claim_repair"
    assert repaired.startswith(
        "The federal fraud statutes do not require the Government to prove economic loss."
    )
    assert "\n\n" in repaired
    assert "common law did not establish" in repaired


def test_supported_claim_repair_preserves_missing_case_topic_disclaimer():
    repaired, meta = pipeline_module._apply_supported_claim_repair(
        (
            "I do not have Missing v. Unknown in the retrieved database, so I cannot say what that case held.\n\n"
            "Based on related retrieved authorities, adverse possession requires open possession."
        ),
        [
            {
                "text": (
                    "I do not have Missing v. Unknown in the retrieved database, so I cannot say what that case held."
                ),
                "span": {"start_char": 0},
                "annotation": {"support_level": "unsupported"},
            },
            {
                "text": "Adverse possession requires open possession.",
                "span": {"start_char": 120},
                "annotation": {"support_level": "supported"},
            },
        ],
        {
            "missing_case": "Missing v. Unknown",
            "status": "applied:missing_case_topic_fallback",
        },
    )

    assert repaired.startswith(
        "I do not have Missing v. Unknown in the retrieved database, so I cannot say what that case held."
    )
    assert "Adverse possession requires open possession." in repaired
    assert meta["status"] == "applied:unsupported_claim_repair"
    assert meta["protected_prefix_applied"] is True


def test_supported_claim_repair_drops_person_title_fragments():
    repaired, meta = pipeline_module._apply_supported_claim_repair(
        (
            "The uploaded document states that Dr.\n\n"
            "The argument about Dr.\n\n"
            "An unrelated authority discusses labor injunctions."
        ),
        [
            {
                "text": "The uploaded document states that Dr.",
                "span": {"start_char": 0},
                "annotation": {"support_level": "possibly_supported"},
            },
            {
                "text": "The argument about Dr.",
                "span": {"start_char": 38},
                "annotation": {"support_level": "possibly_supported"},
            },
            {
                "text": "An unrelated authority discusses labor injunctions.",
                "span": {"start_char": 62},
                "annotation": {"support_level": "unsupported"},
            },
            {
                "text": "The uploaded document proves the expert opinion is inadmissible.",
                "span": {"start_char": 112},
                "annotation": {"support_level": "unsupported"},
            },
        ],
        {
            "status": "applied:user_upload_context",
            "answerable_sentences": [
                "The uploaded document states that Dr. Lena Marquez did not test the battery pack.",
                "The strongest motion argument is that Dr. Marquez failed to connect her observations to a reliable causation methodology.",
            ],
            "response_depth": "concise",
        },
    )

    assert meta["status"] == "applied:unsupported_claim_repair"
    repaired_parts = [part.strip() for part in repaired.split("\n\n") if part.strip()]
    assert "The uploaded document states that Dr." not in repaired_parts
    assert "The argument about Dr." not in repaired_parts
    assert "Dr. Lena Marquez did not test the battery pack." in repaired


def test_missing_case_topic_disclaimer_is_filtered_from_verification_claims():
    raw_claims = pipeline_module.decompose_document(
        {
            "id": "assistant_response",
            "full_text": (
                "I do not have Missing v. Unknown in the retrieved database, so I cannot say what that case held. "
                "Adverse possession requires open possession."
            ),
        }
    )

    filtered, skipped = pipeline_module._filter_claims_for_verification(raw_claims)

    assert "Adverse possession requires open possession." in [claim.text for claim in filtered]
    assert any("Missing v. Unknown" in item["text"] for item in skipped)


def test_user_upload_absence_answer_is_filtered_from_verification_claims():
    raw_claims = pipeline_module.decompose_document(
        {
            "id": "assistant_response",
            "full_text": "The uploaded document excerpt does not identify an arbitration seat.",
        }
    )

    filtered, skipped = pipeline_module._filter_claims_for_verification(raw_claims)

    assert filtered == []
    assert any("arbitration seat" in item["text"] for item in skipped)


def test_fragment_claims_are_filtered_from_verification():
    raw_claims = pipeline_module.decompose_document(
        {
            "id": "assistant_response",
            "full_text": (
                "W. The D. C. v. Garland is whether the challenged divestiture provisions violate rights. "
                "The Court held that the challenged provisions do not violate the First Amendment."
            ),
        }
    )

    filtered, skipped = pipeline_module._filter_claims_for_verification(raw_claims)
    filtered_texts = [claim.text for claim in filtered]
    skipped_texts = [item["text"] for item in skipped]

    assert "The Court held that the challenged provisions do not violate the First Amendment." in filtered_texts
    assert "W." in skipped_texts
    assert "The D." not in filtered_texts
    assert all(not text.startswith("v. Garland") for text in filtered_texts)


def test_query_intent_classification_distinguishes_posture_from_issue_analysis():
    assert (
        pipeline_module._classify_query_intent(
            "In Burnett v. United States, what did the Supreme Court do in the case?",
            target_case="Burnett v. United States",
            citation=None,
        )
        == "posture"
    )
    assert (
        pipeline_module._classify_query_intent(
            "What did Burnett say about whether supervised release can push prison time past the statutory maximum without a jury?",
            target_case="Burnett v. United States",
            citation=None,
        )
        == "issue_analysis"
    )
    assert (
        pipeline_module._classify_query_intent(
            "In Burnett v. United States, who wrote the dissent from the denial of certiorari?",
            target_case="Burnett v. United States",
            citation=None,
        )
        == "author"
    )


def test_llm_topic_overview_route_prevents_retrieval_convergence_target_lock():
    query = "Give me a general overview of adverse possession."
    chunks = [
        LegalChunk(
            id="chunk_1",
            doc_id="doc_1",
            text="Example v. Owner discussed adverse possession and open possession.",
            chunk_index=0,
            doc_type="case",
            case_name="Example v. Owner",
            court_level="scotus",
        ),
        LegalChunk(
            id="chunk_2",
            doc_id="doc_1",
            text="Example v. Owner also discussed hostility and continuity.",
            chunk_index=1,
            doc_type="case",
            case_name="Example v. Owner",
            court_level="scotus",
        ),
        LegalChunk(
            id="chunk_3",
            doc_id="doc_1",
            text="Example v. Owner addressed statutory possession periods.",
            chunk_index=2,
            doc_type="case",
            case_name="Example v. Owner",
            court_level="scotus",
        ),
    ]
    route_llm = _RouteLLM("topic_overview")

    grounding = pipeline_module._resolve_query_grounding(
        query,
        chunks,
        llm_router=route_llm,
    )
    prompt_chunks, prompt_meta = pipeline_module._select_prompt_chunks_for_generation(
        query,
        chunks,
        query_grounding=grounding,
    )

    assert route_llm.queries == [query]
    assert grounding["target_case"] is None
    assert grounding["source"] == "llm_route:topic_overview"
    assert grounding["query_intent"] == "topic_overview"
    assert grounding["llm_route_meta"]["confidence"] == 0.9
    assert prompt_meta["status"] == "not_applied:no_target_case"
    assert prompt_chunks == chunks


def test_llm_research_leads_route_enables_research_leads_mode_without_regex():
    query = "I need starting points on agency penalties and jury trial rights."
    grounding = pipeline_module._resolve_query_grounding(
        query,
        [
            LegalChunk(
                id="chunk_1",
                doc_id="doc_1",
                text="The Court discussed agency penalties and jury trial rights.",
                chunk_index=0,
                doc_type="case",
                case_name="Agency Penalty Case",
                court_level="scotus",
            )
        ],
        llm_router=_RouteLLM("research_leads"),
    )

    assert grounding["target_case"] is None
    assert grounding["query_intent"] == "research_leads"
    assert pipeline_module._query_requests_research_leads(query, grounding) is True


def test_select_chunks_for_verification_prioritizes_generation_and_prompt_scope():
    source_chunk = LegalChunk(
        id="source_chunk",
        doc_id="burnett_doc",
        text="The petition for a writ of certiorari is denied.",
        chunk_index=0,
        doc_type="case",
        case_name="Burnett v. United States",
        court_level="scotus",
    )
    prompt_chunk = LegalChunk(
        id="prompt_chunk",
        doc_id="burnett_doc",
        text="Justice Gorsuch dissented from the denial of certiorari.",
        chunk_index=1,
        doc_type="case",
        case_name="Burnett v. United States",
        court_level="scotus",
    )
    unrelated_chunk = LegalChunk(
        id="unrelated_chunk",
        doc_id="other_doc",
        text="Other case background.",
        chunk_index=0,
        doc_type="case",
        case_name="Other v. United States",
        court_level="scotus",
    )

    chunks, meta = pipeline_module._select_chunks_for_verification(
        retrieved_chunks=[source_chunk, unrelated_chunk, prompt_chunk],
        prompt_chunks=[prompt_chunk],
        generation_context_meta={"source_chunk_ids": ["source_chunk"]},
        query_grounding={
            "target_case": "Burnett v. United States",
            "target_doc_ids": ["burnett_doc"],
        },
    )

    assert meta["status"] == "applied:scoped"
    assert meta["scope"] == "generation_source_chunks"
    assert meta["tiers"] == ["generation_source"]
    assert [chunk.id for chunk in chunks] == ["source_chunk"]


def test_select_chunks_for_verification_prefers_sentence_evidence():
    source_chunk = LegalChunk(
        id="source_chunk",
        doc_id="target_doc",
        text=(
            "The court of appeals held the opposite rule. "
            "Held: The Supreme Court held the rule applies."
        ),
        chunk_index=0,
        doc_type="case",
        case_name="Example v. United States",
        court_level="scotus",
        citation="600 U.S. 1",
    )
    broad_chunk = LegalChunk(
        id="broad_chunk",
        doc_id="target_doc",
        text="A party argued the opposite rule.",
        chunk_index=1,
        doc_type="case",
        case_name="Example v. United States",
        court_level="scotus",
    )

    chunks, meta = pipeline_module._select_chunks_for_verification(
        retrieved_chunks=[source_chunk, broad_chunk],
        prompt_chunks=[source_chunk, broad_chunk],
        generation_context_meta={
            "source_chunk_ids": ["source_chunk"],
            "source_evidence_sentences": [
                {
                    "sentence": "Held: The Supreme Court held the rule applies.",
                    "source_chunk_id": "source_chunk",
                    "role": "holding",
                }
            ],
        },
        query_grounding={
            "target_case": "Example v. United States",
            "target_doc_ids": ["target_doc"],
        },
    )

    assert meta == {
        "status": "applied:scoped",
        "scope": "sentence_evidence",
        "tiers": ["sentence_evidence"],
    }
    assert len(chunks) == 1
    assert chunks[0].id == "source_chunk:sentence_evidence:0"
    assert chunks[0].text == "Held: The Supreme Court held the rule applies."
    assert chunks[0].case_name == "Example v. United States"
    assert chunks[0].citation == "600 U.S. 1"
    assert getattr(chunks[0], "verification_tier") == "sentence_evidence"
    assert getattr(chunks[0], "verification_tier_rank") == 0
