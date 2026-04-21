from __future__ import annotations

from src import pipeline as pipeline_module
from src.ingestion.document import LegalChunk


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
    assert [chunk.id for chunk in chunks] == ["source_chunk"]
