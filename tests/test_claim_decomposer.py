"""Tests for claim decomposition."""

import pytest

from src.verification.claim_decomposer import (
    decompose_document,
    load_document,
    split_clauses,
)


pytestmark = pytest.mark.smoke


def test_golden_case_conjoined_verbs_become_atomic_claims():
    text = "The officer entered the residence and arrested Doe."
    claims = decompose_document(text)
    claim_texts = {claim.text for claim in claims}

    assert "The officer entered the residence." in claim_texts
    assert "The officer arrested Doe." in claim_texts


def test_golden_case_attribution_emits_attribution_and_embedded_claim():
    text = "Plaintiff alleges that Officer Smith entered the home without a warrant."
    claims = decompose_document(text)
    claim_texts = {claim.text for claim in claims}
    claim_types = {claim.claim_type for claim in claims}

    assert "Plaintiff alleges that Officer Smith entered the home without a warrant." in claim_texts
    assert "Officer Smith entered the home without a warrant." in claim_texts
    assert "attribution" in claim_types


def test_split_clauses_conjoined_verbs():
    clauses = split_clauses("The officer entered the residence and arrested Doe.")
    assert clauses == [
        "The officer entered the residence.",
        "The officer arrested Doe.",
    ]


def test_split_clauses_conjoined_verbs_with_multiword_subject():
    clauses = split_clauses("The United States entered the agreement and accepted the terms.")
    assert clauses == [
        "The United States entered the agreement.",
        "The United States accepted the terms.",
    ]


def test_split_clauses_causal_claim_into_atomic_parts():
    clauses = split_clauses(
        "This defect was not cured because the removal petition did not include a statement of jurisdiction as required by 28 U.S.C. Â§ 1441(b)."
    )

    assert clauses[0] == "This defect was not cured."
    assert clauses[1].startswith("The removal petition did not include a statement of jurisdiction")


def test_decompose_document_splits_causal_claim_for_verification():
    claims = decompose_document(
        "This defect was not cured because the removal petition did not include a statement of jurisdiction as required by 28 U.S.C. Â§ 1441(b)."
    )
    claim_texts = [claim.text for claim in claims]

    assert "This defect was not cured." in claim_texts
    assert any(
        text.startswith("The removal petition did not include a statement of jurisdiction")
        for text in claim_texts
    )
    assert all("because the removal petition" not in text for text in claim_texts)


def test_decompose_document_preserves_common_legal_abbreviations():
    text = (
        "Klein v. Martin says 28 U.S.C. 2254(d) governs habeas review. "
        "Harrington v. Richter frames AEDPA review as guarding against extreme malfunctions."
    )

    claims = decompose_document(text)
    claim_texts = [claim.text for claim in claims]

    assert "Klein v. Martin says 28 U.S.C. 2254(d) governs habeas review." in claim_texts
    assert "Harrington v. Richter frames AEDPA review as guarding against extreme malfunctions." in claim_texts
    assert "S." not in claim_texts
    assert "C." not in claim_texts


def test_decompose_document_preserves_person_title_abbreviations():
    text = (
        "The uploaded document states that Dr. Lena Marquez did not test the battery pack. "
        "The argument concerns causation methodology."
    )

    claims = decompose_document(text)
    claim_texts = [claim.text for claim in claims]

    assert "The uploaded document states that Dr. Lena Marquez did not test the battery pack." in claim_texts
    assert "The argument concerns causation methodology." in claim_texts
    assert "The uploaded document states that Dr." not in claim_texts


def test_decompose_document_preserves_entity_and_court_abbreviations():
    text = (
        "Speech First, Inc. v. Sands requires Speech First to show member injury. "
        "The D.C. Circuit addressed causation."
    )

    claims = decompose_document(text)
    claim_texts = [claim.text for claim in claims]

    assert "Speech First, Inc. v. Sands requires Speech First to show member injury." in claim_texts
    assert "The D.C. Circuit addressed causation." in claim_texts
    assert "Speech First, Inc." not in claim_texts
    assert "v. Sands requires Speech First to show member injury." not in claim_texts
    assert "The D." not in claim_texts
    assert "C." not in claim_texts


def test_decompose_document_preserves_party_initial_abbreviations():
    text = (
        "The officer had reasonable suspicion to stop R. W., because fleeing from a traffic stop "
        "can indicate danger."
    )

    claims = decompose_document(text)
    claim_texts = [claim.text for claim in claims]

    assert "The officer had reasonable suspicion to stop R. W." in claim_texts
    assert "The officer had reasonable suspicion to stop R." not in claim_texts


def test_decompose_document_preserves_dangling_fragment_for_pipeline_skip_metadata():
    claims = decompose_document("Nor does it change just.")

    assert [claim.text for claim in claims] == ["Nor does it change just."]


def test_decompose_document_preserves_unmatched_quote_fragment_for_pipeline_skip_metadata():
    claims = decompose_document("This \"did not include agriculture, manufacturing, mining, malum in se crime, or land use.")

    assert [claim.text for claim in claims] == [
        "This \"did not include agriculture, manufacturing, mining, malum in se crime, or land use."
    ]


def test_decompose_document_preserves_trailing_comma_fragment_for_pipeline_skip_metadata():
    text = (
        "The majority holds that obtaining a preliminary injunction never entitles a plaintiff "
        "to fees under Section 1988(b),."
    )

    claims = decompose_document(text)

    assert [claim.text for claim in claims] == [
        "The majority holds that obtaining a preliminary injunction never entitles a plaintiff to fees under Section 1988(b),."
    ]


def test_decompose_document_omits_markdown_and_context_prefixes_from_claims():
    text = (
        "**Short Answer** Under AEDPA, habeas review is limited. [2]\n\n"
        "**Analysis:**\n\n[2] The court must defer to reasonable state-court decisions.\n\n"
        "**Limits:**\n\n* The context does not address every Brady scenario."
    )

    claims = decompose_document(text)
    claim_texts = [claim.text for claim in claims]

    assert "Under AEDPA, habeas review is limited." in claim_texts
    assert "The court must defer to reasonable state-court decisions." in claim_texts
    assert "The context does not address every Brady scenario." in claim_texts
    assert all(not text.startswith(("[2]", "*", "**")) for text in claim_texts)
    assert all("[2]" not in text for text in claim_texts)


def test_decompose_document_strips_label_lead_in_before_claim():
    text = (
        "The holding in Esteras v. United States is:\n"
        "District courts cannot consider Â§ 3553(a)(2)(A) when revoking supervised release."
    )

    claims = decompose_document(text)
    claim_texts = [claim.text for claim in claims]

    assert "District courts cannot consider Â§ 3553(a)(2)(A) when revoking supervised release." in claim_texts
    assert all("The holding in Esteras v. United States is" not in text for text in claim_texts)


def test_decompose_document_does_not_strip_substantive_rule_sentence():
    text = "The rule prohibiting consideration of Â§ 3553(a)(2)(A) is unworkable if applied literally."

    claims = decompose_document(text)
    claim_texts = [claim.text for claim in claims]

    assert "The rule prohibiting consideration of Â§ 3553(a)(2)(A) is unworkable if applied literally." in claim_texts


def test_corpus_builder_record_format_is_accepted():
    # Mirrors fields saved by CorpusBuilder._append_to_jsonl.
    record = {
        "id": "cl_123",
        "doc_type": "case",
        "case_name": "Example v. Example",
        "citation": "123 U.S. 456",
        "court": "scotus",
        "court_level": "scotus",
        "date_decided": "2026-01-26",
        "full_text": "The court held that relief was improper.",
    }

    claims = decompose_document(record)

    assert claims, "Expected at least one claim from corpus-builder style record."
    assert all(claim.span.doc_id == "cl_123" for claim in claims)


def test_load_document_reads_jsonl_first_record(tmp_path):
    path = tmp_path / "doc.jsonl"
    path.write_text(
        '{"id":"cl_1","full_text":"One sentence."}\n'
        '{"id":"cl_2","full_text":"Second sentence."}\n',
        encoding="utf-8",
    )

    loaded = load_document(str(path))
    assert loaded["id"] == "cl_1"

