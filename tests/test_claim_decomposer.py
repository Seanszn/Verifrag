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
        "District courts cannot consider § 3553(a)(2)(A) when revoking supervised release."
    )

    claims = decompose_document(text)
    claim_texts = [claim.text for claim in claims]

    assert "District courts cannot consider § 3553(a)(2)(A) when revoking supervised release." in claim_texts
    assert all("The holding in Esteras v. United States is" not in text for text in claim_texts)


def test_decompose_document_does_not_strip_substantive_rule_sentence():
    text = "The rule prohibiting consideration of § 3553(a)(2)(A) is unworkable if applied literally."

    claims = decompose_document(text)
    claim_texts = [claim.text for claim in claims]

    assert "The rule prohibiting consideration of § 3553(a)(2)(A) is unworkable if applied literally." in claim_texts


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
