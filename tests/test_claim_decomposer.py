"""Tests for claim decomposition."""

from src.verification.claim_decomposer import (
    decompose_document,
    load_document,
    split_clauses,
)


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
