"""Deterministic end-to-end demo test for document processing features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.indexing.embedder import Embedder
from src.ingestion.chunker import chunk_document
from src.ingestion.document import LegalDocument
from src.verification.claim_decomposer import decompose_document
from src.verification.nli_verifier import NLIVerifier


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "demo" / "fixtures" / "document_processing_demo.json"


class _FakeModel:
    def encode(self, items, batch_size=32, convert_to_numpy=True, show_progress_bar=False):
        _ = batch_size, convert_to_numpy, show_progress_bar
        vectors = []
        for index, _item in enumerate(items, start=1):
            vectors.append([float(index), float(index + 1), 0.5])
        return np.asarray(vectors, dtype=np.float32)


class _FakeNLIVerifier(NLIVerifier):
    def __init__(self, score_map):
        super().__init__(device="cpu")
        self.score_map = score_map

    def _predict_pairs(self, pairs):
        return [self.score_map[pair] for pair in pairs]


def test_document_processing_demo_flow_is_clear_and_repeatable():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    document = LegalDocument(
        id=fixture["document"]["id"],
        doc_type="user_upload",
        full_text=fixture["document"]["full_text"],
        source_file=fixture["document"]["source_file"],
        court_level=fixture["document"]["court_level"],
    )
    chunks = chunk_document(
        document,
        chunk_size=fixture["chunking"]["chunk_size"],
        chunk_overlap=fixture["chunking"]["chunk_overlap"],
    )

    assert len(chunks) >= 3
    assert chunks[0].id == f"{document.id}:0"
    assert all(chunk.doc_id == document.id for chunk in chunks)

    claims = decompose_document(fixture["assistant_response"])
    claim_texts = [claim.text for claim in claims]
    assert claim_texts == [
        "Officer Smith entered the residence.",
        "Officer Smith arrested Doe.",
        "The motion to suppress was granted.",
    ]

    embedder = Embedder()
    embedder._model = _FakeModel()
    chunk_vectors = embedder.encode([chunk.text for chunk in chunks], normalize=True)

    assert chunk_vectors.shape == (len(chunks), 3)
    np.testing.assert_allclose(np.linalg.norm(chunk_vectors, axis=1), np.ones(len(chunks)), rtol=1e-6)

    target_claims = fixture["claims_to_verify"]
    default_score = {"entailment": 0.05, "neutral": 0.90, "contradiction": 0.05}
    score_map = {
        (chunk.text, claim_text): dict(default_score)
        for chunk in chunks
        for claim_text in target_claims
    }
    score_map[(chunks[0].text, "Officer Smith entered the residence.")] = {
        "entailment": 0.93,
        "neutral": 0.05,
        "contradiction": 0.02,
    }
    score_map[(chunks[0].text, "Officer Smith arrested Doe.")] = {
        "entailment": 0.91,
        "neutral": 0.06,
        "contradiction": 0.03,
    }
    score_map[(chunks[1].text, "The motion to suppress was granted.")] = {
        "entailment": 0.15,
        "neutral": 0.05,
        "contradiction": 0.80,
    }

    verifier = _FakeNLIVerifier(score_map)
    verification_results = verifier.verify_claims_batch(target_claims, chunks)

    assert verification_results[0].best_chunk == chunks[0]
    assert verification_results[0].is_contradicted is False
    assert verification_results[0].final_score > 0.5

    assert verification_results[1].best_chunk == chunks[0]
    assert verification_results[1].is_contradicted is False
    assert verification_results[1].final_score > 0.5

    assert verification_results[2].best_chunk == chunks[1]
    assert verification_results[2].is_contradicted is True
    assert verification_results[2].final_score < 0.2
