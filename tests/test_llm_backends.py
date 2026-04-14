import json
from pathlib import Path

import pytest
import requests

from src.config import LLM
from src.generation.ollama_backend import OllamaBackend
import ollama

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_SCOTUS_CORPUS = PROJECT_ROOT / "data" / "raw" / "scotus_cases.jsonl"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "test_reports"
TEST_MODEL = LLM.model


def _available_model_names():
    """Return installed Ollama model names, or None when the daemon is unavailable."""
    try:
        response = ollama.list()
    except Exception:
        return None

    names = set()
    for model in response.models:
        name = getattr(model, "model", None) or getattr(model, "name", None)
        if name:
            names.add(str(name))
    return names


AVAILABLE_MODELS = _available_model_names()


def _load_case_record(case_name: str) -> dict:
    if not RAW_SCOTUS_CORPUS.exists():
        pytest.skip(f"Downloaded SCOTUS corpus not found: {RAW_SCOTUS_CORPUS}")

    with RAW_SCOTUS_CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("case_name") == case_name:
                return record

    pytest.skip(f"{case_name} was not found in downloaded SCOTUS corpus.")


def _snippet_around(text: str, marker: str, *, radius: int = 900) -> str:
    marker_index = text.find(marker)
    if marker_index == -1:
        return text[: radius * 2].strip()

    start = max(0, marker_index - radius)
    end = min(len(text), marker_index + len(marker) + radius)
    return text[start:end].strip()


def _write_report(filename: str, payload: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / filename
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.skipif(AVAILABLE_MODELS is None, reason="Ollama daemon is not running.")
@pytest.mark.skipif(
    AVAILABLE_MODELS is not None and TEST_MODEL not in AVAILABLE_MODELS,
    reason=f"Configured Ollama model '{TEST_MODEL}' is not installed.",
)
def test_ollama_generation():
    """Test basic direct generation."""
    backend = OllamaBackend(model_name=TEST_MODEL)
    
    try:
        response = backend.generate("Respond with exactly one word: Hello.", max_tokens=10)
        assert isinstance(response, str)
        assert len(response) > 0
    except ollama.ResponseError:
        pytest.skip(f"Model '{TEST_MODEL}' not found in local Ollama instance.")


def test_ollama_backend_includes_configured_generation_options(monkeypatch):
    """Ensure optional Ollama generation limits are sent when configured."""

    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "ok"}

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)

    backend = OllamaBackend(
        model_name="llama3.1:8b",
        host="http://127.0.0.1:11434",
        timeout=45,
        num_ctx=512,
        num_batch=16,
        num_gpu=1,
    )

    response = backend.generate("Test prompt.", max_tokens=64)

    assert response == "ok"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["timeout"] == 45
    assert captured["payload"]["options"]["num_predict"] == 64
    assert captured["payload"]["options"]["num_ctx"] == 512
    assert captured["payload"]["options"]["num_batch"] == 16
    assert captured["payload"]["options"]["num_gpu"] == 1


@pytest.mark.skipif(AVAILABLE_MODELS is None, reason="Ollama daemon is not running.")
@pytest.mark.skipif(
    AVAILABLE_MODELS is not None and TEST_MODEL not in AVAILABLE_MODELS,
    reason=f"Configured Ollama model '{TEST_MODEL}' is not installed.",
)
def test_ollama_legal_question_records_downloaded_document_io():
    """Exercise live Ollama with a question grounded in the downloaded SCOTUS corpus."""
    record = _load_case_record("Klein v. Martin")
    full_text = record["full_text"]
    query = (
        "Under the Supreme Court's decision in Klein v. Martin, how should a "
        "federal habeas court review a state court's Brady materiality ruling "
        "when AEDPA applies?"
    )
    context = [
        (
            f"{record['case_name']} ({record.get('date_decided')}, "
            f"{record.get('court_level')}): "
            + _snippet_around(full_text, "Under the Antiterrorism and Effective Death Penalty Act")
        ),
        _snippet_around(full_text, "As we have noted many times, AEDPA sharply limits"),
    ]

    backend = OllamaBackend(model_name=TEST_MODEL)

    try:
        response = backend.generate_with_context(query, context, max_tokens=450)
    except requests.RequestException as exc:
        pytest.skip(f"Ollama request failed: {exc}")
    except ollama.ResponseError:
        pytest.skip(f"Model '{TEST_MODEL}' not found in local Ollama instance.")

    report_path = _write_report(
        "ollama_legal_question_trace.json",
        {
            "model": TEST_MODEL,
            "ollama_host": LLM.host,
            "corpus_path": str(RAW_SCOTUS_CORPUS),
            "document": {
                "id": record.get("id"),
                "case_name": record.get("case_name"),
                "citation": record.get("citation"),
                "court": record.get("court"),
                "court_level": record.get("court_level"),
                "date_decided": record.get("date_decided"),
            },
            "input": {
                "query": query,
                "context": context,
            },
            "output": {
                "response": response,
            },
        },
    )

    assert report_path.exists()
    assert response
    assert any(term in response.lower() for term in ("aedpa", "brady", "habeas", "materiality"))


@pytest.mark.skipif(AVAILABLE_MODELS is None, reason="Ollama daemon is not running.")
@pytest.mark.skipif(
    AVAILABLE_MODELS is not None and TEST_MODEL not in AVAILABLE_MODELS,
    reason=f"Configured Ollama model '{TEST_MODEL}' is not installed.",
)
def test_ollama_rag_context():
    """Test that the LLM uses the provided context to answer."""
    backend = OllamaBackend(model_name=TEST_MODEL)
    
    context = [
        "The Supreme Court of Verifrag ruled in 2026 that apples are legally classified as widgets.",
        "A widget is subject to a 5% tariff."
    ]
    query = "According to the context, what is the tariff on apples?"
    
    try:
        response = backend.generate_with_context(query, context)
        
        # The model should mention 5% based on the provided context
        response_lower = response.lower()
        assert "5%" in response or "5 percent" in response_lower
        
        # Current prompt rules require plain text without bracketed context citations.
        assert "[" not in response and "]" not in response
    except ollama.ResponseError:
        pytest.skip(f"Model '{TEST_MODEL}' not found in local Ollama instance.")