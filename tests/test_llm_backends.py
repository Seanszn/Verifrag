import pytest
import requests
from src.config import LLM
from src.generation.ollama_backend import OllamaBackend, OllamaBackendError

ollama = pytest.importorskip("ollama")

TEST_MODEL = LLM.model


def skip_if_local_ollama_unavailable(exc: Exception) -> None:
    message = str(exc).lower()
    unavailable_markers = (
        "not found",
        "unable to allocate",
        "runner process has terminated",
        "model requires more system memory",
    )
    if any(marker in message for marker in unavailable_markers):
        pytest.skip(f"Local Ollama model '{TEST_MODEL}' is unavailable: {exc}")


def is_ollama_running():
    """Helper to check if Ollama daemon is active and model is present."""
    try:
        ollama.list()
        return True
    except Exception:
        return False

@pytest.mark.skipif(not is_ollama_running(), reason="Ollama daemon is not running.")
def test_ollama_generation():
    """Test basic direct generation."""
    backend = OllamaBackend(model_name=TEST_MODEL)
    
    try:
        response = backend.generate("Respond with exactly one word: Hello.", max_tokens=10)
        assert isinstance(response, str)
        assert len(response) > 0
    except (ollama.ResponseError, OllamaBackendError) as exc:
        skip_if_local_ollama_unavailable(exc)
        raise


def test_ollama_backend_includes_configured_generation_options(monkeypatch):
    """Ensure optional Ollama runtime limits are forwarded to the API."""

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

@pytest.mark.skipif(not is_ollama_running(), reason="Ollama daemon is not running.")
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
        
        # Current prompt rules require plain text rather than bracket citations.
        assert "[" not in response and "]" not in response
    except (ollama.ResponseError, OllamaBackendError) as exc:
        skip_if_local_ollama_unavailable(exc)
        raise
