import pytest
from src.generation.ollama_backend import OllamaBackend

ollama = pytest.importorskip("ollama")

# Change this to whatever small model you have pulled locally (e.g., "llama3", "mistral", "phi3")
TEST_MODEL = "llama3"

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
    except ollama.ResponseError:
        pytest.skip(f"Model '{TEST_MODEL}' not found in local Ollama instance.")

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
        assert "5%" in response
        
        # Based on your prompt rules, it should ideally cite [1] or [2]
        assert "[" in response and "]" in response
    except ollama.ResponseError:
        pytest.skip(f"Model '{TEST_MODEL}' not found in local Ollama instance.")
