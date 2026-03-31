import ollama
from typing import List, Optional
from src.generation.base_llm import BaseLLM
from src.generation.prompts import build_rag_legal_prompt

class OllamaBackend(BaseLLM):
    """Ollama local backend implementation."""

    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Standard generation for general queries."""
        options = {"num_predict": max_tokens} if max_tokens else {}
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options=options
        )
        return response['response']

    def generate_with_context(self, query: str, context: List[str], max_tokens: Optional[int] = None) -> str:
        """Retrieval-grounded generation using your legal RAG prompts."""
        # Use the builder from your prompts.py
        full_prompt = build_rag_legal_prompt(query, context)
        return self.generate(full_prompt, max_tokens=max_tokens)