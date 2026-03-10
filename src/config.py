"""
LegalVerifiRAG Configuration

All configurable parameters in one place.
Supports local-first deployment.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# ============== PATHS ==============

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"

# ============== DEPLOYMENT MODES ==============

class DeploymentMode(Enum):
    LOCAL = "local"           # Ollama + ChromaDB

# Current deployment mode (from environment)
DEPLOYMENT_MODE = DeploymentMode.LOCAL

# ============== LLM CONFIGURATION ==============

@dataclass
class LLMConfig:
    """Configuration for the Ollama backend."""

    model: str = os.getenv("LLM_MODEL", "llama3.1:8b")
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Generation parameters
    temperature: float = 0.1
    max_tokens: int = 2048

# ============== VECTOR STORE CONFIGURATION ==============

@dataclass
class VectorStoreConfig:
    """Configuration for the local ChromaDB store."""

    # Local paths
    chroma_path: Path = Path(os.getenv("CHROMA_PATH", str(INDEX_DIR / "chroma")))

# ============== MODELS ==============

@dataclass
class ModelConfig:
    # Embeddings (always local - no reason to pay for cloud embeddings)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # NLI (always local - critical for verification)
    nli_model: str = "microsoft/deberta-v3-base-mnli-fever-anli"
    nli_labels: List[str] = field(default_factory=lambda: ["contradiction", "neutral", "entailment"])

    # Cross-encoder for reranking (always local)
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ============== RETRIEVAL ==============

@dataclass
class RetrievalConfig:
    # Chunking
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 64

    # Search
    dense_k: int = 20      # Vector search candidates
    sparse_k: int = 20     # BM25 candidates
    rerank_k: int = 10     # Final top-k after reranking

    # RRF fusion
    rrf_k: int = 60        # RRF constant

# ============== VERIFICATION ==============

@dataclass
class VerificationConfig:
    # Algorithm 1: Evidence Aggregation
    agg_alpha: float = 0.35      # Max-pool weight
    agg_beta: float = 0.20       # Support ratio weight
    agg_gamma: float = 0.30      # Authority-weighted entailment weight
    agg_delta: float = 0.25      # Contradiction penalty weight
    support_threshold: float = 0.5
    contradiction_threshold: float = 0.6

    # Court authority weights
    authority_weights: Dict[str, float] = field(default_factory=lambda: {
        "scotus": 1.0,
        "circuit": 0.85,
        "district": 0.70,
        "state_supreme": 0.60,
        "state_appellate": 0.50,
        "state_trial": 0.35,
        "unknown": 0.40,
    })

    # Algorithm 4: Verdict thresholds
    threshold_verified: float = 0.92
    threshold_supported: float = 0.82
    threshold_weak: float = 0.50
    threshold_contradicted: float = 0.60

    # Algorithm 5: Citation verification
    fuzzy_match_threshold: float = 85.0

    # Algorithm 3: Contradiction detection
    similarity_threshold: float = 0.6
    max_clusters: int = 5

# ============== DATA SOURCES ==============

@dataclass
class DataConfig:
    # CourtListener API v4
    courtlistener_base_url: str = "https://www.courtlistener.com/api/rest/v4/"
    courtlistener_token: Optional[str] = os.getenv("COURTLISTENER_TOKEN")
    courtlistener_rate_limit: float = 0.72  # seconds between requests (5000/hr)

    # Court ID → court_level mapping
    court_level_map: Dict[str, str] = field(default_factory=lambda: {
        "scotus": "scotus",
        "ca1": "circuit", "ca2": "circuit", "ca3": "circuit",
        "ca4": "circuit", "ca5": "circuit", "ca6": "circuit",
        "ca7": "circuit", "ca8": "circuit", "ca9": "circuit",
        "ca10": "circuit", "ca11": "circuit",
        "cadc": "circuit", "cafc": "circuit",
    })

    # Target corpus size
    target_scotus_cases: int = 1200
    target_circuit_cases: int = 5000
    target_district_cases: int = 3000
    target_state_cases: int = 2000
    target_statute_sections: int = 5000

    # Circuits to include
    circuits: List[str] = field(default_factory=lambda: ["ca1", "ca2", "ca3", "ca9", "cadc"])

    # States to include
    states: List[str] = field(default_factory=lambda: ["CA", "NY", "TX"])

    # USC titles to include
    usc_titles: List[int] = field(default_factory=lambda: [18, 28, 42])

# ============== COST TRACKING ==============

# Pricing per 1M tokens
LLM_PRICING = {
    "llama3.1:8b": {"input": 0.0, "output": 0.0},  # Local = free
}

# ============== INSTANCES ==============

LLM = LLMConfig()
VECTOR_STORE = VectorStoreConfig()
MODELS = ModelConfig()
RETRIEVAL = RetrievalConfig()
VERIFICATION = VerificationConfig()
DATA = DataConfig()
