"""
LegalVerifiRAG configuration.

Server-side settings own model, storage, and auth configuration.
Client-side settings only need the API base URL.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False


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
    LOCAL = "local"


class LLMProvider(Enum):
    OLLAMA = "ollama"


DEPLOYMENT_MODE = DeploymentMode(os.getenv("DEPLOYMENT_MODE", "local"))

# ============== LLM CONFIGURATION ==============


@dataclass
class LLMConfig:
    """Configuration for the Ollama backend."""

    provider: LLMProvider = field(
        default_factory=lambda: LLMProvider(os.getenv("LLM_PROVIDER", "ollama"))
    )
    model: str = os.getenv("LLM_MODEL", "llama3.1:8b")
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class APIConfig:
    """HTTP API configuration."""

    host: str = os.getenv("API_HOST", "127.0.0.1")
    port: int = int(os.getenv("API_PORT", "8000"))
    token_ttl_hours: int = int(os.getenv("AUTH_TOKEN_TTL_HOURS", "24"))
    client_api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


@dataclass
class DatabaseConfig:
    """SQLite storage for users and conversation history."""

    path: Path = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "legalverifirag.db")))

# ============== VECTOR STORE CONFIGURATION ==============


@dataclass
class VectorStoreConfig:
    """Configuration for the local ChromaDB store."""

    chroma_path: Path = Path(os.getenv("CHROMA_PATH", str(INDEX_DIR / "chroma")))
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "legal_chunks")
    chroma_distance: str = os.getenv("CHROMA_DISTANCE", "cosine")
    chroma_batch_size: int = int(os.getenv("CHROMA_BATCH_SIZE", "100"))

# ============== MODELS ==============


@dataclass
class ModelConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    nli_model: str = "microsoft/deberta-v3-base-mnli-fever-anli"
    nli_labels: List[str] = field(default_factory=lambda: ["contradiction", "neutral", "entailment"])
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ============== RETRIEVAL ==============


@dataclass
class RetrievalConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    dense_k: int = 20
    sparse_k: int = 20
    rerank_k: int = 10
    rrf_k: int = 60

# ============== VERIFICATION ==============


@dataclass
class VerificationConfig:
    agg_alpha: float = 0.35
    agg_beta: float = 0.20
    agg_gamma: float = 0.30
    agg_delta: float = 0.25
    support_threshold: float = 0.5
    contradiction_threshold: float = 0.6
    authority_weights: Dict[str, float] = field(default_factory=lambda: {
        "scotus": 1.0,
        "circuit": 0.85,
        "district": 0.70,
        "state_supreme": 0.60,
        "state_appellate": 0.50,
        "state_trial": 0.35,
        "unknown": 0.40,
    })
    threshold_verified: float = 0.92
    threshold_supported: float = 0.82
    threshold_weak: float = 0.50
    threshold_contradicted: float = 0.60
    fuzzy_match_threshold: float = 85.0
    similarity_threshold: float = 0.6
    max_clusters: int = 5

# ============== DATA SOURCES ==============


@dataclass
class DataConfig:
    courtlistener_base_url: str = "https://www.courtlistener.com/api/rest/v4/"
    courtlistener_token: Optional[str] = os.getenv("COURTLISTENER_TOKEN")
    courtlistener_rate_limit: float = 0.72
    court_level_map: Dict[str, str] = field(default_factory=lambda: {
        "scotus": "scotus",
        "ca1": "circuit", "ca2": "circuit", "ca3": "circuit",
        "ca4": "circuit", "ca5": "circuit", "ca6": "circuit",
        "ca7": "circuit", "ca8": "circuit", "ca9": "circuit",
        "ca10": "circuit", "ca11": "circuit",
        "cadc": "circuit", "cafc": "circuit",
    })
    target_scotus_cases: int = 1200
    target_circuit_cases: int = 5000
    target_district_cases: int = 3000
    target_state_cases: int = 2000
    target_statute_sections: int = 5000
    circuits: List[str] = field(default_factory=lambda: ["ca1", "ca2", "ca3", "ca9", "cadc"])
    states: List[str] = field(default_factory=lambda: ["CA", "NY", "TX"])
    usc_titles: List[int] = field(default_factory=lambda: [18, 28, 42])

# ============== COST TRACKING ==============

LLM_PRICING = {
    "llama3.1:8b": {"input": 0.0, "output": 0.0},
}

# ============== INSTANCES ==============

LLM = LLMConfig()
API = APIConfig()
DATABASE = DatabaseConfig()
VECTOR_STORE = VectorStoreConfig()
MODELS = ModelConfig()
RETRIEVAL = RetrievalConfig()
VERIFICATION = VerificationConfig()
DATA = DataConfig()
