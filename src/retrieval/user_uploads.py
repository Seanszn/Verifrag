"""Helpers for building and loading user-scoped upload retrieval overlays."""

from __future__ import annotations

from pathlib import Path

from src.config import DATA_DIR
from src.indexing.bm25_index import BM25Index
from src.indexing.chroma_store import ChromaStore
from src.indexing.embedder import Embedder
from src.indexing.index_builder import BuildArtifacts, build_indices
from src.indexing.index_discovery import discover_index_artifacts
from src.retrieval.hybrid_retriever import HybridRetriever


USER_UPLOADS_ROOT = DATA_DIR / "uploads"


def user_upload_root(user_id: int, *, uploads_root: str | Path = USER_UPLOADS_ROOT) -> Path:
    return Path(uploads_root) / f"user_{int(user_id)}"


def build_user_upload_indices(
    user_id: int,
    *,
    uploads_root: str | Path = USER_UPLOADS_ROOT,
    embedder: Embedder | None = None,
) -> BuildArtifacts:
    root = user_upload_root(user_id, uploads_root=uploads_root)
    processed_dir = root / "processed"
    index_dir = root / "index"
    return build_indices(
        processed_dir=processed_dir,
        output_dir=index_dir,
        collection_name=f"user_{int(user_id)}_uploads",
        embedder=embedder,
    )


def load_user_upload_retriever(
    user_id: int,
    *,
    uploads_root: str | Path = USER_UPLOADS_ROOT,
    shared_embedder: Embedder | None = None,
) -> tuple[HybridRetriever | None, str]:
    index_dir = user_upload_root(user_id, uploads_root=uploads_root) / "index"
    artifacts = discover_index_artifacts(index_dir=index_dir)
    bm25_index = _load_bm25_index(artifacts.bm25_path)
    vector_store, embedder = _load_vector_store(
        artifacts.chroma_path,
        collection_name=artifacts.collection_name,
        shared_embedder=shared_embedder,
    )

    if bm25_index is None and vector_store is None:
        return None, "unavailable:no_user_upload_index"

    try:
        retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            embedder=embedder,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"error:{exc.__class__.__name__}"

    return retriever, "ok"


def _load_bm25_index(index_path: Path) -> BM25Index | None:
    if not index_path.exists():
        return None

    bm25_index = BM25Index(index_path=index_path)
    bm25_index.load()
    return bm25_index


def _load_vector_store(
    chroma_path: Path,
    *,
    collection_name: str | None = None,
    shared_embedder: Embedder | None = None,
) -> tuple[ChromaStore | None, Embedder | None]:
    if not chroma_path.exists():
        return None, None

    try:
        has_entries = any(chroma_path.iterdir())
    except OSError:
        has_entries = False

    if not has_entries:
        return None, None

    vector_store = ChromaStore(path=chroma_path, collection_name=collection_name)
    return vector_store, shared_embedder or Embedder()
