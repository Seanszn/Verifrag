from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.bm25_index import BM25Index
from src.indexing.chroma_store import ChromaStore
from src.indexing.embedder import Embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.verification.claim_decomposer import decompose_document
from src.verification.nli_verifier import NLIVerifier
from src.config import MODELS


DEFAULT_QUERY = (
    "What did Justice Gorsuch say in Burnett v. United States about supervised release "
    "and the Sixth Amendment?"
)
DEFAULT_ANALYSIS = (
    "Justice Gorsuch dissented from the denial of certiorari in the Burnett case. "
    "He argued that the Sixth Amendment required a jury to find contested facts before "
    "imprisonment could exceed the statutory maximum."
)
FALLBACK_NLI_MODELS = [
    MODELS.nli_model,
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a retrieval plus NLI verification demo over the dedicated nli_100 indices.",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="User query used for retrieval.")
    parser.add_argument(
        "--analysis",
        default=DEFAULT_ANALYSIS,
        help="Analyst statement to decompose into claims and verify against retrieved evidence.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of retrieved chunks to pass into NLI verification.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "index" / "nli_100_chroma",
        help="Path to the dedicated Chroma index for the 100-document batch.",
    )
    parser.add_argument(
        "--collection-name",
        default="nli_100_chunks",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--bm25-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "index" / "nli_100_bm25.pkl",
        help="Path to the dedicated BM25 index for the 100-document batch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "demo" / "output",
        help="Directory where JSON and Markdown reports are written.",
    )
    parser.add_argument(
        "--nli-model",
        default=None,
        help="Optional explicit NLI model override. If omitted, the script tries the configured model and then a public fallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embedder = Embedder()
    vector_store = ChromaStore(path=args.chroma_path, collection_name=args.collection_name)
    bm25 = BM25Index(index_path=args.bm25_path)
    bm25.load()

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25,
        embedder=embedder,
    )
    retrieved_chunks = retriever.retrieve(args.query, k=args.top_k)

    claims = [claim.to_dict() for claim in decompose_document(args.analysis)]
    claim_texts = [claim["text"] for claim in claims]

    verification_results, model_used, model_attempts = run_verification_with_fallbacks(
        claim_texts,
        retrieved_chunks,
        explicit_model=args.nli_model,
    )

    payload = {
        "query": args.query,
        "analysis": args.analysis,
        "top_k": args.top_k,
        "index_paths": {
            "chroma_path": str(args.chroma_path),
            "collection_name": args.collection_name,
            "bm25_path": str(args.bm25_path),
        },
        "nli_runtime": {
            "model_used": model_used,
            "model_attempts": model_attempts,
        },
        "retrieved_chunks": [chunk.to_dict() for chunk in retrieved_chunks],
        "claims": claims,
        "verification_results": normalize_for_json(verification_results),
        "architecture": {
            "pipeline": [
                "Load dedicated nli_100 dense and sparse indices.",
                "Use HybridRetriever to combine dense Chroma search with sparse BM25 search.",
                "Decompose the analyst statement into atomic claims.",
                "Run NLIVerifier over every claim against the shared retrieved evidence set.",
                "Aggregate authority-weighted support and contradiction signals into final claim scores.",
            ],
            "design_decisions": [
                "Use the dedicated nli_100 indices so this demo is isolated from the default project corpus.",
                "Use hybrid retrieval because BM25 and dense embeddings complement each other on legal text.",
                "Verify claims from an analyst statement instead of the query because the verifier expects hypotheses, not search intents.",
                "Use a public fallback NLI model if the configured repository identifier is unavailable, so the demo still exercises the real NLI pipeline.",
                "Keep the output in both JSON and Markdown so it is machine-readable and presentation-ready.",
            ],
        },
        "commands": {
            "example": (
                "python scripts/run_retrieval_nli_demo.py "
                "--top-k 2"
            ),
        },
    }

    json_path = args.output_dir / "retrieval_nli_demo_output.json"
    md_path = args.output_dir / "retrieval_nli_demo_output.md"
    json_path.write_text(json.dumps(normalize_for_json(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(build_markdown_report(payload), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def run_verification_with_fallbacks(
    claim_texts: list[str],
    retrieved_chunks,
    *,
    explicit_model: str | None = None,
):
    attempts: list[dict[str, str]] = []
    model_candidates = [explicit_model] if explicit_model else FALLBACK_NLI_MODELS

    last_error: Exception | None = None
    for model_name in model_candidates:
        if not model_name:
            continue
        try:
            verifier = NLIVerifier(model_name=model_name)
            results = verifier.verify_claims_batch(claim_texts, retrieved_chunks)
            attempts.append({"model": model_name, "status": "ok"})
            return results, model_name, attempts
        except Exception as exc:
            last_error = exc
            attempts.append(
                {
                    "model": model_name,
                    "status": f"error:{exc.__class__.__name__}",
                }
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("No NLI model candidates were available.")


def normalize_for_json(value: Any) -> Any:
    if is_dataclass(value):
        return normalize_for_json(asdict(value))
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_for_json(item) for key, item in value.items()}
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return normalize_for_json(vars(value))
    return value


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Retrieval and NLI Demo Output",
        "",
        "## Input",
        "",
        f"- Query: `{payload['query']}`",
        f"- Analysis statement: `{payload['analysis']}`",
        f"- Top-k evidence chunks: `{payload['top_k']}`",
        "",
        "## Architecture",
        "",
    ]

    for item in payload["architecture"]["pipeline"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Why It Works",
            "",
        ]
    )
    for item in payload["architecture"]["design_decisions"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Runtime Model Selection",
            "",
            f"- Model used: `{payload['nli_runtime']['model_used']}`",
            f"- Model attempts: `{json.dumps(payload['nli_runtime']['model_attempts'], ensure_ascii=True)}`",
            "",
        ]
    )

    lines.extend(
        [
            "## Retrieved Evidence",
            "",
        ]
    )
    for index, chunk in enumerate(payload["retrieved_chunks"], start=1):
        lines.append(f"### Chunk {index}")
        lines.append("")
        lines.append(f"- Chunk ID: `{chunk['id']}`")
        lines.append(f"- Document ID: `{chunk['doc_id']}`")
        lines.append(f"- Court level: `{chunk.get('court_level')}`")
        lines.append(f"- Date decided: `{chunk.get('date_decided')}`")
        lines.append(f"- Text: `{chunk['text']}`")
        lines.append("")

    lines.extend(
        [
            "## Claim Verification Output",
            "",
        ]
    )

    for claim, result in zip(payload["claims"], payload["verification_results"]):
        lines.append(f"### Claim: `{claim['text']}`")
        lines.append("")
        lines.append(f"- Final score: `{result['final_score']}`")
        lines.append(f"- Contradicted: `{result['is_contradicted']}`")
        lines.append(f"- Support ratio: `{result['support_ratio']}`")
        best_chunk = result.get("best_chunk")
        if best_chunk:
            lines.append(f"- Best chunk ID: `{best_chunk['id']}`")
            lines.append(f"- Best chunk text: `{best_chunk['text']}`")
        lines.append(f"- Component scores: `{json.dumps(result['component_scores'], ensure_ascii=True)}`")
        lines.append("")

    lines.extend(
        [
            "## Commands",
            "",
            f"- Example command: `{payload['commands']['example']}`",
            "",
            "## Raw JSON",
            "",
            "```json",
            json.dumps(normalize_for_json(payload), indent=2, ensure_ascii=True),
            "```",
            "",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
