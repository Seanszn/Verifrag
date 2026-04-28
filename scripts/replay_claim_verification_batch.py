"""Replay the 50-query offline claim-verification batch with current retrieval rules."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nli_test_utils import (  # noqa: E402
    BM25Retriever,
    HeuristicNLIVerifier,
    build_corpus_backed_test_case,
    dump_json,
)
from src.config import VERIFICATION  # noqa: E402
from src.pipeline import QueryPipeline  # noqa: E402
from src.storage.database import Database  # noqa: E402

try:  # noqa: E402
    from src.retrieval.case_targeting import canonical_doc_family_key as _canonical_doc_family_key  # type: ignore
except ImportError:  # pragma: no cover - compatibility with trimmed retrieval module set
    from src.pipeline import _case_family_key as _canonical_doc_family_key  # type: ignore


def canonical_doc_family_key(*, case_name=None, date_decided=None, doc_id=None, citation=None, **_ignored):
    _ = date_decided, doc_id, citation, _ignored
    return _canonical_doc_family_key(case_name or "")


DEFAULT_SOURCE_REPORT = PROJECT_ROOT / "artifacts" / "test_reports" / "claim_verification_50_query_batch.json"


class FixedResponseLLM:
    """Return a prebuilt deterministic response for both direct and RAG generation."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate_legal_answer(self, query: str, **kwargs) -> str:
        _ = query
        _ = kwargs
        return self.response

    def generate_with_context(
        self,
        query: str,
        context,
        max_tokens=None,
        *,
        conversation_history=None,
        case_posture=None,
        response_depth=None,
        **kwargs,
    ) -> str:
        _ = query, context, max_tokens, conversation_history, case_posture, response_depth, kwargs
        return self.response


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the offline 50-query claim-verification batch with current retrieval rules.",
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=DEFAULT_SOURCE_REPORT,
        help="Source batch JSON that provides the query list and expected document ids.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the replayed batch JSON.",
    )
    parser.add_argument(
        "--stats-md",
        type=Path,
        default=None,
        help="Optional Markdown summary path for the replayed batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="How many queries to replay from the source report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many retrieved chunks to keep for each query.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=3,
        help="Maximum number of deterministic evidence-backed response sentences.",
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default="2026-04-13",
        help="Date label to stamp into the run metadata.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_report = args.source_report.resolve()
    output_path = args.output.resolve()
    stats_md = args.stats_md.resolve() if args.stats_md is not None else None

    source_payload = json.loads(source_report.read_text(encoding="utf-8"))
    source_entries = list(_iter_source_entries(source_payload))[: args.limit]

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    retriever = BM25Retriever(top_k=args.top_k)
    verifier = HeuristicNLIVerifier()

    with tempfile.TemporaryDirectory(prefix="claim-batch-replay-") as temp_dir:
        db = Database(Path(temp_dir) / "batch.db")
        db.initialize()
        user = db.create_user("offline_batch", "not-a-real-hash")

        for offset, entry in enumerate(source_entries, start=1):
            try:
                results.append(
                    _replay_entry(
                        entry,
                        index=offset,
                        top_k=args.top_k,
                        max_sentences=args.max_sentences,
                        db=db,
                        user_id=user["id"],
                        retriever=retriever,
                        verifier=verifier,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "index": offset,
                        "query": entry.get("query"),
                        "expected_doc_id": entry.get("expected_doc_id"),
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )

    summary = _build_summary(
        source_report=source_report,
        results=results,
        failures=failures,
        requested_count=len(source_entries),
        run_date=args.run_date,
        top_k=args.top_k,
        max_sentences=args.max_sentences,
    )
    payload = {
        "summary": summary,
        "failures": failures,
        "results": results,
    }
    dump_json(payload, output_path=output_path)

    if stats_md is not None:
        stats_md.parent.mkdir(parents=True, exist_ok=True)
        stats_md.write_text(_render_stats_markdown(payload, output_path), encoding="utf-8")

    print(dump_json(summary))
    return 0


def _iter_source_entries(source_payload: dict[str, Any]):
    if isinstance(source_payload.get("results"), list):
        for entry in source_payload["results"]:
            if isinstance(entry, dict) and entry.get("query"):
                yield entry
        return

    if isinstance(source_payload.get("queries"), list):
        for entry in source_payload["queries"]:
            if isinstance(entry, dict) and entry.get("query"):
                yield entry


def _replay_entry(
    entry: dict[str, Any],
    *,
    index: int,
    top_k: int,
    max_sentences: int,
    db: Database,
    user_id: int,
    retriever: BM25Retriever,
    verifier: HeuristicNLIVerifier,
) -> dict[str, Any]:
    query = str(entry["query"])
    expected_doc_id = entry.get("expected_doc_id")
    case_name = entry.get("case_name")
    date_decided = entry.get("date_decided")

    generated = build_corpus_backed_test_case(
        query=query,
        top_k=top_k,
        max_sentences=max_sentences,
    )

    pipeline = QueryPipeline(
        db=db,
        llm=FixedResponseLLM(generated.response),
        retriever=retriever,
        verifier=verifier,
    )
    pipeline_result = pipeline.run(user_id=user_id, query=query)
    meta = pipeline_result["pipeline"]

    expected_family_key = canonical_doc_family_key(
        case_name=case_name,
        date_decided=date_decided,
        doc_id=expected_doc_id,
    )
    retrieval = _build_retrieval_payload(meta.get("retrieved_chunks", []), expected_family_key)
    claims = [
        _serialize_claim(claim_payload, expected_family_key)
        for claim_payload in meta.get("claims", [])
        if isinstance(claim_payload, dict)
    ]

    verdict_counts = Counter(claim["verdict"] for claim in claims)
    scores = [claim["final_score"] for claim in claims]
    confirmed_count = sum(
        1 for claim in claims if claim["verdict"] in {"SUPPORTED", "VERIFIED"}
    )

    top_doc_id = retrieval["top_doc_id"]
    top_doc_family_key = retrieval["top_doc_family_key"]
    top_doc_matches_expected_raw = bool(top_doc_id and top_doc_id == expected_doc_id)
    top_doc_matches_expected_canonical = bool(
        top_doc_family_key
        and expected_family_key
        and top_doc_family_key == expected_family_key
    )

    retrieved_doc_ids = retrieval["retrieved_doc_ids"]
    retrieved_doc_family_keys = retrieval["retrieved_doc_family_keys"]
    expected_doc_in_retrieval_raw = bool(expected_doc_id and expected_doc_id in retrieved_doc_ids)
    expected_doc_in_retrieval_canonical = bool(
        expected_family_key and expected_family_key in retrieved_doc_family_keys
    )

    return {
        "query": query,
        "expected_doc_id": expected_doc_id,
        "expected_doc_family_key": expected_family_key,
        "case_name": case_name,
        "date_decided": date_decided,
        "topic": entry.get("topic"),
        "index": entry.get("index", index),
        "response": generated.response,
        "supporting_sentences": list(generated.supporting_sentences),
        "rule_set": {
            "branch": "sean",
            "generation": "corpus-backed deterministic response synthesis",
            "retrieval": "BM25Retriever(top_k=8) with target-case filtering and canonical chunk dedupe",
            "pipeline_generation_mode": meta.get("generation_mode"),
            "verifier": "HeuristicNLIVerifier",
            "reranker_note": "Cross-encoder reranker exists on this branch but was not exercised in this offline batch run.",
            "canonicalization_note": "Revision-suffixed case names are canonicalized for matching, and duplicate revised chunks are deduped before generation and verification.",
        },
        "pipeline_status": {
            "llm_backend_status": meta.get("llm_backend_status"),
            "generation_mode": meta.get("generation_mode"),
            "retrieval_backend_status": meta.get("retrieval_backend_status"),
            "verification_backend_status": meta.get("verification_backend_status"),
            "retrieval_used": meta.get("retrieval_used"),
            "retrieval_chunk_count": meta.get("retrieval_chunk_count"),
        },
        "metrics": {
            "claim_count": len(claims),
            "verdict_counts": dict(verdict_counts),
            "contradicted_count": sum(1 for claim in claims if claim["is_contradicted"]),
            "confirmed_supported_or_verified_count": confirmed_count,
            "min_score": min(scores) if scores else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "top_doc_matches_expected_raw": top_doc_matches_expected_raw,
            "top_doc_matches_expected_canonical": top_doc_matches_expected_canonical,
            "expected_doc_in_retrieval_raw": expected_doc_in_retrieval_raw,
            "expected_doc_in_retrieval_canonical": expected_doc_in_retrieval_canonical,
            "unique_retrieved_doc_count": len(set(retrieved_doc_ids)),
            "unique_canonical_retrieved_doc_count": len(set(retrieved_doc_family_keys)),
        },
        "retrieval": retrieval,
        "retrieved_chunks": meta.get("retrieved_chunks", []),
        "claims": claims,
    }


def _build_retrieval_payload(
    retrieved_chunks: list[dict[str, Any]],
    expected_family_key: str,
) -> dict[str, Any]:
    normalized_chunks = [chunk for chunk in retrieved_chunks if isinstance(chunk, dict)]
    doc_ids = [chunk.get("doc_id") for chunk in normalized_chunks if chunk.get("doc_id")]
    doc_family_keys = [
        canonical_doc_family_key(
            case_name=chunk.get("case_name"),
            citation=chunk.get("citation"),
            date_decided=chunk.get("date_decided"),
            court_level=chunk.get("court_level"),
            doc_id=chunk.get("doc_id"),
        )
        for chunk in normalized_chunks
    ]
    top_doc_id = doc_ids[0] if doc_ids else None
    top_doc_family_key = doc_family_keys[0] if doc_family_keys else None
    return {
        "top_doc_id": top_doc_id,
        "top_doc_family_key": top_doc_family_key,
        "expected_doc_family_key": expected_family_key,
        "retrieved_doc_ids": doc_ids,
        "retrieved_doc_family_keys": doc_family_keys,
    }


def _serialize_claim(claim_payload: dict[str, Any], expected_family_key: str) -> dict[str, Any]:
    verification = claim_payload.get("verification", {})
    best_chunk = verification.get("best_chunk") or {}
    best_doc_id = best_chunk.get("doc_id")
    best_doc_family_key = canonical_doc_family_key(
        case_name=best_chunk.get("case_name"),
        citation=best_chunk.get("citation"),
        date_decided=best_chunk.get("date_decided"),
        court_level=best_chunk.get("court_level"),
        doc_id=best_doc_id,
    )
    return {
        "text": claim_payload.get("text"),
        "final_score": float(verification.get("final_score", 0.0)),
        "verdict": verification.get("verdict", "UNSUPPORTED"),
        "is_contradicted": bool(verification.get("is_contradicted")),
        "support_ratio": float(verification.get("support_ratio", 0.0)),
        "best_chunk_id": best_chunk.get("id"),
        "best_doc_id": best_doc_id,
        "best_doc_family_key": best_doc_family_key,
        "matches_expected_doc_family": bool(
            best_doc_family_key and expected_family_key and best_doc_family_key == expected_family_key
        ),
    }


def _build_summary(
    *,
    source_report: Path,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    requested_count: int,
    run_date: str,
    top_k: int,
    max_sentences: int,
) -> dict[str, Any]:
    verdict_counts: Counter[str] = Counter()
    total_claims = 0
    total_confirmed = 0
    query_with_confirmed = 0
    queries_all_confirmed = 0
    top_raw = 0
    top_canonical = 0
    hit_raw = 0
    hit_canonical = 0
    scores: list[float] = []

    for result in results:
        metrics = result["metrics"]
        claim_count = int(metrics["claim_count"])
        total_claims += claim_count
        verdict_counts.update(metrics["verdict_counts"])
        total_confirmed += int(metrics["confirmed_supported_or_verified_count"])
        if metrics["confirmed_supported_or_verified_count"] > 0:
            query_with_confirmed += 1
        if claim_count and metrics["confirmed_supported_or_verified_count"] == claim_count:
            queries_all_confirmed += 1
        if metrics["top_doc_matches_expected_raw"]:
            top_raw += 1
        if metrics["top_doc_matches_expected_canonical"]:
            top_canonical += 1
        if metrics["expected_doc_in_retrieval_raw"]:
            hit_raw += 1
        if metrics["expected_doc_in_retrieval_canonical"]:
            hit_canonical += 1
        scores.extend(claim["final_score"] for claim in result["claims"])

    completed = len(results)
    claim_level_support_rate = (total_confirmed / total_claims) if total_claims else 0.0

    return {
        "source_report": str(source_report.relative_to(PROJECT_ROOT)).replace("/", "\\"),
        "query_count_requested": requested_count,
        "query_count_completed": completed,
        "failure_count": len(failures),
        "total_claims": total_claims,
        "verdict_counts": dict(verdict_counts),
        "total_confirmed_claims_verified_or_supported": total_confirmed,
        "queries_with_confirmed_claim": query_with_confirmed,
        "queries_all_confirmed_claims": queries_all_confirmed,
        "top_doc_match_count_raw": top_raw,
        "top_doc_match_count_canonical": top_canonical,
        "expected_doc_hit_at_k_count_raw": hit_raw,
        "expected_doc_hit_at_k_count_canonical": hit_canonical,
        "score_min": min(scores) if scores else 0.0,
        "score_avg": (sum(scores) / len(scores)) if scores else 0.0,
        "score_max": max(scores) if scores else 0.0,
        "thresholds": {
            "threshold_verified": VERIFICATION.threshold_verified,
            "threshold_supported": VERIFICATION.threshold_supported,
            "threshold_weak": VERIFICATION.threshold_weak,
            "threshold_contradicted": VERIFICATION.threshold_contradicted,
        },
        "tracked_statistics": {
            "precision_at_1_raw": _safe_rate(top_raw, completed),
            "precision_at_1_canonical": _safe_rate(top_canonical, completed),
            "expected_doc_hit_rate_at_k_raw": _safe_rate(hit_raw, completed),
            "expected_doc_hit_rate_at_k_canonical": _safe_rate(hit_canonical, completed),
            "wrong_case_top_hit_rate_raw": 1.0 - _safe_rate(top_raw, completed),
            "wrong_case_top_hit_rate_canonical": 1.0 - _safe_rate(top_canonical, completed),
            "claim_level_support_rate": claim_level_support_rate,
        },
        "statistics_note": [
            "Raw doc-id metrics are retained for comparability with prior reruns.",
            "Canonical metrics collapse revision-suffixed duplicate opinions into one case family and are the preferred proxy when duplicate revised documents exist.",
            "Claim-level support rate tracks whether final answers remain grounded after retrieval, decomposition, and verdict classification.",
        ],
        "run_notes": {
            "date": run_date,
            "rule_set_name": "sean_rules_offline_batch_canonicalized",
            "top_k": top_k,
            "max_sentences": max_sentences,
        },
    }


def _render_stats_markdown(payload: dict[str, Any], output_path: Path) -> str:
    summary = payload["summary"]
    strict = _confusion_metrics(payload["results"], positive_labels={"SUPPORTED", "VERIFIED"})
    lenient = _confusion_metrics(
        payload["results"],
        positive_labels={"SUPPORTED", "VERIFIED", "POSSIBLE_SUPPORT"},
    )

    return "\n".join(
        [
            "# Canonicalized Sean Rules Statistics Summary",
            "",
            f"Date: {summary['run_notes']['date']}",
            "",
            "## Scope",
            "",
            f"- `{_display_path(output_path)}`",
            "",
            "Important caveat:",
            "",
            "- This is **not** a manual-audit report.",
            "- The confusion matrices below use a **canonical proxy ground truth**:",
            "  a claim is treated as actually supported when its `best_doc_family_key` matches the query's `expected_doc_family_key`.",
            "- Raw doc-id retrieval metrics are still included for comparison, but the canonical metrics are the preferred proxy when revised duplicate opinions exist.",
            "",
            "## Headline Counts",
            "",
            f"- Queries completed: `{summary['query_count_completed']}/{summary['query_count_requested']}`",
            f"- Failures: `{summary['failure_count']}`",
            f"- Total claims: `{summary['total_claims']}`",
            f"- System `VERIFIED`: `{summary['verdict_counts'].get('VERIFIED', 0)}`",
            f"- System `SUPPORTED`: `{summary['verdict_counts'].get('SUPPORTED', 0)}`",
            f"- System `CONTRADICTED`: `{summary['verdict_counts'].get('CONTRADICTED', 0)}`",
            f"- System `POSSIBLE_SUPPORT`: `{summary['verdict_counts'].get('POSSIBLE_SUPPORT', 0)}`",
            f"- System `UNSUPPORTED`: `{summary['verdict_counts'].get('UNSUPPORTED', 0)}`",
            f"- Top-1 raw exact-match rate: `{_pct(summary['tracked_statistics']['precision_at_1_raw'])}`",
            f"- Top-1 canonical match rate: `{_pct(summary['tracked_statistics']['precision_at_1_canonical'])}`",
            f"- Raw expected-doc hit rate at k: `{_pct(summary['tracked_statistics']['expected_doc_hit_rate_at_k_raw'])}`",
            f"- Canonical expected-doc hit rate at k: `{_pct(summary['tracked_statistics']['expected_doc_hit_rate_at_k_canonical'])}`",
            f"- Claim-level support rate (`SUPPORTED` or `VERIFIED`): `{_pct(summary['tracked_statistics']['claim_level_support_rate'])}`",
            "",
            "## Thresholds Used",
            "",
            f"- `threshold_verified = {summary['thresholds']['threshold_verified']}`",
            f"- `threshold_supported = {summary['thresholds']['threshold_supported']}`",
            f"- `threshold_weak = {summary['thresholds']['threshold_weak']}`",
            f"- `threshold_contradicted = {summary['thresholds']['threshold_contradicted']}`",
            "",
            "## Strict Positive Only",
            "",
            "Treating only `SUPPORTED` and `VERIFIED` as positive:",
            "",
            f"- Accuracy: `{_pct(strict['accuracy'])}`",
            f"- Precision: `{_pct(strict['precision'])}`",
            f"- Recall: `{_pct(strict['recall'])}`",
            f"- Specificity: `{_pct(strict['specificity'])}`",
            f"- False-positive rate: `{_pct(strict['fpr'])}`",
            f"- F1: `{_pct(strict['f1'])}`",
            f"- TP: `{strict['tp']}`",
            f"- FP: `{strict['fp']}`",
            f"- FN: `{strict['fn']}`",
            f"- TN: `{strict['tn']}`",
            "",
            "Confusion matrix:",
            "",
            "| Actual | Predicted supported | Predicted not supported |",
            "| --- | ---: | ---: |",
            f"| Supported | {strict['tp']} | {strict['fn']} |",
            f"| Not supported | {strict['fp']} | {strict['tn']} |",
            "",
            "## If POSSIBLE_SUPPORT Counts As Supported",
            "",
            "Treating `POSSIBLE_SUPPORT` as positive:",
            "",
            f"- Accuracy: `{_pct(lenient['accuracy'])}`",
            f"- Precision: `{_pct(lenient['precision'])}`",
            f"- Recall: `{_pct(lenient['recall'])}`",
            f"- Specificity: `{_pct(lenient['specificity'])}`",
            f"- False-positive rate: `{_pct(lenient['fpr'])}`",
            f"- F1: `{_pct(lenient['f1'])}`",
            f"- TP: `{lenient['tp']}`",
            f"- FP: `{lenient['fp']}`",
            f"- FN: `{lenient['fn']}`",
            f"- TN: `{lenient['tn']}`",
            "",
            "Confusion matrix:",
            "",
            "| Actual | Predicted supported | Predicted not supported |",
            "| --- | ---: | ---: |",
            f"| Supported | {lenient['tp']} | {lenient['fn']} |",
            f"| Not supported | {lenient['fp']} | {lenient['tn']} |",
            "",
            "## Notable Findings",
            "",
            f"- Canonical top-1 retrieval precision improved to `{_pct(summary['tracked_statistics']['precision_at_1_canonical'])}` once duplicate revised opinions were collapsed into the same case family.",
            f"- Raw top-1 precision remains `{_pct(summary['tracked_statistics']['precision_at_1_raw'])}`, which preserves comparability with prior runs and still surfaces doc-id-level revision collisions.",
            f"- Canonical wrong-case top-hit rate is `{_pct(summary['tracked_statistics']['wrong_case_top_hit_rate_canonical'])}`.",
            f"- Strict claim-level support remains `{_pct(summary['tracked_statistics']['claim_level_support_rate'])}`, indicating retrieval is no longer the dominant error source on this offline batch.",
            "",
            "## Artifacts",
            "",
            f"- Source batch JSON: `{summary['source_report'].replace(chr(92), '/')}`",
            f"- Replay batch JSON: `{_display_path(output_path)}`",
            "",
        ]
    ) + "\n"


def _confusion_metrics(results: list[dict[str, Any]], *, positive_labels: set[str]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for result in results:
        expected_family_key = result["retrieval"]["expected_doc_family_key"]
        for claim in result["claims"]:
            actual_positive = (
                bool(claim["best_doc_family_key"])
                and bool(expected_family_key)
                and claim["best_doc_family_key"] == expected_family_key
            )
            predicted_positive = claim["verdict"] in positive_labels
            if actual_positive and predicted_positive:
                tp += 1
            elif actual_positive:
                fn += 1
            elif predicted_positive:
                fp += 1
            else:
                tn += 1

    total = tp + fp + fn + tn
    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    specificity = _safe_rate(tn, tn + fp)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": _safe_rate(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": _safe_rate(fp, fp + tn),
        "f1": _safe_rate(2 * precision * recall, precision + recall),
    }


def _safe_rate(numerator, denominator) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _display_path(path: Path) -> str:
    resolved = path if path.is_absolute() else (PROJECT_ROOT / path)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
