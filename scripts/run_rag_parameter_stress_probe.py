"""Run the live API against a targeted RAG-parameter stress set."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval" / "rag_parameter_stress_set_2026-04-20.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "test_reports" / "rag_parameter_stress_probe_2026-04-20.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--username", default=f"rag_probe_{uuid4().hex[:8]}")
    parser.add_argument("--password", default="rag_probe_password_123")
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--query-timeout", type=int, default=420)
    return parser.parse_args()


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _contains_any(answer: str, terms: list[str]) -> list[str]:
    haystack = _normalized(answer)
    return [term for term in terms if _normalized(term) in haystack]


def _evaluate_item(item: dict[str, Any], answer: str, pipeline: dict[str, Any]) -> dict[str, Any]:
    required_groups = item.get("required_groups", [])
    required_results = []
    all_required = True
    for group in required_groups:
        hits = _contains_any(answer, [str(term) for term in group])
        required_results.append({"group": group, "hits": hits, "passed": bool(hits)})
        all_required = all_required and bool(hits)

    forbidden_hits = _contains_any(answer, [str(term) for term in item.get("forbidden_terms", [])])

    expected_filter = item.get("expect_case_filter")
    actual_filter = pipeline.get("prompt_case_filter_status")
    case_filter_ok = True
    if expected_filter is True:
        case_filter_ok = actual_filter == "applied"
    elif expected_filter is False:
        case_filter_ok = True

    expected_posture = item.get("expected_posture") or {}
    actual_posture = pipeline.get("case_posture") or {}
    posture_checks = []
    posture_ok = True
    for key, expected in expected_posture.items():
        actual = actual_posture.get(key)
        passed = actual == expected
        posture_checks.append({"field": key, "expected": expected, "actual": actual, "passed": passed})
        posture_ok = posture_ok and passed

    auto_pass = all_required and not forbidden_hits and case_filter_ok and posture_ok

    retrieved_case_names = []
    for chunk in pipeline.get("retrieved_chunks", []) or []:
        if isinstance(chunk, dict):
            case_name = chunk.get("case_name")
            if case_name and case_name not in retrieved_case_names:
                retrieved_case_names.append(case_name)

    return {
        "required_results": required_results,
        "forbidden_hits": forbidden_hits,
        "case_filter_ok": case_filter_ok,
        "expected_case_filter": expected_filter,
        "actual_case_filter_status": actual_filter,
        "posture_checks": posture_checks,
        "posture_ok": posture_ok,
        "auto_pass": auto_pass,
        "retrieved_case_names": retrieved_case_names,
    }


def _create_conversation(session: requests.Session, base_url: str, title: str, timeout: int) -> int:
    response = session.post(
        f"{base_url}/api/conversations",
        json={"title": title},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return int(payload["id"])


def main() -> int:
    args = parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    items = dataset.get("items", [])
    if not isinstance(items, list) or not items:
        raise SystemExit("Dataset did not contain any items")

    session = requests.Session()
    session.post(
        f"{args.base_url.rstrip('/')}/api/auth/register",
        json={"username": args.username, "password": args.password},
        timeout=args.request_timeout,
    )
    login = session.post(
        f"{args.base_url.rstrip('/')}/api/auth/login",
        json={"username": args.username, "password": args.password},
        timeout=args.request_timeout,
    )
    login.raise_for_status()
    token = login.json()["token"]
    session.headers.update({"Authorization": f"Bearer {token}"})

    conversation_ids: dict[str, int] = {}
    results: list[dict[str, Any]] = []
    for item in items:
        conversation_key = str(item.get("conversation_key") or item["id"])
        if conversation_key not in conversation_ids:
            conversation_ids[conversation_key] = _create_conversation(
                session,
                args.base_url.rstrip("/"),
                title=f"RAG Probe {conversation_key}",
                timeout=args.request_timeout,
            )
        conversation_id = conversation_ids[conversation_key]
        response = session.post(
            f"{args.base_url.rstrip('/')}/api/query",
            json={"query": item["query"], "conversation_id": conversation_id},
            timeout=args.query_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        pipeline = payload.get("pipeline", {})
        answer = payload.get("assistant_message", {}).get("content", "")
        evaluation = _evaluate_item(item, answer, pipeline)
        results.append(
            {
                "item": item,
                "conversation_id": conversation_id,
                "assistant_content": answer,
                "pipeline_summary": {
                    "llm_backend_status": pipeline.get("llm_backend_status"),
                    "retrieval_backend_status": pipeline.get("retrieval_backend_status"),
                    "verification_backend_status": pipeline.get("verification_backend_status"),
                    "verification_verifier_mode": pipeline.get("verification_verifier_mode"),
                    "retrieval_chunk_count": pipeline.get("retrieval_chunk_count"),
                    "prompt_chunk_count": pipeline.get("prompt_chunk_count"),
                    "prompt_case_filter_status": pipeline.get("prompt_case_filter_status"),
                    "target_case_name": pipeline.get("target_case_name"),
                    "target_case_prompt_candidate_count": pipeline.get("target_case_prompt_candidate_count"),
                    "target_case_prompt_limit": pipeline.get("target_case_prompt_limit"),
                    "case_posture_status": pipeline.get("case_posture_status"),
                    "case_posture": pipeline.get("case_posture"),
                    "claim_count": pipeline.get("claim_count"),
                    "timings_ms": pipeline.get("timings_ms"),
                },
                "evaluation": evaluation,
            }
        )

    category_counts = Counter(result["item"].get("category", "unknown") for result in results)
    category_passes = Counter(
        result["item"].get("category", "unknown")
        for result in results
        if result["evaluation"]["auto_pass"]
    )
    failures_by_category: dict[str, list[str]] = defaultdict(list)
    for result in results:
        if not result["evaluation"]["auto_pass"]:
            failures_by_category[result["item"].get("category", "unknown")].append(result["item"]["id"])

    summary = {
        "dataset_name": dataset.get("name"),
        "item_count": len(results),
        "auto_pass_count": sum(1 for result in results if result["evaluation"]["auto_pass"]),
        "auto_fail_count": sum(1 for result in results if not result["evaluation"]["auto_pass"]),
        "category_counts": dict(category_counts),
        "category_auto_passes": dict(category_passes),
        "failures_by_category": dict(failures_by_category),
    }

    output = {
        "dataset": dataset,
        "username": args.username,
        "base_url": args.base_url,
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
