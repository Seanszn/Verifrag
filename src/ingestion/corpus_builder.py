"""
Corpus Builder - Download legal documents from CourtListener API v4.

Provides:
  - CourtListenerClient: async HTTP client with rate limiting and retry
  - CorpusBuilder: orchestrates corpus building and incremental updates
  - SyncResult: result dataclass for sync operations
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Set
from ingestion.api_connectors.courtlistener_client import CourtListenerClient

import aiohttp
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ingestion.document import LegalDocument

logger = logging.getLogger(__name__)


# ── SyncResult ────────────────────────────────────────────


@dataclass
class SyncResult:
    """Result of syncing a single court."""

    court_id: str
    docs_fetched: int = 0
    docs_new: int = 0
    docs_updated: int = 0
    errors: int = 0
    elapsed_seconds: float = 0.0


# ── CourtListenerClient ──────────────────────────────────


class RateLimitError(Exception):
    """Raised when API returns 429."""

    def __init__(self, retry_after: float = 60.0):
        self.retry_after = retry_after
        super().__init__(f"Rate limited, retry after {retry_after}s")


class CourtListenerClient:
    """Async HTTP client for CourtListener REST API v4."""

    def __init__(
        self,
        token: Optional[str],
        base_url: str = "https://www.courtlistener.com/api/rest/v4/",
        rate_limit: float = 0.72,
    ):
        self.token = token
        self.base_url = base_url.rstrip("/") + "/"
        self.rate_limit = rate_limit
        self._last_request_time: float = 0
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "CourtListenerClient":
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Token {self.token}"
        self._session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()
            self._session = None

    # --- Core API methods ---

    async def fetch_clusters(
        self,
        court_id: str,
        since: Optional[str] = None,
        page_size: int = 20,
    ) -> AsyncIterator[dict]:
        """
        Paginate through all clusters for a given court.

        Yields raw cluster dicts from the API.
        Uses cursor-based pagination (follows 'next' URLs).
        """
        params = {
            "docket__court": court_id,
            "page_size": page_size,
        }
        if since:
            params["date_modified__gte"] = since

        url = f"{self.base_url}clusters/"

        while url:
            data = await self._get(url, params=params)
            for cluster in data.get("results", []):
                yield cluster

            url = data.get("next")
            # After the first request, params are encoded in the cursor URL
            params = None

    async def fetch_opinion(self, opinion_url: str) -> dict:
        """
        Fetch a single opinion by its API URL.

        Returns raw opinion dict with text fields.
        """
        return await self._get(opinion_url)

    async def fetch_opinions_for_cluster(self, cluster_id: int) -> List[dict]:
        """
        Fetch all opinions linked to a cluster.

        Returns list of opinion dicts.
        """
        url = f"{self.base_url}opinions/"
        params = {"cluster_id": cluster_id}
        data = await self._get(url, params=params)
        return data.get("results", [])

    # --- Internal helpers ---

    async def _get(self, url: str, params: Optional[dict] = None) -> dict:
        """Make authenticated GET request with rate limiting and retry."""
        await self._enforce_rate_limit()
        return await self._request_with_retry(url, params)

    async def _enforce_rate_limit(self):
        """Enforce minimum time between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.monotonic()

    @retry(
        wait=wait_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((aiohttp.ClientError, RateLimitError)),
    )
    async def _request_with_retry(self, url: str, params: Optional[dict]) -> dict:
        """Make request with retry on transient errors."""
        if not self._session:
            raise RuntimeError("Client session not initialized. Use 'async with' context manager.")

        async with self._session.get(url, params=params) as resp:
            if resp.status == 429:
                retry_after = float(resp.headers.get("Retry-After", "60"))
                logger.warning("Rate limited. Retry-After: %s seconds", retry_after)
                await asyncio.sleep(retry_after)
                raise RateLimitError(retry_after)

            if resp.status >= 500:
                body = await resp.text()
                logger.warning("Server error %d: %s", resp.status, body[:200])
                raise aiohttp.ClientResponseError(
                    resp.request_info,
                    resp.history,
                    status=resp.status,
                    message=f"Server error {resp.status}",
                )

            resp.raise_for_status()
            return await resp.json()


# ── CorpusBuilder ─────────────────────────────────────────


class CorpusBuilder:
    """Orchestrates corpus building and incremental updates."""

    def __init__(self, client: CourtListenerClient, output_dir: Path):
        self.client = client
        self.output_dir = Path(output_dir)
        self.sync_state_path = self.output_dir / ".sync_state.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._progress_callback = None

    def set_progress_callback(self, callback):
        """Set a callback(docs_fetched) for progress reporting."""
        self._progress_callback = callback

    async def sync_court(
        self,
        court_id: str,
        court_level: str,
        output_file: str,
        target_count: Optional[int] = None,
    ) -> SyncResult:
        """
        Sync a single court. Works for both initial build and updates.

        1. Load sync state → get last_sync timestamp for this court
        2. If last_sync is None → initial build (fetch all)
        3. If last_sync exists → incremental (fetch modified since)
        4. For each cluster, fetch opinions and build LegalDocuments
        5. Save updated sync state
        """
        start_time = time.monotonic()
        result = SyncResult(court_id=court_id)
        sync_state = self._load_sync_state()
        last_sync = sync_state.get(court_id, {}).get("last_sync")

        filepath = self.output_dir / output_file
        existing_ids = self._load_existing_ids(filepath)

        sync_start = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Syncing court=%s level=%s since=%s target=%s",
            court_id, court_level, last_sync or "beginning", target_count or "all",
        )

        async for cluster in self.client.fetch_clusters(
            court_id=court_id,
            since=last_sync,
        ):
            if target_count is not None and result.docs_fetched >= target_count:
                break

            result.docs_fetched += 1

            try:
                # Fetch opinions for this cluster
                opinion_text = await self._fetch_best_opinion_text(cluster)
                if not opinion_text:
                    logger.debug("No opinion text for cluster %s, skipping", cluster.get("id"))
                    result.errors += 1
                    continue

                doc = self._parse_cluster_to_document(
                    cluster=cluster,
                    opinion_text=opinion_text,
                    court_id=court_id,
                    court_level=court_level,
                )

                if doc.id in existing_ids:
                    result.docs_updated += 1
                else:
                    result.docs_new += 1

                self._append_to_jsonl(doc, filepath)
                existing_ids.add(doc.id)

            except Exception:
                logger.exception("Error processing cluster %s", cluster.get("id"))
                result.errors += 1

            if self._progress_callback:
                self._progress_callback(result.docs_fetched)

        # Save sync state
        sync_state[court_id] = {
            "last_sync": sync_start,
            "docs_total": len(existing_ids),
            "last_run": datetime.now(timezone.utc).isoformat(),
        }
        self._save_sync_state(sync_state)

        result.elapsed_seconds = time.monotonic() - start_time
        logger.info(
            "Sync complete: court=%s fetched=%d new=%d updated=%d errors=%d (%.1fs)",
            court_id, result.docs_fetched, result.docs_new,
            result.docs_updated, result.errors, result.elapsed_seconds,
        )
        return result

    async def sync_all(
        self,
        courts: Optional[Dict[str, str]] = None,
        target_count: Optional[int] = None,
    ) -> List[SyncResult]:
        """Sync all configured courts sequentially."""
        if courts is None:
            from src.config import DATA
            courts = DATA.court_level_map

        results = []
        for court_id, court_level in courts.items():
            output_file = f"{court_id}_cases.jsonl"
            result = await self.sync_court(
                court_id=court_id,
                court_level=court_level,
                output_file=output_file,
                target_count=target_count,
            )
            results.append(result)
        return results

    async def _fetch_best_opinion_text(self, cluster: dict) -> Optional[str]:
        """Fetch opinions for a cluster and return the best available text."""
        sub_opinions = cluster.get("sub_opinions", [])

        if sub_opinions:
            # Fetch each opinion by its URL
            for opinion_url in sub_opinions:
                try:
                    opinion = await self.client.fetch_opinion(opinion_url)
                    text = self._extract_opinion_text(opinion)
                    if text:
                        return text
                except Exception:
                    logger.debug("Failed to fetch opinion at %s", opinion_url)
                    continue

        # Fallback: fetch by cluster_id
        cluster_id = cluster.get("id")
        if cluster_id:
            try:
                opinions = await self.client.fetch_opinions_for_cluster(cluster_id)
                for opinion in opinions:
                    text = self._extract_opinion_text(opinion)
                    if text:
                        return text
            except Exception:
                logger.debug("Failed to fetch opinions for cluster %s", cluster_id)

        return None

    def _parse_cluster_to_document(
        self,
        cluster: dict,
        opinion_text: str,
        court_id: str,
        court_level: str,
    ) -> LegalDocument:
        """Convert API cluster + opinion text → LegalDocument dataclass."""
        cluster_id = cluster.get("id", "unknown")
        date_filed = cluster.get("date_filed")
        date_decided = None
        if date_filed:
            try:
                date_decided = date.fromisoformat(date_filed)
            except (ValueError, TypeError):
                pass

        citation = self._format_citation(cluster.get("citations", []))

        return LegalDocument(
            id=f"cl_{cluster_id}",
            doc_type="case",
            full_text=opinion_text,
            case_name=cluster.get("case_name", ""),
            citation=citation,
            court=court_id,
            court_level=court_level,
            date_decided=date_decided,
        )

    def _extract_opinion_text(self, opinion: dict) -> Optional[str]:
        """
        Get best available text from opinion dict.

        Priority: plain_text > html_with_citations > html >
                  html_lawbox > html_columbia > xml_harvard
        """
        # Try plain_text first (cleanest)
        plain = opinion.get("plain_text", "")
        if plain and plain.strip():
            return self._clean_legal_text(plain)

        # Try HTML fields in priority order, stripping tags
        html_fields = [
            "html_with_citations",
            "html",
            "html_lawbox",
            "html_columbia",
            "xml_harvard",
        ]
        for field_name in html_fields:
            html = opinion.get(field_name, "")
            if html and html.strip():
                stripped = self._strip_html(html)
                return self._clean_legal_text(stripped)

        return None

    @staticmethod
    def _strip_html(html: str) -> str:
        """Strip HTML/XML tags and clean up whitespace."""
        text = BeautifulSoup(html, "html.parser").get_text(separator=" ")
        # Collapse multiple whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _clean_legal_text(text: str) -> str:
        # Normalize line endings first.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Repair hyphenated line-wraps:
        text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

        # Collapse intra-line spacing noise while preserving line boundaries.
        text = re.sub(r"[ \t]+", " ", text)

        # Trim spaces around newlines.
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)

        # Keep paragraph structure but avoid excessive blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _format_citation(self, citations: List[dict]) -> Optional[str]:
        """
        Format citation from API citations list.

        e.g. [{"volume": 384, "reporter": "U.S.", "page": "436"}]
              → "384 U.S. 436"
        """
        if not citations:
            return None
        for cit in citations:
            volume = cit.get("volume")
            reporter = cit.get("reporter")
            page = cit.get("page")
            if volume and reporter and page:
                return f"{volume} {reporter} {page}"
        return None

    # --- JSONL I/O ---

    def _append_to_jsonl(self, doc: LegalDocument, filepath: Path):
        """Append one document as a JSON line."""
        record = {
            "id": doc.id,
            "doc_type": doc.doc_type,
            "case_name": doc.case_name,
            "citation": doc.citation,
            "court": doc.court,
            "court_level": doc.court_level,
            "date_decided": str(doc.date_decided) if doc.date_decided else None,
            "full_text": doc.full_text,
        }
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_existing_ids(self, filepath: Path) -> Set[str]:
        """Load all document IDs from existing JSONL for dedup."""
        ids: Set[str] = set()
        if not filepath.exists():
            return ids
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    ids.add(record["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return ids

    # --- Sync state ---

    def _load_sync_state(self) -> dict:
        """Load sync state from JSON file."""
        if not self.sync_state_path.exists():
            return {}
        with open(self.sync_state_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_sync_state(self, state: dict):
        """Save sync state to JSON file."""
        with open(self.sync_state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    
    