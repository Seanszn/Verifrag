import asyncio
import logging
import time
from typing import AsyncIterator, List, Optional

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
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

    async def fetch_clusters(
        self,
        court_id: str,
        since: Optional[str] = None,
        page_size: int = 20,
    ) -> AsyncIterator[dict]:
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
            params = None

    async def fetch_opinion(self, opinion_url: str) -> dict:
        return await self._get(opinion_url)

    async def fetch_opinions_for_cluster(self, cluster_id: int) -> List[dict]:
        url = f"{self.base_url}opinions/"
        params = {"cluster_id": cluster_id}
        data = await self._get(url, params=params)
        return data.get("results", [])

    async def _get(self, url: str, params: Optional[dict] = None) -> dict:
        await self._enforce_rate_limit()
        return await self._request_with_retry(url, params)

    async def _enforce_rate_limit(self):
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