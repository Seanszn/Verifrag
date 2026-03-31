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
        if not self.token:
            raise ValueError(
                "CourtListener API token is missing. Check your .env and config."
            )

        headers = {
            "Accept": "application/json",
            "Authorization": f"Token {self.token}",
        }

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
        retry=retry_if_exception_type((aiohttp.ClientConnectionError, asyncio.TimeoutError, RateLimitError)),
        reraise=True,
    )
    async def _request_with_retry(self, url: str, params: Optional[dict]) -> dict:
        if not self._session:
            raise RuntimeError("Client session not initialized. Use 'async with' context manager.")

        try:
            async with self._session.get(url, params=params) as resp:
                body = await resp.text()

                logger.info("CourtListener request URL: %s", str(resp.url))
                logger.info("CourtListener status: %s", resp.status)

                if resp.status == 429:
                    retry_after = float(resp.headers.get("Retry-After", "60"))
                    logger.warning("Rate limited. Retry-After: %s seconds", retry_after)
                    await asyncio.sleep(retry_after)
                    raise RateLimitError(retry_after)

                if resp.status >= 500:
                    logger.warning("Server error %d: %s", resp.status, body[:500])
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status,
                        message=body[:500],
                        headers=resp.headers,
                    )

                if resp.status >= 400:
                    logger.error("Client error %d for %s", resp.status, resp.url)
                    logger.error("Response body: %s", body[:1000])

                    # Do NOT retry normal client-side errors like 400/401/403/404
                    raise RuntimeError(
                        f"CourtListener API request failed with status {resp.status}\n"
                        f"URL: {resp.url}\n"
                        f"Response: {body[:1000]}"
                    )

                try:
                    return await resp.json()
                except Exception as json_error:
                    logger.error("Failed to parse JSON. Body: %s", body[:1000])
                    raise RuntimeError(
                        f"CourtListener returned non-JSON response.\n"
                        f"URL: {resp.url}\n"
                        f"Body: {body[:1000]}"
                    ) from json_error

        except aiohttp.ClientConnectionError as e:
            logger.warning("Connection error calling CourtListener: %s", e)
            raise
        except asyncio.TimeoutError as e:
            logger.warning("Timeout calling CourtListener: %s", e)
            raise
        