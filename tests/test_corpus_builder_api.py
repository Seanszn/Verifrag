"""Tests for CourtListener API query construction."""

import pytest

from src.ingestion.corpus_builder import CourtListenerClient


pytestmark = pytest.mark.smoke


@pytest.mark.asyncio
async def test_fetch_clusters_omits_deprecated_ordering_param():
    client = CourtListenerClient(token=None)
    captured = {}

    async def fake_get(url, params=None):
        captured["url"] = url
        captured["params"] = params
        return {"results": [{"id": 1}], "next": None}

    client._get = fake_get

    results = [
        cluster
        async for cluster in client.fetch_clusters(
            court_id="scotus",
            since="2026-02-01T00:00:00+00:00",
            page_size=7,
        )
    ]

    assert len(results) == 1
    assert captured["url"].endswith("/clusters/")
    assert captured["params"]["docket__court"] == "scotus"
    assert captured["params"]["date_modified__gte"] == "2026-02-01T00:00:00+00:00"
    assert captured["params"]["page_size"] == 7
    assert "ordering" not in captured["params"]
