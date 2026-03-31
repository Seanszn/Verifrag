import os
import asyncio
from pathlib import Path

from src.ingestion.corpus_builder import CorpusBuilder
from src.ingestion.api_connectors.courtlistener_client import CourtListenerClient

async def main():
    token = os.getenv("COURTLISTENER_API_TOKEN")

    async with CourtListenerClient(token=token) as client:
        builder = CorpusBuilder(client=client, output_dir=Path("data/raw"))
        result = await builder.sync_court(
            court_id="scotus",
            court_level="federal",
            output_file="scotus_cases.jsonl",
            target_count=10,
        )
        print(result)

asyncio.run(main())