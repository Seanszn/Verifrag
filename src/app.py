import os
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from ingestion.corpus_builder import CorpusBuilder
from ingestion.api_connectors.courtlistener_client import CourtListenerClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env from project root (one level above src)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


async def main():
    token = os.getenv("COURTLISTENER_API_TOKEN")

    if not token:
        raise ValueError(
            f"COURTLISTENER_API_TOKEN not found in .env file at: {ENV_PATH}"
        )

    logger.info("Token loaded successfully: %s", bool(token))

    # data/raw relative to project root
    output_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    async with CourtListenerClient(token=token) as client:
        builder = CorpusBuilder(client=client, output_dir=output_dir)

        result = await builder.sync_court(
            court_id="scotus",
            court_level="federal",
            output_file="scotus_cases.jsonl",
            target_count=10,
        )

        print("Sync result:", result)


if __name__ == "__main__":
    asyncio.run(main())