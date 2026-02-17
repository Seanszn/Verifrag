"""
CLI for corpus building and incremental updates.

Usage:
  python scripts/download_corpus.py --all              # Initial full build
  python scripts/download_corpus.py --scotus            # SCOTUS only
  python scripts/download_corpus.py --circuits          # All configured circuits
  python scripts/download_corpus.py --update            # Incremental update (all)
  python scripts/download_corpus.py --scotus --limit 10 # Test with 10 docs
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA, RAW_DIR
from src.ingestion.corpus_builder import (
    CorpusBuilder,
    CourtListenerClient,
    SyncResult,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)

# Global reference for signal handler
_builder: CorpusBuilder | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download legal corpus from CourtListener API v4",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Sync all configured courts")
    group.add_argument("--scotus", action="store_true", help="Sync SCOTUS only")
    group.add_argument("--circuits", action="store_true", help="Sync all configured circuits")
    group.add_argument("--update", action="store_true", help="Incremental update (all courts)")
    group.add_argument(
        "--court", type=str, metavar="ID",
        help="Sync a specific court by ID (e.g. ca9)",
    )

    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max documents to fetch per court (for testing)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {RAW_DIR})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def signal_handler(sig, frame):
    """Save sync state and exit gracefully on Ctrl+C."""
    print("\nInterrupted. Sync state has been saved.")
    sys.exit(0)


def print_summary(results: list[SyncResult]):
    """Print a summary table of sync results."""
    print("\n" + "=" * 60)
    print("SYNC SUMMARY")
    print("=" * 60)
    print(f"{'Court':<10} {'Fetched':>8} {'New':>8} {'Updated':>8} {'Errors':>8} {'Time':>8}")
    print("-" * 60)

    totals = SyncResult(court_id="TOTAL")
    for r in results:
        print(
            f"{r.court_id:<10} {r.docs_fetched:>8} {r.docs_new:>8} "
            f"{r.docs_updated:>8} {r.errors:>8} {r.elapsed_seconds:>7.1f}s"
        )
        totals.docs_fetched += r.docs_fetched
        totals.docs_new += r.docs_new
        totals.docs_updated += r.docs_updated
        totals.errors += r.errors
        totals.elapsed_seconds += r.elapsed_seconds

    print("-" * 60)
    print(
        f"{totals.court_id:<10} {totals.docs_fetched:>8} {totals.docs_new:>8} "
        f"{totals.docs_updated:>8} {totals.errors:>8} {totals.elapsed_seconds:>7.1f}s"
    )
    print("=" * 60)


def resolve_courts(args: argparse.Namespace) -> dict[str, str]:
    """Determine which courts to sync based on CLI flags."""
    court_map = DATA.court_level_map

    if args.all or args.update:
        return dict(court_map)

    if args.scotus:
        return {"scotus": court_map["scotus"]}

    if args.circuits:
        return {
            cid: level for cid, level in court_map.items()
            if level == "circuit" and cid in DATA.circuits
        }

    if args.court:
        court_id = args.court
        if court_id not in court_map:
            print(f"Unknown court ID: {court_id}")
            print(f"Available: {', '.join(court_map.keys())}")
            sys.exit(1)
        return {court_id: court_map[court_id]}

    return {}


async def main(args: argparse.Namespace):
    global _builder

    token = DATA.courtlistener_token
    if not token:
        print("Warning: COURTLISTENER_TOKEN not set. Requests may be rate-limited more aggressively.")

    output_dir = Path(args.output_dir) if args.output_dir else RAW_DIR
    courts = resolve_courts(args)

    if not courts:
        print("No courts selected. Use --all, --scotus, --circuits, or --court <id>.")
        sys.exit(1)

    print(f"Courts to sync: {', '.join(courts.keys())}")
    if args.limit:
        print(f"Limit: {args.limit} docs per court")
    print(f"Output: {output_dir}")
    print()

    async with CourtListenerClient(
        token=token,
        base_url=DATA.courtlistener_base_url,
        rate_limit=DATA.courtlistener_rate_limit,
    ) as client:
        builder = CorpusBuilder(client=client, output_dir=output_dir)
        _builder = builder

        results = []
        for court_id, court_level in courts.items():
            output_file = f"{court_id}_cases.jsonl"

            # Set up progress bar
            pbar = None
            if tqdm and args.limit:
                pbar = tqdm(total=args.limit, desc=court_id, unit="docs")
                builder.set_progress_callback(lambda n, p=pbar: p.update(1))
            elif tqdm:
                pbar = tqdm(desc=court_id, unit="docs")
                builder.set_progress_callback(lambda n, p=pbar: p.update(1))
            else:
                builder.set_progress_callback(
                    lambda n: print(f"\r  {court_id}: {n} docs fetched", end="", flush=True)
                )

            result = await builder.sync_court(
                court_id=court_id,
                court_level=court_level,
                output_file=output_file,
                target_count=args.limit,
            )
            results.append(result)

            if pbar:
                pbar.close()
            else:
                print()  # newline after \r progress

        print_summary(results)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(main(args))
