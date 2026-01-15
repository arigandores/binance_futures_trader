"""Test script to verify features backfill."""

import asyncio
import logging
import sys
from detector.config import Config
from detector.storage import Storage
from detector.features import FeatureCalculator
from detector.binance_rest import BinanceRestClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Test features backfill."""
    logger.info("Loading configuration...")
    config = Config.from_yaml("config.yaml")

    logger.info("Initializing storage...")
    storage = Storage(config.storage.sqlite_path, config.storage.wal_mode)
    await storage.init_db()

    logger.info("Initializing REST client...")
    rest_client = BinanceRestClient(
        api_key=config.api.key or "",
        api_secret=config.api.secret or ""
    )

    logger.info("Creating feature calculator...")
    feature_calc = FeatureCalculator(
        bar_queue=asyncio.Queue(),
        feature_queue=asyncio.Queue(),
        storage=storage,
        rest_client=rest_client,
        config=config
    )

    logger.info("Starting features backfill...")
    await feature_calc.backfill()

    logger.info("Backfill complete! Checking results...")

    # Check how many features were written
    async with storage.db.execute("SELECT COUNT(*) FROM features") as cursor:
        row = await cursor.fetchone()
        total_features = row[0]

    logger.info(f"Total features in database: {total_features}")

    # Check features per symbol
    async with storage.db.execute("""
        SELECT symbol, COUNT(*) as count
        FROM features
        GROUP BY symbol
        ORDER BY count DESC
        LIMIT 10
    """) as cursor:
        rows = await cursor.fetchall()

    logger.info("Top 10 symbols by feature count:")
    for symbol, count in rows:
        logger.info(f"  {symbol}: {count} features")

    # Check bars count for comparison
    async with storage.db.execute("SELECT COUNT(*) FROM bars_1m") as cursor:
        row = await cursor.fetchone()
        total_bars = row[0]

    logger.info(f"Total bars in database: {total_bars}")
    logger.info(f"Features coverage: {total_features / total_bars * 100:.1f}%")

    await storage.close()
    await rest_client.close()


if __name__ == "__main__":
    asyncio.run(main())
