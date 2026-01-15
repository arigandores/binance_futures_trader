"""Main CLI orchestrator for the Sector Shot Detector."""

import asyncio
import click
import structlog
from detector.config import Config
from detector.storage import Storage
from detector.binance_ws import BinanceWebSocketClient
from detector.binance_rest import BinanceRestClient
from detector.aggregator import BarAggregator
from detector.features import FeatureCalculator
from detector.detector import AnomalyDetector
from detector.alerts import AlertDispatcher
from detector.position_manager import PositionManager
from detector.report import ReportGenerator
from detector.backfill import Backfiller
from detector.utils import setup_logging

logger = structlog.get_logger()


class SectorShotDetector:
    """
    Main orchestrator for the Sector Shot Detector.

    Coordinates all components:
    - WebSocket client
    - REST API client
    - Bar aggregator
    - Feature calculator
    - Anomaly detector
    - Alert dispatcher
    - Storage
    """

    def __init__(self, config: Config):
        self.config = config
        self.skip_backfill = False  # Can be set to True to skip auto-backfill

        # Initialize storage
        self.storage = Storage(
            config.storage.sqlite_path,
            wal_mode=config.storage.wal_mode,
            batch_interval=config.storage.batch_write_interval_sec
        )

        # Queues
        self.tick_queue = asyncio.Queue(maxsize=10000)
        self.bar_queue = asyncio.Queue(maxsize=1000)
        self.bar_queue_pm = asyncio.Queue(maxsize=1000)  # Separate queue for position manager
        self.feature_queue = asyncio.Queue(maxsize=1000)
        self.feature_queue_pm = asyncio.Queue(maxsize=1000)  # For position manager exit checks
        self.event_queue = asyncio.Queue(maxsize=100)
        self.event_queue_pm = asyncio.Queue(maxsize=100)  # For position manager to open positions

        # Components
        self.ws_client = BinanceWebSocketClient(
            config.universe.all_symbols,
            self.tick_queue,
            config.runtime.ws_reconnect_backoff_sec
        )

        self.rest_client = BinanceRestClient(
            config.api.key if config.api.key else None,
            config.api.secret if config.api.secret else None
        )

        self.aggregator = BarAggregator(
            self.tick_queue,
            self.bar_queue,
            config.universe.all_symbols,
            self.rest_client,
            config.runtime.clock_skew_tolerance_sec,
            extra_bar_queues=[self.bar_queue_pm]  # Broadcast bars to position manager
        )

        # Prepare extra queues for broadcasting (if position manager enabled)
        extra_feature_queues = [self.feature_queue_pm] if config.position_management.enabled else []
        extra_event_queues = [self.event_queue_pm] if config.position_management.enabled else []

        self.features = FeatureCalculator(
            self.bar_queue,
            self.feature_queue,
            self.storage,
            self.rest_client,
            config,
            extra_feature_queues=extra_feature_queues
        )

        self.detector = AnomalyDetector(
            self.feature_queue,
            self.event_queue,
            self.rest_client,
            self.storage,
            config,
            extra_event_queues=extra_event_queues
        )

        self.alerts = AlertDispatcher(self.event_queue, config)

        # Position manager (if enabled)
        self.position_manager = None
        if config.position_management.enabled:
            self.position_manager = PositionManager(
                self.event_queue_pm,
                self.feature_queue_pm,
                self.bar_queue_pm,
                self.storage,
                config
            )

    async def _check_and_backfill(self) -> None:
        """Check if database has sufficient and recent historical data, backfill if needed."""
        if self.skip_backfill:
            logger.info("Skipping automatic backfill (--skip-backfill flag set)")
            return

        logger.info("Checking database for historical data...")

        # Count bars in database for benchmark symbol
        benchmark = self.config.universe.benchmark_symbol
        bars = await self.storage.get_recent_bars(benchmark, limit=1000)

        bars_count = len(bars)
        needed_bars = self.config.windows.zscore_lookback_bars  # Usually 720

        logger.info(f"Found {bars_count} bars in database (need {needed_bars} for stable z-scores)")

        # Check if data is recent (not stale)
        should_backfill = False

        if bars_count < needed_bars:
            logger.warning(f"Insufficient data in database ({bars_count}/{needed_bars} bars)")
            should_backfill = True
        elif bars_count > 0:
            # Check if most recent bar is fresh (within last 2 hours)
            import time
            most_recent_bar = bars[-1]
            current_time_ms = int(time.time() * 1000)
            time_since_last_bar_minutes = (current_time_ms - most_recent_bar.ts_minute) / 1000 / 60

            logger.info(f"Most recent bar is {time_since_last_bar_minutes:.1f} minutes old")

            if time_since_last_bar_minutes > 120:  # More than 2 hours old
                logger.warning(f"Data is stale (last bar {time_since_last_bar_minutes:.1f} minutes old)")
                logger.info("Clearing old data and backfilling fresh data to ensure z-scores are based on recent market conditions")
                # Clear old data to avoid gaps
                await self.storage.clear_bars_and_features()
                should_backfill = True
            else:
                logger.info("Data is recent and sufficient, skipping backfill")

        if not should_backfill:
            # Even if we skip klines backfill, check if we should backfill OI
            # (In case bars exist but OI is missing)
            await self._backfill_oi_if_needed(hours_needed=13)
            return

        # Need to backfill klines
        hours_needed = (needed_bars + 60) // 60  # Convert bars to hours, add 1 hour buffer
        logger.info(f"Starting automatic backfill of {hours_needed} hours of historical data...")
        logger.info("This will take 1-2 minutes. Please wait...")

        try:
            backfiller = Backfiller(self.rest_client, self.storage)

            # Step 1: Backfill klines
            await backfiller.backfill_symbols(self.config.universe.all_symbols, hours=hours_needed)
            logger.info("Klines backfill complete!")

            # Step 2: Backfill OI data
            await self._backfill_oi_if_needed(hours_needed)

            logger.info("Automatic backfill complete!")
        except Exception as e:
            logger.error(f"Backfill failed: {e}")
            logger.warning("Continuing with limited data. Z-scores may be unstable initially.")

    async def _backfill_oi_if_needed(self, hours_needed: int) -> None:
        """
        Backfill OI data if API key is available.

        This is a separate step because:
        1. OI backfill is optional (requires API key)
        2. May be called even when klines are already present
        """
        if not self.rest_client.api_key:
            logger.info("Skipping OI backfill (no API key)")
            return

        logger.info(f"Starting OI backfill for {hours_needed} hours...")

        try:
            backfiller = Backfiller(self.rest_client, self.storage)
            await backfiller.backfill_oi_for_symbols(
                self.config.universe.all_symbols,
                hours=hours_needed
            )
            logger.info("OI backfill complete!")
        except Exception as e:
            logger.error(f"OI backfill failed: {e}")
            logger.warning("Continuing without OI data. Events will have z_oi_delta_1h = None")

    async def run(self) -> None:
        """Main entry point - start all components."""
        logger.info("Starting Sector Shot Detector...")

        # Initialize database
        await self.storage.init_db()

        # Check if we need to backfill historical data
        await self._check_and_backfill()

        # Backfill rolling windows from DB
        logger.info("Loading rolling windows from database...")
        await self.features.backfill()

        # Start all background tasks
        tasks = [
            asyncio.create_task(self.ws_client.connect(), name="ws_client"),
            asyncio.create_task(self.rest_client.poll_loop(
                self.config.universe.all_symbols,
                self.config.runtime.rest_poll_sec
            ), name="rest_poll"),
            asyncio.create_task(self.aggregator.run(), name="aggregator"),
            asyncio.create_task(self.features.run(), name="features"),
            asyncio.create_task(self.detector.run(), name="detector"),
            asyncio.create_task(self.alerts.run(), name="alerts"),
            asyncio.create_task(self._storage_flush_loop(), name="storage_flush"),
            asyncio.create_task(self._health_monitor(), name="health"),
        ]

        # Add position manager if enabled
        if self.position_manager:
            tasks.append(asyncio.create_task(self.position_manager.run(), name="position_manager"))
            logger.info("Position manager enabled")

        logger.info("All components started. Press Ctrl+C to stop.")

        # Wait for shutdown signal
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutdown signal received, stopping...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        # Cleanup
        await self.storage.close()
        await self.rest_client.close()
        await self.alerts.close()

        if self.position_manager:
            await self.position_manager.close()

        logger.info("Shutdown complete")

    async def _storage_flush_loop(self) -> None:
        """Flush storage buffers periodically."""
        while True:
            await asyncio.sleep(self.config.storage.batch_write_interval_sec)

            try:
                await self.storage.flush_all()
            except Exception as e:
                logger.error(f"Error flushing storage: {e}")

    async def _health_monitor(self) -> None:
        """Monitor queue depths and connection status."""
        while True:
            await asyncio.sleep(60)

            try:
                health_info = {
                    "tick_q": self.tick_queue.qsize(),
                    "bar_q": self.bar_queue.qsize(),
                    "feature_q": self.feature_queue.qsize(),
                    "event_q": self.event_queue.qsize(),
                    "ws_connected": self.ws_client.is_connected
                }

                if self.position_manager:
                    health_info["open_positions"] = len(self.position_manager.open_positions)
                    health_info["bar_q_pm"] = self.bar_queue_pm.qsize()
                    health_info["feature_q_pm"] = self.feature_queue_pm.qsize()

                logger.info("Health check", **health_info)

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")


# CLI Commands

@click.group()
def cli():
    """Binance Sector Shot Detector - Real-time anomaly detection for coordinated sector movements."""
    pass


@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
@click.option('--skip-backfill', is_flag=True, help='Skip automatic backfill on startup')
def run(config: str, skip_backfill: bool):
    """Run the detector service."""
    try:
        cfg = Config.from_yaml(config)
        cfg.validate()

        # Setup logging
        setup_logging(cfg.runtime.log_level)

        logger.info("Configuration loaded", config_path=config)

        # Run detector
        detector = SectorShotDetector(cfg)
        detector.skip_backfill = skip_backfill
        asyncio.run(detector.run())

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Please create config.yaml (copy from config.example.yaml)", err=True)
        exit(1)
    except Exception as e:
        click.echo(f"Fatal error: {e}", err=True)
        logger.exception("Fatal error")
        exit(1)


@cli.command()
@click.option('--hours', default=13, help='Number of hours to backfill (default: 13 for 720+ bars)')
@click.option('--config', default='config.yaml', help='Path to config file')
def backfill(hours: int, config: str):
    """
    Backfill historical data from Binance REST API.

    Fetches 1-minute klines (OHLCV + taker buy/sell volume) for all configured symbols.
    Default 13 hours provides 780 bars, ensuring enough data for z-score calculations (need 720).

    Note: Klines provide volume splits but not liquidation data.
    """
    try:
        cfg = Config.from_yaml(config)
        setup_logging(cfg.runtime.log_level)

        async def run_backfill():
            # Initialize components
            storage = Storage(cfg.storage.sqlite_path, wal_mode=cfg.storage.wal_mode)
            await storage.init_db()

            rest_client = BinanceRestClient(
                api_key=cfg.api.key if cfg.api.key else None,
                api_secret=cfg.api.secret if cfg.api.secret else None
            )

            backfiller = Backfiller(rest_client, storage)

            # Run backfill
            click.echo(f"Backfilling last {hours} hours for {len(cfg.universe.all_symbols)} symbols...")
            click.echo("This may take 1-2 minutes depending on symbols and network speed.\n")

            await backfiller.backfill_symbols(cfg.universe.all_symbols, hours)

            # Cleanup
            await rest_client.close()
            await storage.close()

            click.echo("\nBackfill complete! You can now run the detector with sufficient historical data.")
            click.echo("Run: poetry run python -m detector run --config config.yaml")

        asyncio.run(run_backfill())

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Backfill failed")
        exit(1)


@cli.command()
@click.option('--since', default='24h', help='Time range (e.g., 24h, 7d)')
@click.option('--output', default='report.json', help='Output file path')
@click.option('--config', default='config.yaml', help='Path to config file')
def report(since: str, output: str, config: str):
    """Generate JSON report of detected events."""
    try:
        cfg = Config.from_yaml(config)
        setup_logging(cfg.runtime.log_level)

        async def generate():
            storage = Storage(cfg.storage.sqlite_path)
            await storage.init_db()

            generator = ReportGenerator(storage)
            report_data = await generator.generate_report(since, output)

            generator.print_summary(report_data)

            await storage.close()

        asyncio.run(generate())

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        exit(1)


@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
def db_migrate(config: str):
    """Initialize database schema."""
    try:
        cfg = Config.from_yaml(config)
        setup_logging(cfg.runtime.log_level)

        async def init():
            storage = Storage(cfg.storage.sqlite_path, wal_mode=cfg.storage.wal_mode)
            await storage.init_db()
            await storage.close()
            logger.info("Database schema created successfully")

        asyncio.run(init())

        click.echo(f"Database initialized at {cfg.storage.sqlite_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        exit(1)


if __name__ == '__main__':
    cli()
