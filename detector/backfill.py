"""Backfill historical data from Binance REST API with parallel fetching."""

import asyncio
import logging
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from detector.models import Bar
from detector.storage import Storage
from detector.binance_rest import BinanceRestClient

logger = logging.getLogger(__name__)


@dataclass
class BackfillResult:
    """Result of backfilling a single symbol."""
    symbol: str
    bars_count: int
    success: bool
    error: Optional[str] = None


class Backfiller:
    """
    Backfills historical klines data into the database.

    Optimized for parallel fetching with rate limiting:
    - Uses asyncio.Semaphore for concurrent request control
    - Respects Binance rate limits (2400 weight/min)
    - Klines with limit=1000 = 5 weight, so ~480 requests/min max
    - Default: 50 concurrent requests (conservative, ~250 weight/sec max)
    """

    # Rate limiting configuration
    DEFAULT_MAX_CONCURRENT = 50  # Max concurrent API requests
    DEFAULT_BATCH_SIZE = 20  # Symbols per DB flush batch

    # Binance API limits
    MAX_KLINES_PER_REQUEST = 1500  # Binance max limit
    KLINES_WEIGHT_1000 = 5  # Weight for limit=1000
    MAX_WEIGHT_PER_MINUTE = 2400

    def __init__(
        self,
        rest_client: BinanceRestClient,
        storage: Storage,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT
    ):
        self.rest_client = rest_client
        self.storage = storage
        self.max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._progress_lock = asyncio.Lock()
        self._total_bars = 0
        self._completed_symbols = 0

    async def backfill_symbols(
        self,
        symbols: List[str],
        hours: int = 13,
        benchmark_symbol: str = "BTCUSDT"
    ) -> None:
        """
        Backfill historical data for all symbols in PARALLEL.

        BTC (benchmark) is loaded FIRST to ensure it's available for beta/excess return calculations.

        Args:
            symbols: List of symbols to backfill
            hours: Number of hours to backfill (default 13 to ensure 720+ bars)
            benchmark_symbol: Symbol to load first (default BTCUSDT)
        """
        start_time_wall = time.time()

        print(f"Starting backfill for {len(symbols)} symbols, {hours} hours of data")

        # Calculate start time
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        start_time_ms = int(start_time.timestamp() * 1000)

        print(f"Backfill range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

        # Initialize semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._total_bars = 0
        self._completed_symbols = 0

        # STEP 1: Load benchmark (BTC) FIRST - required for beta calculations
        if benchmark_symbol in symbols:
            print(f"Loading benchmark {benchmark_symbol} first...")
            btc_result = await self._fetch_and_store_symbol(
                benchmark_symbol, start_time_ms, hours, len(symbols)
            )
            if btc_result.success:
                print(f"Benchmark {benchmark_symbol} loaded: {btc_result.bars_count} bars")
                await self.storage.flush_all()
            else:
                logger.error(f"Failed to load benchmark {benchmark_symbol}: {btc_result.error}")

        # STEP 2: Load remaining symbols in parallel
        other_symbols = [s for s in symbols if s != benchmark_symbol]

        # Create tasks for remaining symbols
        tasks = [
            self._fetch_and_store_symbol(symbol, start_time_ms, hours, len(symbols))
            for symbol in other_symbols
        ]

        # Execute all tasks in parallel with rate limiting
        results: List[BackfillResult] = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful = 0
        failed_symbols = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task exception: {result}")
            elif isinstance(result, BackfillResult):
                if result.success:
                    successful += 1
                else:
                    failed_symbols.append(result.symbol)

        # Final flush
        await self.storage.flush_all()

        elapsed = time.time() - start_time_wall
        print(f"Backfill complete: {self._total_bars} bars written for {successful}/{len(symbols)} symbols in {elapsed:.1f}s")

        if failed_symbols:
            logger.warning(f"Failed to backfill: {', '.join(failed_symbols[:10])}{'...' if len(failed_symbols) > 10 else ''}")

    async def _fetch_and_store_symbol(
        self,
        symbol: str,
        start_time_ms: int,
        hours: int,
        total_symbols: int
    ) -> BackfillResult:
        """
        Fetch klines for a single symbol with rate limiting and store to DB.
        """
        async with self._semaphore:
            try:
                bars = await self._fetch_klines_for_symbol(symbol, start_time_ms, hours)

                if bars:
                    # Write to database (batch write, will be flushed later)
                    await self.storage.batch_write_bars(bars)

                    # Update progress
                    async with self._progress_lock:
                        self._total_bars += len(bars)
                        self._completed_symbols += 1

                        # Log progress every 10 symbols
                        if self._completed_symbols % 10 == 0 or self._completed_symbols == total_symbols:
                            print(f"Progress: {self._completed_symbols}/{total_symbols} symbols, {self._total_bars} bars")

                    # Periodic flush (every 20 symbols)
                    if self._completed_symbols % self.DEFAULT_BATCH_SIZE == 0:
                        await self.storage.flush_all()

                    return BackfillResult(symbol=symbol, bars_count=len(bars), success=True)
                else:
                    return BackfillResult(symbol=symbol, bars_count=0, success=False, error="No data received")

            except Exception as e:
                logger.error(f"Error backfilling {symbol}: {e}")
                return BackfillResult(symbol=symbol, bars_count=0, success=False, error=str(e))

    async def _fetch_klines_for_symbol(self, symbol: str, start_time_ms: int, hours: int) -> List[Bar]:
        """
        Fetch klines for a single symbol and convert to Bar objects.

        Args:
            symbol: Symbol to fetch
            start_time_ms: Start time in milliseconds
            hours: Number of hours to fetch

        Returns:
            List of Bar objects
        """
        bars = []

        # Calculate how many requests we need (1000 bars per request, 60 bars per hour)
        bars_needed = hours * 60
        num_requests = (bars_needed + 999) // 1000  # Ceiling division

        current_start_time = start_time_ms

        for i in range(num_requests):
            # Fetch klines
            klines = await self.rest_client.get_klines(
                symbol=symbol,
                interval="1m",
                limit=1000,
                start_time=current_start_time
            )

            if not klines:
                logger.debug(f"No klines returned for {symbol} starting at {current_start_time}")
                break

            # Convert klines to Bar objects
            for kline in klines:
                bar = self._kline_to_bar(symbol, kline)
                if bar:
                    bars.append(bar)

            # Update start time for next request (last bar's close time + 1ms)
            if klines:
                last_close_time = klines[-1][6]  # Close time
                current_start_time = last_close_time + 1

            # Minimal delay between pagination requests for same symbol
            if i < num_requests - 1:
                await asyncio.sleep(0.05)

        return bars

    def _kline_to_bar(self, symbol: str, kline: List) -> Optional[Bar]:
        """
        Convert Binance kline data to Bar object.

        Kline format:
        [
            0: Open time (ms)
            1: Open
            2: High
            3: Low
            4: Close
            5: Volume
            6: Close time (ms)
            7: Quote asset volume
            8: Number of trades
            9: Taker buy base asset volume
            10: Taker buy quote asset volume
            11: Ignore
        ]

        Note: Klines provide taker buy volume but not taker sell volume.
        We can calculate taker_sell = volume - taker_buy.
        """
        try:
            ts_minute = int(kline[0])
            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            volume = float(kline[5])
            notional = float(kline[7])  # Quote asset volume
            trades = int(kline[8])
            taker_buy_volume = float(kline[9])

            # Calculate taker sell volume
            taker_sell_volume = volume - taker_buy_volume

            bar = Bar(
                symbol=symbol,
                ts_minute=ts_minute,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                notional=notional,
                trades=trades,
                taker_buy=taker_buy_volume,
                taker_sell=taker_sell_volume,
                funding=None,  # Not available in klines, will be populated by markPrice stream
                oi=None  # Will be populated by OI backfill
            )

            return bar

        except (IndexError, ValueError, TypeError) as e:
            logger.error(f"Error parsing kline for {symbol}: {e}")
            return None

    async def backfill_oi_for_symbols(self, symbols: List[str], hours: int = 13) -> None:
        """
        Backfill historical OI data for all symbols in PARALLEL.

        Fetches OI history from REST API (5m resolution) and updates existing bars.
        Uses forward-fill to map 5m OI data to 1m bars.

        Args:
            symbols: List of symbols to backfill
            hours: Number of hours to backfill (default 13 to match klines backfill)
        """
        start_time_wall = time.time()

        print(f"Starting OI backfill for {len(symbols)} symbols, {hours} hours of data")

        # Calculate how many 5m periods needed
        periods_5m = (hours * 60) // 5  # Convert hours to 5-minute periods
        limit = min(periods_5m, 500)  # Binance API limit is 500

        # Initialize semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._total_bars = 0
        self._completed_symbols = 0

        # Create tasks for all symbols
        tasks = [
            self._fetch_and_update_oi_symbol(symbol, limit, len(symbols))
            for symbol in symbols
        ]

        # Execute all tasks in parallel
        results: List[BackfillResult] = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful = 0
        failed_symbols = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"OI task exception: {result}")
            elif isinstance(result, BackfillResult):
                if result.success:
                    successful += 1
                else:
                    failed_symbols.append(result.symbol)

        elapsed = time.time() - start_time_wall
        print(f"OI backfill complete: {self._total_bars} bars updated for {successful}/{len(symbols)} symbols in {elapsed:.1f}s")

        if failed_symbols:
            logger.warning(f"Failed to backfill OI for: {', '.join(failed_symbols[:10])}{'...' if len(failed_symbols) > 10 else ''}")

    async def _fetch_and_update_oi_symbol(
        self,
        symbol: str,
        limit: int,
        total_symbols: int
    ) -> BackfillResult:
        """
        Fetch OI for a single symbol with rate limiting and update bars.
        """
        async with self._semaphore:
            try:
                # Fetch OI history from REST API
                oi_data = await self.rest_client.get_oi_history(
                    symbol=symbol,
                    period="5m",
                    limit=limit
                )

                if not oi_data:
                    return BackfillResult(symbol=symbol, bars_count=0, success=False, error="No OI data (may need API key)")

                # Convert OI data to 1m bar updates
                bars_updated = await self._update_bars_with_oi(symbol, oi_data)

                # Update progress
                async with self._progress_lock:
                    self._total_bars += bars_updated
                    self._completed_symbols += 1

                    # Log progress every 20 symbols
                    if self._completed_symbols % 20 == 0 or self._completed_symbols == total_symbols:
                        print(f"OI Progress: {self._completed_symbols}/{total_symbols} symbols, {self._total_bars} bars updated")

                return BackfillResult(symbol=symbol, bars_count=bars_updated, success=True)

            except Exception as e:
                logger.error(f"Error backfilling OI for {symbol}: {e}")
                return BackfillResult(symbol=symbol, bars_count=0, success=False, error=str(e))

    async def _update_bars_with_oi(self, symbol: str, oi_data: List[Dict]) -> int:
        """
        Update existing bars with OI data using forward-fill strategy.

        OI data is at 5m resolution, bars are at 1m resolution.
        Forward-fill: each 5m OI value is applied to the next 5 consecutive 1m bars.

        Args:
            symbol: Symbol to update
            oi_data: List of OI history entries from Binance API
                     Format: [{'timestamp': ms, 'sumOpenInterest': float, ...}, ...]

        Returns:
            Number of bars updated
        """
        if not oi_data:
            return 0

        # Build OI lookup map: timestamp -> oi_value
        # Each OI timestamp represents the START of a 5m period
        oi_map = {}
        for entry in oi_data:
            ts_ms = int(entry['timestamp'])
            oi_value = float(entry.get('sumOpenInterest', 0))
            oi_map[ts_ms] = oi_value

        # Fetch existing bars for this symbol in the same time range
        # Get time range from OI data
        if not oi_map:
            return 0

        start_ts = min(oi_map.keys())
        end_ts = max(oi_map.keys()) + 5 * 60 * 1000  # Add 5m to cover last period

        # Load bars from database in this range
        bars_to_update = []

        try:
            async with self.storage.db.execute("""
                SELECT symbol, ts_minute FROM bars_1m
                WHERE symbol = ? AND ts_minute >= ? AND ts_minute < ?
                ORDER BY ts_minute ASC
            """, (symbol, start_ts, end_ts)) as cursor:
                rows = await cursor.fetchall()

            # For each 1m bar, find the corresponding 5m OI value using forward-fill
            bars_updated = 0

            for row in rows:
                bar_symbol = row[0]
                bar_ts = row[1]

                # Find the OI value for this bar
                # Round down to nearest 5m boundary to find the OI period
                oi_period_start = (bar_ts // (5 * 60 * 1000)) * (5 * 60 * 1000)

                oi_value = oi_map.get(oi_period_start)

                if oi_value is not None:
                    # Update this bar with OI value
                    bars_to_update.append((oi_value, bar_symbol, bar_ts))
                    bars_updated += 1

            # Batch update bars
            if bars_to_update:
                await self.storage.db.executemany("""
                    UPDATE bars_1m SET oi = ? WHERE symbol = ? AND ts_minute = ?
                """, bars_to_update)
                await self.storage.db.commit()

            return bars_updated

        except Exception as e:
            logger.error(f"Error updating bars with OI for {symbol}: {e}")
            return 0


# Legacy function for backwards compatibility
async def backfill_symbols_sequential(
    rest_client: BinanceRestClient,
    storage: Storage,
    symbols: List[str],
    hours: int = 13
) -> None:
    """
    Legacy sequential backfill (for debugging or rate limit issues).

    Use Backfiller class with parallel fetching for better performance.
    """
    backfiller = Backfiller(rest_client, storage, max_concurrent=1)
    await backfiller.backfill_symbols(symbols, hours)
