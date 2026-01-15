"""Backfill historical data from Binance REST API."""

import asyncio
import logging
from typing import List, Dict
from datetime import datetime, timedelta
from detector.models import Bar
from detector.storage import Storage
from detector.binance_rest import BinanceRestClient

logger = logging.getLogger(__name__)


class Backfiller:
    """Backfills historical klines data into the database."""

    def __init__(self, rest_client: BinanceRestClient, storage: Storage):
        self.rest_client = rest_client
        self.storage = storage

    async def backfill_symbols(self, symbols: List[str], hours: int = 13) -> None:
        """
        Backfill historical data for all symbols.

        Args:
            symbols: List of symbols to backfill
            hours: Number of hours to backfill (default 13 to ensure 720+ bars)
        """
        logger.info(f"Starting backfill for {len(symbols)} symbols, {hours} hours of data")

        # Calculate start time
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        start_time_ms = int(start_time.timestamp() * 1000)

        logger.info(f"Backfill range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

        total_bars = 0
        failed_symbols = []

        for symbol in symbols:
            try:
                bars = await self._fetch_klines_for_symbol(symbol, start_time_ms, hours)

                if bars:
                    # Write to database
                    await self.storage.batch_write_bars(bars)
                    await self.storage.flush_all()

                    total_bars += len(bars)
                    logger.info(f"Backfilled {len(bars)} bars for {symbol} (total: {total_bars})")
                else:
                    logger.warning(f"No data received for {symbol}")
                    failed_symbols.append(symbol)

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error backfilling {symbol}: {e}")
                failed_symbols.append(symbol)

        logger.info(f"Backfill complete: {total_bars} bars written for {len(symbols) - len(failed_symbols)}/{len(symbols)} symbols")

        if failed_symbols:
            logger.warning(f"Failed to backfill: {', '.join(failed_symbols)}")

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
                logger.warning(f"No klines returned for {symbol} starting at {current_start_time}")
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

            logger.debug(f"Fetched {len(klines)} klines for {symbol} (request {i+1}/{num_requests})")

            # Small delay between requests
            await asyncio.sleep(0.2)

        return bars

    def _kline_to_bar(self, symbol: str, kline: List) -> Bar:
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
                liq_notional=0.0,  # Not available in klines
                liq_count=0,
                mid=None,  # Not available in klines
                spread_bps=None,
                mark=None,
                funding=None,
                next_funding_ts=None,
                oi=None  # Will be populated by OI backfill
            )

            return bar

        except (IndexError, ValueError, TypeError) as e:
            logger.error(f"Error parsing kline for {symbol}: {e}")
            return None

    async def backfill_oi_for_symbols(self, symbols: List[str], hours: int = 13) -> None:
        """
        Backfill historical OI data for all symbols.

        Fetches OI history from REST API (5m resolution) and updates existing bars.
        Uses forward-fill to map 5m OI data to 1m bars.

        Args:
            symbols: List of symbols to backfill
            hours: Number of hours to backfill (default 13 to match klines backfill)
        """
        logger.info(f"Starting OI backfill for {len(symbols)} symbols, {hours} hours of data")

        # Calculate how many 5m periods needed
        periods_5m = (hours * 60) // 5  # Convert hours to 5-minute periods
        limit = min(periods_5m, 500)  # Binance API limit is 500

        total_bars_updated = 0
        failed_symbols = []

        for symbol in symbols:
            try:
                # Fetch OI history from REST API
                oi_data = await self.rest_client.get_oi_history(
                    symbol=symbol,
                    period="5m",
                    limit=limit
                )

                if not oi_data:
                    logger.warning(f"No OI data received for {symbol} (may not have API key)")
                    failed_symbols.append(symbol)
                    continue

                # Convert OI data to 1m bar updates
                bars_updated = await self._update_bars_with_oi(symbol, oi_data)

                total_bars_updated += bars_updated
                logger.info(f"Updated {bars_updated} bars with OI for {symbol}")

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error backfilling OI for {symbol}: {e}")
                failed_symbols.append(symbol)

        logger.info(f"OI backfill complete: {total_bars_updated} bars updated for {len(symbols) - len(failed_symbols)}/{len(symbols)} symbols")

        if failed_symbols:
            logger.warning(f"Failed to backfill OI for: {', '.join(failed_symbols)}")

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
