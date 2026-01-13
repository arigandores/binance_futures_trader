"""Backfill historical data from Binance REST API."""

import asyncio
import logging
from typing import List
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
                next_funding_ts=None
            )

            return bar

        except (IndexError, ValueError, TypeError) as e:
            logger.error(f"Error parsing kline for {symbol}: {e}")
            return None
