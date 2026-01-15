"""Bar aggregator - converts tick stream to 1-minute OHLCV bars."""

import asyncio
import logging
import time
from typing import Dict, List, Optional
from detector.models import Bar, StreamType

logger = logging.getLogger(__name__)


class BarAggregator:
    """
    Aggregates WebSocket ticks into 1-minute bars.

    Handles:
    - OHLCV aggregation from aggTrade
    - Taker buy/sell split tracking
    - Funding rate from markPrice stream
    - Minute boundary detection and bar closing
    """

    def __init__(
        self,
        tick_queue: asyncio.Queue,
        bar_queue: asyncio.Queue,
        symbols: List[str],
        rest_client: Optional['BinanceRestClient'] = None,
        clock_tolerance_sec: int = 2,
        extra_bar_queues: Optional[List[asyncio.Queue]] = None
    ):
        self.tick_queue = tick_queue
        self.bar_queue = bar_queue
        self.extra_bar_queues = extra_bar_queues or []  # Additional queues for bar broadcast
        self.symbols_set = set(symbols)  # For fast lookup
        self.rest_client = rest_client
        self.clock_tolerance_sec = clock_tolerance_sec

        # Current incomplete bars (one per symbol)
        self.current_bars: Dict[str, Bar] = {}
        self.current_minute: int = 0

    async def run(self) -> None:
        """Main loop: consume ticks and trigger minute boundaries."""
        # Start minute boundary timer
        asyncio.create_task(self._minute_boundary_timer())

        logger.info("Bar aggregator started")

        tick_count = 0
        last_log_time = 0

        while True:
            try:
                event_type, stream_type, data = await self.tick_queue.get()

                if event_type == 'minute_boundary':
                    await self._close_bars(data)  # data is timestamp
                else:
                    self._update_bar(stream_type, data)
                    tick_count += 1

                    # Log activity every 100 ticks
                    current_time = int(time.time())
                    if tick_count % 100 == 0 or current_time - last_log_time >= 10:
                        logger.info(f"Processing ticks: {tick_count} received, {len(self.current_bars)} symbols active")
                        last_log_time = current_time

            except Exception as e:
                logger.error(f"Error in bar aggregator: {e}")

    async def _minute_boundary_timer(self) -> None:
        """Triggers minute boundaries on :00 seconds."""
        while True:
            now = int(time.time())
            next_minute = (now // 60 + 1) * 60
            sleep_time = next_minute - now + 0.1  # Small buffer

            await asyncio.sleep(sleep_time)

            # Emit minute boundary event
            timestamp_ms = next_minute * 1000
            await self.tick_queue.put(('minute_boundary', None, timestamp_ms))

    async def _close_bars(self, timestamp_ms: int) -> None:
        """Close all current bars and emit to bar_queue."""
        if not self.current_bars:
            return

        closed_count = 0
        bar_summary = []

        for symbol, bar in self.current_bars.items():
            # Set bar timestamp to minute boundary
            bar.ts_minute = timestamp_ms - 60_000  # Bar represents previous minute

            # Enrich bar with OI from REST client cache (if available)
            if self.rest_client:
                bar.oi = self.rest_client.get_latest_oi(symbol)

            # Emit to queue(s)
            await self.bar_queue.put(bar)

            # Broadcast to extra queues (for position manager, etc.)
            for queue in self.extra_bar_queues:
                try:
                    await queue.put(bar)
                except Exception as e:
                    logger.error(f"Error broadcasting bar to extra queue: {e}")

            closed_count += 1

            # Collect summary info
            bar_summary.append(f"{symbol}(v:{bar.volume:.2f}, t:{bar.trades})")

        logger.info(f"Closed {closed_count} bars: {', '.join(bar_summary[:3])}{'...' if len(bar_summary) > 3 else ''}")

        # Clear current bars
        self.current_bars.clear()
        self.current_minute = timestamp_ms

    def _update_bar(self, stream_type: StreamType, data: Dict) -> None:
        """Update current bar based on tick data."""
        if stream_type == StreamType.AGG_TRADE:
            self._update_from_agg_trade(data)
        elif stream_type == StreamType.MARK_PRICE:
            self._update_from_mark_price(data)

    def _update_from_agg_trade(self, data: Dict) -> None:
        """Update bar from aggTrade data."""
        symbol = data.get('s')
        if not symbol or symbol not in self.symbols_set:
            return

        price = float(data.get('p', 0))
        qty = float(data.get('q', 0))
        is_buyer_maker = data.get('m', False)

        # Get or create bar
        bar = self.current_bars.get(symbol)
        if not bar:
            bar = Bar(symbol=symbol, ts_minute=0)
            bar.open = price
            bar.high = price
            bar.low = price
            self.current_bars[symbol] = bar

        # Update OHLC
        bar.close = price
        if price > bar.high:
            bar.high = price
        if price < bar.low:
            bar.low = price

        # Update volume
        bar.volume += qty
        bar.notional += price * qty
        bar.trades += 1

        # Update taker buy/sell
        if is_buyer_maker:
            # Seller is the taker (aggressive seller)
            bar.taker_sell += qty
        else:
            # Buyer is the taker (aggressive buyer)
            bar.taker_buy += qty

    def _update_from_mark_price(self, data: Dict) -> None:
        """Update bar from markPrice data."""
        symbol = data.get('s')
        if not symbol or symbol not in self.symbols_set:
            return

        funding_rate = float(data.get('r', 0))

        # Get or create bar
        bar = self.current_bars.get(symbol)
        if not bar:
            bar = Bar(symbol=symbol, ts_minute=0)
            # Initialize OHLC from mark price if this is the first data for this bar
            mark_price = float(data.get('p', 0))
            if mark_price > 0:
                bar.open = mark_price
                bar.high = mark_price
                bar.low = mark_price
                bar.close = mark_price
            self.current_bars[symbol] = bar

        bar.funding = funding_rate
