"""Tests for bar aggregator."""

import pytest
import asyncio
from detector.aggregator import BarAggregator
from detector.models import Bar, StreamType


@pytest.mark.asyncio
async def test_agg_trade_bar_aggregation():
    """Test 1m bar aggregation from synthetic aggTrade events."""
    tick_queue = asyncio.Queue()
    bar_queue = asyncio.Queue()

    aggregator = BarAggregator(tick_queue, bar_queue, clock_tolerance_sec=2)

    # Simulate aggTrade ticks for BTCUSDT
    trades = [
        {'s': 'BTCUSDT', 'p': '50000', 'q': '0.1', 'm': False},  # Taker buy
        {'s': 'BTCUSDT', 'p': '50100', 'q': '0.2', 'm': True},   # Taker sell
        {'s': 'BTCUSDT', 'p': '50050', 'q': '0.15', 'm': False}, # Taker buy
    ]

    for trade in trades:
        await tick_queue.put(('tick', StreamType.AGG_TRADE, trade))

    # Trigger minute boundary
    await tick_queue.put(('minute_boundary', None, 1000 * 60 * 1000))

    # Process ticks
    for _ in range(len(trades) + 1):
        event_type, stream_type, data = await tick_queue.get()
        if event_type == 'minute_boundary':
            await aggregator._close_bars(data)
        else:
            aggregator._update_bar(stream_type, data)

    # Check bar
    bar = await bar_queue.get()

    assert bar.symbol == 'BTCUSDT'
    assert bar.open == 50000
    assert bar.high == 50100
    assert bar.low == 50000
    assert bar.close == 50050
    assert bar.volume == pytest.approx(0.45)
    assert bar.taker_buy == pytest.approx(0.25)  # First and third trades
    assert bar.taker_sell == pytest.approx(0.2)  # Second trade


@pytest.mark.asyncio
async def test_deduplication():
    """Test trade ID deduplication on reconnect."""
    # This would test the WebSocket client's deduplication logic
    # For now, placeholder
    pass
