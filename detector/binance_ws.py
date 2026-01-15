"""Binance WebSocket client with automatic reconnection."""

import asyncio
import json
import logging
import time
import websockets
from typing import Dict, List, Optional
from detector.models import StreamType

logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """
    WebSocket client for Binance USD-M Futures streams.

    Handles:
    - Combined stream subscription (aggTrade, markPrice)
    - Automatic reconnection with exponential backoff
    - Trade ID deduplication
    - Emits ticks to asyncio.Queue
    """

    BASE_URL = "wss://fstream.binance.com/stream"

    def __init__(
        self,
        symbols: List[str],
        tick_queue: asyncio.Queue,
        reconnect_delays: List[int] = None
    ):
        self.symbols = symbols
        self.tick_queue = tick_queue
        self.reconnect_delays = reconnect_delays or [1, 2, 5, 10, 30]

        # State
        self.last_trade_id: Dict[str, int] = {}
        self.is_connected = False
        self.ws: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self) -> None:
        """Main connection loop with automatic reconnection."""
        delay_index = 0

        while True:
            try:
                url = self._build_stream_url()
                logger.info(f"Connecting to Binance WebSocket: {url[:100]}...")

                async with websockets.connect(url) as ws:
                    self.ws = ws
                    self.is_connected = True
                    delay_index = 0  # Reset backoff on successful connection
                    logger.info("WebSocket connected successfully")

                    # Listen for messages
                    await self._listen()

            except websockets.exceptions.WebSocketException as e:
                self.is_connected = False
                delay = self.reconnect_delays[min(delay_index, len(self.reconnect_delays) - 1)]
                logger.warning(f"WebSocket disconnected: {e}. Reconnecting in {delay}s...")
                await asyncio.sleep(delay)
                delay_index += 1

            except Exception as e:
                self.is_connected = False
                logger.error(f"Unexpected error in WebSocket: {e}")
                delay = self.reconnect_delays[min(delay_index, len(self.reconnect_delays) - 1)]
                await asyncio.sleep(delay)
                delay_index += 1

    def _build_stream_url(self) -> str:
        """Build combined stream URL for all symbols."""
        streams = []

        # Add aggTrade and markPrice for each symbol
        for symbol in self.symbols:
            symbol_lower = symbol.lower()
            streams.append(f"{symbol_lower}@aggTrade")
            streams.append(f"{symbol_lower}@markPrice@1s")

        stream_string = "/".join(streams)
        return f"{self.BASE_URL}?streams={stream_string}"

    async def _listen(self) -> None:
        """Listen for WebSocket messages and emit to queue."""
        if not self.ws:
            return

        msg_count = 0
        last_log_time = 0

        async for message in self.ws:
            try:
                data = json.loads(message)
                await self._process_message(data)
                msg_count += 1

                # Log activity every 1000 messages
                current_time = int(time.time())
                if msg_count % 1000 == 0 or (msg_count % 100 == 0 and current_time - last_log_time >= 10):
                    logger.info(f"WebSocket active: {msg_count} messages received")
                    last_log_time = current_time

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse WebSocket message: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def _process_message(self, msg: Dict) -> None:
        """Process incoming WebSocket message and emit to queue."""
        # Combined stream format: {"stream": "...", "data": {...}}
        stream = msg.get('stream', '')
        data = msg.get('data', {})

        if not stream or not data:
            # Single stream format or invalid
            return

        # Detect stream type
        stream_type = self._detect_stream_type(stream)
        if not stream_type:
            return

        # Handle aggTrade deduplication
        if stream_type == StreamType.AGG_TRADE:
            symbol = data.get('s')
            trade_id = data.get('a')

            if symbol and trade_id is not None:
                last_id = self.last_trade_id.get(symbol, 0)
                if trade_id <= last_id:
                    # Duplicate trade, skip
                    return

                self.last_trade_id[symbol] = trade_id

        # Emit tick to queue
        try:
            tick_data = ('tick', stream_type, data)
            await self.tick_queue.put(tick_data)
        except asyncio.QueueFull:
            logger.warning("Tick queue full, dropping message")

    def _detect_stream_type(self, stream: str) -> Optional[StreamType]:
        """Detect stream type from stream name."""
        if 'aggTrade' in stream:
            return StreamType.AGG_TRADE
        elif 'markPrice' in stream:
            return StreamType.MARK_PRICE
        else:
            return None

    async def close(self) -> None:
        """Close WebSocket connection."""
        if self.ws:
            await self.ws.close()
            self.is_connected = False
            logger.info("WebSocket connection closed")
