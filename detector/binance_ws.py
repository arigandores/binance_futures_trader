"""Binance WebSocket client with automatic reconnection.

Supports 500+ symbols via dynamic SUBSCRIBE method (not URL streams).
Binance limits: 200 streams per connection, 1024 total connections.
"""

import asyncio
import logging
import time
import websockets

# Performance: orjson is 4-6x faster than standard json
try:
    import orjson
    def json_loads(data):
        return orjson.loads(data)
    def json_dumps(data):
        return orjson.dumps(data).decode('utf-8')
except ImportError:
    import json
    def json_loads(data):
        return json.loads(data)
    def json_dumps(data):
        return json.dumps(data)
from typing import Dict, List, Optional
from detector.models import StreamType

logger = logging.getLogger(__name__)

# Binance WebSocket limits
MAX_STREAMS_PER_CONNECTION = 200  # Binance limit: 200 streams per connection
MAX_SUBSCRIBE_BATCH = 200  # Max streams per SUBSCRIBE request


class BinanceWebSocketClient:
    """
    WebSocket client for Binance USD-M Futures streams.

    Handles:
    - Dynamic subscription via SUBSCRIBE method (no URL length limits)
    - Multiple connections for 500+ symbols (200 streams each)
    - Automatic reconnection with exponential backoff
    - Trade ID deduplication
    - Emits ticks to asyncio.Queue
    """

    BASE_URL = "wss://fstream.binance.com/ws"  # Use /ws endpoint for SUBSCRIBE method

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
        self.connections: List[websockets.WebSocketClientProtocol] = []
        self._subscribe_id = 1  # Request ID counter

    def _build_stream_list(self) -> List[str]:
        """Build list of all streams to subscribe to."""
        streams = []
        for symbol in self.symbols:
            symbol_lower = symbol.lower()
            streams.append(f"{symbol_lower}@aggTrade")
            streams.append(f"{symbol_lower}@markPrice@1s")
        return streams

    def _chunk_streams(self, streams: List[str], chunk_size: int) -> List[List[str]]:
        """Split streams into chunks for multiple connections."""
        return [streams[i:i + chunk_size] for i in range(0, len(streams), chunk_size)]

    async def _subscribe(self, ws: websockets.WebSocketClientProtocol, streams: List[str]) -> bool:
        """Send SUBSCRIBE request and wait for confirmation."""
        # Split into batches if needed (max 200 per request)
        batches = self._chunk_streams(streams, MAX_SUBSCRIBE_BATCH)

        for batch in batches:
            request_id = self._subscribe_id
            self._subscribe_id += 1

            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": batch,
                "id": request_id
            }

            logger.debug(f"Subscribing to {len(batch)} streams (id={request_id})")
            await ws.send(json_dumps(subscribe_msg))

            # Wait for subscription confirmation
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json_loads(response)

                if data.get("id") == request_id:
                    if data.get("result") is None:
                        logger.info(f"Subscribed to {len(batch)} streams successfully")
                    else:
                        logger.warning(f"Subscribe response: {data}")
                else:
                    # This might be a data message, not our response
                    # Process it and continue waiting
                    await self._process_message(data)

            except asyncio.TimeoutError:
                logger.warning(f"Subscribe confirmation timeout for batch (id={request_id})")
                # Continue anyway - subscription might still work

        return True

    async def connect(self) -> None:
        """Main connection loop with automatic reconnection."""
        all_streams = self._build_stream_list()
        total_streams = len(all_streams)
        logger.info(f"Total streams to subscribe: {total_streams} ({len(self.symbols)} symbols)")

        # Calculate number of connections needed
        stream_chunks = self._chunk_streams(all_streams, MAX_STREAMS_PER_CONNECTION)
        num_connections = len(stream_chunks)
        logger.info(f"Using {num_connections} WebSocket connection(s) (max {MAX_STREAMS_PER_CONNECTION} streams each)")

        # Run all connections concurrently
        tasks = [
            self._run_connection(i, chunk)
            for i, chunk in enumerate(stream_chunks)
        ]
        await asyncio.gather(*tasks)

    async def _run_connection(self, conn_id: int, streams: List[str]) -> None:
        """Run a single WebSocket connection with its streams."""
        delay_index = 0

        while True:
            try:
                logger.info(f"[Conn {conn_id}] Connecting to {self.BASE_URL} for {len(streams)} streams...")

                async with websockets.connect(self.BASE_URL) as ws:
                    self.connections.append(ws)
                    self.is_connected = True
                    delay_index = 0

                    # Subscribe to streams dynamically
                    await self._subscribe(ws, streams)
                    logger.info(f"[Conn {conn_id}] WebSocket connected and subscribed successfully")

                    # Listen for messages
                    await self._listen(ws, conn_id)

            except websockets.exceptions.WebSocketException as e:
                self.is_connected = len(self.connections) > 0
                delay = self.reconnect_delays[min(delay_index, len(self.reconnect_delays) - 1)]
                logger.warning(f"[Conn {conn_id}] WebSocket disconnected: {e}. Reconnecting in {delay}s...")
                await asyncio.sleep(delay)
                delay_index += 1

            except Exception as e:
                self.is_connected = len(self.connections) > 0
                logger.error(f"[Conn {conn_id}] Unexpected error in WebSocket: {e}")
                delay = self.reconnect_delays[min(delay_index, len(self.reconnect_delays) - 1)]
                await asyncio.sleep(delay)
                delay_index += 1

    async def _listen(self, ws: websockets.WebSocketClientProtocol, conn_id: int) -> None:
        """Listen for WebSocket messages and emit to queue."""
        msg_count = 0
        last_log_time = 0

        async for message in ws:
            try:
                data = json_loads(message)
                await self._process_message(data)
                msg_count += 1

                # Log activity every 5000 messages (less spam with many connections)
                current_time = int(time.time())
                if msg_count % 5000 == 0 or (msg_count % 500 == 0 and current_time - last_log_time >= 30):
                    logger.info(f"[Conn {conn_id}] Active: {msg_count} messages received")
                    last_log_time = current_time

            except Exception as e:
                logger.warning(f"[Conn {conn_id}] Error processing message: {e}")

    async def _process_message(self, msg: Dict) -> None:
        """Process incoming WebSocket message and emit to queue."""
        # Skip subscription responses
        if 'id' in msg and 'result' in msg:
            return

        # Handle both formats:
        # 1. Combined stream: {"stream": "...", "data": {...}}
        # 2. Direct format: {"e": "aggTrade", "s": "BTCUSDT", ...}
        stream = msg.get('stream', '')
        data = msg.get('data', {})

        if not stream and not data:
            # Try direct format (e.g., {"e": "aggTrade", ...})
            if 'e' in msg:
                data = msg
                # Reconstruct stream name from event type and symbol
                event_type = msg.get('e', '')
                symbol = msg.get('s', '').lower()
                if event_type == 'aggTrade':
                    stream = f"{symbol}@aggTrade"
                elif event_type == 'markPriceUpdate':
                    stream = f"{symbol}@markPrice"
                else:
                    return
            else:
                return

        if not stream or not data:
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
        """Close all WebSocket connections."""
        for i, ws in enumerate(self.connections):
            try:
                await ws.close()
                logger.info(f"[Conn {i}] WebSocket connection closed")
            except Exception as e:
                logger.warning(f"[Conn {i}] Error closing connection: {e}")
        self.connections.clear()
        self.is_connected = False
        logger.info("All WebSocket connections closed")
