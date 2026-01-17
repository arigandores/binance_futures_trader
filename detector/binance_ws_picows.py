"""High-performance Binance WebSocket client using picows.

picows is a Cython-based WebSocket library that provides 2-3x faster
message handling compared to the pure Python websockets library.

This module is optional and will only be used if picows is installed.
Install with: poetry install -E performance
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

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

from detector.models import StreamType

logger = logging.getLogger(__name__)

# Check if picows is available
try:
    from picows import ws_connect, WSFrame, WSTransport, WSListener, WSMsgType, WSCloseCode
    PICOWS_AVAILABLE = True
except ImportError:
    PICOWS_AVAILABLE = False
    logger.debug("picows not available, using standard websockets library")


# Binance WebSocket limits
MAX_STREAMS_PER_CONNECTION = 200
MAX_SUBSCRIBE_BATCH = 200


def get_websocket_client(symbols: List[str], tick_queue: asyncio.Queue, reconnect_delays: List[int] = None):
    """
    Factory function to get the best available WebSocket client.

    Returns picows client if available (Linux/macOS), otherwise standard websockets.
    """
    if PICOWS_AVAILABLE:
        logger.info("Performance: Using picows WebSocket client (2-3x faster)")
        return BinanceWebSocketClientPicows(symbols, tick_queue, reconnect_delays)
    else:
        from detector.binance_ws import BinanceWebSocketClient
        logger.info("Using standard websockets client (picows not available)")
        return BinanceWebSocketClient(symbols, tick_queue, reconnect_delays)


# Only define picows classes if picows is available
if PICOWS_AVAILABLE:
    class BinanceWSListener(WSListener):
        """picows listener for Binance WebSocket messages."""

        def __init__(self, client: 'BinanceWebSocketClientPicows', conn_id: int, streams: List[str]):
            self.client = client
            self.conn_id = conn_id
            self.streams = streams
            self.msg_count = 0
            self.last_log_time = 0
            self._transport: Optional[WSTransport] = None
            self._subscribe_complete = asyncio.Event()
            self._pending_subscribe_ids: Dict[int, int] = {}  # id -> stream count
            self._confirmed_streams = 0

        def on_ws_connected(self, transport: WSTransport):
            """Called when WebSocket connection is established."""
            self._transport = transport
            logger.info(f"[Conn {self.conn_id}] picows connected, subscribing to {len(self.streams)} streams")
            # Schedule subscription after connection
            asyncio.create_task(self._subscribe())

        async def _subscribe(self):
            """Send SUBSCRIBE request for all streams with confirmation waiting."""
            if not self._transport:
                return

            # Split into batches
            batches = [self.streams[i:i + MAX_SUBSCRIBE_BATCH]
                       for i in range(0, len(self.streams), MAX_SUBSCRIBE_BATCH)]

            total_streams = len(self.streams)

            for i, batch in enumerate(batches):
                subscribe_id = self.client._get_subscribe_id()
                self._pending_subscribe_ids[subscribe_id] = len(batch)

                subscribe_msg = {
                    "method": "SUBSCRIBE",
                    "params": batch,
                    "id": subscribe_id
                }
                self._transport.send(WSMsgType.TEXT, json_dumps(subscribe_msg).encode())
                logger.debug(f"[Conn {self.conn_id}] Sent subscribe batch {i+1}/{len(batches)} (id={subscribe_id}, {len(batch)} streams)")

                # Wait a bit between batches to avoid rate limiting
                if i < len(batches) - 1:
                    await asyncio.sleep(0.1)

            # Wait for all confirmations (with timeout)
            try:
                for _ in range(50):  # Max 5 seconds total
                    if self._confirmed_streams >= total_streams:
                        break
                    await asyncio.sleep(0.1)

                if self._confirmed_streams >= total_streams:
                    logger.info(f"[Conn {self.conn_id}] All {total_streams} streams confirmed")
                else:
                    logger.warning(f"[Conn {self.conn_id}] Only {self._confirmed_streams}/{total_streams} streams confirmed (continuing anyway)")
            except Exception as e:
                logger.warning(f"[Conn {self.conn_id}] Error waiting for confirmations: {e}")

            self._subscribe_complete.set()
            logger.info(f"[Conn {self.conn_id}] Subscription complete")

        def on_ws_frame(self, transport: WSTransport, frame: WSFrame):
            """Called for each received WebSocket frame."""
            if frame.msg_type == WSMsgType.TEXT:
                try:
                    data = json_loads(frame.get_payload_as_bytes())

                    # Handle subscription confirmations
                    if 'id' in data and 'result' in data:
                        sub_id = data.get('id')
                        if sub_id in self._pending_subscribe_ids:
                            stream_count = self._pending_subscribe_ids.pop(sub_id)
                            self._confirmed_streams += stream_count
                            logger.debug(f"[Conn {self.conn_id}] Subscribe confirmed (id={sub_id}, +{stream_count} streams, total={self._confirmed_streams})")
                        return

                    # Use create_task to avoid blocking the event loop
                    asyncio.create_task(self._process_message(data))
                    self.msg_count += 1

                    # Log activity periodically
                    current_time = int(time.time())
                    if self.msg_count % 5000 == 0 or (self.msg_count % 500 == 0 and current_time - self.last_log_time >= 30):
                        logger.info(f"[Conn {self.conn_id}] Active: {self.msg_count} messages (picows)")
                        self.last_log_time = current_time

                except Exception as e:
                    logger.warning(f"[Conn {self.conn_id}] Failed to parse message: {e}")

        async def _process_message(self, msg: Dict):
            """Process incoming WebSocket message and emit to queue."""
            await self.client._process_message(msg)

        def on_ws_disconnected(self, transport: WSTransport):
            """Called when WebSocket connection is closed."""
            logger.warning(f"[Conn {self.conn_id}] picows disconnected")
            self.client.is_connected = False


    class BinanceWebSocketClientPicows:
        """
        High-performance WebSocket client for Binance USD-M Futures using picows.

        Provides 2-3x faster message handling compared to standard websockets.
        """

        BASE_URL = "wss://fstream.binance.com/ws"

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
            self._subscribe_id = 1
            self._listeners: List[BinanceWSListener] = []

        def _get_subscribe_id(self) -> int:
            """Get next subscribe ID (thread-safe increment)."""
            result = self._subscribe_id
            self._subscribe_id += 1
            return result

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

        async def connect(self) -> None:
            """Main connection loop with automatic reconnection."""
            all_streams = self._build_stream_list()
            total_streams = len(all_streams)
            logger.info(f"[picows] Total streams: {total_streams} ({len(self.symbols)} symbols)")

            stream_chunks = self._chunk_streams(all_streams, MAX_STREAMS_PER_CONNECTION)
            num_connections = len(stream_chunks)
            logger.info(f"[picows] Using {num_connections} connections (max {MAX_STREAMS_PER_CONNECTION} streams each)")

            # Run all connections concurrently
            tasks = [
                self._run_connection(i, chunk)
                for i, chunk in enumerate(stream_chunks)
            ]
            await asyncio.gather(*tasks)

        async def _run_connection(self, conn_id: int, streams: List[str]) -> None:
            """Run a single WebSocket connection."""
            delay_index = 0
            listener: Optional[BinanceWSListener] = None

            while True:
                try:
                    logger.info(f"[Conn {conn_id}] Connecting via picows...")

                    # Clean up previous listener if any (prevent memory leak)
                    if listener is not None and listener in self._listeners:
                        self._listeners.remove(listener)

                    listener = BinanceWSListener(self, conn_id, streams)
                    self._listeners.append(listener)

                    # picows requires a factory function (callable), not an instance
                    # Use default argument to capture listener value (avoids closure issue)
                    transport, _ = await ws_connect(lambda l=listener: l, self.BASE_URL)
                    self.is_connected = True
                    delay_index = 0

                    # Wait for disconnection
                    await listener._subscribe_complete.wait()

                    # Keep connection alive
                    while self.is_connected:
                        await asyncio.sleep(1)

                except Exception as e:
                    self.is_connected = len([l for l in self._listeners if l._transport]) > 0
                    delay = self.reconnect_delays[min(delay_index, len(self.reconnect_delays) - 1)]
                    logger.warning(f"[Conn {conn_id}] picows error: {e}. Reconnecting in {delay}s...")
                    await asyncio.sleep(delay)
                    delay_index += 1

        async def _process_message(self, msg: Dict) -> None:
            """Process incoming WebSocket message and emit to queue."""
            # Skip subscription responses
            if 'id' in msg and 'result' in msg:
                return

            stream = msg.get('stream', '')
            data = msg.get('data', {})

            if not stream and not data:
                if 'e' in msg:
                    data = msg
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
            return None

        async def close(self) -> None:
            """Close all WebSocket connections."""
            for listener in self._listeners:
                if listener._transport:
                    listener._transport.send_close(WSCloseCode.OK)
            self._listeners.clear()
            self.is_connected = False
            logger.info("[picows] All connections closed")

else:
    # picows not available - define dummy class to avoid import errors
    class BinanceWebSocketClientPicows:
        """Dummy class when picows is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("picows is not available. Install with: poetry install -E performance")
