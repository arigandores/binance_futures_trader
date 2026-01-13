"""SQLite storage layer with batched writes and WAL mode."""

import aiosqlite
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from detector.models import Bar, Features, Event, Direction

logger = logging.getLogger(__name__)


class Storage:
    """SQLite persistence with batched writes."""

    def __init__(self, db_path: str, wal_mode: bool = True, batch_interval: int = 5):
        self.db_path = Path(db_path)
        self.wal_mode = wal_mode
        self.batch_interval = batch_interval

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Batched write buffers
        self.bars_buffer: List[Bar] = []
        self.features_buffer: List[Features] = []

        self.db: Optional[aiosqlite.Connection] = None

    async def init_db(self) -> None:
        """Initialize database schema with WAL mode."""
        self.db = await aiosqlite.connect(str(self.db_path))

        if self.wal_mode:
            await self.db.execute("PRAGMA journal_mode=WAL")

        # Create tables
        await self._create_tables()
        await self.db.commit()

        logger.info(f"Database initialized at {self.db_path} (WAL mode: {self.wal_mode})")

    async def _create_tables(self) -> None:
        """Create all database tables."""
        # Table: bars_1m
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS bars_1m (
                symbol TEXT,
                ts_minute INTEGER,
                o REAL, h REAL, l REAL, c REAL,
                vol REAL, notional REAL, trades INTEGER,
                taker_buy REAL, taker_sell REAL,
                liq_notional REAL, liq_count INTEGER,
                mid REAL, spread_bps REAL,
                mark REAL, funding REAL, next_funding_ts INTEGER,
                PRIMARY KEY (symbol, ts_minute)
            )
        """)

        # Table: features
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS features (
                symbol TEXT,
                ts_minute INTEGER,
                er_15m REAL, z_er_15m REAL,
                vol_15m REAL, z_vol_15m REAL,
                taker_buy_share_15m REAL,
                oi_delta_1h REAL, z_oi_delta_1h REAL,
                liq_15m REAL, z_liq_15m REAL,
                beta REAL,
                PRIMARY KEY (symbol, ts_minute)
            )
        """)

        # Table: events
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                ts INTEGER,
                initiator_symbol TEXT,
                direction TEXT,
                status TEXT,
                followers_json TEXT,
                metrics_json TEXT
            )
        """)

        # Table: cooldown_tracker
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS cooldown_tracker (
                symbol TEXT PRIMARY KEY,
                last_alert_ts INTEGER,
                last_direction TEXT
            )
        """)

        # Create indices
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_bars_ts ON bars_1m(ts_minute DESC)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_features_ts ON features(ts_minute DESC)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC)")

    async def batch_write_bars(self, bars: List[Bar]) -> None:
        """Add bars to buffer (will be flushed periodically)."""
        self.bars_buffer.extend(bars)

    async def batch_write_features(self, features: List[Features]) -> None:
        """Add features to buffer (will be flushed periodically)."""
        self.features_buffer.extend(features)

    async def flush_all(self) -> None:
        """Flush all buffered writes to database."""
        if not self.db:
            logger.warning("Database not initialized, skipping flush")
            return

        bars_count = len(self.bars_buffer)
        features_count = len(self.features_buffer)

        if bars_count == 0 and features_count == 0:
            return

        try:
            # Flush bars
            if self.bars_buffer:
                await self._flush_bars()

            # Flush features
            if self.features_buffer:
                await self._flush_features()

            await self.db.commit()

            if bars_count > 0 or features_count > 0:
                logger.info(f"DB flush: {bars_count} bars, {features_count} features written")

        except Exception as e:
            logger.error(f"Error flushing data to DB: {e}")
            await self.db.rollback()

    async def _flush_bars(self) -> None:
        """Flush bars buffer using executemany."""
        if not self.bars_buffer:
            return

        bars_data = [
            (
                bar.symbol, bar.ts_minute,
                bar.open, bar.high, bar.low, bar.close,
                bar.volume, bar.notional, bar.trades,
                bar.taker_buy, bar.taker_sell,
                bar.liq_notional, bar.liq_count,
                bar.mid, bar.spread_bps,
                bar.mark, bar.funding, bar.next_funding_ts
            )
            for bar in self.bars_buffer
        ]

        await self.db.executemany("""
            INSERT OR REPLACE INTO bars_1m (
                symbol, ts_minute,
                o, h, l, c,
                vol, notional, trades,
                taker_buy, taker_sell,
                liq_notional, liq_count,
                mid, spread_bps,
                mark, funding, next_funding_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, bars_data)

        self.bars_buffer.clear()

    async def _flush_features(self) -> None:
        """Flush features buffer using executemany."""
        if not self.features_buffer:
            return

        features_data = [
            (
                f.symbol, f.ts_minute,
                f.er_15m, f.z_er_15m,
                f.vol_15m, f.z_vol_15m,
                f.taker_buy_share_15m,
                f.oi_delta_1h, f.z_oi_delta_1h,
                f.liq_15m, f.z_liq_15m,
                f.beta
            )
            for f in self.features_buffer
        ]

        await self.db.executemany("""
            INSERT OR REPLACE INTO features (
                symbol, ts_minute,
                er_15m, z_er_15m,
                vol_15m, z_vol_15m,
                taker_buy_share_15m,
                oi_delta_1h, z_oi_delta_1h,
                liq_15m, z_liq_15m,
                beta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, features_data)

        self.features_buffer.clear()

    async def write_event(self, event: Event) -> None:
        """Write event immediately (not buffered)."""
        if not self.db:
            logger.warning("Database not initialized, skipping event write")
            return

        try:
            await self.db.execute("""
                INSERT INTO events (
                    event_id, ts, initiator_symbol, direction, status,
                    followers_json, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.ts,
                event.initiator_symbol,
                event.direction.value,
                event.status.value,
                json.dumps(event.followers),
                json.dumps(event.metrics)
            ))
            await self.db.commit()
            logger.info(f"Event {event.event_id} written to DB")
        except Exception as e:
            logger.error(f"Error writing event to DB: {e}")

    async def get_recent_bars(self, symbol: str, limit: int = 720) -> List[Bar]:
        """Get recent bars for backfilling rolling windows."""
        if not self.db:
            logger.warning("Database not initialized")
            return []

        try:
            async with self.db.execute("""
                SELECT symbol, ts_minute, o, h, l, c, vol, notional, trades,
                       taker_buy, taker_sell, liq_notional, liq_count,
                       mid, spread_bps, mark, funding, next_funding_ts
                FROM bars_1m
                WHERE symbol = ?
                ORDER BY ts_minute DESC
                LIMIT ?
            """, (symbol, limit)) as cursor:
                rows = await cursor.fetchall()

            # Convert rows to Bar objects (reversed to get chronological order)
            bars = []
            for row in reversed(rows):
                bar = Bar(
                    symbol=row[0],
                    ts_minute=row[1],
                    open=row[2],
                    high=row[3],
                    low=row[4],
                    close=row[5],
                    volume=row[6],
                    notional=row[7],
                    trades=row[8],
                    taker_buy=row[9],
                    taker_sell=row[10],
                    liq_notional=row[11],
                    liq_count=row[12],
                    mid=row[13],
                    spread_bps=row[14],
                    mark=row[15],
                    funding=row[16],
                    next_funding_ts=row[17]
                )
                bars.append(bar)

            return bars

        except Exception as e:
            logger.error(f"Error fetching recent bars for {symbol}: {e}")
            return []

    async def update_cooldown(self, symbol: str, direction: Direction, ts: int) -> None:
        """Update cooldown tracker for a symbol."""
        if not self.db:
            return

        try:
            await self.db.execute("""
                INSERT OR REPLACE INTO cooldown_tracker (symbol, last_alert_ts, last_direction)
                VALUES (?, ?, ?)
            """, (symbol, ts, direction.value))
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error updating cooldown for {symbol}: {e}")

    async def check_cooldown(
        self,
        symbol: str,
        direction: Direction,
        current_ts: int,
        cooldown_ms: int,
        grace_ms: int
    ) -> bool:
        """
        Check if alert is allowed based on cooldown logic.

        Returns True if alert is allowed, False if blocked.
        """
        if not self.db:
            return True  # Allow if DB not available

        try:
            async with self.db.execute("""
                SELECT last_alert_ts, last_direction FROM cooldown_tracker WHERE symbol = ?
            """, (symbol,)) as cursor:
                row = await cursor.fetchone()

            if not row:
                return True  # No previous alert, allow

            last_ts, last_dir_str = row
            last_dir = Direction(last_dir_str)
            time_since_last = current_ts - last_ts

            # Same direction: require full cooldown
            if direction == last_dir:
                return time_since_last >= cooldown_ms

            # Opposite direction: require grace period only
            return time_since_last >= grace_ms

        except Exception as e:
            logger.error(f"Error checking cooldown for {symbol}: {e}")
            return True  # Allow on error

    async def query_events(self, since_ts: Optional[int] = None, limit: int = 1000) -> List[Event]:
        """Query events from database."""
        if not self.db:
            return []

        try:
            if since_ts is not None:
                query = """
                    SELECT event_id, ts, initiator_symbol, direction, status,
                           followers_json, metrics_json
                    FROM events
                    WHERE ts >= ?
                    ORDER BY ts DESC
                    LIMIT ?
                """
                params = (since_ts, limit)
            else:
                query = """
                    SELECT event_id, ts, initiator_symbol, direction, status,
                           followers_json, metrics_json
                    FROM events
                    ORDER BY ts DESC
                    LIMIT ?
                """
                params = (limit,)

            async with self.db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            events = []
            for row in rows:
                event = Event(
                    event_id=row[0],
                    ts=row[1],
                    initiator_symbol=row[2],
                    direction=Direction(row[3]),
                    status=row[4],
                    followers=json.loads(row[5]) if row[5] else [],
                    metrics=json.loads(row[6]) if row[6] else {}
                )
                events.append(event)

            return events

        except Exception as e:
            logger.error(f"Error querying events: {e}")
            return []

    async def clear_bars_and_features(self) -> None:
        """Clear all bars and features data (useful before fresh backfill)."""
        if not self.db:
            logger.warning("Database not initialized")
            return

        try:
            await self.db.execute("DELETE FROM bars_1m")
            await self.db.execute("DELETE FROM features")
            await self.db.commit()
            logger.info("Cleared all bars and features from database")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            await self.db.rollback()

    async def close(self) -> None:
        """Close database connection."""
        if self.db:
            await self.flush_all()
            await self.db.close()
            logger.info("Database connection closed")
