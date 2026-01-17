"""SQLite storage layer with batched writes and WAL mode."""

import aiosqlite
import logging
from pathlib import Path

# Performance: orjson is 4-6x faster than standard json
try:
    import orjson
    def json_dumps(data):
        return orjson.dumps(data).decode('utf-8') if data else None
    def json_loads(data):
        return orjson.loads(data) if data else {}
except ImportError:
    import json
    def json_dumps(data):
        return json.dumps(data) if data else None
    def json_loads(data):
        return json.loads(data) if data else {}
from typing import List, Optional, Tuple
from detector.models import Bar, Features, Event, Direction, Position, PositionStatus, ExitReason

logger = logging.getLogger(__name__)


class Storage:
    """SQLite persistence with batched writes."""

    def __init__(self, db_path: str, wal_mode: bool = True, batch_interval: int = 5):
        self.db_path = Path(db_path)
        self.wal_mode = wal_mode
        self.batch_interval = batch_interval

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Batched write buffers (bars and features only - events write immediately)
        self.bars_buffer: List[Bar] = []
        self.features_buffer: List[Features] = []

        self.db: Optional[aiosqlite.Connection] = None

    async def init_db(self) -> None:
        """Initialize database schema with WAL mode and optimized settings for 500+ symbols."""
        self.db = await aiosqlite.connect(str(self.db_path))

        # Performance: Optimized PRAGMA settings for high-write workload
        # These settings provide 2-3x improvement in write throughput
        if self.wal_mode:
            await self.db.execute("PRAGMA journal_mode=WAL")
            await self.db.execute("PRAGMA synchronous=NORMAL")      # Safe in WAL mode, faster than FULL
            await self.db.execute("PRAGMA cache_size=-64000")       # 64MB cache (negative = KB)
            await self.db.execute("PRAGMA mmap_size=268435456")     # 256MB memory-mapped I/O
            await self.db.execute("PRAGMA temp_store=MEMORY")       # Temp tables in memory
            await self.db.execute("PRAGMA wal_autocheckpoint=1000") # Checkpoint every 1000 pages
            logger.info("SQLite optimized: WAL mode, 64MB cache, 256MB mmap, NORMAL sync")

        # Create tables
        await self._create_tables()

        # Run migration to add oi column (for existing databases)
        await self._migrate_add_oi_column()

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
                funding REAL,
                oi REAL,
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

        # Table: positions
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                event_id TEXT,
                symbol TEXT,
                direction TEXT,
                status TEXT,

                open_price REAL,
                open_ts INTEGER,
                entry_z_er REAL,
                entry_z_vol REAL,
                entry_taker_share REAL,

                close_price REAL,
                close_ts INTEGER,
                exit_z_er REAL,
                exit_z_vol REAL,
                exit_reason TEXT,

                pnl_percent REAL,
                max_favorable_excursion REAL,
                max_adverse_excursion REAL,

                duration_minutes INTEGER,
                metrics_json TEXT,

                FOREIGN KEY (event_id) REFERENCES events(event_id)
            )
        """)

        # Table: alert_audit - для аудита всех алертов
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS alert_audit (
                alert_id TEXT PRIMARY KEY,
                ts INTEGER,
                alert_type TEXT,
                symbol TEXT,
                direction TEXT,
                message_text TEXT,
                metadata_json TEXT
            )
        """)

        # Create indices
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_bars_ts ON bars_1m(ts_minute DESC)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_features_ts ON features(ts_minute DESC)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_positions_open_ts ON positions(open_ts DESC)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_alert_audit_ts ON alert_audit(ts DESC)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_alert_audit_type ON alert_audit(alert_type)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_alert_audit_symbol ON alert_audit(symbol)")

    async def _migrate_add_oi_column(self) -> None:
        """Add oi column to bars_1m if it doesn't exist (migration for existing databases)."""
        try:
            # Check if column exists
            async with self.db.execute("PRAGMA table_info(bars_1m)") as cursor:
                columns = await cursor.fetchall()
                column_names = [col[1] for col in columns]

                if 'oi' not in column_names:
                    logger.info("Migrating database: adding 'oi' column to bars_1m")
                    await self.db.execute("ALTER TABLE bars_1m ADD COLUMN oi REAL")
                    await self.db.commit()
                    logger.info("Migration complete: oi column added")
                else:
                    logger.debug("Migration skipped: oi column already exists")
        except Exception as e:
            logger.error(f"Error during OI column migration: {e}")
            # Non-fatal - continue without OI if migration fails

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
                bar.funding,
                bar.oi
            )
            for bar in self.bars_buffer
        ]

        await self.db.executemany("""
            INSERT OR REPLACE INTO bars_1m (
                symbol, ts_minute,
                o, h, l, c,
                vol, notional, trades,
                taker_buy, taker_sell,
                funding,
                oi
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                beta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, features_data)

        self.features_buffer.clear()

    async def write_event(self, event: Event) -> None:
        """
        Write event immediately to database.

        Events are critical trading data that must not be lost on crash,
        so they are written immediately rather than buffered.
        """
        if not self.db:
            logger.warning("Database not initialized, skipping event write")
            return

        try:
            await self.db.execute("""
                INSERT INTO events (
                    event_id, ts, initiator_symbol, direction, metrics_json
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.ts,
                event.initiator_symbol,
                event.direction.value,
                json_dumps(event.metrics)
            ))
            await self.db.commit()
            logger.debug(f"Event {event.event_id} written to DB")
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
                       taker_buy, taker_sell, funding, oi
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
                    funding=row[11],
                    oi=row[12]
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
                    SELECT event_id, ts, initiator_symbol, direction, metrics_json
                    FROM events
                    WHERE ts >= ?
                    ORDER BY ts DESC
                    LIMIT ?
                """
                params = (since_ts, limit)
            else:
                query = """
                    SELECT event_id, ts, initiator_symbol, direction, metrics_json
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
                    metrics=json_loads(row[4]) if row[4] else {}
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

    async def write_position(self, position: Position) -> None:
        """Write or update position in database."""
        if not self.db:
            logger.warning("Database not initialized, skipping position write")
            return

        try:
            await self.db.execute("""
                INSERT OR REPLACE INTO positions (
                    position_id, event_id, symbol, direction, status,
                    open_price, open_ts, entry_z_er, entry_z_vol, entry_taker_share,
                    close_price, close_ts, exit_z_er, exit_z_vol, exit_reason,
                    pnl_percent, max_favorable_excursion, max_adverse_excursion,
                    duration_minutes, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.position_id,
                position.event_id,
                position.symbol,
                position.direction.value,
                position.status.value,
                position.open_price,
                position.open_ts,
                position.entry_z_er,
                position.entry_z_vol,
                position.entry_taker_share,
                position.close_price,
                position.close_ts,
                position.exit_z_er,
                position.exit_z_vol,
                position.exit_reason.value if position.exit_reason else None,
                position.pnl_percent,
                position.max_favorable_excursion,
                position.max_adverse_excursion,
                position.duration_minutes,
                json_dumps(position.metrics)
            ))
            await self.db.commit()
            logger.debug(f"Position {position.position_id} written to DB")
        except Exception as e:
            logger.error(f"Error writing position to DB: {e}")

    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all open positions, optionally filtered by symbol."""
        if not self.db:
            logger.warning("Database not initialized")
            return []

        try:
            if symbol:
                query = "SELECT * FROM positions WHERE status = ? AND symbol = ?"
                params = (PositionStatus.OPEN.value, symbol)
            else:
                query = "SELECT * FROM positions WHERE status = ?"
                params = (PositionStatus.OPEN.value,)

            async with self.db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            return [self._row_to_position(row) for row in rows]

        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")
            return []

    async def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        if not self.db:
            logger.warning("Database not initialized")
            return None

        try:
            async with self.db.execute(
                "SELECT * FROM positions WHERE position_id = ?",
                (position_id,)
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                return self._row_to_position(row)
            return None

        except Exception as e:
            logger.error(f"Error fetching position {position_id}: {e}")
            return None

    async def query_positions(
        self,
        status: Optional[PositionStatus] = None,
        symbol: Optional[str] = None,
        since_ts: Optional[int] = None,
        limit: int = 1000
    ) -> List[Position]:
        """Query positions with filters."""
        if not self.db:
            return []

        try:
            query_parts = ["SELECT * FROM positions WHERE 1=1"]
            params = []

            if status:
                query_parts.append("AND status = ?")
                params.append(status.value)

            if symbol:
                query_parts.append("AND symbol = ?")
                params.append(symbol)

            if since_ts:
                query_parts.append("AND open_ts >= ?")
                params.append(since_ts)

            query_parts.append("ORDER BY open_ts DESC LIMIT ?")
            params.append(limit)

            query = " ".join(query_parts)

            async with self.db.execute(query, tuple(params)) as cursor:
                rows = await cursor.fetchall()

            return [self._row_to_position(row) for row in rows]

        except Exception as e:
            logger.error(f"Error querying positions: {e}")
            return []

    def _row_to_position(self, row: Tuple) -> Position:
        """Convert database row to Position object."""
        return Position(
            position_id=row[0],
            event_id=row[1],
            symbol=row[2],
            direction=Direction(row[3]),
            status=PositionStatus(row[4]),
            open_price=row[5],
            open_ts=row[6],
            entry_z_er=row[7],
            entry_z_vol=row[8],
            entry_taker_share=row[9],
            close_price=row[10],
            close_ts=row[11],
            exit_z_er=row[12],
            exit_z_vol=row[13],
            exit_reason=ExitReason(row[14]) if row[14] else None,
            pnl_percent=row[15],
            max_favorable_excursion=row[16] or 0.0,
            max_adverse_excursion=row[17] or 0.0,
            duration_minutes=row[18],
            metrics=json_loads(row[19]) if row[19] else {}
        )

    async def write_alert(
        self,
        alert_id: str,
        ts: int,
        alert_type: str,
        symbol: str,
        direction: str,
        message_text: str,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Write alert to audit table for future analysis.

        Args:
            alert_id: Unique alert identifier
            ts: Timestamp in milliseconds
            alert_type: Type of alert (PENDING_SIGNAL_CREATED, POSITION_OPENED, etc.)
            symbol: Trading symbol
            direction: UP or DOWN
            message_text: Full formatted alert message
            metadata: Additional metrics as dict
        """
        if not self.db:
            logger.warning("Database not initialized, skipping alert write")
            return

        try:
            await self.db.execute("""
                INSERT OR REPLACE INTO alert_audit (
                    alert_id, ts, alert_type, symbol, direction, message_text, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_id,
                ts,
                alert_type,
                symbol,
                direction,
                message_text,
                json_dumps(metadata) if metadata else None
            ))
            await self.db.commit()
            logger.debug(f"Alert {alert_type} for {symbol} written to audit log")
        except Exception as e:
            logger.error(f"Error writing alert to audit log: {e}")

    async def query_alerts(
        self,
        alert_type: Optional[str] = None,
        symbol: Optional[str] = None,
        since_ts: Optional[int] = None,
        limit: int = 1000
    ) -> List[dict]:
        """
        Query alerts from audit table.

        Args:
            alert_type: Filter by alert type
            symbol: Filter by symbol
            since_ts: Filter alerts after this timestamp
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        if not self.db:
            return []

        try:
            query_parts = ["SELECT * FROM alert_audit WHERE 1=1"]
            params = []

            if alert_type:
                query_parts.append("AND alert_type = ?")
                params.append(alert_type)

            if symbol:
                query_parts.append("AND symbol = ?")
                params.append(symbol)

            if since_ts:
                query_parts.append("AND ts >= ?")
                params.append(since_ts)

            query_parts.append("ORDER BY ts DESC LIMIT ?")
            params.append(limit)

            query = " ".join(query_parts)

            async with self.db.execute(query, tuple(params)) as cursor:
                rows = await cursor.fetchall()

            alerts = []
            for row in rows:
                alert = {
                    'alert_id': row[0],
                    'ts': row[1],
                    'alert_type': row[2],
                    'symbol': row[3],
                    'direction': row[4],
                    'message_text': row[5],
                    'metadata': json_loads(row[6]) if row[6] else {}
                }
                alerts.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"Error querying alerts: {e}")
            return []

    async def close(self) -> None:
        """Close database connection."""
        if self.db:
            await self.flush_all()
            await self.db.close()
            logger.info("Database connection closed")
