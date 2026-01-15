"""Data models for the Sector Shot Detector."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class StreamType(str, Enum):
    """WebSocket stream types."""
    AGG_TRADE = "aggTrade"
    BOOK_TICKER = "bookTicker"
    MARK_PRICE = "markPrice"
    FORCE_ORDER = "forceOrder"


class Direction(str, Enum):
    """Market direction."""
    UP = "UP"
    DOWN = "DOWN"


class EventStatus(str, Enum):
    """Event confirmation status."""
    CONFIRMED = "CONFIRMED"
    PARTIAL = "PARTIAL"
    UNCONFIRMED = "UNCONFIRMED"
    SECTOR_DIFFUSION = "SECTOR_DIFFUSION"


@dataclass
class DataAvailability:
    """Tracks available data sources for graceful degradation."""
    has_api_key: bool = False
    has_oi: bool = False
    has_taker_ratio: bool = False
    has_liquidations: bool = False
    has_funding: bool = False

    def confirmation_level(self) -> EventStatus:
        """Determine confirmation level based on available data."""
        if not self.has_api_key:
            return EventStatus.UNCONFIRMED

        available = sum([
            self.has_oi,
            self.has_taker_ratio,
            self.has_liquidations,
            self.has_funding
        ])

        if available >= 2:
            return EventStatus.CONFIRMED
        elif available == 1:
            return EventStatus.PARTIAL
        else:
            return EventStatus.UNCONFIRMED


@dataclass
class Tick:
    """Raw tick data from WebSocket streams."""
    symbol: str
    timestamp: int  # Milliseconds since epoch
    stream_type: StreamType
    data: Dict[str, Any]

    # Parsed fields (stream-specific)
    price: Optional[float] = None
    quantity: Optional[float] = None
    trade_id: Optional[int] = None
    is_buyer_maker: Optional[bool] = None


@dataclass
class Bar:
    """1-minute OHLCV bar with microstructure data."""
    symbol: str
    ts_minute: int  # Bar opening time (aligned to minute), milliseconds since epoch

    # OHLCV
    open: float = 0.0
    high: float = 0.0
    low: float = float('inf')
    close: float = 0.0
    volume: float = 0.0
    notional: float = 0.0  # Price * Quantity
    trades: int = 0

    # Taker buy/sell splits
    taker_buy: float = 0.0
    taker_sell: float = 0.0

    # Liquidations (if available)
    liq_notional: float = 0.0
    liq_count: int = 0

    # Book data
    mid: Optional[float] = None
    spread_bps: Optional[float] = None

    # Mark price and funding
    mark: Optional[float] = None
    funding: Optional[float] = None
    next_funding_ts: Optional[int] = None

    # Open Interest
    oi: Optional[float] = None

    def taker_buy_share(self) -> Optional[float]:
        """Calculate taker buy share (0 to 1)."""
        total = self.taker_buy + self.taker_sell
        if total == 0:
            return None
        return self.taker_buy / total


@dataclass
class Features:
    """Calculated features for one symbol at one time point."""
    symbol: str
    ts_minute: int  # Milliseconds since epoch

    # Returns
    r_1m: float = 0.0
    r_15m: float = 0.0

    # Beta and excess returns
    beta: float = 0.0
    er_15m: float = 0.0  # Excess return vs BTC

    # Z-scores
    z_er_15m: float = 0.0
    z_vol_15m: float = 0.0

    # Aggregated metrics
    vol_15m: float = 0.0  # Sum of volume over last 15 bars
    taker_buy_share_15m: Optional[float] = None

    # Confirmation metrics (optional, depends on API key)
    oi_delta_1h: Optional[float] = None
    z_oi_delta_1h: Optional[float] = None
    liq_15m: Optional[float] = None
    z_liq_15m: Optional[float] = None
    funding_rate: Optional[float] = None

    # Direction
    direction: Optional[Direction] = None

    def determine_direction(self) -> Direction:
        """Determine direction based on excess return sign."""
        if self.er_15m > 0:
            return Direction.UP
        else:
            return Direction.DOWN


@dataclass
class Event:
    """Detected anomaly event."""
    event_id: str
    ts: int  # Event timestamp, milliseconds since epoch
    initiator_symbol: str
    direction: Direction
    status: EventStatus
    followers: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'event_id': self.event_id,
            'ts': self.ts,
            'initiator_symbol': self.initiator_symbol,
            'direction': self.direction.value,
            'status': self.status.value,
            'followers': self.followers,
            'metrics': self.metrics
        }


@dataclass
class PendingSignal:
    """
    A detected signal waiting for entry triggers to be met.

    Represents a "watch window" - system checks EVERY bar for up to max_wait_minutes.
    Entry happens on FIRST bar where all triggers met (not fixed delay).

    One pending signal per symbol (if allow_multiple_positions=false) to prevent duplicates.
    """
    signal_id: str  # f"{symbol}_{event.event_id}_{direction}"
    event: Event  # Original initiator event
    created_ts: int  # Bar timestamp when signal detected (ms) - MUST use bar.ts_minute!
    expires_ts: int  # TTL expiry (created_ts + max_wait_window_ms) - bar time scale
    direction: Direction  # UP or DOWN
    symbol: str

    # Signal metrics (at detection time)
    signal_z_er: float
    signal_z_vol: float
    signal_price: float

    # Tracking (Must-Fix #1: bar time scale only)
    bars_since_signal: int = 0  # Number of bars processed since signal

    # Must-Fix #3: Track peak/trough since signal detection
    peak_since_signal: Optional[float] = None  # For UP: max(high), DOWN: min(low)

    # Must-Fix #5: Track last evaluated timestamp for freshness check
    last_evaluated_bar_ts: Optional[int] = None  # Last bar ts_minute evaluated

    # Must-Fix #8: Invalidation flag for immediate removal
    invalidated: bool = False
    invalidation_reason: Optional[str] = None

    # Trigger state (updated on each bar)
    z_cooldown_met: bool = False
    pullback_met: bool = False
    stability_met: bool = False

    # WIN_RATE_MAX profile: Additional tracking fields
    re_expansion_met: bool = False  # Re-expansion confirmed (1 of 3 methods)
    pullback_exceeded_max: bool = False  # Structure broken (pullback too deep)
    z_cooldown_in_range: bool = False  # Z-score in [min, max] range
    flow_death_bar_count: int = 0  # Consecutive bars with low dominance

    def is_expired(self, current_bar_ts: int) -> bool:
        """
        Check if max watch window (TTL) exceeded.
        Must-Fix #1: Use bar time scale, not wall clock.
        """
        return current_bar_ts >= self.expires_ts

    def update_peak(self, bar: 'Bar') -> None:
        """
        Update peak/trough since signal detection.
        Must-Fix #3: Track extremes from signal moment, not arbitrary lookback.
        """
        if self.direction == Direction.UP:
            if self.peak_since_signal is None:
                self.peak_since_signal = bar.high
            else:
                self.peak_since_signal = max(self.peak_since_signal, bar.high)
        else:  # DOWN
            if self.peak_since_signal is None:
                self.peak_since_signal = bar.low
            else:
                self.peak_since_signal = min(self.peak_since_signal, bar.low)

    def all_triggers_met(self) -> bool:
        """Check if all required triggers are met."""
        # If triggers disabled, always True
        # If enabled, require all 3
        return self.z_cooldown_met and self.pullback_met and self.stability_met


@dataclass
class PendingSectorEvent:
    """Tracks pending sector diffusion events."""
    initiator: Event
    window_end: int  # Milliseconds since epoch
    followers: List[Features] = field(default_factory=list)


@dataclass
class CooldownState:
    """Cooldown state for a symbol."""
    symbol: str
    last_alert_ts: int  # Milliseconds since epoch
    last_direction: Direction


class PositionStatus(str, Enum):
    """Virtual position status."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class ExitReason(str, Enum):
    """Reason for position exit."""
    Z_SCORE_REVERSAL = "Z_SCORE_REVERSAL"  # z_er fell below threshold
    STOP_LOSS = "STOP_LOSS"  # Price moved against position
    TAKE_PROFIT = "TAKE_PROFIT"  # Target profit reached
    TIME_EXIT = "TIME_EXIT"  # Maximum holding time reached
    OPPOSITE_SIGNAL = "OPPOSITE_SIGNAL"  # Strong opposite direction signal
    ORDER_FLOW_REVERSAL = "ORDER_FLOW_REVERSAL"  # Taker buy/sell ratio reversed
    TRAILING_STOP = "TRAILING_STOP"  # Trailing stop triggered after reaching profit target


@dataclass
class Position:
    """Virtual trading position."""
    position_id: str
    event_id: str  # Link to the triggering event
    symbol: str
    direction: Direction
    status: PositionStatus

    # Entry data
    open_price: float
    open_ts: int  # Milliseconds since epoch
    entry_z_er: float
    entry_z_vol: float
    entry_taker_share: float

    # Exit data (None if still open)
    close_price: Optional[float] = None
    close_ts: Optional[int] = None
    exit_z_er: Optional[float] = None
    exit_z_vol: Optional[float] = None
    exit_reason: Optional[ExitReason] = None

    # PnL metrics
    pnl_percent: Optional[float] = None  # (close - open) / open * 100 * direction_multiplier
    pnl_ticks: Optional[float] = None  # Absolute price difference
    max_favorable_excursion: float = 0.0  # Best profit seen during position
    max_adverse_excursion: float = 0.0  # Worst drawdown seen during position

    # Duration
    duration_minutes: Optional[int] = None
    bars_held: Optional[int] = None

    # WIN_RATE_MAX profile: Partial profit tracking
    partial_profit_executed: bool = False
    partial_profit_price: Optional[float] = None
    partial_profit_pnl_percent: Optional[float] = None
    partial_profit_ts: Optional[int] = None

    # Additional metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    def update_excursions(self, current_price: float) -> None:
        """Update MFE/MAE based on current price."""
        direction_multiplier = 1 if self.direction == Direction.UP else -1
        pnl_pct = ((current_price - self.open_price) / self.open_price * 100) * direction_multiplier

        # Update max favorable excursion (best profit)
        if pnl_pct > self.max_favorable_excursion:
            self.max_favorable_excursion = pnl_pct

        # Update max adverse excursion (worst drawdown)
        if pnl_pct < self.max_adverse_excursion:
            self.max_adverse_excursion = pnl_pct

    def close_position(
        self,
        close_price: float,
        close_ts: int,
        exit_reason: ExitReason,
        exit_z_er: float,
        exit_z_vol: float
    ) -> None:
        """Close the position and calculate final PnL."""
        self.status = PositionStatus.CLOSED
        self.close_price = close_price
        self.close_ts = close_ts
        self.exit_reason = exit_reason
        self.exit_z_er = exit_z_er
        self.exit_z_vol = exit_z_vol

        # Calculate PnL
        direction_multiplier = 1 if self.direction == Direction.UP else -1
        self.pnl_percent = ((close_price - self.open_price) / self.open_price * 100) * direction_multiplier
        self.pnl_ticks = (close_price - self.open_price) * direction_multiplier

        # Calculate duration
        self.duration_minutes = (close_ts - self.open_ts) // (60 * 1000)
        self.bars_held = self.duration_minutes  # 1-minute bars

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'position_id': self.position_id,
            'event_id': self.event_id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'status': self.status.value,
            'open_price': self.open_price,
            'open_ts': self.open_ts,
            'close_price': self.close_price,
            'close_ts': self.close_ts,
            'exit_reason': self.exit_reason.value if self.exit_reason else None,
            'pnl_percent': self.pnl_percent,
            'pnl_ticks': self.pnl_ticks,
            'duration_minutes': self.duration_minutes,
            'mfe': self.max_favorable_excursion,
            'mae': self.max_adverse_excursion,
            'metrics': self.metrics
        }
