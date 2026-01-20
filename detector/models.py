"""Data models for the Binance Anomaly Detector."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class StreamType(str, Enum):
    """WebSocket stream types."""
    AGG_TRADE = "aggTrade"
    MARK_PRICE = "markPrice"


class Direction(str, Enum):
    """Market direction."""
    UP = "UP"
    DOWN = "DOWN"


class SignalClass(str, Enum):
    """
    Signal classification based on z-score magnitude.

    EXTREME_SPIKE: z >= 5.0 - Mean-reversion (fade the move)
    STRONG_SIGNAL: 3.0 <= z < 5.0 - Conditional momentum (need confirmation)
    EARLY_SIGNAL: 1.5 <= z < 3.0 - Wait for continuation
    """
    EXTREME_SPIKE = "EXTREME_SPIKE"
    STRONG_SIGNAL = "STRONG_SIGNAL"
    EARLY_SIGNAL = "EARLY_SIGNAL"


class TradingMode(str, Enum):
    """
    Trading mode determines entry triggers and exit parameters.

    MEAN_REVERSION: Trade AGAINST the anomaly (fade extreme moves)
    CONDITIONAL_MOMENTUM: Trade WITH the anomaly (if confirmed)
    EARLY_MOMENTUM: Trade WITH early signals (if continuation confirmed)
    """
    MEAN_REVERSION = "MEAN_REVERSION"
    CONDITIONAL_MOMENTUM = "CONDITIONAL_MOMENTUM"
    EARLY_MOMENTUM = "EARLY_MOMENTUM"




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

    # Funding rate (from markPrice stream, None during backfill)
    funding: Optional[float] = None

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

    # Funding rate (from markPrice stream)
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
    """Detected anomaly event (initiator signal)."""
    event_id: str
    ts: int  # Event timestamp, milliseconds since epoch
    initiator_symbol: str
    direction: Direction
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Hybrid Strategy: Signal classification
    signal_class: Optional[SignalClass] = None  # EXTREME_SPIKE, STRONG_SIGNAL, EARLY_SIGNAL
    original_z_score: Optional[float] = None  # Exact z_er_15m at detection
    original_vol_z: Optional[float] = None  # Exact z_vol_15m at detection
    original_taker_share: Optional[float] = None  # taker_share at detection
    original_price: Optional[float] = None  # bar.close at detection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'event_id': self.event_id,
            'ts': self.ts,
            'initiator_symbol': self.initiator_symbol,
            'direction': self.direction.value,
            'metrics': self.metrics,
            'signal_class': self.signal_class.value if self.signal_class else None,
            'original_z_score': self.original_z_score,
            'original_vol_z': self.original_vol_z,
            'original_taker_share': self.original_taker_share,
            'original_price': self.original_price
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
    invalidation_reason: Optional[str] = None  # Short code: "flow_died", "momentum_died", etc.
    invalidation_details: Optional[str] = None  # Human-readable detailed explanation

    # Trigger state (updated on each bar)
    z_cooldown_met: bool = False
    pullback_met: bool = False
    stability_met: bool = False

    # WIN_RATE_MAX profile: Additional tracking fields
    re_expansion_met: bool = False  # Re-expansion confirmed (1 of 3 methods)
    pullback_exceeded_max: bool = False  # Structure broken (pullback too deep)
    z_cooldown_in_range: bool = False  # Z-score in [min, max] range
    flow_death_bar_count: int = 0  # Consecutive bars with low dominance

    # =========== HYBRID STRATEGY FIELDS ===========
    # Signal classification
    signal_class: Optional[SignalClass] = None  # EXTREME_SPIKE, STRONG_SIGNAL, EARLY_SIGNAL
    trading_mode: Optional[TradingMode] = None  # MEAN_REVERSION, CONDITIONAL_MOMENTUM, EARLY_MOMENTUM
    trade_direction: Optional[Direction] = None  # May differ from direction for mean-reversion!
    original_direction: Optional[Direction] = None  # Original z-score direction (preserved)

    # History tracking for hybrid strategy evaluation
    z_history: List[float] = field(default_factory=list)  # z_er_15m per bar
    vol_history: List[float] = field(default_factory=list)  # z_vol_15m per bar
    taker_history: List[float] = field(default_factory=list)  # taker_share per bar
    price_history: List[float] = field(default_factory=list)  # close price per bar

    # State tracking for mode switching and confirmation
    mode_switched: bool = False  # Strong signal switched to mean-reversion
    continuation_confirmed: bool = False  # Early signal continuation confirmed
    reversal_started: bool = False  # Mean-reversion: reversal detection confirmed

    # Mean-Reversion specific
    reversal_bar_count: int = 0  # Bars since reversal started
    volume_growth_streak: int = 0  # Consecutive bars of volume growth (for invalidation)

    # Strong Signal specific
    z_stable_bar_count: int = 0  # Bars where z-score remained stable

    # Early Signal specific
    early_wait_complete: bool = False  # Passed minimum wait bars (3)

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

    # =========== HYBRID STRATEGY METHODS ===========

    def update_history(self, features: 'Features', bar: 'Bar') -> None:
        """
        Update history tracking for hybrid strategy evaluation.
        Called on each bar evaluation.
        """
        if features.z_er_15m is not None:
            self.z_history.append(features.z_er_15m)
        if features.z_vol_15m is not None:
            self.vol_history.append(features.z_vol_15m)
        if features.taker_buy_share_15m is not None:
            self.taker_history.append(features.taker_buy_share_15m)
        self.price_history.append(bar.close)

    def get_z_variance(self, last_n_bars: int = 3) -> Optional[float]:
        """
        Calculate variance of z-scores over last N bars.
        Used for STRONG_SIGNAL stability check.
        """
        if len(self.z_history) < last_n_bars:
            return None
        recent = self.z_history[-last_n_bars:]
        mean_z = sum(recent) / len(recent)
        variance = sum((z - mean_z) ** 2 for z in recent) / len(recent)
        return variance

    def get_z_growth_pct(self) -> Optional[float]:
        """
        Calculate z-score growth percentage from signal.
        Used for EARLY_SIGNAL continuation check.
        """
        if not self.z_history or self.signal_z_er == 0:
            return None
        current_z = self.z_history[-1]
        # For both directions, we want to see z moving further from zero
        growth = (abs(current_z) - abs(self.signal_z_er)) / abs(self.signal_z_er)
        return growth

    def get_z_drop_pct(self) -> Optional[float]:
        """
        Calculate z-score drop percentage from signal.
        Used for EXTREME_SPIKE reversal check.
        """
        if not self.z_history or self.signal_z_er == 0:
            return None
        current_z = self.z_history[-1]
        # Measure how much z has dropped towards zero
        drop = 1.0 - (abs(current_z) / abs(self.signal_z_er))
        return drop

    def get_avg_volume_early(self) -> Optional[float]:
        """Get average volume z-score for first few bars (0-2)."""
        if len(self.vol_history) < 3:
            return None
        return sum(self.vol_history[:3]) / 3

    def get_avg_volume_current(self) -> Optional[float]:
        """Get average volume z-score for recent bars (3-5)."""
        if len(self.vol_history) < 6:
            # Use what we have for bars 3+
            if len(self.vol_history) > 3:
                recent = self.vol_history[3:]
                return sum(recent) / len(recent)
            return None
        return sum(self.vol_history[3:6]) / 3

    def get_min_taker_recent(self, last_n_bars: int = 3) -> Optional[float]:
        """Get minimum taker share over last N bars."""
        if len(self.taker_history) < last_n_bars:
            return None
        return min(self.taker_history[-last_n_bars:])

    def get_price_change_pct(self) -> Optional[float]:
        """
        Calculate price change percentage from signal.
        Positive = price moved in signal direction.
        """
        if not self.price_history or self.signal_price == 0:
            return None
        current_price = self.price_history[-1]
        change = (current_price - self.signal_price) / self.signal_price * 100
        # Adjust for direction: positive means price moved in signal direction
        if self.original_direction == Direction.DOWN:
            change = -change
        return change




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
    Z_SCORE_REVERSAL_PARTIAL = "Z_SCORE_REVERSAL_PARTIAL"  # Delayed z-exit: partial close
    STOP_LOSS = "STOP_LOSS"  # Price moved against position
    TAKE_PROFIT = "TAKE_PROFIT"  # Target profit reached (legacy single TP)
    TAKE_PROFIT_TP1 = "TAKE_PROFIT_TP1"  # Tiered TP: first level (30%)
    TAKE_PROFIT_TP2 = "TAKE_PROFIT_TP2"  # Tiered TP: second level (30%)
    TAKE_PROFIT_TP3 = "TAKE_PROFIT_TP3"  # Tiered TP: final level (40%)
    TIME_EXIT = "TIME_EXIT"  # Maximum holding time reached
    TIME_EXIT_LOSING = "TIME_EXIT_LOSING"  # Aggressive exit: losing position timeout
    TIME_EXIT_FLAT = "TIME_EXIT_FLAT"  # Aggressive exit: flat position timeout
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
    entry_reason_details: Optional[str] = None  # Human-readable entry trigger explanation

    # Exit data (None if still open)
    close_price: Optional[float] = None
    close_ts: Optional[int] = None
    exit_z_er: Optional[float] = None
    exit_z_vol: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    exit_reason_details: Optional[str] = None  # Human-readable detailed explanation

    # PnL metrics
    pnl_percent: Optional[float] = None  # (close - open) / open * 100 * direction_multiplier
    max_favorable_excursion: float = 0.0  # Best profit seen during position
    max_adverse_excursion: float = 0.0  # Worst drawdown seen during position

    # Duration
    duration_minutes: Optional[int] = None

    # WIN_RATE_MAX profile: Partial profit tracking
    partial_profit_executed: bool = False
    partial_profit_price: Optional[float] = None
    partial_profit_pnl_percent: Optional[float] = None
    partial_profit_ts: Optional[int] = None

    # =========== TIERED TAKE-PROFIT (Улучшение 2) ===========
    # TP levels calculated at entry (based on ATR and signal class)
    tp1_price: Optional[float] = None
    tp2_price: Optional[float] = None
    tp3_price: Optional[float] = None
    tp1_hit: bool = False  # First level reached (30%)
    tp2_hit: bool = False  # Second level reached (30%)
    tp3_hit: bool = False  # Final level reached (40%)
    remaining_quantity_pct: float = 100.0  # Remaining position size (%)
    sl_moved_to_breakeven: bool = False  # SL moved to entry after TP1

    # =========== ADAPTIVE STOP-LOSS (Улучшение 1) ===========
    adaptive_stop_price: Optional[float] = None  # Dynamically calculated stop
    adaptive_stop_multiplier: Optional[float] = None  # Final calculated multiplier

    # =========== TRAILING STOP BY CLASS (Улучшение 5) ===========
    trailing_active: bool = False
    trailing_price: Optional[float] = None  # Current trail level
    trailing_activation_profit: Optional[float] = None  # Profit % when activated
    trailing_distance_atr: Optional[float] = None  # ATR distance for this position

    # =========== SIGNAL CLASS (for class-based logic) ===========
    signal_class: Optional[str] = None  # EXTREME_SPIKE, STRONG_SIGNAL, EARLY_SIGNAL

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

        # Calculate duration
        self.duration_minutes = (close_ts - self.open_ts) // (60 * 1000)

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
            'duration_minutes': self.duration_minutes,
            'mfe': self.max_favorable_excursion,
            'mae': self.max_adverse_excursion,
            'metrics': self.metrics
        }
