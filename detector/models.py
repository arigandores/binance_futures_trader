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
