"""Configuration loading and validation."""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import yaml


@dataclass
class ApiConfig:
    """API credentials configuration."""
    key: str = ""
    secret: str = ""

    @property
    def has_credentials(self) -> bool:
        """Check if API credentials are provided."""
        return bool(self.key and self.secret)


@dataclass
class SectorConfig:
    """Single sector configuration."""
    name: str
    symbols: List[str]

    def __post_init__(self):
        """Validate sector configuration."""
        if not self.name:
            raise ValueError("Sector name cannot be empty")
        if not self.symbols:
            raise ValueError(f"Sector '{self.name}' must have at least one symbol")


@dataclass
class UniverseConfig:
    """Symbol universe configuration."""
    benchmark_symbol: str = "BTCUSDT"
    sectors: List[SectorConfig] = field(default_factory=lambda: [
        SectorConfig(name="Privacy", symbols=["ZECUSDT", "DASHUSDT", "XMRUSDT"])
    ])

    @property
    def all_symbols(self) -> List[str]:
        """Get all symbols (benchmark + all sectors)."""
        sector_symbols = []
        for sector in self.sectors:
            sector_symbols.extend(sector.symbols)
        return [self.benchmark_symbol] + sector_symbols

    @property
    def sector_symbols(self) -> List[str]:
        """Get all sector symbols (for backward compatibility)."""
        symbols = []
        for sector in self.sectors:
            symbols.extend(sector.symbols)
        return symbols

    def get_sector_for_symbol(self, symbol: str) -> Optional[SectorConfig]:
        """Get the sector config for a given symbol."""
        for sector in self.sectors:
            if symbol in sector.symbols:
                return sector
        return None

    def get_sector_symbols(self, symbol: str) -> List[str]:
        """Get all symbols in the same sector as the given symbol."""
        sector = self.get_sector_for_symbol(symbol)
        if sector:
            return sector.symbols
        return []


@dataclass
class WindowsConfig:
    """Time window configuration."""
    bar_interval_sec: int = 60
    zscore_lookback_bars: int = 720
    beta_lookback_bars: int = 240
    beta_aggregation_minutes: int = 5
    initiator_eval_window_bars: int = 15
    confirm_window_bars: int = 60
    sector_diffusion_window_bars: int = 120


@dataclass
class ThresholdsConfig:
    """Detection threshold configuration."""
    excess_return_z_initiator: float = 3.0
    volume_z_initiator: float = 3.0
    taker_dominance_min: float = 0.65
    liquidation_z_confirm: float = 2.0
    funding_abs_threshold: float = 0.0010
    sector_k_min: int = 2
    sector_share_min: float = 0.40


@dataclass
class TelegramConfig:
    """Telegram alert configuration."""
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


@dataclass
class AlertsConfig:
    """Alert configuration."""
    cooldown_minutes_per_symbol: int = 60
    direction_swap_grace_minutes: int = 15
    telegram: TelegramConfig = field(default_factory=TelegramConfig)


@dataclass
class StorageConfig:
    """Storage configuration."""
    sqlite_path: str = "./data/market.db"
    wal_mode: bool = True
    batch_write_interval_sec: int = 5


@dataclass
class DiffusionConfig:
    """Sector diffusion configuration."""
    mode: str = "after_initiator"


@dataclass
class WinRateMaxProfileConfig:
    """WIN_RATE_MAX profile configuration for win-rate optimization."""
    # Watch window (shorter than DEFAULT)
    entry_trigger_max_wait_minutes: int = 6
    entry_trigger_min_wait_bars: int = 1

    # Z-score cooldown range (min AND max)
    entry_trigger_z_cooldown_min: float = 2.0
    entry_trigger_z_cooldown_max: float = 2.7

    # Pullback range (min AND max)
    entry_trigger_pullback_min_pct: float = 0.8
    entry_trigger_pullback_max_pct: float = 2.0
    entry_trigger_pullback_min_atr: float = 0.5
    entry_trigger_pullback_max_atr: float = 2.2

    # Taker flow (stricter)
    entry_trigger_taker_stability: float = 0.06
    entry_trigger_min_taker_dominance: float = 0.58

    # Re-expansion confirmation (NEW)
    require_re_expansion: bool = True
    re_expansion_price_action: bool = True
    re_expansion_micro_impulse: bool = True
    re_expansion_flow_acceleration: bool = True

    # Invalidation thresholds (NEW)
    invalidate_z_er_min: float = 1.8
    invalidate_taker_dominance_min: float = 0.52
    invalidate_taker_dominance_bars: int = 2

    # Exit parameters (win-rate biased)
    atr_stop_multiplier: float = 1.5
    atr_target_multiplier: float = 2.0  # Reduced from 3.0
    trailing_stop_activation: float = 0.35  # Reduced from 0.50
    trailing_stop_distance_atr: float = 0.8  # Reduced from 1.0
    order_flow_reversal_threshold: float = 0.12  # Reduced from 0.15

    # Partial profit taking (NEW)
    use_partial_profit: bool = True
    partial_profit_percent: float = 0.5
    partial_profit_target_atr: float = 1.0
    partial_profit_move_sl_breakeven: bool = True

    # Time exit (NEW)
    time_exit_enabled: bool = True
    time_exit_minutes: int = 25
    time_exit_min_pnl_atr_mult: float = 0.5

    # Market regime gating (NEW)
    btc_anomaly_filter: bool = True
    btc_anomaly_lookback_minutes: int = 15

    # Symbol quality filter
    symbol_quality_filter: bool = True
    symbol_blacklist: List[str] = field(default_factory=list)
    min_volume_usd: float = 100000.0  # Minimum 1m volume in USD
    min_trades_per_bar: int = 50  # Minimum trades per 1m bar

    # Beta quality filter (optional)
    beta_quality_filter: bool = False
    beta_min_r_squared: float = 0.2
    beta_min_abs: float = 0.1  # Renamed from beta_min_value for clarity
    beta_max_abs: float = 3.0  # Renamed from beta_max_value for clarity


@dataclass
class PositionManagementConfig:
    """Virtual position management configuration."""
    enabled: bool = True
    allow_multiple_positions: bool = False  # Allow multiple positions per symbol

    # Profile selection
    profile: str = "DEFAULT"  # Options: "DEFAULT", "WIN_RATE_MAX"

    # Entry Triggers (Signal+Trigger separation) - FIXED ARCHITECTURE
    use_entry_triggers: bool = False  # Enable pending signals queue (watch window)
    entry_trigger_max_wait_minutes: int = 10  # Max watch window (TTL) - NOT fixed delay!
    entry_trigger_min_wait_bars: int = 0  # Optional: min bars before allowing entry (0 = immediate if ready)
    entry_trigger_z_cooldown: float = 2.0  # Min z-score after cooling [2.0, 3.0]
    entry_trigger_pullback_pct: float = 0.5  # Required pullback % from peak_since_signal
    entry_trigger_taker_stability: float = 0.10  # Max taker flow change (stability check)
    entry_trigger_min_taker_dominance: float = 0.55  # Must-Fix #7: Min buy/sell dominance (0.55 = 55%)
    entry_trigger_require_data: bool = True  # Must-Fix #4: Fail-closed if data missing (safer)

    # Dynamic Stops/Targets (IMPROVED)
    atr_period: int = 10  # Shorter period for crypto volatility (was 14)
    atr_stop_multiplier: float = 1.5  # Tighter stops (1.5x ATR, was 2.0x)
    atr_target_multiplier: float = 3.0  # Wider targets (3x ATR for 1:2 ratio)
    min_risk_reward_ratio: float = 2.0  # Minimum R:R ratio enforced
    use_atr_stops: bool = True

    # Trailing Stop (NEW)
    use_trailing_stop: bool = False  # Enable trailing stops
    trailing_stop_activation: float = 0.5  # Activate at 50% of TP
    trailing_stop_distance_atr: float = 1.0  # Trail at 1x ATR distance

    # Exit Conditions (RELAXED)
    z_score_exit_threshold: float = 0.5  # More lenient (was 1.0)
    stop_loss_percent: float = 2.0  # Used if ATR unavailable
    take_profit_percent: float = 4.0  # Increased from 3.0 for better R:R
    max_hold_minutes: int = 120  # Allow more time (was 60)

    # Order Flow Reversal
    exit_on_order_flow_reversal: bool = True
    order_flow_reversal_threshold: float = 0.15  # 15% change in taker flow

    # Opposite Signal Exit
    exit_on_opposite_signal: bool = False
    opposite_signal_threshold: float = 2.5  # Z-score threshold for opposite direction

    # WIN_RATE_MAX profile configuration
    win_rate_max_profile: WinRateMaxProfileConfig = field(default_factory=WinRateMaxProfileConfig)


@dataclass
class RuntimeConfig:
    """Runtime configuration."""
    log_level: str = "INFO"
    rest_poll_sec: int = 60
    ws_reconnect_backoff_sec: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 30])
    clock_skew_tolerance_sec: int = 2


@dataclass
class Config:
    """Main configuration container."""
    api: ApiConfig = field(default_factory=ApiConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    windows: WindowsConfig = field(default_factory=WindowsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    position_management: PositionManagementConfig = field(default_factory=PositionManagementConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse nested configs
        config = cls()

        if 'api' in data:
            config.api = ApiConfig(**data['api'])

        if 'universe' in data:
            universe_data = data['universe']
            # Parse sectors if present
            if 'sectors' in universe_data:
                sectors = []
                for sector_data in universe_data['sectors']:
                    sectors.append(SectorConfig(**sector_data))
                universe_data['sectors'] = sectors
            # Support old format (sector_symbols) for backward compatibility
            elif 'sector_symbols' in universe_data:
                sectors = [SectorConfig(name="Default", symbols=universe_data['sector_symbols'])]
                universe_data['sectors'] = sectors
                del universe_data['sector_symbols']
            config.universe = UniverseConfig(**universe_data)

        if 'windows' in data:
            config.windows = WindowsConfig(**data['windows'])

        if 'thresholds' in data:
            config.thresholds = ThresholdsConfig(**data['thresholds'])

        if 'alerts' in data:
            alerts_data = data['alerts']
            if 'telegram' in alerts_data:
                telegram = TelegramConfig(**alerts_data['telegram'])
                alerts_data['telegram'] = telegram
            config.alerts = AlertsConfig(**alerts_data)

        if 'storage' in data:
            config.storage = StorageConfig(**data['storage'])

        if 'diffusion' in data:
            config.diffusion = DiffusionConfig(**data['diffusion'])

        if 'position_management' in data:
            pm_data = data['position_management']
            # Parse WIN_RATE_MAX profile if present
            if 'win_rate_max_profile' in pm_data:
                win_rate_cfg = WinRateMaxProfileConfig(**pm_data['win_rate_max_profile'])
                pm_data['win_rate_max_profile'] = win_rate_cfg
            config.position_management = PositionManagementConfig(**pm_data)

        if 'runtime' in data:
            config.runtime = RuntimeConfig(**data['runtime'])

        return config

    def validate(self) -> None:
        """Validate configuration."""
        # Check symbols
        if not self.universe.benchmark_symbol:
            raise ValueError("benchmark_symbol cannot be empty")

        if not self.universe.sectors:
            raise ValueError("At least one sector must be configured")

        # Check for duplicate symbols across sectors
        all_sector_symbols = []
        for sector in self.universe.sectors:
            for symbol in sector.symbols:
                if symbol in all_sector_symbols:
                    raise ValueError(f"Symbol {symbol} appears in multiple sectors")
                all_sector_symbols.append(symbol)

        # Check windows
        if self.windows.bar_interval_sec <= 0:
            raise ValueError("bar_interval_sec must be positive")

        if self.windows.zscore_lookback_bars < 10:
            raise ValueError("zscore_lookback_bars must be at least 10")

        # Check thresholds
        if self.thresholds.taker_dominance_min < 0.5 or self.thresholds.taker_dominance_min > 1.0:
            raise ValueError("taker_dominance_min must be between 0.5 and 1.0")

        if self.thresholds.sector_k_min < 1:
            raise ValueError("sector_k_min must be at least 1")

        if self.thresholds.sector_share_min <= 0 or self.thresholds.sector_share_min > 1.0:
            raise ValueError("sector_share_min must be between 0 and 1.0")

        # Check storage
        storage_path = Path(self.storage.sqlite_path)
        if not storage_path.parent.exists():
            storage_path.parent.mkdir(parents=True, exist_ok=True)
