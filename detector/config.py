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
class UniverseConfig:
    """Symbol universe configuration."""
    benchmark_symbol: str = "BTCUSDT"
    symbols: List[str] = field(default_factory=lambda: [
        "ZECUSDT", "DASHUSDT", "XMRUSDT"
    ])

    @property
    def all_symbols(self) -> List[str]:
        """Get all symbols (benchmark + configured symbols)."""
        # Ensure benchmark is first and not duplicated
        if self.benchmark_symbol in self.symbols:
            return [self.benchmark_symbol] + [s for s in self.symbols if s != self.benchmark_symbol]
        return [self.benchmark_symbol] + self.symbols


@dataclass
class WindowsConfig:
    """Time window configuration."""
    bar_interval_sec: int = 60
    zscore_lookback_bars: int = 720
    beta_lookback_bars: int = 240
    beta_aggregation_minutes: int = 5
    initiator_eval_window_bars: int = 15
    confirm_window_bars: int = 60


@dataclass
class ThresholdsConfig:
    """Detection threshold configuration."""
    excess_return_z_initiator: float = 3.0
    volume_z_initiator: float = 3.0
    taker_dominance_min: float = 0.65
    liquidation_z_confirm: float = 2.0
    funding_abs_threshold: float = 0.0010


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


# =========== HYBRID STRATEGY CONFIGURATION ===========


@dataclass
class ClassFilterConfig:
    """Per-class filter configuration for class-aware filtering."""
    # Liquidity thresholds
    min_volume_usd: float = 100000.0
    min_trades_per_bar: int = 50

    # Quality filters
    apply_btc_anomaly_filter: bool = True
    apply_beta_quality_filter: bool = True
    beta_min_abs: float = 0.1
    beta_max_abs: float = 3.0
    beta_min_r_squared: float = 0.2

    # Symbol restrictions
    use_global_blacklist: bool = True
    additional_blacklist: List[str] = field(default_factory=list)

    # Additional filters (for EARLY_SIGNAL)
    require_recent_volume_spike: bool = False
    recent_volume_spike_threshold: float = 1.5  # Multiple of average volume


@dataclass
class ClassAwareFiltersConfig:
    """
    Class-aware filter configuration.

    Different signal classes have different filter strictness:
    - EXTREME_SPIKE: Relaxed filters (signal is reliable itself)
    - STRONG_SIGNAL: Standard filters
    - EARLY_SIGNAL: Strict filters (weak signals need extra validation)
    """
    enabled: bool = False  # Master switch for class-aware filtering

    # EXTREME_SPIKE (z >= 5.0) - Mean-Reversion - Relaxed filters
    extreme_spike: ClassFilterConfig = field(default_factory=lambda: ClassFilterConfig(
        min_volume_usd=25000.0,        # Relaxed from 100000
        min_trades_per_bar=15,         # Relaxed from 50
        apply_btc_anomaly_filter=True,  # Keep - BTC chaos affects everything
        apply_beta_quality_filter=False,  # Disabled - beta less important for MR
        beta_min_abs=0.1,
        beta_max_abs=3.0,
        beta_min_r_squared=0.2,
        use_global_blacklist=True,
        additional_blacklist=[],
        require_recent_volume_spike=False,
        recent_volume_spike_threshold=1.5,
    ))

    # STRONG_SIGNAL (3.0 <= z < 5.0) - Conditional Momentum - Standard filters
    strong_signal: ClassFilterConfig = field(default_factory=lambda: ClassFilterConfig(
        min_volume_usd=100000.0,       # Standard
        min_trades_per_bar=50,         # Standard
        apply_btc_anomaly_filter=True,
        apply_beta_quality_filter=True,
        beta_min_abs=0.1,
        beta_max_abs=3.0,
        beta_min_r_squared=0.2,
        use_global_blacklist=True,
        additional_blacklist=[],
        require_recent_volume_spike=False,
        recent_volume_spike_threshold=1.5,
    ))

    # EARLY_SIGNAL (1.5 <= z < 3.0) - Early Momentum - Strict filters
    early_signal: ClassFilterConfig = field(default_factory=lambda: ClassFilterConfig(
        min_volume_usd=150000.0,       # Stricter - weak signals need liquidity
        min_trades_per_bar=75,         # Stricter
        apply_btc_anomaly_filter=True,
        apply_beta_quality_filter=True,
        beta_min_abs=0.15,             # Stricter
        beta_max_abs=2.5,              # Stricter
        beta_min_r_squared=0.3,        # Stricter
        use_global_blacklist=True,
        additional_blacklist=[],
        require_recent_volume_spike=True,  # Extra validation for weak signals
        recent_volume_spike_threshold=1.5,
    ))


@dataclass
class MeanReversionConfig:
    """Mean-Reversion mode configuration (EXTREME_SPIKE signals, z >= 5.0)."""
    # Entry triggers - wait for reversal signs
    reversal_z_drop_pct: float = 0.20  # Z must drop 20% from signal for entry
    min_bars_before_entry: int = 2  # Minimum 2 bars before entry allowed
    max_bars_before_expiry: int = 8  # TTL: max 8 bars
    require_price_confirmation: bool = True  # Price must move in trade_direction
    require_volume_fade: bool = True  # Volume must be declining

    # Invalidation conditions
    invalidate_on_z_growth: bool = True  # Invalidate if z continues growing
    z_growth_invalidate_threshold: float = 1.10  # If z grows 10% more, invalidate
    max_volume_growth_bars: int = 3  # Invalidate if volume grows N bars consecutively

    # Exit parameters - quick scalps
    atr_target_multiplier: float = 1.0  # Quick TP: 1x ATR
    atr_stop_multiplier: float = 1.2  # Slightly wider stop: 1.2x ATR
    max_hold_minutes: int = 20  # Short holds
    z_score_exit_threshold: float = 1.5  # Exit when z normalizes
    use_trailing_stop: bool = False  # No trailing for quick scalps
    use_z_exit: bool = True  # Use z-score exit


@dataclass
class ConditionalMomentumConfig:
    """Conditional Momentum mode configuration (STRONG_SIGNAL, 3.0 <= z < 5.0)."""
    # Confirmation requirements - ALL must be met
    min_taker_dominance: float = 0.70  # 70% buyers for LONG, 30% for SHORT
    min_volume_retention: float = 0.90  # Volume must not drop more than 10%
    max_z_variance_pct: float = 0.25  # Z variance must be < 25% of signal_z
    require_no_divergence: bool = True  # Price and z must move together
    confirmation_bars: int = 3  # Need 3 bars for stability check
    max_wait_bars: int = 10  # TTL

    # Mode switch conditions (switch to mean-reversion if momentum fails)
    enable_mode_switch: bool = True
    mode_switch_z_drop_pct: float = 0.30  # If z drops 30%, switch to MR
    mode_switch_taker_reversal: float = 0.50  # If taker crosses 50%, switch
    mode_switch_price_reversal: bool = True  # If price reverses past signal, switch

    # Exit parameters - balanced
    atr_target_multiplier: float = 2.0
    atr_stop_multiplier: float = 1.5
    max_hold_minutes: int = 45
    z_score_exit_threshold: float = 0.8
    use_trailing_stop: bool = True
    trailing_stop_activation_atr: float = 1.0  # Activate at +1 ATR
    trailing_stop_distance_atr: float = 0.8  # Trail at 0.8 ATR
    use_z_exit: bool = True


@dataclass
class EarlyMomentumConfig:
    """Early Momentum mode configuration (EARLY_SIGNAL, 1.5 <= z < 3.0)."""
    # Continuation criteria - must see momentum building
    min_z_growth_pct: float = 0.30  # Z must grow 30% from signal
    min_price_follow_through_pct: float = 0.15  # Price must move 0.15% in direction
    volume_must_sustain: bool = True  # Volume must not decline
    min_taker_persistence: float = 0.55  # Min taker over last 3 bars (55% for LONG)

    # Timing
    min_wait_bars: int = 3  # Must wait at least 3 bars
    max_wait_bars: int = 5  # TTL: 5 bars

    # Invalidation
    z_decline_invalidate_pct: float = 0.20  # If z drops 20%, invalidate

    # Exit parameters - long holds for big moves
    atr_target_multiplier: float = 3.0  # Wide target
    atr_stop_multiplier: float = 1.5
    max_hold_minutes: int = 90  # Long holds
    use_trailing_stop: bool = True
    trailing_stop_activation_atr: float = 1.5  # Activate later
    trailing_stop_distance_atr: float = 1.0  # Wider trail
    use_partial_profit: bool = True  # Take partial at 1.5 ATR
    partial_profit_target_atr: float = 1.5
    partial_profit_percent: float = 0.50  # Close 50%
    use_z_exit: bool = False  # Don't exit on z for early signals - let them run


@dataclass
class HybridCommonConfig:
    """Common settings for hybrid strategy."""
    # Volume filter for all signals
    min_volume_z_for_signal: float = 1.5  # Minimum vol_z for any signal

    # Taker thresholds
    taker_bullish_threshold: float = 0.55  # >55% = bullish
    taker_bearish_threshold: float = 0.45  # <45% = bearish
    taker_extreme_bullish: float = 0.70  # >70% = strong bullish
    taker_extreme_bearish: float = 0.30  # <30% = strong bearish

    # Position limits
    max_positions_total: int = 5
    max_positions_per_mode: int = 2


@dataclass
class HybridStrategyConfig:
    """
    Hybrid Strategy configuration.

    Dynamically selects between three trading modes based on signal strength:
    - EXTREME_SPIKE (z >= 5.0): Mean-reversion (fade the move)
    - STRONG_SIGNAL (3.0 <= z < 5.0): Conditional momentum (need confirmation)
    - EARLY_SIGNAL (1.5 <= z < 3.0): Wait for continuation
    """
    enabled: bool = False  # Master switch for hybrid strategy

    # Signal classification thresholds
    extreme_spike_threshold: float = 5.0  # z >= 5.0 = EXTREME_SPIKE
    strong_signal_min: float = 3.0  # z >= 3.0 = STRONG_SIGNAL
    early_signal_min: float = 1.5  # z >= 1.5 = EARLY_SIGNAL

    # Mode-specific configurations
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    conditional_momentum: ConditionalMomentumConfig = field(default_factory=ConditionalMomentumConfig)
    early_momentum: EarlyMomentumConfig = field(default_factory=EarlyMomentumConfig)
    common: HybridCommonConfig = field(default_factory=HybridCommonConfig)

    # Class-aware filtering (replaces WIN_RATE_MAX filters when enabled)
    class_aware_filters: ClassAwareFiltersConfig = field(default_factory=ClassAwareFiltersConfig)


@dataclass
class AdaptiveStopLossConfig:
    """
    Улучшение 1: Адаптивный Stop-Loss.
    SL multiplier зависит от signal class, volatility regime и direction.

    CRITICAL FIX v2: Avg Loss вырос с -0.78% до -1.99%.
    Уменьшены базовые множители и volatility adjustment для более тугих стопов.
    """
    enabled: bool = True

    # Base multipliers по классам сигналов (УМЕНЬШЕНЫ в v2)
    base_multiplier_extreme_spike: float = 1.2  # Было 1.5, теперь 1.2
    base_multiplier_strong_signal: float = 1.4  # Было 1.8, теперь 1.4
    base_multiplier_early_signal: float = 1.6   # Было 2.0, теперь 1.6

    # Volatility adjustment (УМЕНЬШЕНЫ в v2)
    volatility_adjustment_enabled: bool = True
    volatility_lookback_bars: int = 1440  # 24 часа на 1m bars
    high_volatility_percentile: int = 75
    high_volatility_multiplier: float = 1.3  # Было 1.5, теперь 1.3
    low_volatility_percentile: int = 25
    low_volatility_multiplier: float = 0.85  # Было 0.75, теперь 0.85

    # Direction adjustment (лонги рискованнее)
    direction_adjustment_enabled: bool = True
    long_additional_multiplier: float = 1.15  # На 15% шире для лонгов
    short_additional_multiplier: float = 1.0

    # Safety limits (УМЕНЬШЕН max в v2)
    min_stop_distance_pct: float = 0.3  # Минимум 0.3% от entry
    max_stop_distance_pct: float = 3.0  # Было 5.0, теперь 3.0


@dataclass
class TieredTakeProfitLevelConfig:
    """Configuration for one TP level."""
    atr_multiplier: float = 0.5
    close_percent: int = 30


@dataclass
class TieredTakeProfitConfig:
    """
    Улучшение 2: Tiered Take-Profit с частичным закрытием.
    Вместо одного далёкого TP — три уровня с частичным закрытием.
    """
    enabled: bool = True

    # EXTREME_SPIKE уровни (быстрые MR сделки)
    extreme_spike_tp1_atr: float = 0.4
    extreme_spike_tp1_close_pct: int = 30
    extreme_spike_tp2_atr: float = 0.8
    extreme_spike_tp2_close_pct: int = 30
    extreme_spike_tp3_atr: float = 1.2
    extreme_spike_tp3_close_pct: int = 40

    # STRONG_SIGNAL уровни (стандартный momentum)
    strong_signal_tp1_atr: float = 0.5
    strong_signal_tp1_close_pct: int = 30
    strong_signal_tp2_atr: float = 1.0
    strong_signal_tp2_close_pct: int = 30
    strong_signal_tp3_atr: float = 1.5
    strong_signal_tp3_close_pct: int = 40

    # EARLY_SIGNAL уровни (ловим большое движение)
    early_signal_tp1_atr: float = 0.6
    early_signal_tp1_close_pct: int = 30
    early_signal_tp2_atr: float = 1.2
    early_signal_tp2_close_pct: int = 30
    early_signal_tp3_atr: float = 2.0
    early_signal_tp3_close_pct: int = 40

    # Actions после достижения уровней
    move_sl_breakeven_on_tp1: bool = True  # Перенести SL в breakeven после TP1
    activate_trailing_on_tp2: bool = True   # Активировать trailing после TP2


@dataclass
class DelayedZExitConfig:
    """
    Улучшение 3: Отложенный Z-Score Exit.
    Добавляет условия задержки для Z-exit: минимальный profit и время.

    CRITICAL FIX v2: Delayed Z-Exit держит losers слишком долго.
    Добавлен флаг require_min_profit для отключения требования min profit.
    Если require_min_profit=false, Z-exit срабатывает без задержки (стандартное поведение).
    """
    enabled: bool = False  # ИЗМЕНЕНО в v2: отключено по умолчанию

    # CRITICAL FIX v2: Убрать требование min profit
    # Если false - Z-exit срабатывает без задержки на profit (обычное поведение)
    require_min_profit: bool = False  # Было true по умолчанию

    # EXTREME_SPIKE условия (самые мягкие для быстрых MR)
    extreme_spike_min_profit_pct: float = 0.15
    extreme_spike_min_hold_minutes: int = 2
    extreme_spike_partial_close_pct: int = 60

    # STRONG_SIGNAL условия
    strong_signal_min_profit_pct: float = 0.20
    strong_signal_min_hold_minutes: int = 3
    strong_signal_partial_close_pct: int = 50

    # EARLY_SIGNAL условия
    early_signal_min_profit_pct: float = 0.25
    early_signal_min_hold_minutes: int = 4
    early_signal_partial_close_pct: int = 50

    # Partial close при Z-exit
    partial_close_enabled: bool = True
    skip_partial_if_tp1_hit: bool = True  # Если TP1 уже был — закрыть полностью


@dataclass
class DirectionFiltersConfig:
    """
    Улучшение 4: Асимметричная фильтрация Long/Short.
    Разные требования для входа в long и short позиции.
    """
    enabled: bool = True

    # LONG требования (строже, т.к. лонги хуже)
    long_min_extreme_spike_z: float = 6.0  # Выше чем для short (5.0)
    long_min_strong_signal_z: float = 3.5   # Выше чем для short (3.0)
    long_pullback_multiplier: float = 1.25   # На 25% больше pullback
    long_volume_multiplier: float = 1.2      # На 20% больше volume

    # BTC filter для LONG
    long_btc_filter_enabled: bool = True
    long_btc_block_threshold: float = -2.0   # Блокировать если BTC z < -2.0
    long_btc_restrict_threshold: float = -1.0  # Ограничить если BTC z < -1.0

    # Опционально отключить LONG для EARLY_SIGNAL
    long_disable_for_early_signal: bool = False

    # SHORT требования (стандартные)
    short_min_extreme_spike_z: float = 5.0
    short_min_strong_signal_z: float = 3.0
    short_pullback_multiplier: float = 1.0
    short_volume_multiplier: float = 1.0
    short_btc_filter_enabled: bool = False


@dataclass
class TrailingStopByClassConfig:
    """
    Улучшение 5: Активация Trailing Stop по классам.
    Разные параметры активации и дистанции для разных классов.
    """
    enabled: bool = True

    # EXTREME_SPIKE (ранняя активация, tight trail)
    extreme_spike_profit_threshold_pct: float = 0.25
    extreme_spike_distance_atr: float = 0.5

    # STRONG_SIGNAL (стандартная)
    strong_signal_profit_threshold_pct: float = 0.35
    strong_signal_distance_atr: float = 0.7

    # EARLY_SIGNAL (поздняя активация, wide trail)
    early_signal_profit_threshold_pct: float = 0.45
    early_signal_distance_atr: float = 1.0

    # Behavior
    activate_on_tp2: bool = True  # Авто-активация при TP2
    update_frequency: str = "every_bar"
    use_close_price: bool = True  # Сравнивать с close, не high/low


@dataclass
class TimeExitConfig:
    """
    Улучшение 6: Интеллектуальный Time Exit.
    Aggressive time exit для убыточных и flat позиций.

    CRITICAL FIX v2: aggressive_exits убивает систему (38.5% позиций в убытке).
    По умолчанию aggressive_exits ОТКЛЮЧЕНЫ. Включайте только после тестирования
    с мягкими настройками (threshold -0.8%, time 12+ мин, с grace period).
    """
    enabled: bool = True

    # =========================================================================
    # CRITICAL FIX v2: Aggressive exits controls
    # =========================================================================
    # Главный выключатель агрессивных time exits (LOSING и FLAT).
    # ОТКЛЮЧЕНО по умолчанию - причина 38.5% убыточных закрытий.
    aggressive_exits_enabled: bool = False  # ГЛАВНОЕ ИЗМЕНЕНИЕ - отключено!

    # Отдельный контроль для flat positions (|pnl| < threshold)
    flat_position_exit_enabled: bool = False  # Отключено - flat может ждать breakout

    # Grace period: не применять aggressive exits первые N минут
    # Позиции могут быть в минусе сразу после открытия из-за spread/slippage
    grace_period_minutes: int = 3  # Первые 3 мин - только SL/TP

    # =========================================================================
    # Conditional mode: закрывать по времени ТОЛЬКО при комбинации факторов
    # =========================================================================
    conditional_mode_enabled: bool = False  # По умолчанию отключено
    conditional_min_loss_pct: float = -0.5  # Мин убыток для срабатывания
    conditional_require_z_reversal: bool = True  # z должен быть против нас
    conditional_require_flow_reversal: bool = True  # taker flow против нас
    conditional_min_time_minutes: int = 8  # Мин время в позиции

    # =========================================================================
    # EXTREME_SPIKE settings (СМЯГЧЕНЫ в v2)
    # =========================================================================
    extreme_spike_losing_threshold_pct: float = -0.8  # Было -0.25, теперь -0.8
    extreme_spike_losing_max_minutes: int = 12  # Было 4, теперь 12
    extreme_spike_flat_threshold_pct: float = 0.1
    extreme_spike_flat_max_minutes: int = 6
    extreme_spike_max_hold_minutes: int = 20

    # =========================================================================
    # STRONG_SIGNAL settings (СМЯГЧЕНЫ в v2)
    # =========================================================================
    strong_signal_losing_threshold_pct: float = -0.9  # Было -0.30, теперь -0.9
    strong_signal_losing_max_minutes: int = 14  # Было 5, теперь 14
    strong_signal_flat_threshold_pct: float = 0.1
    strong_signal_flat_max_minutes: int = 8
    strong_signal_max_hold_minutes: int = 45

    # =========================================================================
    # EARLY_SIGNAL settings (СМЯГЧЕНЫ в v2)
    # =========================================================================
    early_signal_losing_threshold_pct: float = -1.0  # Было -0.35, теперь -1.0
    early_signal_losing_max_minutes: int = 15  # Было 6, теперь 15
    early_signal_flat_threshold_pct: float = 0.1
    early_signal_flat_max_minutes: int = 10
    early_signal_max_hold_minutes: int = 90


@dataclass
class MinProfitFilterConfig:
    """
    Улучшение 7: Минимальный Profit Filter.
    Не открывать позицию если expected profit слишком мал.
    """
    enabled: bool = True

    # Fees estimation
    estimated_fees_pct: float = 0.10  # Round-trip fees (0.04% * 2 + buffer)

    # Minimum expected profit (до комиссий)
    min_expected_profit_pct: float = 0.35

    # Check against TP level
    check_against: str = "tp1"  # "tp1", "tp2", "tp3"

    # Per-class thresholds (опционально)
    extreme_spike_min_profit_pct: float = 0.30
    strong_signal_min_profit_pct: float = 0.35
    early_signal_min_profit_pct: float = 0.40


@dataclass
class PositionManagementConfig:
    """
    Virtual position management configuration.

    CRITICAL FIX v2: Added grace_period_minutes to prevent aggressive exits
    from closing positions that are in temporary drawdown due to spread/slippage.
    """
    enabled: bool = True
    allow_multiple_positions: bool = False  # Allow multiple positions per symbol

    # CRITICAL FIX v2: Grace period for new positions
    # Первые N минут не применять aggressive exits (TIME_EXIT_LOSING, TIME_EXIT_FLAT)
    # Позиции могут быть в минусе сразу после открытия из-за spread/slippage
    grace_period_minutes: int = 3  # Первые 3 мин - только SL/TP, без aggressive time exits

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

    # =========== TRADING IMPROVEMENTS (7 улучшений) ===========
    adaptive_stop_loss: AdaptiveStopLossConfig = field(default_factory=AdaptiveStopLossConfig)
    tiered_take_profit: TieredTakeProfitConfig = field(default_factory=TieredTakeProfitConfig)
    delayed_z_exit: DelayedZExitConfig = field(default_factory=DelayedZExitConfig)
    direction_filters: DirectionFiltersConfig = field(default_factory=DirectionFiltersConfig)
    trailing_stop_by_class: TrailingStopByClassConfig = field(default_factory=TrailingStopByClassConfig)
    time_exit: TimeExitConfig = field(default_factory=TimeExitConfig)
    min_profit_filter: MinProfitFilterConfig = field(default_factory=MinProfitFilterConfig)


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
    position_management: PositionManagementConfig = field(default_factory=PositionManagementConfig)
    hybrid_strategy: HybridStrategyConfig = field(default_factory=HybridStrategyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Parse nested configs
        config = cls()

        if 'api' in data:
            config.api = ApiConfig(**data['api'])

        if 'universe' in data:
            universe_data = data['universe']
            # Support old 'sectors' format - flatten to symbols list
            if 'sectors' in universe_data:
                symbols = []
                for sector_data in universe_data['sectors']:
                    symbols.extend(sector_data.get('symbols', []))
                universe_data['symbols'] = symbols
                del universe_data['sectors']
            # Support old 'sector_symbols' format
            elif 'sector_symbols' in universe_data:
                universe_data['symbols'] = universe_data['sector_symbols']
                del universe_data['sector_symbols']
            config.universe = UniverseConfig(**universe_data)

        if 'windows' in data:
            # Filter out removed parameters for backward compatibility
            windows_data = {k: v for k, v in data['windows'].items()
                          if k not in {'sector_diffusion_window_bars'}}
            config.windows = WindowsConfig(**windows_data)

        if 'thresholds' in data:
            # Filter out removed parameters for backward compatibility
            thresholds_data = {k: v for k, v in data['thresholds'].items()
                             if k not in {'sector_k_min', 'sector_share_min', 'oi_delta_z_confirm'}}
            config.thresholds = ThresholdsConfig(**thresholds_data)

        if 'alerts' in data:
            alerts_data = data['alerts']
            if 'telegram' in alerts_data:
                telegram = TelegramConfig(**alerts_data['telegram'])
                alerts_data['telegram'] = telegram
            config.alerts = AlertsConfig(**alerts_data)

        if 'storage' in data:
            config.storage = StorageConfig(**data['storage'])

        # Ignore 'diffusion' section for backward compatibility (removed feature)

        if 'position_management' in data:
            pm_data = data['position_management'].copy()
            # Parse WIN_RATE_MAX profile if present
            if 'win_rate_max_profile' in pm_data:
                win_rate_cfg = WinRateMaxProfileConfig(**pm_data['win_rate_max_profile'])
                pm_data['win_rate_max_profile'] = win_rate_cfg
            # Parse Trading Improvements configs (7 улучшений)
            if 'adaptive_stop_loss' in pm_data:
                pm_data['adaptive_stop_loss'] = AdaptiveStopLossConfig(**pm_data['adaptive_stop_loss'])
            if 'tiered_take_profit' in pm_data:
                pm_data['tiered_take_profit'] = TieredTakeProfitConfig(**pm_data['tiered_take_profit'])
            if 'delayed_z_exit' in pm_data:
                pm_data['delayed_z_exit'] = DelayedZExitConfig(**pm_data['delayed_z_exit'])
            if 'direction_filters' in pm_data:
                pm_data['direction_filters'] = DirectionFiltersConfig(**pm_data['direction_filters'])
            if 'trailing_stop_by_class' in pm_data:
                pm_data['trailing_stop_by_class'] = TrailingStopByClassConfig(**pm_data['trailing_stop_by_class'])
            if 'time_exit' in pm_data:
                pm_data['time_exit'] = TimeExitConfig(**pm_data['time_exit'])
            if 'min_profit_filter' in pm_data:
                pm_data['min_profit_filter'] = MinProfitFilterConfig(**pm_data['min_profit_filter'])
            config.position_management = PositionManagementConfig(**pm_data)

        # Parse hybrid_strategy configuration
        if 'hybrid_strategy' in data:
            hs_data = data['hybrid_strategy'].copy()
            # Parse nested mode configs
            if 'mean_reversion' in hs_data:
                hs_data['mean_reversion'] = MeanReversionConfig(**hs_data['mean_reversion'])
            if 'conditional_momentum' in hs_data:
                hs_data['conditional_momentum'] = ConditionalMomentumConfig(**hs_data['conditional_momentum'])
            if 'early_momentum' in hs_data:
                hs_data['early_momentum'] = EarlyMomentumConfig(**hs_data['early_momentum'])
            if 'common' in hs_data:
                hs_data['common'] = HybridCommonConfig(**hs_data['common'])
            # Parse class_aware_filters config
            if 'class_aware_filters' in hs_data:
                caf_data = hs_data['class_aware_filters'].copy()
                # Parse per-class configs
                if 'extreme_spike' in caf_data:
                    caf_data['extreme_spike'] = ClassFilterConfig(**caf_data['extreme_spike'])
                if 'strong_signal' in caf_data:
                    caf_data['strong_signal'] = ClassFilterConfig(**caf_data['strong_signal'])
                if 'early_signal' in caf_data:
                    caf_data['early_signal'] = ClassFilterConfig(**caf_data['early_signal'])
                hs_data['class_aware_filters'] = ClassAwareFiltersConfig(**caf_data)
            config.hybrid_strategy = HybridStrategyConfig(**hs_data)

        if 'runtime' in data:
            config.runtime = RuntimeConfig(**data['runtime'])

        return config

    def validate(self) -> None:
        """Validate configuration."""
        # Check symbols
        if not self.universe.benchmark_symbol:
            raise ValueError("benchmark_symbol cannot be empty")

        if not self.universe.symbols:
            raise ValueError("At least one symbol must be configured")

        # Check for duplicate symbols
        seen = set()
        for symbol in self.universe.symbols:
            if symbol in seen:
                raise ValueError(f"Duplicate symbol: {symbol}")
            seen.add(symbol)

        # Check windows
        if self.windows.bar_interval_sec <= 0:
            raise ValueError("bar_interval_sec must be positive")

        if self.windows.zscore_lookback_bars < 10:
            raise ValueError("zscore_lookback_bars must be at least 10")

        # Check thresholds
        if self.thresholds.taker_dominance_min < 0.5 or self.thresholds.taker_dominance_min > 1.0:
            raise ValueError("taker_dominance_min must be between 0.5 and 1.0")

        # Check storage
        storage_path = Path(self.storage.sqlite_path)
        if not storage_path.parent.exists():
            storage_path.parent.mkdir(parents=True, exist_ok=True)
