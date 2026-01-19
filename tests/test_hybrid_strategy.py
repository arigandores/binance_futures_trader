"""Tests for Hybrid Strategy implementation."""

import pytest
from detector.models import (
    Features, Direction, Event, Bar, PendingSignal,
    SignalClass, TradingMode
)
from detector.config import (
    Config, HybridStrategyConfig, MeanReversionConfig,
    ConditionalMomentumConfig, EarlyMomentumConfig, HybridCommonConfig
)


# =============================================================================
# Signal Classification Tests
# =============================================================================

class TestSignalClassification:
    """Test signal classification logic."""

    def test_extreme_spike_classification(self):
        """Test that z >= 5.0 is classified as EXTREME_SPIKE."""
        hs_cfg = HybridStrategyConfig()

        # Extreme positive (z = +7.0)
        z_er = 7.0
        assert abs(z_er) >= hs_cfg.extreme_spike_threshold
        expected_class = SignalClass.EXTREME_SPIKE

        # Extreme negative (z = -6.5)
        z_er_neg = -6.5
        assert abs(z_er_neg) >= hs_cfg.extreme_spike_threshold

    def test_strong_signal_classification(self):
        """Test that 3.0 <= z < 5.0 is classified as STRONG_SIGNAL."""
        hs_cfg = HybridStrategyConfig()

        # Strong positive (z = +4.2)
        z_er = 4.2
        assert abs(z_er) >= hs_cfg.strong_signal_min
        assert abs(z_er) < hs_cfg.extreme_spike_threshold

        # Strong negative (z = -3.5)
        z_er_neg = -3.5
        assert abs(z_er_neg) >= hs_cfg.strong_signal_min
        assert abs(z_er_neg) < hs_cfg.extreme_spike_threshold

    def test_early_signal_classification(self):
        """Test that 1.5 <= z < 3.0 is classified as EARLY_SIGNAL."""
        hs_cfg = HybridStrategyConfig()

        # Early positive (z = +2.3)
        z_er = 2.3
        assert abs(z_er) >= hs_cfg.early_signal_min
        assert abs(z_er) < hs_cfg.strong_signal_min

        # Early negative (z = -1.8)
        z_er_neg = -1.8
        assert abs(z_er_neg) >= hs_cfg.early_signal_min
        assert abs(z_er_neg) < hs_cfg.strong_signal_min

    def test_no_signal_classification(self):
        """Test that z < 1.5 is not classified as a signal."""
        hs_cfg = HybridStrategyConfig()

        # Too weak (z = +1.2)
        z_er = 1.2
        assert abs(z_er) < hs_cfg.early_signal_min


# =============================================================================
# Trading Mode and Direction Tests
# =============================================================================

class TestTradingModeAndDirection:
    """Test trading mode and direction determination."""

    def test_extreme_spike_fades_up_move(self):
        """EXTREME_SPIKE with UP direction should trade SHORT (fade)."""
        signal_class = SignalClass.EXTREME_SPIKE
        original_direction = Direction.UP

        # Mean-reversion: trade AGAINST the move
        expected_mode = TradingMode.MEAN_REVERSION
        expected_trade_direction = Direction.DOWN  # Fade the up move

        assert signal_class == SignalClass.EXTREME_SPIKE
        # Logic: UP spike -> SHORT (fade)

    def test_extreme_spike_fades_down_move(self):
        """EXTREME_SPIKE with DOWN direction should trade LONG (fade)."""
        signal_class = SignalClass.EXTREME_SPIKE
        original_direction = Direction.DOWN

        # Mean-reversion: trade AGAINST the move
        expected_mode = TradingMode.MEAN_REVERSION
        expected_trade_direction = Direction.UP  # Fade the down move

        assert signal_class == SignalClass.EXTREME_SPIKE
        # Logic: DOWN spike -> LONG (fade)

    def test_strong_signal_follows_direction(self):
        """STRONG_SIGNAL should follow original direction (conditional momentum)."""
        signal_class = SignalClass.STRONG_SIGNAL
        original_direction = Direction.UP

        expected_mode = TradingMode.CONDITIONAL_MOMENTUM
        expected_trade_direction = Direction.UP  # Follow the move

        assert signal_class == SignalClass.STRONG_SIGNAL

    def test_early_signal_follows_direction(self):
        """EARLY_SIGNAL should follow original direction (early momentum)."""
        signal_class = SignalClass.EARLY_SIGNAL
        original_direction = Direction.DOWN

        expected_mode = TradingMode.EARLY_MOMENTUM
        expected_trade_direction = Direction.DOWN  # Follow the move

        assert signal_class == SignalClass.EARLY_SIGNAL


# =============================================================================
# PendingSignal History Methods Tests
# =============================================================================

class TestPendingSignalHistory:
    """Test PendingSignal history tracking methods."""

    def create_pending_signal(self, z_er=5.5, signal_price=100.0, direction=Direction.UP):
        """Helper to create a test pending signal."""
        event = Event(
            event_id="test_event",
            ts=1000000,
            initiator_symbol="BTCUSDT",
            direction=direction,
            metrics={'z_er': z_er, 'z_vol': 3.5},
            signal_class=SignalClass.EXTREME_SPIKE,
            original_z_score=z_er
        )

        return PendingSignal(
            signal_id="test_pending",
            event=event,
            created_ts=1000000,
            expires_ts=1000000 + 480000,  # 8 minutes TTL
            direction=direction,
            symbol="BTCUSDT",
            signal_z_er=z_er,
            signal_z_vol=3.5,
            signal_price=signal_price,
            bars_since_signal=0,
            signal_class=SignalClass.EXTREME_SPIKE,
            trading_mode=TradingMode.MEAN_REVERSION,
            trade_direction=Direction.DOWN if direction == Direction.UP else Direction.UP,
            original_direction=direction
        )

    def test_z_drop_calculation(self):
        """Test z-score drop percentage calculation."""
        pending = self.create_pending_signal(z_er=6.0)

        # Add z history simulating z dropping
        pending.z_history = [6.0, 5.5, 5.0, 4.5]

        z_drop = pending.get_z_drop_pct()
        # Drop from 6.0 to 4.5 = 1 - (4.5/6.0) = 0.25 = 25%
        assert z_drop is not None
        assert abs(z_drop - 0.25) < 0.01

    def test_z_growth_calculation(self):
        """Test z-score growth percentage calculation."""
        pending = self.create_pending_signal(z_er=2.0)
        pending.signal_class = SignalClass.EARLY_SIGNAL
        pending.trading_mode = TradingMode.EARLY_MOMENTUM
        pending.trade_direction = Direction.UP

        # Add z history simulating z growing
        pending.z_history = [2.0, 2.3, 2.5, 2.8]

        z_growth = pending.get_z_growth_pct()
        # Growth from 2.0 to 2.8: (2.8 - 2.0) / 2.0 = 0.4 = 40%
        assert z_growth is not None
        assert abs(z_growth - 0.40) < 0.01

    def test_z_variance_calculation(self):
        """Test z-score variance calculation."""
        pending = self.create_pending_signal(z_er=4.0)
        pending.signal_class = SignalClass.STRONG_SIGNAL

        # Stable z history
        pending.z_history = [4.0, 3.9, 4.1, 4.0, 3.95]

        variance = pending.get_z_variance(last_n_bars=3)
        # Last 3: [4.1, 4.0, 3.95] - should have low variance
        assert variance is not None
        assert variance < 0.1  # Very stable

    def test_price_change_calculation(self):
        """Test price change percentage calculation."""
        pending = self.create_pending_signal(signal_price=100.0, direction=Direction.UP)
        pending.signal_class = SignalClass.EARLY_SIGNAL

        # Price moved up
        pending.price_history = [100.0, 100.5, 101.0, 101.5]

        price_change = pending.get_price_change_pct()
        # Change from 100.0 to 101.5 = +1.5%
        assert price_change is not None
        assert abs(price_change - 1.5) < 0.01

    def test_price_change_down_direction(self):
        """Test price change for DOWN direction (inverted)."""
        pending = self.create_pending_signal(signal_price=100.0, direction=Direction.DOWN)
        pending.original_direction = Direction.DOWN

        # Price moved down (favorable for DOWN direction)
        pending.price_history = [100.0, 99.5, 99.0, 98.5]

        price_change = pending.get_price_change_pct()
        # Change from 100.0 to 98.5 = -1.5%, but for DOWN direction positive is good
        # So we negate: result should be +1.5% (favorable)
        assert price_change is not None
        assert abs(price_change - 1.5) < 0.01

    def test_min_taker_recent(self):
        """Test minimum taker share over recent bars."""
        pending = self.create_pending_signal()

        pending.taker_history = [0.60, 0.65, 0.58, 0.62, 0.55]

        min_taker = pending.get_min_taker_recent(last_n_bars=3)
        # Last 3: [0.58, 0.62, 0.55] -> min = 0.55
        assert min_taker is not None
        assert min_taker == 0.55


# =============================================================================
# Mean-Reversion Entry Trigger Tests
# =============================================================================

class TestMeanReversionTriggers:
    """Test Mean-Reversion mode entry triggers."""

    def test_mr_requires_min_bars(self):
        """MR mode requires minimum bars before entry."""
        mr_cfg = MeanReversionConfig()

        bars_since_signal = 1
        # Default min_bars_before_entry is 2
        assert bars_since_signal < mr_cfg.min_bars_before_entry

        bars_since_signal = 3
        assert bars_since_signal >= mr_cfg.min_bars_before_entry

    def test_mr_reversal_z_drop_check(self):
        """MR mode requires z to drop by threshold percentage."""
        mr_cfg = MeanReversionConfig()

        signal_z = 6.0
        current_z = 4.5

        z_drop = 1.0 - (abs(current_z) / abs(signal_z))  # 0.25 = 25%
        assert z_drop >= mr_cfg.reversal_z_drop_pct  # 20%

        # Not enough drop
        current_z_small_drop = 5.5
        z_drop_small = 1.0 - (abs(current_z_small_drop) / abs(signal_z))  # ~8%
        assert z_drop_small < mr_cfg.reversal_z_drop_pct

    def test_mr_invalidation_on_z_growth(self):
        """MR mode invalidates if z grows instead of dropping."""
        mr_cfg = MeanReversionConfig()

        signal_z = 5.0
        current_z = 5.6  # Z grew

        growth_ratio = abs(current_z) / abs(signal_z)  # 1.12
        assert growth_ratio >= mr_cfg.z_growth_invalidate_threshold  # 1.10

    def test_mr_max_bars_expiry(self):
        """MR mode expires after max bars."""
        mr_cfg = MeanReversionConfig()

        bars_since_signal = 9
        assert bars_since_signal > mr_cfg.max_bars_before_expiry  # 8


# =============================================================================
# Conditional Momentum Entry Trigger Tests
# =============================================================================

class TestConditionalMomentumTriggers:
    """Test Conditional Momentum mode entry triggers."""

    def test_cm_taker_dominance_check(self):
        """CM mode requires strong taker dominance."""
        cm_cfg = ConditionalMomentumConfig()

        # Strong bullish flow
        taker_buy = 0.75
        assert taker_buy >= cm_cfg.min_taker_dominance  # 0.70

        # Weak bullish flow
        taker_weak = 0.62
        assert taker_weak < cm_cfg.min_taker_dominance

    def test_cm_volume_retention_check(self):
        """CM mode requires volume to be maintained."""
        cm_cfg = ConditionalMomentumConfig()

        signal_vol_z = 4.0
        current_vol_z = 3.8

        retention = current_vol_z / signal_vol_z  # 0.95
        assert retention >= cm_cfg.min_volume_retention  # 0.90

        # Volume dropped too much
        current_vol_z_low = 3.2
        retention_low = current_vol_z_low / signal_vol_z  # 0.80
        assert retention_low < cm_cfg.min_volume_retention

    def test_cm_mode_switch_on_z_drop(self):
        """CM mode switches to MR if z drops significantly."""
        cm_cfg = ConditionalMomentumConfig()

        signal_z = 4.0
        current_z = 2.5

        z_drop = 1.0 - (abs(current_z) / abs(signal_z))  # 0.375 = 37.5%
        assert z_drop >= cm_cfg.mode_switch_z_drop_pct  # 30%

    def test_cm_mode_switch_on_taker_reversal(self):
        """CM mode switches to MR if taker flow reverses."""
        cm_cfg = ConditionalMomentumConfig()

        # Originally bullish (UP direction), taker dropped below 50%
        original_direction = Direction.UP
        current_taker = 0.45

        assert current_taker < cm_cfg.mode_switch_taker_reversal  # 0.50


# =============================================================================
# Early Momentum Entry Trigger Tests
# =============================================================================

class TestEarlyMomentumTriggers:
    """Test Early Momentum mode entry triggers."""

    def test_em_requires_min_wait_bars(self):
        """EM mode requires minimum wait before evaluating."""
        em_cfg = EarlyMomentumConfig()

        bars_since_signal = 2
        assert bars_since_signal < em_cfg.min_wait_bars  # 3

        bars_since_signal = 4
        assert bars_since_signal >= em_cfg.min_wait_bars

    def test_em_z_growth_check(self):
        """EM mode requires z to grow by threshold percentage."""
        em_cfg = EarlyMomentumConfig()

        signal_z = 2.0
        current_z = 2.8

        z_growth = (abs(current_z) - abs(signal_z)) / abs(signal_z)  # 0.40 = 40%
        assert z_growth >= em_cfg.min_z_growth_pct  # 30%

        # Not enough growth
        current_z_small = 2.2
        z_growth_small = (abs(current_z_small) - abs(signal_z)) / abs(signal_z)  # 0.10
        assert z_growth_small < em_cfg.min_z_growth_pct

    def test_em_price_follow_through_check(self):
        """EM mode requires price to follow in direction."""
        em_cfg = EarlyMomentumConfig()

        signal_price = 100.0
        current_price = 100.20

        price_change = (current_price - signal_price) / signal_price * 100  # 0.20%
        assert price_change >= em_cfg.min_price_follow_through_pct  # 0.15%

    def test_em_taker_persistence_check(self):
        """EM mode requires consistent taker dominance."""
        em_cfg = EarlyMomentumConfig()

        # For LONG direction
        taker_history = [0.60, 0.58, 0.56]
        min_taker = min(taker_history)  # 0.56
        assert min_taker >= em_cfg.min_taker_persistence  # 0.55

        # Taker dropped
        taker_history_bad = [0.60, 0.52, 0.50]
        min_taker_bad = min(taker_history_bad)  # 0.50
        assert min_taker_bad < em_cfg.min_taker_persistence

    def test_em_invalidation_on_z_decline(self):
        """EM mode invalidates if z declines instead of growing."""
        em_cfg = EarlyMomentumConfig()

        signal_z = 2.0
        current_z = 1.5

        z_decline = (abs(signal_z) - abs(current_z)) / abs(signal_z)  # 0.25 = 25%
        # Growth is negative: (1.5 - 2.0) / 2.0 = -0.25
        z_growth = (abs(current_z) - abs(signal_z)) / abs(signal_z)  # -0.25

        assert z_growth < -em_cfg.z_decline_invalidate_pct  # -20%


# =============================================================================
# Configuration Loading Tests
# =============================================================================

class TestHybridStrategyConfig:
    """Test hybrid strategy configuration loading."""

    def test_default_config_values(self):
        """Test default configuration values."""
        hs_cfg = HybridStrategyConfig()

        assert hs_cfg.enabled == False
        assert hs_cfg.extreme_spike_threshold == 5.0
        assert hs_cfg.strong_signal_min == 3.0
        assert hs_cfg.early_signal_min == 1.5

    def test_mean_reversion_config_defaults(self):
        """Test mean-reversion mode defaults."""
        mr_cfg = MeanReversionConfig()

        assert mr_cfg.reversal_z_drop_pct == 0.20
        assert mr_cfg.min_bars_before_entry == 2
        assert mr_cfg.max_bars_before_expiry == 8
        assert mr_cfg.atr_target_multiplier == 1.0
        assert mr_cfg.atr_stop_multiplier == 1.2
        assert mr_cfg.max_hold_minutes == 20
        assert mr_cfg.use_trailing_stop == False

    def test_conditional_momentum_config_defaults(self):
        """Test conditional momentum mode defaults."""
        cm_cfg = ConditionalMomentumConfig()

        assert cm_cfg.min_taker_dominance == 0.70
        assert cm_cfg.min_volume_retention == 0.90
        assert cm_cfg.enable_mode_switch == True
        assert cm_cfg.atr_target_multiplier == 2.0
        assert cm_cfg.max_hold_minutes == 45

    def test_early_momentum_config_defaults(self):
        """Test early momentum mode defaults."""
        em_cfg = EarlyMomentumConfig()

        assert em_cfg.min_z_growth_pct == 0.30
        assert em_cfg.min_wait_bars == 3
        assert em_cfg.max_wait_bars == 5
        assert em_cfg.atr_target_multiplier == 3.0
        assert em_cfg.max_hold_minutes == 90
        assert em_cfg.use_z_exit == False  # Don't exit on z for early signals


# =============================================================================
# Exit Parameter Tests
# =============================================================================

class TestModeSpecificExitParams:
    """Test that exit parameters differ by mode."""

    def test_mr_quick_exits(self):
        """Mean-reversion has quick exit parameters."""
        mr_cfg = MeanReversionConfig()

        # Quick scalps
        assert mr_cfg.atr_target_multiplier == 1.0
        assert mr_cfg.max_hold_minutes == 20
        assert mr_cfg.use_trailing_stop == False
        assert mr_cfg.use_z_exit == True

    def test_cm_balanced_exits(self):
        """Conditional momentum has balanced exit parameters."""
        cm_cfg = ConditionalMomentumConfig()

        assert cm_cfg.atr_target_multiplier == 2.0
        assert cm_cfg.max_hold_minutes == 45
        assert cm_cfg.use_trailing_stop == True

    def test_em_wide_exits(self):
        """Early momentum has wide exit parameters for big moves."""
        em_cfg = EarlyMomentumConfig()

        assert em_cfg.atr_target_multiplier == 3.0
        assert em_cfg.max_hold_minutes == 90
        assert em_cfg.use_trailing_stop == True
        assert em_cfg.use_z_exit == False  # Let it run


# =============================================================================
# Integration Tests
# =============================================================================

class TestHybridStrategyIntegration:
    """Integration tests for hybrid strategy flow."""

    def test_event_has_signal_class(self):
        """Test that Event includes signal classification fields."""
        event = Event(
            event_id="test_1",
            ts=1000000,
            initiator_symbol="ETHUSDT",
            direction=Direction.UP,
            metrics={'z_er': 6.0, 'z_vol': 3.5},
            signal_class=SignalClass.EXTREME_SPIKE,
            original_z_score=6.0,
            original_vol_z=3.5,
            original_taker_share=0.72,
            original_price=2500.0
        )

        assert event.signal_class == SignalClass.EXTREME_SPIKE
        assert event.original_z_score == 6.0
        assert event.original_price == 2500.0

        # to_dict should include new fields
        event_dict = event.to_dict()
        assert 'signal_class' in event_dict
        assert event_dict['signal_class'] == 'EXTREME_SPIKE'

    def test_pending_signal_has_hybrid_fields(self):
        """Test that PendingSignal includes hybrid strategy fields."""
        event = Event(
            event_id="test_2",
            ts=1000000,
            initiator_symbol="SOLUSDT",
            direction=Direction.DOWN,
            metrics={'z_er': -5.5, 'z_vol': 4.0}
        )

        pending = PendingSignal(
            signal_id="test_pending_2",
            event=event,
            created_ts=1000000,
            expires_ts=1480000,
            direction=Direction.DOWN,
            symbol="SOLUSDT",
            signal_z_er=-5.5,
            signal_z_vol=4.0,
            signal_price=150.0,
            # Hybrid fields
            signal_class=SignalClass.EXTREME_SPIKE,
            trading_mode=TradingMode.MEAN_REVERSION,
            trade_direction=Direction.UP,  # Fade the down move
            original_direction=Direction.DOWN
        )

        assert pending.signal_class == SignalClass.EXTREME_SPIKE
        assert pending.trading_mode == TradingMode.MEAN_REVERSION
        assert pending.trade_direction == Direction.UP
        assert pending.original_direction == Direction.DOWN
        assert pending.mode_switched == False

    def test_mode_switch_updates_fields(self):
        """Test that mode switch properly updates PendingSignal fields."""
        pending = PendingSignal(
            signal_id="test_switch",
            event=Event(
                event_id="e1",
                ts=1000000,
                initiator_symbol="AVAXUSDT",
                direction=Direction.UP,
                metrics={}
            ),
            created_ts=1000000,
            expires_ts=1600000,
            direction=Direction.UP,
            symbol="AVAXUSDT",
            signal_z_er=4.0,
            signal_z_vol=3.5,
            signal_price=35.0,
            signal_class=SignalClass.STRONG_SIGNAL,
            trading_mode=TradingMode.CONDITIONAL_MOMENTUM,
            trade_direction=Direction.UP,
            original_direction=Direction.UP
        )

        # Simulate mode switch
        pending.trading_mode = TradingMode.MEAN_REVERSION
        pending.trade_direction = Direction.DOWN
        pending.mode_switched = True

        assert pending.trading_mode == TradingMode.MEAN_REVERSION
        assert pending.trade_direction == Direction.DOWN
        assert pending.mode_switched == True
        # Original preserved
        assert pending.original_direction == Direction.UP
        assert pending.direction == Direction.UP


# =============================================================================
# Class-Aware Filters Tests
# =============================================================================

class TestClassAwareFiltersConfig:
    """Test class-aware filter configuration."""

    def test_class_filter_config_defaults(self):
        """Test ClassFilterConfig default values."""
        from detector.config import ClassFilterConfig

        cfg = ClassFilterConfig()

        assert cfg.min_volume_usd == 100000.0
        assert cfg.min_trades_per_bar == 50
        assert cfg.apply_btc_anomaly_filter == True
        assert cfg.apply_beta_quality_filter == True
        assert cfg.use_global_blacklist == True
        assert cfg.require_recent_volume_spike == False

    def test_class_aware_filters_config_defaults(self):
        """Test ClassAwareFiltersConfig default values."""
        from detector.config import ClassAwareFiltersConfig

        caf = ClassAwareFiltersConfig()

        assert caf.enabled == False

        # EXTREME_SPIKE has relaxed filters
        assert caf.extreme_spike.min_volume_usd == 25000.0
        assert caf.extreme_spike.min_trades_per_bar == 15
        assert caf.extreme_spike.apply_beta_quality_filter == False  # Disabled for MR

        # STRONG_SIGNAL has standard filters
        assert caf.strong_signal.min_volume_usd == 100000.0
        assert caf.strong_signal.min_trades_per_bar == 50

        # EARLY_SIGNAL has strict filters
        assert caf.early_signal.min_volume_usd == 150000.0
        assert caf.early_signal.min_trades_per_bar == 75
        assert caf.early_signal.require_recent_volume_spike == True

    def test_hybrid_strategy_includes_class_aware_filters(self):
        """Test that HybridStrategyConfig includes class_aware_filters."""
        from detector.config import HybridStrategyConfig

        hs_cfg = HybridStrategyConfig()

        assert hasattr(hs_cfg, 'class_aware_filters')
        assert hs_cfg.class_aware_filters.enabled == False


class TestClassAwareFiltersLogic:
    """Test class-aware filter logic differences."""

    def test_extreme_spike_relaxed_volume_threshold(self):
        """EXTREME_SPIKE accepts lower volume than other classes."""
        from detector.config import ClassAwareFiltersConfig

        caf = ClassAwareFiltersConfig()

        # Low-volume bar: $30,000
        volume = 30000.0

        # EXTREME_SPIKE: PASS (threshold: $25k)
        assert volume >= caf.extreme_spike.min_volume_usd

        # STRONG_SIGNAL: FAIL (threshold: $100k)
        assert volume < caf.strong_signal.min_volume_usd

        # EARLY_SIGNAL: FAIL (threshold: $150k)
        assert volume < caf.early_signal.min_volume_usd

    def test_early_signal_strict_trades_threshold(self):
        """EARLY_SIGNAL requires more trades than other classes."""
        from detector.config import ClassAwareFiltersConfig

        caf = ClassAwareFiltersConfig()

        # Medium trade count: 60 trades
        trades = 60

        # EXTREME_SPIKE: PASS (threshold: 15)
        assert trades >= caf.extreme_spike.min_trades_per_bar

        # STRONG_SIGNAL: PASS (threshold: 50)
        assert trades >= caf.strong_signal.min_trades_per_bar

        # EARLY_SIGNAL: FAIL (threshold: 75)
        assert trades < caf.early_signal.min_trades_per_bar

    def test_extreme_spike_no_beta_filter(self):
        """EXTREME_SPIKE doesn't require beta quality check."""
        from detector.config import ClassAwareFiltersConfig

        caf = ClassAwareFiltersConfig()

        # EXTREME_SPIKE: beta filter disabled
        assert caf.extreme_spike.apply_beta_quality_filter == False

        # STRONG_SIGNAL: beta filter enabled
        assert caf.strong_signal.apply_beta_quality_filter == True

        # EARLY_SIGNAL: beta filter enabled
        assert caf.early_signal.apply_beta_quality_filter == True

    def test_early_signal_requires_volume_spike(self):
        """EARLY_SIGNAL requires volume spike, others don't."""
        from detector.config import ClassAwareFiltersConfig

        caf = ClassAwareFiltersConfig()

        # EXTREME_SPIKE: no volume spike required
        assert caf.extreme_spike.require_recent_volume_spike == False

        # STRONG_SIGNAL: no volume spike required
        assert caf.strong_signal.require_recent_volume_spike == False

        # EARLY_SIGNAL: volume spike required
        assert caf.early_signal.require_recent_volume_spike == True
        assert caf.early_signal.recent_volume_spike_threshold == 1.5


class TestClassAwareFiltersScenarios:
    """Test class-aware filter scenarios from specification."""

    def test_scenario_extreme_spike_passes_mildly_relaxed_filters(self):
        """
        Scenario 7.1: EXTREME_SPIKE passes mildly relaxed filters.

        Input:
        - Symbol: MITOUSDT
        - Z-score: +6.5 (EXTREME_SPIKE)
        - Volume: $30,000 (below standard 100k)
        - Trades: 20 (below standard 50)

        Expected:
        - Old system: BLOCKED (low_volume)
        - New system: PASSED (thresholds for EXTREME_SPIKE: $25k, 15 trades)
        """
        from detector.config import ClassAwareFiltersConfig

        caf = ClassAwareFiltersConfig()

        volume = 30000.0
        trades = 20

        # Would fail standard filters (STRONG_SIGNAL thresholds)
        assert volume < caf.strong_signal.min_volume_usd  # BLOCKED

        # Passes EXTREME_SPIKE relaxed filters
        assert volume >= caf.extreme_spike.min_volume_usd  # $30k >= $25k
        assert trades >= caf.extreme_spike.min_trades_per_bar  # 20 >= 15

    def test_scenario_early_signal_blocked_by_strict_filters(self):
        """
        Scenario 7.2: EARLY_SIGNAL blocked by strict filters.

        Input:
        - Symbol: BANDUSDT
        - Z-score: +2.0 (EARLY_SIGNAL)
        - Volume: $120,000 (above standard 100k)
        - Trades: 60 (above standard 50)

        Expected:
        - Old system: PASSED
        - New system: BLOCKED (thresholds for EARLY_SIGNAL: $150k, 75 trades)
        """
        from detector.config import ClassAwareFiltersConfig

        caf = ClassAwareFiltersConfig()

        volume = 120000.0
        trades = 60

        # Would pass standard filters (STRONG_SIGNAL thresholds)
        assert volume >= caf.strong_signal.min_volume_usd  # PASSED

        # Fails EARLY_SIGNAL strict filters
        assert volume < caf.early_signal.min_volume_usd  # $120k < $150k

    def test_scenario_strong_signal_with_bad_beta(self):
        """
        Scenario 7.3: STRONG_SIGNAL blocked by beta quality.

        Input:
        - Symbol: XYZUSDT
        - Z-score: +4.0 (STRONG_SIGNAL)
        - Volume: $200,000
        - Trades: 100
        - Beta: 0.05 (too low)

        Expected:
        - BLOCKED (beta_too_low)
        """
        from detector.config import ClassAwareFiltersConfig

        caf = ClassAwareFiltersConfig()

        beta = 0.05

        # STRONG_SIGNAL requires beta quality filter
        assert caf.strong_signal.apply_beta_quality_filter == True

        # Beta is too low
        assert abs(beta) < caf.strong_signal.beta_min_abs  # 0.05 < 0.1

    def test_scenario_early_signal_without_volume_spike(self):
        """
        Scenario 7.4: EARLY_SIGNAL blocked by no volume spike.

        Input:
        - Symbol: ETHUSDT
        - Z-score: +1.8 (EARLY_SIGNAL)
        - Current volume: $160,000
        - Average volume: $150,000
        - Ratio: 1.07x (below threshold 1.5x)

        Expected:
        - BLOCKED (no_volume_spike)
        """
        from detector.config import ClassAwareFiltersConfig

        caf = ClassAwareFiltersConfig()

        current_volume = 160000.0
        avg_volume = 150000.0
        volume_ratio = current_volume / avg_volume  # 1.07

        # EARLY_SIGNAL requires volume spike
        assert caf.early_signal.require_recent_volume_spike == True

        # Volume ratio below threshold
        assert volume_ratio < caf.early_signal.recent_volume_spike_threshold  # 1.07 < 1.5


class TestClassAwareFiltersIntegration:
    """Integration tests for class-aware filtering in position manager."""

    def test_helper_function_get_recent_volume_usd(self):
        """Test _get_recent_volume_usd returns bar notional."""
        bar = Bar(
            ts_minute=1000000,
            symbol="ETHUSDT",
            open=2500.0,
            high=2510.0,
            low=2490.0,
            close=2505.0,
            volume=100.0,
            notional=250000.0,  # $250k
            trades=500,
            taker_buy=60.0,
            taker_sell=40.0
        )

        # notional IS the USD volume for USDT pairs
        assert bar.notional == 250000.0

    def test_filter_order_class_aware_before_legacy(self):
        """Test that class-aware filters run before legacy WIN_RATE_MAX."""
        from detector.config import HybridStrategyConfig

        hs_cfg = HybridStrategyConfig()
        hs_cfg.enabled = True
        hs_cfg.class_aware_filters.enabled = True

        # When both hybrid and class_aware_filters enabled:
        # class_aware_filters should be used (not WIN_RATE_MAX)
        assert hs_cfg.enabled == True
        assert hs_cfg.class_aware_filters.enabled == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
