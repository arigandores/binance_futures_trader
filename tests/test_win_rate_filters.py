"""
Tests for WIN_RATE_MAX profile market regime filters.

These tests verify the three new filter methods:
1. _check_btc_anomaly_filter - Blocks trades during BTC volatility spikes
2. _check_symbol_quality_filter - Blocks trades on low-quality symbols
3. _check_beta_quality_filter - Blocks trades with unreliable beta calculations
"""

import pytest
import asyncio
from unittest.mock import AsyncMock
from detector.position_manager import PositionManager
from detector.config import Config, PositionManagementConfig, WinRateMaxProfileConfig
from detector.models import Bar, Features, Direction, Event, EventStatus, PendingSignal
from detector.features_extended import ExtendedFeatureCalculator
from detector.storage import Storage
from datetime import datetime
from collections import deque


@pytest.fixture
def mock_storage():
    """Create mock storage."""
    storage = AsyncMock(spec=Storage)
    storage.write_position = AsyncMock()
    storage.get_open_positions = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def config_win_rate_max():
    """Create WIN_RATE_MAX profile configuration."""
    config = Config()
    config.position_management = PositionManagementConfig(
        enabled=True,
        profile="WIN_RATE_MAX"
    )
    # Ensure WIN_RATE_MAX profile config exists
    config.position_management.win_rate_max_profile = WinRateMaxProfileConfig()
    return config


@pytest.fixture
def config_default():
    """Create DEFAULT profile configuration."""
    config = Config()
    config.position_management = PositionManagementConfig(
        enabled=True,
        profile="DEFAULT"
    )
    return config


def create_position_manager(config, mock_storage):
    """Helper to create PositionManager instance."""
    # Create dummy queues (not used in filter tests)
    event_queue = asyncio.Queue()
    feature_queue = asyncio.Queue()
    bar_queue = asyncio.Queue()

    pm = PositionManager(
        event_queue=event_queue,
        feature_queue=feature_queue,
        bar_queue=bar_queue,
        storage=mock_storage,
        config=config
    )

    return pm


def create_features(
    symbol: str,
    z_er: float = 0.0,
    z_vol: float = 0.0,
    beta: float = 1.0
) -> Features:
    """Helper to create Features object."""
    ts_minute = int(datetime.now().timestamp() * 1000)
    return Features(
        symbol=symbol,
        ts_minute=ts_minute,
        z_er_15m=z_er,
        z_vol_15m=z_vol,
        beta=beta,
        r_1m=0.0,
        r_15m=0.0,
        er_15m=0.0,
        vol_15m=0.0,
        taker_buy_share_15m=0.6
    )


def create_bar(
    symbol: str,
    close: float = 50000.0,
    notional: float = 100000.0,
    trades: int = 100
) -> Bar:
    """Helper to create Bar object."""
    ts_minute = int(datetime.now().timestamp() * 1000)
    return Bar(
        symbol=symbol,
        ts_minute=ts_minute,
        open=close - 10,
        high=close + 10,
        low=close - 20,
        close=close,
        volume=100.0,
        notional=notional,
        trades=trades,
        taker_buy=60.0,
        taker_sell=40.0
    )


# =========================================================================
# Test Group 1: _check_btc_anomaly_filter (3 tests)
# =========================================================================

def test_btc_anomaly_filter_blocks_during_btc_anomaly(config_win_rate_max, mock_storage):
    """BTC anomaly filter blocks trades when BTC in anomaly (abs(z_ER) >= 3.0 AND z_VOL >= 3.0)."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.btc_anomaly_filter = True
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create BTC features showing anomaly
    btc_features = create_features(
        symbol="BTCUSDT",
        z_er=3.5,  # Anomaly threshold
        z_vol=3.2   # Anomaly threshold
    )
    pm.latest_features["BTCUSDT"] = btc_features

    # ACT
    result = pm._check_btc_anomaly_filter("ETHUSDT")

    # ASSERT
    assert result is False, "Should block trade during BTC anomaly (z_ER >= 3.0 AND z_VOL >= 3.0)"


def test_btc_anomaly_filter_allows_normal_btc(config_win_rate_max, mock_storage):
    """BTC anomaly filter allows trades when BTC not in anomaly."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.btc_anomaly_filter = True
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create BTC features showing normal conditions (either z_ER < 3.0 OR z_VOL < 3.0)
    btc_features = create_features(
        symbol="BTCUSDT",
        z_er=2.5,  # Below threshold
        z_vol=2.8   # Below threshold
    )
    pm.latest_features["BTCUSDT"] = btc_features

    # ACT
    result = pm._check_btc_anomaly_filter("ETHUSDT")

    # ASSERT
    assert result is True, "Should allow trade when BTC not in anomaly (z_ER < 3.0 OR z_VOL < 3.0)"


def test_btc_anomaly_filter_skips_for_default_profile(config_default, mock_storage):
    """BTC anomaly filter skipped when profile is DEFAULT (not WIN_RATE_MAX)."""
    # ARRANGE
    pm = create_position_manager(config_default, mock_storage)

    # Create BTC features showing anomaly (should be ignored for DEFAULT profile)
    btc_features = create_features(
        symbol="BTCUSDT",
        z_er=3.5,  # Anomaly
        z_vol=3.2   # Anomaly
    )
    pm.latest_features["BTCUSDT"] = btc_features

    # ACT
    result = pm._check_btc_anomaly_filter("ETHUSDT")

    # ASSERT
    assert result is True, "Should skip BTC anomaly filter for DEFAULT profile (always return True)"


# =========================================================================
# Test Group 2: _check_symbol_quality_filter (3 tests)
# =========================================================================

def test_symbol_quality_blocks_blacklisted(config_win_rate_max, mock_storage):
    """Symbol quality filter blocks trades on blacklisted symbols."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.symbol_quality_filter = True
    config_win_rate_max.position_management.win_rate_max_profile.symbol_blacklist = ["DOGEUSDT", "SHIBUSDT"]
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create bar for blacklisted symbol (high volume, should still be blocked)
    bar = create_bar(symbol="DOGEUSDT", notional=200000.0, trades=100)
    pm.extended_features.bars_windows["DOGEUSDT"] = [bar]

    # ACT
    result = pm._check_symbol_quality_filter("DOGEUSDT")

    # ASSERT
    assert result is False, "Should block trade on blacklisted symbol (regardless of volume)"


def test_symbol_quality_blocks_low_volume(config_win_rate_max, mock_storage):
    """Symbol quality filter blocks trades on low volume symbols."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.symbol_quality_filter = True
    config_win_rate_max.position_management.win_rate_max_profile.min_volume_usd = 100000.0
    config_win_rate_max.position_management.win_rate_max_profile.min_trades_per_bar = 50
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create bar with insufficient volume (< min_volume_usd)
    bar = create_bar(symbol="ETHUSDT", notional=50000.0, trades=100)  # Below 100k minimum
    pm.extended_features.bars_windows["ETHUSDT"] = [bar]

    # ACT
    result = pm._check_symbol_quality_filter("ETHUSDT")

    # ASSERT
    assert result is False, "Should block trade on low volume symbol (notional < min_volume_usd)"


def test_symbol_quality_allows_good_symbol(config_win_rate_max, mock_storage):
    """Symbol quality filter allows trades on high-quality symbols."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.symbol_quality_filter = True
    config_win_rate_max.position_management.win_rate_max_profile.min_volume_usd = 100000.0
    config_win_rate_max.position_management.win_rate_max_profile.min_trades_per_bar = 50
    config_win_rate_max.position_management.win_rate_max_profile.symbol_blacklist = []
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create bar with sufficient volume and trades
    bar = create_bar(symbol="ETHUSDT", notional=200000.0, trades=150)  # Above both minimums
    pm.extended_features.bars_windows["ETHUSDT"] = [bar]

    # ACT
    result = pm._check_symbol_quality_filter("ETHUSDT")

    # ASSERT
    assert result is True, "Should allow trade on high-quality symbol (volume OK, not blacklisted)"


# =========================================================================
# Test Group 3: _check_beta_quality_filter (3 tests)
# =========================================================================

def test_beta_quality_blocks_low_beta(config_win_rate_max, mock_storage):
    """Beta quality filter blocks trades when beta too low (unreliable correlation)."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.beta_quality_filter = True
    config_win_rate_max.position_management.win_rate_max_profile.beta_min_abs = 0.1
    config_win_rate_max.position_management.win_rate_max_profile.beta_max_abs = 3.0
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create features with beta too low
    features = create_features(symbol="ETHUSDT", beta=0.05)  # Below 0.1 minimum
    pm.latest_features["ETHUSDT"] = features

    # ACT
    result = pm._check_beta_quality_filter("ETHUSDT")

    # ASSERT
    assert result is False, "Should block trade when |beta| < beta_min_abs (unreliable correlation)"


def test_beta_quality_blocks_high_beta(config_win_rate_max, mock_storage):
    """Beta quality filter blocks trades when beta too high (unreliable regression)."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.beta_quality_filter = True
    config_win_rate_max.position_management.win_rate_max_profile.beta_min_abs = 0.1
    config_win_rate_max.position_management.win_rate_max_profile.beta_max_abs = 3.0
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create features with beta too high
    features = create_features(symbol="ETHUSDT", beta=4.0)  # Above 3.0 maximum
    pm.latest_features["ETHUSDT"] = features

    # ACT
    result = pm._check_beta_quality_filter("ETHUSDT")

    # ASSERT
    assert result is False, "Should block trade when |beta| > beta_max_abs (unreliable regression)"


def test_beta_quality_allows_reasonable_beta(config_win_rate_max, mock_storage):
    """Beta quality filter allows trades when beta in reasonable range."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.beta_quality_filter = True
    config_win_rate_max.position_management.win_rate_max_profile.beta_min_abs = 0.1
    config_win_rate_max.position_management.win_rate_max_profile.beta_max_abs = 3.0
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create features with beta in acceptable range
    features = create_features(symbol="ETHUSDT", beta=0.8)  # Within [0.1, 3.0]
    pm.latest_features["ETHUSDT"] = features

    # ACT
    result = pm._check_beta_quality_filter("ETHUSDT")

    # ASSERT
    assert result is True, "Should allow trade when beta_min_abs <= |beta| <= beta_max_abs"


# =========================================================================
# Additional Edge Case Tests (Bonus Coverage)
# =========================================================================

def test_btc_anomaly_filter_blocks_when_no_btc_data(config_win_rate_max, mock_storage):
    """BTC anomaly filter blocks when BTC features unavailable (fail-closed)."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.btc_anomaly_filter = True
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # No BTC features available (latest_features is empty)
    assert "BTCUSDT" not in pm.latest_features

    # ACT
    result = pm._check_btc_anomaly_filter("ETHUSDT")

    # ASSERT
    assert result is False, "Should block when BTC features unavailable (fail-closed safety)"


def test_symbol_quality_blocks_when_no_bar_data(config_win_rate_max, mock_storage):
    """Symbol quality filter blocks when bar data unavailable (fail-closed)."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.symbol_quality_filter = True
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # No bar data available (bars_windows is empty)
    assert "ETHUSDT" not in pm.extended_features.bars_windows

    # ACT
    result = pm._check_symbol_quality_filter("ETHUSDT")

    # ASSERT
    assert result is False, "Should block when bar data unavailable (fail-closed safety)"


def test_beta_quality_blocks_when_no_features(config_win_rate_max, mock_storage):
    """Beta quality filter blocks when features unavailable (fail-closed)."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.beta_quality_filter = True
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # No features available (latest_features is empty)
    assert "ETHUSDT" not in pm.latest_features

    # ACT
    result = pm._check_beta_quality_filter("ETHUSDT")

    # ASSERT
    assert result is False, "Should block when features unavailable (fail-closed safety)"


def test_symbol_quality_blocks_insufficient_trades(config_win_rate_max, mock_storage):
    """Symbol quality filter blocks when trades per bar below minimum."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.symbol_quality_filter = True
    config_win_rate_max.position_management.win_rate_max_profile.min_volume_usd = 100000.0
    config_win_rate_max.position_management.win_rate_max_profile.min_trades_per_bar = 50
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create bar with good volume but insufficient trades
    bar = create_bar(symbol="ETHUSDT", notional=200000.0, trades=30)  # < 50 minimum
    pm.extended_features.bars_windows["ETHUSDT"] = [bar]

    # ACT
    result = pm._check_symbol_quality_filter("ETHUSDT")

    # ASSERT
    assert result is False, "Should block when trades < min_trades_per_bar (low liquidity)"


def test_filters_disabled_always_pass(config_win_rate_max, mock_storage):
    """All filters return True when disabled in config."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.btc_anomaly_filter = False
    config_win_rate_max.position_management.win_rate_max_profile.symbol_quality_filter = False
    config_win_rate_max.position_management.win_rate_max_profile.beta_quality_filter = False
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create conditions that would normally block (anomaly, low volume, bad beta)
    btc_features = create_features(symbol="BTCUSDT", z_er=3.5, z_vol=3.5)
    pm.latest_features["BTCUSDT"] = btc_features

    bad_features = create_features(symbol="ETHUSDT", beta=0.05)  # Too low
    pm.latest_features["ETHUSDT"] = bad_features

    bad_bar = create_bar(symbol="ETHUSDT", notional=10000.0)  # Too low
    pm.extended_features.bars_windows["ETHUSDT"] = [bad_bar]

    # ACT
    btc_result = pm._check_btc_anomaly_filter("ETHUSDT")
    quality_result = pm._check_symbol_quality_filter("ETHUSDT")
    beta_result = pm._check_beta_quality_filter("ETHUSDT")

    # ASSERT
    assert btc_result is True, "BTC filter should pass when disabled"
    assert quality_result is True, "Symbol quality filter should pass when disabled"
    assert beta_result is True, "Beta quality filter should pass when disabled"


def test_btc_anomaly_filter_only_z_vol_high(config_win_rate_max, mock_storage):
    """BTC anomaly filter allows when only z_VOL high (needs BOTH z_ER AND z_VOL)."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.btc_anomaly_filter = True
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Only z_VOL is high, z_ER is normal
    btc_features = create_features(symbol="BTCUSDT", z_er=2.0, z_vol=3.5)
    pm.latest_features["BTCUSDT"] = btc_features

    # ACT
    result = pm._check_btc_anomaly_filter("ETHUSDT")

    # ASSERT
    assert result is True, "Should allow when only z_VOL >= 3.0 (needs BOTH z_ER AND z_VOL)"


def test_btc_anomaly_filter_only_z_er_high(config_win_rate_max, mock_storage):
    """BTC anomaly filter allows when only z_ER high (needs BOTH z_ER AND z_VOL)."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.btc_anomaly_filter = True
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Only z_ER is high, z_VOL is normal
    btc_features = create_features(symbol="BTCUSDT", z_er=3.5, z_vol=2.0)
    pm.latest_features["BTCUSDT"] = btc_features

    # ACT
    result = pm._check_btc_anomaly_filter("ETHUSDT")

    # ASSERT
    assert result is True, "Should allow when only z_ER >= 3.0 (needs BOTH z_ER AND z_VOL)"


def test_beta_quality_negative_beta_within_range(config_win_rate_max, mock_storage):
    """Beta quality filter uses absolute value for range check (negative beta OK)."""
    # ARRANGE
    config_win_rate_max.position_management.win_rate_max_profile.beta_quality_filter = True
    config_win_rate_max.position_management.win_rate_max_profile.beta_min_abs = 0.1
    config_win_rate_max.position_management.win_rate_max_profile.beta_max_abs = 3.0
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Negative beta within acceptable range
    features = create_features(symbol="ETHUSDT", beta=-0.8)  # |beta| = 0.8, within [0.1, 3.0]
    pm.latest_features["ETHUSDT"] = features

    # ACT
    result = pm._check_beta_quality_filter("ETHUSDT")

    # ASSERT
    assert result is True, "Should allow negative beta when |beta| within [beta_min_abs, beta_max_abs]"


# =========================================================================
# Test Group 4: _check_z_cooldown_declining (2 tests)
# =========================================================================

def test_z_cooldown_declining_passes_for_now(config_win_rate_max, mock_storage):
    """Z-cooldown declining check passes (pass-through until features history implemented)."""
    # ARRANGE
    config_win_rate_max.position_management.profile = "WIN_RATE_MAX"
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create pending signal
    event = Event(
        event_id="test_event_1",
        ts=int(datetime.now().timestamp() * 1000),
        initiator_symbol="ETHUSDT",
        direction=Direction.UP,
        status=EventStatus.CONFIRMED
    )
    pending = PendingSignal(
        signal_id="ETHUSDT_test_UP",
        event=event,
        created_ts=int(datetime.now().timestamp() * 1000),
        expires_ts=int((datetime.now().timestamp() + 600) * 1000),
        direction=Direction.UP,
        symbol="ETHUSDT",
        signal_z_er=3.5,
        signal_z_vol=3.2,
        signal_price=100.0
    )

    # Create features
    features = create_features(symbol="ETHUSDT", z_er=2.5)

    # ACT
    result = pm._check_z_cooldown_declining(pending, features)

    # ASSERT
    assert result is True, "Should pass (return True) until features history implemented"


def test_z_cooldown_declining_skips_for_default(config_default, mock_storage):
    """Z-cooldown declining check skipped for DEFAULT profile."""
    # ARRANGE
    pm = create_position_manager(config_default, mock_storage)

    # Create pending signal
    event = Event(
        event_id="test_event_2",
        ts=int(datetime.now().timestamp() * 1000),
        initiator_symbol="ETHUSDT",
        direction=Direction.UP,
        status=EventStatus.CONFIRMED
    )
    pending = PendingSignal(
        signal_id="ETHUSDT_test_UP",
        event=event,
        created_ts=int(datetime.now().timestamp() * 1000),
        expires_ts=int((datetime.now().timestamp() + 600) * 1000),
        direction=Direction.UP,
        symbol="ETHUSDT",
        signal_z_er=3.5,
        signal_z_vol=3.2,
        signal_price=100.0
    )

    # Create features (even with declining z-score pattern, should skip)
    features = create_features(symbol="ETHUSDT", z_er=2.5)

    # ACT
    result = pm._check_z_cooldown_declining(pending, features)

    # ASSERT
    assert result is True, "Should skip check for DEFAULT profile (always return True)"


# =========================================================================
# Test Group 5: _check_re_expansion (3 tests)
# =========================================================================

def test_re_expansion_price_action_long(config_win_rate_max, mock_storage):
    """Re-expansion price action method works for LONG signals."""
    # ARRANGE
    config_win_rate_max.position_management.profile = "WIN_RATE_MAX"
    config_win_rate_max.position_management.win_rate_max_profile.require_re_expansion = True
    config_win_rate_max.position_management.win_rate_max_profile.re_expansion_price_action = True
    config_win_rate_max.position_management.win_rate_max_profile.re_expansion_micro_impulse = False
    config_win_rate_max.position_management.win_rate_max_profile.re_expansion_flow_acceleration = False
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create bars: prev_bar (high=100), current_bar (close=101 > prev_high)
    prev_bar = create_bar(symbol="ETHUSDT", close=99.0)
    prev_bar.high = 100.0

    current_bar = create_bar(symbol="ETHUSDT", close=101.0)
    current_bar.high = 101.0

    # Add bars to extended features window
    pm.extended_features.bars_windows["ETHUSDT"] = deque([prev_bar, current_bar], maxlen=100)

    # Create pending signal (Direction.UP)
    event = Event(
        event_id="test_event_3",
        ts=int(datetime.now().timestamp() * 1000),
        initiator_symbol="ETHUSDT",
        direction=Direction.UP,
        status=EventStatus.CONFIRMED
    )
    pending = PendingSignal(
        signal_id="ETHUSDT_test_UP",
        event=event,
        created_ts=int(datetime.now().timestamp() * 1000),
        expires_ts=int((datetime.now().timestamp() + 600) * 1000),
        direction=Direction.UP,
        symbol="ETHUSDT",
        signal_z_er=3.5,
        signal_z_vol=3.2,
        signal_price=100.0
    )

    features = create_features(symbol="ETHUSDT", z_er=2.5)

    # ACT
    result = pm._check_re_expansion(pending, current_bar, features)

    # ASSERT
    assert result is True, "Price action expansion should pass (close > prev_high for LONG)"


def test_re_expansion_micro_impulse_short(config_win_rate_max, mock_storage):
    """Re-expansion micro impulse method works for SHORT signals."""
    # ARRANGE
    config_win_rate_max.position_management.profile = "WIN_RATE_MAX"
    config_win_rate_max.position_management.win_rate_max_profile.require_re_expansion = True
    config_win_rate_max.position_management.win_rate_max_profile.re_expansion_price_action = False
    config_win_rate_max.position_management.win_rate_max_profile.re_expansion_micro_impulse = True
    config_win_rate_max.position_management.win_rate_max_profile.re_expansion_flow_acceleration = False
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Mock get_bar_return to return negative value (bearish)
    pm.extended_features.get_bar_return = lambda symbol: -0.01

    # Create pending signal (Direction.DOWN)
    event = Event(
        event_id="test_event_4",
        ts=int(datetime.now().timestamp() * 1000),
        initiator_symbol="ETHUSDT",
        direction=Direction.DOWN,
        status=EventStatus.CONFIRMED
    )
    pending = PendingSignal(
        signal_id="ETHUSDT_test_DOWN",
        event=event,
        created_ts=int(datetime.now().timestamp() * 1000),
        expires_ts=int((datetime.now().timestamp() + 600) * 1000),
        direction=Direction.DOWN,
        symbol="ETHUSDT",
        signal_z_er=-3.5,
        signal_z_vol=3.2,
        signal_price=100.0
    )

    current_bar = create_bar(symbol="ETHUSDT", close=99.0)
    features = create_features(symbol="ETHUSDT", z_er=-2.5)

    # ACT
    result = pm._check_re_expansion(pending, current_bar, features)

    # ASSERT
    assert result is True, "Micro impulse expansion should pass (negative return for SHORT)"


def test_re_expansion_flow_acceleration_long(config_win_rate_max, mock_storage):
    """Re-expansion flow acceleration method works for LONG signals."""
    # ARRANGE
    config_win_rate_max.position_management.profile = "WIN_RATE_MAX"
    config_win_rate_max.position_management.win_rate_max_profile.require_re_expansion = True
    config_win_rate_max.position_management.win_rate_max_profile.re_expansion_price_action = False
    config_win_rate_max.position_management.win_rate_max_profile.re_expansion_micro_impulse = False
    config_win_rate_max.position_management.win_rate_max_profile.re_expansion_flow_acceleration = True
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Mock get_flow_acceleration_bars to return increasing buy dominance
    pm.extended_features.get_flow_acceleration_bars = lambda symbol, lookback: [0.50, 0.55, 0.60]

    # Create pending signal (Direction.UP)
    event = Event(
        event_id="test_event_5",
        ts=int(datetime.now().timestamp() * 1000),
        initiator_symbol="ETHUSDT",
        direction=Direction.UP,
        status=EventStatus.CONFIRMED
    )
    pending = PendingSignal(
        signal_id="ETHUSDT_test_UP",
        event=event,
        created_ts=int(datetime.now().timestamp() * 1000),
        expires_ts=int((datetime.now().timestamp() + 600) * 1000),
        direction=Direction.UP,
        symbol="ETHUSDT",
        signal_z_er=3.5,
        signal_z_vol=3.2,
        signal_price=100.0
    )

    current_bar = create_bar(symbol="ETHUSDT", close=101.0)
    features = create_features(symbol="ETHUSDT", z_er=2.5)

    # ACT
    result = pm._check_re_expansion(pending, current_bar, features)

    # ASSERT
    assert result is True, "Flow acceleration expansion should pass (increasing buy dominance for LONG)"


# =========================================================================
# Test Group 6: Additional Edge Case Tests (Bonus)
# =========================================================================

def test_re_expansion_requires_at_least_one_method(config_win_rate_max, mock_storage):
    """Re-expansion fails when no methods pass (no expansion confirmed)."""
    # ARRANGE
    config_win_rate_max.position_management.profile = "WIN_RATE_MAX"
    config_win_rate_max.position_management.win_rate_max_profile.require_re_expansion = True
    config_win_rate_max.position_management.win_rate_max_profile.re_expansion_price_action = True
    config_win_rate_max.position_management.win_rate_max_profile.re_expansion_micro_impulse = True
    config_win_rate_max.position_management.win_rate_max_profile.re_expansion_flow_acceleration = True
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create bars: current price NOT above prev high (price action fails)
    prev_bar = create_bar(symbol="ETHUSDT", close=100.0)
    prev_bar.high = 101.0
    current_bar = create_bar(symbol="ETHUSDT", close=100.5)  # 100.5 < 101.0 (prev_high)
    current_bar.high = 100.5

    pm.extended_features.bars_windows["ETHUSDT"] = deque([prev_bar, current_bar], maxlen=100)

    # Mock get_bar_return to return negative (micro impulse fails for LONG)
    pm.extended_features.get_bar_return = lambda symbol: -0.005

    # Mock get_flow_acceleration_bars to return non-increasing (flow acceleration fails)
    pm.extended_features.get_flow_acceleration_bars = lambda symbol, lookback: [0.60, 0.55, 0.50]

    # Create pending signal (Direction.UP)
    event = Event(
        event_id="test_event_6",
        ts=int(datetime.now().timestamp() * 1000),
        initiator_symbol="ETHUSDT",
        direction=Direction.UP,
        status=EventStatus.CONFIRMED
    )
    pending = PendingSignal(
        signal_id="ETHUSDT_test_UP",
        event=event,
        created_ts=int(datetime.now().timestamp() * 1000),
        expires_ts=int((datetime.now().timestamp() + 600) * 1000),
        direction=Direction.UP,
        symbol="ETHUSDT",
        signal_z_er=3.5,
        signal_z_vol=3.2,
        signal_price=100.0
    )

    features = create_features(symbol="ETHUSDT", z_er=2.5)

    # ACT
    result = pm._check_re_expansion(pending, current_bar, features)

    # ASSERT
    assert result is False, "Should fail when all 3 methods fail (no expansion confirmed)"


def test_re_expansion_disabled_always_passes(config_win_rate_max, mock_storage):
    """Re-expansion passes when require_re_expansion=False."""
    # ARRANGE
    config_win_rate_max.position_management.profile = "WIN_RATE_MAX"
    config_win_rate_max.position_management.win_rate_max_profile.require_re_expansion = False
    pm = create_position_manager(config_win_rate_max, mock_storage)

    # Create bars with no expansion (would fail if checked)
    prev_bar = create_bar(symbol="ETHUSDT", close=100.0)
    prev_bar.high = 101.0
    current_bar = create_bar(symbol="ETHUSDT", close=99.0)  # Below prev_high
    current_bar.high = 99.0

    pm.extended_features.bars_windows["ETHUSDT"] = deque([prev_bar, current_bar], maxlen=100)

    # Create pending signal
    event = Event(
        event_id="test_event_7",
        ts=int(datetime.now().timestamp() * 1000),
        initiator_symbol="ETHUSDT",
        direction=Direction.UP,
        status=EventStatus.CONFIRMED
    )
    pending = PendingSignal(
        signal_id="ETHUSDT_test_UP",
        event=event,
        created_ts=int(datetime.now().timestamp() * 1000),
        expires_ts=int((datetime.now().timestamp() + 600) * 1000),
        direction=Direction.UP,
        symbol="ETHUSDT",
        signal_z_er=3.5,
        signal_z_vol=3.2,
        signal_price=100.0
    )

    features = create_features(symbol="ETHUSDT", z_er=2.5)

    # ACT
    result = pm._check_re_expansion(pending, current_bar, features)

    # ASSERT
    assert result is True, "Should pass when require_re_expansion=False (regardless of expansion)"


def test_re_expansion_skips_for_default_profile(config_default, mock_storage):
    """Re-expansion check skipped for DEFAULT profile."""
    # ARRANGE
    pm = create_position_manager(config_default, mock_storage)

    # Create bars with no expansion (would fail if checked)
    prev_bar = create_bar(symbol="ETHUSDT", close=100.0)
    prev_bar.high = 101.0
    current_bar = create_bar(symbol="ETHUSDT", close=99.0)
    current_bar.high = 99.0

    pm.extended_features.bars_windows["ETHUSDT"] = deque([prev_bar, current_bar], maxlen=100)

    # Create pending signal
    event = Event(
        event_id="test_event_8",
        ts=int(datetime.now().timestamp() * 1000),
        initiator_symbol="ETHUSDT",
        direction=Direction.UP,
        status=EventStatus.CONFIRMED
    )
    pending = PendingSignal(
        signal_id="ETHUSDT_test_UP",
        event=event,
        created_ts=int(datetime.now().timestamp() * 1000),
        expires_ts=int((datetime.now().timestamp() + 600) * 1000),
        direction=Direction.UP,
        symbol="ETHUSDT",
        signal_z_er=3.5,
        signal_z_vol=3.2,
        signal_price=100.0
    )

    features = create_features(symbol="ETHUSDT", z_er=2.5)

    # ACT
    result = pm._check_re_expansion(pending, current_bar, features)

    # ASSERT
    assert result is True, "Should skip check for DEFAULT profile (always return True)"
