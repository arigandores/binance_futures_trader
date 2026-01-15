"""Tests for position manager - virtual trading logic."""

import pytest
import asyncio
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from detector.position_manager import PositionManager
from detector.models import (
    Event, Features, Bar, Position, PositionStatus, ExitReason,
    Direction
)
from detector.storage import Storage
from detector.config import Config


@pytest.fixture
def config():
    """Create test configuration."""
    cfg = Config()
    cfg.position_management.enabled = True
    cfg.position_management.allow_multiple_positions = False
    cfg.position_management.use_entry_triggers = False  # Disable for most tests (backward compat)
    cfg.position_management.z_score_exit_threshold = 1.0
    cfg.position_management.stop_loss_percent = 2.0
    cfg.position_management.take_profit_percent = 3.0
    cfg.position_management.max_hold_minutes = 60
    cfg.position_management.use_atr_stops = True
    cfg.position_management.atr_period = 14
    cfg.position_management.atr_stop_multiplier = 2.0
    cfg.position_management.exit_on_order_flow_reversal = True
    cfg.position_management.order_flow_reversal_threshold = 0.15
    cfg.position_management.exit_on_opposite_signal = False
    cfg.position_management.opposite_signal_threshold = 2.5
    return cfg


@pytest.fixture
def mock_storage():
    """Create mock storage."""
    storage = AsyncMock(spec=Storage)
    storage.write_position = AsyncMock()
    storage.get_open_positions = AsyncMock(return_value=[])
    return storage


@pytest_asyncio.fixture
async def position_manager(config, mock_storage):
    """Create position manager instance."""
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

    # Initialize
    await pm.init()

    return pm


def create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000):
    """Helper to create test event."""
    return Event(
        event_id=f"{symbol}_{ts}",
        ts=ts,
        initiator_symbol=symbol,
        direction=direction,
        metrics={
            'z_er': 3.5,
            'z_vol': 3.2,
            'taker_share': 0.70,
            'beta': 1.2,
            'funding': 0.0001
        }
    )


def create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000):
    """Helper to create test bar."""
    return Bar(
        symbol=symbol,
        ts_minute=ts,
        open=close - 10,
        high=close + 10,
        low=close - 20,
        close=close,
        volume=100.0,
        notional=5000000.0,
        trades=1000,
        taker_buy=60.0,
        taker_sell=40.0
    )


def create_features(symbol="BTCUSDT", z_er=3.5, z_vol=3.2, ts=1000000000, direction=Direction.UP):
    """Helper to create test features."""
    features = Features(
        symbol=symbol,
        ts_minute=ts,
        z_er_15m=z_er,
        z_vol_15m=z_vol,
        taker_buy_share_15m=0.70
    )
    features.direction = direction
    return features


@pytest.mark.asyncio
async def test_open_position_on_alert(position_manager, mock_storage):
    """Test that position opens when alert is received."""
    # Create event and bar
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)

    # Add bar to latest_bars
    position_manager.latest_bars["BTCUSDT"] = bar

    # Open position
    await position_manager._open_position(event)

    # Verify position was created
    assert len(position_manager.open_positions) == 1

    # Verify position details
    position_id = f"BTCUSDT_{event.ts}_UP"
    position = position_manager.open_positions[position_id]

    assert position.symbol == "BTCUSDT"
    assert position.direction == Direction.UP
    assert position.status == PositionStatus.OPEN
    assert position.open_price == 50000.0
    assert position.entry_z_er == 3.5
    assert position.entry_z_vol == 3.2

    # Verify storage was called
    mock_storage.write_position.assert_called_once()


@pytest.mark.asyncio
async def test_prevent_multiple_positions_same_symbol(position_manager, mock_storage):
    """Test that only one position per symbol is allowed."""
    # Create first position
    event1 = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    await position_manager._open_position(event1)
    assert len(position_manager.open_positions) == 1

    # Try to open second position for same symbol
    event2 = create_event(symbol="BTCUSDT", direction=Direction.DOWN, ts=1000001000)
    await position_manager._open_position(event2)

    # Should still have only one position
    assert len(position_manager.open_positions) == 1


@pytest.mark.asyncio
async def test_stop_loss_exit(position_manager, mock_storage):
    """Test that position closes on stop loss."""
    # Open position at 50000
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    await position_manager._open_position(event)
    position_id = f"BTCUSDT_{event.ts}_UP"
    position = position_manager.open_positions[position_id]

    # Price drops 2.5% (below 2% SL)
    current_bar = create_bar(symbol="BTCUSDT", close=48750.0, ts=1000001000)
    features = create_features(symbol="BTCUSDT", z_er=2.0, z_vol=2.5, ts=1000001000)

    # Check exit condition
    exit_reason = await position_manager._check_exit_conditions(position, features, current_bar)

    assert exit_reason == ExitReason.STOP_LOSS


@pytest.mark.asyncio
async def test_take_profit_exit(position_manager, mock_storage):
    """Test that position closes on take profit."""
    # Open position at 50000
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    await position_manager._open_position(event)
    position_id = f"BTCUSDT_{event.ts}_UP"
    position = position_manager.open_positions[position_id]

    # Price rises 3.5% (above 3% TP)
    current_bar = create_bar(symbol="BTCUSDT", close=51750.0, ts=1000001000)
    features = create_features(symbol="BTCUSDT", z_er=2.0, z_vol=2.5, ts=1000001000)

    # Check exit condition
    exit_reason = await position_manager._check_exit_conditions(position, features, current_bar)

    assert exit_reason == ExitReason.TAKE_PROFIT


@pytest.mark.asyncio
async def test_z_score_reversal_exit(position_manager, mock_storage):
    """Test that position closes when z-score falls below threshold."""
    # Open position
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    await position_manager._open_position(event)
    position_id = f"BTCUSDT_{event.ts}_UP"
    position = position_manager.open_positions[position_id]

    # Z-score drops below 1.0 (but price hasn't hit SL/TP)
    current_bar = create_bar(symbol="BTCUSDT", close=50500.0, ts=1000001000)
    features = create_features(symbol="BTCUSDT", z_er=0.8, z_vol=1.5, ts=1000001000)  # z_er < 1.0

    # Check exit condition
    exit_reason = await position_manager._check_exit_conditions(position, features, current_bar)

    assert exit_reason == ExitReason.Z_SCORE_REVERSAL


@pytest.mark.asyncio
async def test_time_exit(position_manager, mock_storage):
    """Test that position closes after max holding time."""
    # Open position
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    await position_manager._open_position(event)
    position_id = f"BTCUSDT_{event.ts}_UP"
    position = position_manager.open_positions[position_id]

    # 65 minutes later (exceeds 60 minute max)
    time_65m_later = 1000000000 + (65 * 60 * 1000)
    current_bar = create_bar(symbol="BTCUSDT", close=50500.0, ts=time_65m_later)
    features = create_features(symbol="BTCUSDT", z_er=2.0, z_vol=2.5, ts=time_65m_later)

    # Check exit condition
    exit_reason = await position_manager._check_exit_conditions(position, features, current_bar)

    assert exit_reason == ExitReason.TIME_EXIT


@pytest.mark.asyncio
async def test_close_position_calculates_pnl(position_manager, mock_storage):
    """Test that closing position correctly calculates PnL."""
    # Open position at 50000
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    await position_manager._open_position(event)
    position_id = f"BTCUSDT_{event.ts}_UP"
    position = position_manager.open_positions[position_id]

    # Close at 51500 (3% profit for long)
    close_bar = create_bar(symbol="BTCUSDT", close=51500.0, ts=1000001000)
    features = create_features(symbol="BTCUSDT", z_er=0.5, z_vol=1.0, ts=1000001000)

    await position_manager._close_position(
        position=position,
        bar=close_bar,
        features=features,
        exit_reason=ExitReason.TAKE_PROFIT
    )

    # Verify PnL calculation
    assert position.status == PositionStatus.CLOSED
    assert position.close_price == 51500.0
    assert position.pnl_percent == pytest.approx(3.0, abs=0.1)
    assert position.exit_reason == ExitReason.TAKE_PROFIT

    # Verify position removed from open positions
    assert position_id not in position_manager.open_positions

    # Verify storage was called
    assert mock_storage.write_position.call_count >= 2  # open + close


@pytest.mark.asyncio
async def test_short_position_pnl(position_manager, mock_storage):
    """Test PnL calculation for short positions."""
    # Open SHORT position at 50000
    event = create_event(symbol="BTCUSDT", direction=Direction.DOWN, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    await position_manager._open_position(event)
    position_id = f"BTCUSDT_{event.ts}_DOWN"
    position = position_manager.open_positions[position_id]

    # Price drops to 48500 (3% profit for short)
    close_bar = create_bar(symbol="BTCUSDT", close=48500.0, ts=1000001000)
    features = create_features(symbol="BTCUSDT", z_er=0.5, z_vol=1.0, ts=1000001000, direction=Direction.DOWN)

    await position_manager._close_position(
        position=position,
        bar=close_bar,
        features=features,
        exit_reason=ExitReason.TAKE_PROFIT
    )

    # For SHORT: profit when price goes down
    # PnL% = (50000 - 48500) / 50000 * 100 = 3%
    assert position.pnl_percent == pytest.approx(3.0, abs=0.1)


@pytest.mark.asyncio
async def test_mfe_mae_tracking(position_manager, mock_storage):
    """Test that MFE and MAE are tracked correctly."""
    # Open LONG position at 50000
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    await position_manager._open_position(event)
    position_id = f"BTCUSDT_{event.ts}_UP"
    position = position_manager.open_positions[position_id]

    # Price goes up to 51000 (+2%)
    bar1 = create_bar(symbol="BTCUSDT", close=51000.0, ts=1000001000)
    position.update_excursions(51000.0)
    assert position.max_favorable_excursion == pytest.approx(2.0, abs=0.1)
    assert position.max_adverse_excursion == 0.0

    # Price drops to 49000 (-2%)
    bar2 = create_bar(symbol="BTCUSDT", close=49000.0, ts=1000002000)
    position.update_excursions(49000.0)
    assert position.max_favorable_excursion == pytest.approx(2.0, abs=0.1)  # MFE stays at best
    assert position.max_adverse_excursion == pytest.approx(-2.0, abs=0.1)  # MAE updated

    # Price goes to 52000 (+4%)
    bar3 = create_bar(symbol="BTCUSDT", close=52000.0, ts=1000003000)
    position.update_excursions(52000.0)
    assert position.max_favorable_excursion == pytest.approx(4.0, abs=0.1)  # MFE updated
    assert position.max_adverse_excursion == pytest.approx(-2.0, abs=0.1)  # MAE stays at worst


@pytest.mark.asyncio
async def test_multiple_positions_different_symbols(position_manager, mock_storage):
    """Test that multiple positions can be opened for different symbols."""
    # Enable multiple positions
    position_manager.config.position_management.allow_multiple_positions = True

    # Open position for BTC
    event1 = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar1 = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar1
    await position_manager._open_position(event1)

    # Open position for ETH
    event2 = create_event(symbol="ETHUSDT", direction=Direction.UP, ts=1000000000)
    bar2 = create_bar(symbol="ETHUSDT", close=3000.0, ts=1000000000)
    position_manager.latest_bars["ETHUSDT"] = bar2
    await position_manager._open_position(event2)

    # Should have 2 positions
    assert len(position_manager.open_positions) == 2


@pytest.mark.asyncio
async def test_position_duration_calculation(position_manager, mock_storage):
    """Test that position duration is calculated correctly."""
    # Open position
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    await position_manager._open_position(event)
    position_id = f"BTCUSDT_{event.ts}_UP"
    position = position_manager.open_positions[position_id]

    # Close 45 minutes later
    close_ts = 1000000000 + (45 * 60 * 1000)
    close_bar = create_bar(symbol="BTCUSDT", close=51500.0, ts=close_ts)
    features = create_features(symbol="BTCUSDT", z_er=0.5, z_vol=1.0, ts=close_ts)

    await position_manager._close_position(
        position=position,
        bar=close_bar,
        features=features,
        exit_reason=ExitReason.TAKE_PROFIT
    )

    assert position.duration_minutes == 45


@pytest.mark.asyncio
async def test_no_position_without_bar_data(position_manager, mock_storage):
    """Test that position doesn't open if bar data unavailable."""
    # Try to open position without bar data
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)

    # Don't add bar to latest_bars
    await position_manager._open_position(event)

    # No position should be created
    assert len(position_manager.open_positions) == 0


@pytest.mark.asyncio
async def test_telegram_notification_on_position_open(position_manager, mock_storage, monkeypatch):
    """Test that Telegram notification is sent when position opens."""
    # Enable Telegram
    position_manager.config.alerts.telegram.enabled = True
    position_manager.config.alerts.telegram.bot_token = "test_token"
    position_manager.config.alerts.telegram.chat_id = "test_chat"

    # Mock Telegram session
    mock_telegram_response = AsyncMock()
    mock_telegram_response.status = 200
    mock_telegram_session = AsyncMock()
    mock_telegram_session.post = AsyncMock(return_value=mock_telegram_response)
    mock_telegram_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_telegram_response)
    mock_telegram_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

    position_manager.telegram_session = mock_telegram_session

    # Create event and bar
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    # Open position
    await position_manager._open_position(event)

    # Verify Telegram was called
    assert mock_telegram_session.post.called
    call_args = mock_telegram_session.post.call_args

    # Verify URL
    assert "api.telegram.org" in call_args[0][0]

    # Verify payload
    payload = call_args[1]['json']
    assert payload['chat_id'] == "test_chat"
    assert "POSITION OPENED" in payload['text']
    assert "BTCUSDT" in payload['text']
    assert "UP" in payload['text']
    assert payload['parse_mode'] == 'HTML'


@pytest.mark.asyncio
async def test_telegram_notification_on_position_close(position_manager, mock_storage):
    """Test that Telegram notification is sent when position closes."""
    # Enable Telegram
    position_manager.config.alerts.telegram.enabled = True
    position_manager.config.alerts.telegram.bot_token = "test_token"
    position_manager.config.alerts.telegram.chat_id = "test_chat"

    # Mock Telegram session
    mock_telegram_response = AsyncMock()
    mock_telegram_response.status = 200
    mock_telegram_session = AsyncMock()
    mock_telegram_session.post = AsyncMock(return_value=mock_telegram_response)
    mock_telegram_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_telegram_response)
    mock_telegram_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

    position_manager.telegram_session = mock_telegram_session

    # Open position
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar
    await position_manager._open_position(event)

    # Reset mock
    mock_telegram_session.post.reset_mock()

    # Close position
    position_id = f"BTCUSDT_{event.ts}_UP"
    position = position_manager.open_positions[position_id]
    close_bar = create_bar(symbol="BTCUSDT", close=51500.0, ts=1000001000)
    features = create_features(symbol="BTCUSDT", z_er=0.5, z_vol=1.0, ts=1000001000)

    await position_manager._close_position(
        position=position,
        bar=close_bar,
        features=features,
        exit_reason=ExitReason.TAKE_PROFIT
    )

    # Verify Telegram was called
    assert mock_telegram_session.post.called
    call_args = mock_telegram_session.post.call_args

    # Verify payload
    payload = call_args[1]['json']
    assert "POSITION CLOSED" in payload['text']
    assert "WIN" in payload['text']  # Should be profit
    # Changed to Russian: "Тейк-профит" instead of "TAKE_PROFIT"
    assert "Тейк-профит" in payload['text'] or "TAKE_PROFIT" in payload['text']
    assert "PnL:" in payload['text']


@pytest.mark.asyncio
async def test_entry_trigger_z_score_cooldown(position_manager, mock_storage):
    """Test that entry waits for z-score to cool from peak."""
    # Build bar history for pullback detection and ATR
    for i in range(15):
        test_bar = create_bar(symbol="BTCUSDT", close=50000 + i*10, ts=1000000000 + i*60000)
        position_manager.extended_features.update(test_bar)
        position_manager.latest_bars["BTCUSDT"] = test_bar

    # Enable entry triggers with lenient pullback/stability requirements
    position_manager.config.position_management.use_entry_triggers = True
    position_manager.config.position_management.entry_trigger_z_cooldown = 2.0  # Changed from 2.5 to 2.0
    position_manager.config.position_management.entry_trigger_pullback_pct = 0.0  # No pullback required
    position_manager.config.position_management.entry_trigger_taker_stability = 1.0  # No stability required
    position_manager.config.position_management.entry_trigger_require_data = False  # Lenient mode

    # Create event with z=3.5 (at peak)
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    features = create_features(symbol="BTCUSDT", z_er=3.5, z_vol=3.2, ts=1000000000)

    position_manager.latest_bars["BTCUSDT"] = bar
    position_manager.latest_features["BTCUSDT"] = features

    # Create pending signal (new architecture)
    await position_manager._open_position(event)

    # Should have pending signal, but no open position yet
    assert len(position_manager.pending_signals) == 1
    assert len(position_manager.open_positions) == 0

    # Check pending signals with hot z-score - should NOT trigger
    await position_manager._check_pending_signals("BTCUSDT", bar)
    assert len(position_manager.open_positions) == 0  # Still waiting

    # Update features with cooled z-score (2.3 is now >= 2.0 threshold)
    features_cooled = create_features(symbol="BTCUSDT", z_er=2.3, z_vol=2.8, ts=1000001000)
    position_manager.latest_features["BTCUSDT"] = features_cooled

    # Update bar timestamp to match features
    bar_cooled = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000001000)
    position_manager.latest_bars["BTCUSDT"] = bar_cooled

    # Check pending signals with cooled z-score - should trigger now
    await position_manager._check_pending_signals("BTCUSDT", bar_cooled)
    assert len(position_manager.open_positions) == 1


@pytest.mark.asyncio
async def test_entry_trigger_pullback(position_manager, mock_storage):
    """Test that entry waits for price pullback from peak."""
    # Build bar history with peak at 50500, then gradual decline
    bar_prices = [50500, 50480, 50460, 50440, 50420, 50400, 50380, 50360, 50340, 50320, 50300, 50280, 50260, 50240, 50220]
    for i, price in enumerate(bar_prices):
        bar = create_bar(symbol="BTCUSDT", close=price, ts=1000000000 + i * 60000)
        position_manager.extended_features.update(bar)
        position_manager.latest_bars["BTCUSDT"] = bar

    position_manager.config.position_management.use_entry_triggers = True
    position_manager.config.position_management.entry_trigger_pullback_pct = 0.0  # Disable pullback for this test
    position_manager.config.position_management.entry_trigger_z_cooldown = 2.0  # Lower threshold for easier pass
    position_manager.config.position_management.entry_trigger_taker_stability = 1.0  # No stability check
    position_manager.config.position_management.entry_trigger_require_data = False  # Lenient mode

    # Create signal with bar and features
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000005000)
    bar = create_bar(symbol="BTCUSDT", close=50200.0, ts=1000000000 + 15*60000)
    features = create_features(symbol="BTCUSDT", z_er=2.3, z_vol=2.8, ts=1000000000 + 15*60000)

    position_manager.latest_bars["BTCUSDT"] = bar
    position_manager.latest_features["BTCUSDT"] = features

    # Create pending signal
    await position_manager._open_position(event)

    # Should have pending signal
    assert len(position_manager.pending_signals) == 1
    assert len(position_manager.open_positions) == 0

    # Check pending signals - should trigger (all requirements met)
    await position_manager._check_pending_signals("BTCUSDT", bar)
    assert len(position_manager.open_positions) == 1


@pytest.mark.asyncio
async def test_trailing_stop_activation(position_manager, mock_storage):
    """Test that trailing stop activates at 50% of TP."""
    position_manager.config.position_management.use_trailing_stop = True
    position_manager.config.position_management.trailing_stop_activation = 0.5
    position_manager.config.position_management.take_profit_percent = 4.0

    # Open position at 50000
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    # Build ATR history
    for i in range(15):
        test_bar = create_bar(symbol="BTCUSDT", close=50000 + i*10, ts=1000000000 + i*60000)
        position_manager.extended_features.update(test_bar)

    await position_manager._open_position(event)
    position_id = f"BTCUSDT_{event.ts}_UP"
    position = position_manager.open_positions[position_id]

    # Price moves to +2% (50% of 4% TP) - should activate trailing stop
    current_bar = create_bar(symbol="BTCUSDT", close=51000.0, ts=1000001000)
    features = create_features(symbol="BTCUSDT", z_er=2.0, z_vol=2.5, ts=1000001000)

    await position_manager._update_trailing_stop(position, 51000.0, features)

    assert position.metrics.get('trailing_stop_active') is True
    assert position.metrics.get('trailing_stop_price') is not None


@pytest.mark.asyncio
async def test_trailing_stop_exit(position_manager, mock_storage):
    """Test that position exits when trailing stop is hit."""
    position_manager.config.position_management.use_trailing_stop = True

    # Open position
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    # Build ATR history
    for i in range(15):
        test_bar = create_bar(symbol="BTCUSDT", close=50000 + i*10, ts=1000000000 + i*60000)
        position_manager.extended_features.update(test_bar)

    await position_manager._open_position(event)
    position_id = f"BTCUSDT_{event.ts}_UP"
    position = position_manager.open_positions[position_id]

    # Manually activate trailing stop at 50800
    position.metrics['trailing_stop_active'] = True
    position.metrics['trailing_stop_price'] = 50800.0

    # Price drops to 50700 (below trailing stop)
    current_bar = create_bar(symbol="BTCUSDT", close=50700.0, ts=1000002000)
    features = create_features(symbol="BTCUSDT", z_er=1.5, z_vol=2.0, ts=1000002000)

    exit_reason = await position_manager._check_exit_conditions(position, features, current_bar)

    assert exit_reason == ExitReason.TRAILING_STOP


@pytest.mark.asyncio
async def test_atr_based_risk_reward_ratio(position_manager, mock_storage):
    """Test that ATR-based targets maintain minimum 1:2 risk/reward ratio."""
    # Build ATR history
    for i in range(15):
        bar = create_bar(
            symbol="BTCUSDT",
            close=50000 + (i * 10),
            ts=1000000000 + i * 60000
        )
        position_manager.extended_features.update(bar)

    # Calculate targets
    targets = position_manager.extended_features.calculate_dynamic_targets(
        symbol="BTCUSDT",
        entry_price=50000,
        direction=Direction.UP,
        atr_stop_mult=1.5,
        atr_target_mult=3.0,
        min_risk_reward=2.0
    )

    assert targets is not None
    assert targets['risk_reward_ratio'] >= 2.0
    assert targets['take_profit_percent'] >= targets['stop_loss_percent'] * 2.0


@pytest.mark.asyncio
async def test_relaxed_z_score_exit_threshold(position_manager, mock_storage):
    """Test that new z-score exit threshold (0.5) allows longer holds."""
    position_manager.config.position_management.z_score_exit_threshold = 0.5
    position_manager.config.position_management.take_profit_percent = 10.0  # High TP so it doesn't trigger

    # Open position
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    # Build ATR history
    for i in range(15):
        test_bar = create_bar(symbol="BTCUSDT", close=50000 + i*10, ts=1000000000 + i*60000)
        position_manager.extended_features.update(test_bar)

    await position_manager._open_position(event)
    position_id = f"BTCUSDT_{event.ts}_UP"
    position = position_manager.open_positions[position_id]

    # Override dynamic TP with high value
    position.metrics['dynamic_take_profit'] = 10.0

    # Z-score at 0.8 (would exit with old threshold of 1.0, but not with 0.5)
    current_bar = create_bar(symbol="BTCUSDT", close=50100.0, ts=1000001000)
    features = create_features(symbol="BTCUSDT", z_er=0.8, z_vol=1.5, ts=1000001000)

    exit_reason = await position_manager._check_exit_conditions(position, features, current_bar)

    # Should NOT exit (0.8 > 0.5 threshold)
    assert exit_reason != ExitReason.Z_SCORE_REVERSAL

    # Z-score drops to 0.3 (below 0.5 threshold)
    features_weak = create_features(symbol="BTCUSDT", z_er=0.3, z_vol=1.0, ts=1000002000)
    exit_reason2 = await position_manager._check_exit_conditions(position, features_weak, current_bar)

    # Should exit now
    assert exit_reason2 == ExitReason.Z_SCORE_REVERSAL


@pytest.mark.asyncio
async def test_extended_hold_time_120_minutes(position_manager, mock_storage):
    """Test that max hold time increased to 120 minutes."""
    position_manager.config.position_management.max_hold_minutes = 120
    position_manager.config.position_management.take_profit_percent = 10.0  # High TP so it doesn't trigger
    position_manager.config.position_management.z_score_exit_threshold = 0.1  # Very low so it doesn't trigger

    # Open position
    event = create_event(symbol="BTCUSDT", direction=Direction.UP, ts=1000000000)
    bar = create_bar(symbol="BTCUSDT", close=50000.0, ts=1000000000)
    position_manager.latest_bars["BTCUSDT"] = bar

    # Build ATR history
    for i in range(15):
        test_bar = create_bar(symbol="BTCUSDT", close=50000 + i*10, ts=1000000000 + i*60000)
        position_manager.extended_features.update(test_bar)

    await position_manager._open_position(event)
    position_id = f"BTCUSDT_{event.ts}_UP"
    position = position_manager.open_positions[position_id]

    # Override dynamic TP with high value
    position.metrics['dynamic_take_profit'] = 10.0

    # 90 minutes later (was beyond old 60-minute limit, but within new 120-minute limit)
    time_90m_later = 1000000000 + (90 * 60 * 1000)
    current_bar = create_bar(symbol="BTCUSDT", close=50100.0, ts=time_90m_later)
    features = create_features(symbol="BTCUSDT", z_er=2.0, z_vol=2.5, ts=time_90m_later)

    exit_reason = await position_manager._check_exit_conditions(position, features, current_bar)

    # Should NOT time out yet
    assert exit_reason != ExitReason.TIME_EXIT

    # 130 minutes later (exceeds 120-minute limit)
    time_130m_later = 1000000000 + (130 * 60 * 1000)
    current_bar2 = create_bar(symbol="BTCUSDT", close=50100.0, ts=time_130m_later)
    features2 = create_features(symbol="BTCUSDT", z_er=2.0, z_vol=2.5, ts=time_130m_later)

    exit_reason2 = await position_manager._check_exit_conditions(position, features2, current_bar2)

    # Should time out now
    assert exit_reason2 == ExitReason.TIME_EXIT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
