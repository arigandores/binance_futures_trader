"""Tests for detector trigger rules."""

import pytest
from detector.models import Features, Direction
from detector.config import ThresholdsConfig


def test_initiator_trigger():
    """Test initiator rule with synthetic features."""
    # Create features that should trigger
    features = Features(
        symbol="BTCUSDT",
        ts_minute=1000000,
        z_er_15m=3.5,  # Above threshold (3.0)
        z_vol_15m=3.2,  # Above threshold (3.0)
        taker_buy_share_15m=0.70,  # Above threshold (0.65)
    )

    cfg = ThresholdsConfig()

    # Check rules
    assert features.z_er_15m >= cfg.excess_return_z_initiator
    assert features.z_vol_15m >= cfg.volume_z_initiator
    assert features.taker_buy_share_15m >= cfg.taker_dominance_min


def test_direction_determination():
    """Test direction determination (UP vs DOWN)."""
    # UP direction (positive excess return)
    features_up = Features(
        symbol="BTCUSDT",
        ts_minute=1000000,
        er_15m=0.05  # Positive
    )
    assert features_up.determine_direction() == Direction.UP

    # DOWN direction (negative excess return)
    features_down = Features(
        symbol="BTCUSDT",
        ts_minute=1000000,
        er_15m=-0.03  # Negative
    )
    assert features_down.determine_direction() == Direction.DOWN
