"""Anomaly detector with strict trigger rules."""

import asyncio
import logging
import numpy as np
from typing import Optional, Tuple
from detector.models import Features, Event, Direction, SignalClass, TradingMode
from detector.storage import Storage
from detector.binance_rest import BinanceRestClient
from detector.config import Config

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects initiator signals based on strict trigger rules.

    Implements:
    - Initiator signal detection (z-score + volume + taker dominance)
    - Cooldown logic with direction-aware grace period
    """

    def __init__(
        self,
        feature_queue: asyncio.Queue,
        rest_client: BinanceRestClient,
        storage: Storage,
        config: Config,
        event_queues: list = None
    ):
        self.feature_queue = feature_queue
        self.event_queues = event_queues or []  # Only position manager queue(s)
        self.rest_client = rest_client
        self.storage = storage
        self.config = config

    async def run(self) -> None:
        """Consume features and detect events."""
        logger.info("Anomaly detector started")

        while True:
            try:
                features = await self.feature_queue.get()
                await self._process_features(features)
            except Exception as e:
                logger.error(f"Error in detector: {e}")

    async def _process_features(self, features: Features) -> None:
        """Check for initiator signal."""
        # Determine if hybrid strategy or legacy mode
        hybrid_enabled = self.config.hybrid_strategy.enabled

        # Check initiator trigger (threshold depends on mode)
        if hybrid_enabled:
            signal_class = self._classify_signal(features)
            if signal_class is None:
                return  # No signal detected
        else:
            # Legacy mode: check standard trigger
            if not self._check_initiator_trigger(features):
                return
            signal_class = None

        # Check cooldown
        cooldown_ms = self.config.alerts.cooldown_minutes_per_symbol * 60 * 1000
        grace_ms = self.config.alerts.direction_swap_grace_minutes * 60 * 1000

        is_allowed = await self.storage.check_cooldown(
            features.symbol,
            features.direction,
            features.ts_minute,
            cooldown_ms,
            grace_ms
        )

        if not is_allowed:
            logger.info(f"Cooldown active for {features.symbol} {features.direction.value}, skipping alert")
            return

        # Create event with signal classification
        event = Event(
            event_id=f"{features.symbol}_{features.ts_minute}",
            ts=features.ts_minute,
            initiator_symbol=features.symbol,
            direction=features.direction,
            metrics=self._extract_metrics(features),
            # Hybrid strategy fields
            signal_class=signal_class,
            original_z_score=features.z_er_15m,
            original_vol_z=features.z_vol_15m,
            original_taker_share=features.taker_buy_share_15m,
            original_price=None  # Will be set by position manager (has access to bar)
        )

        # Broadcast to event queues (position manager)
        for queue in self.event_queues:
            try:
                await queue.put(('initiator', event))
            except Exception as e:
                logger.error(f"Error broadcasting event to queue: {e}")

        # Write to database
        await self.storage.write_event(event)

        # NOTE: Cooldown is updated by position_manager AFTER signal is accepted
        # (not blocked by filters). This ensures cooldown only applies when
        # a pending signal or position is actually created.

        # Log with signal class if hybrid mode
        if signal_class:
            logger.info(
                f"Hybrid signal detected: {features.symbol} {features.direction.value} "
                f"class={signal_class.value} z={features.z_er_15m:.2f}"
            )
        else:
            logger.info(f"Initiator detected: {features.symbol} {features.direction.value}")

    def _check_initiator_trigger(self, features: Features) -> bool:
        """
        Apply strict trigger rules for initiator signal.

        Returns True if ALL conditions met:
        - A) abs(z_er_15m) >= excess_return_z_initiator (bidirectional)
        - B) z_vol_15m >= volume_z_initiator
        - C) taker_buy_share_15m >= taker_dominance_min OR <= (1 - taker_dominance_min)
        """
        cfg = self.config.thresholds

        # Rule A: Excess return z-score (bidirectional - use absolute value)
        if abs(features.z_er_15m) < cfg.excess_return_z_initiator:
            return False

        # Rule B: Volume z-score
        if features.z_vol_15m < cfg.volume_z_initiator:
            return False

        # Rule C: Taker dominance (bidirectional)
        taker_share = features.taker_buy_share_15m
        if taker_share is None or np.isnan(taker_share):
            return False

        # Allow both high taker buy (bullish) and high taker sell (bearish)
        if not (taker_share >= cfg.taker_dominance_min or taker_share <= (1 - cfg.taker_dominance_min)):
            return False

        return True

    def _extract_metrics(self, features: Features) -> dict:
        """Extract metrics for event record."""
        return {
            'z_er': features.z_er_15m,
            'z_vol': features.z_vol_15m,
            'taker_share': features.taker_buy_share_15m or 0,
            'beta': features.beta,
            'funding': features.funding_rate or 0,
        }

    def _classify_signal(self, features: Features) -> Optional[SignalClass]:
        """
        Classify signal for hybrid strategy based on z-score magnitude.

        Returns:
            SignalClass or None if no signal detected.

        Classification:
            - EXTREME_SPIKE: z >= 5.0 (mean-reversion)
            - STRONG_SIGNAL: 3.0 <= z < 5.0 (conditional momentum)
            - EARLY_SIGNAL: 1.5 <= z < 3.0 (wait for continuation)
            - None: z < 1.5 (no signal)
        """
        hs_cfg = self.config.hybrid_strategy
        common_cfg = hs_cfg.common

        # Get absolute z-score
        z_abs = abs(features.z_er_15m)

        # Volume filter (all modes require minimum volume z-score)
        if features.z_vol_15m < common_cfg.min_volume_z_for_signal:
            return None

        # Taker dominance check (bidirectional)
        taker_share = features.taker_buy_share_15m
        if taker_share is None or np.isnan(taker_share):
            return None

        # For hybrid strategy, use common thresholds for taker
        is_bullish_flow = taker_share >= common_cfg.taker_bullish_threshold
        is_bearish_flow = taker_share <= common_cfg.taker_bearish_threshold

        if not (is_bullish_flow or is_bearish_flow):
            return None  # Neutral flow, no signal

        # Classify by z-score magnitude
        if z_abs >= hs_cfg.extreme_spike_threshold:
            return SignalClass.EXTREME_SPIKE
        elif z_abs >= hs_cfg.strong_signal_min:
            return SignalClass.STRONG_SIGNAL
        elif z_abs >= hs_cfg.early_signal_min:
            return SignalClass.EARLY_SIGNAL
        else:
            return None  # Below minimum threshold
