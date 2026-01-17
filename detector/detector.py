"""Anomaly detector with strict trigger rules."""

import asyncio
import logging
import numpy as np
from detector.models import Features, Event, Direction
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
        # Check initiator trigger
        if self._check_initiator_trigger(features):
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

            # Create event
            event = Event(
                event_id=f"{features.symbol}_{features.ts_minute}",
                ts=features.ts_minute,
                initiator_symbol=features.symbol,
                direction=features.direction,
                metrics=self._extract_metrics(features)
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
