"""Anomaly detector with strict trigger rules and sector diffusion."""

import asyncio
import logging
import numpy as np
from typing import Dict
from detector.models import Features, Event, PendingSectorEvent, EventStatus, Direction
from detector.storage import Storage
from detector.binance_rest import BinanceRestClient
from detector.config import Config

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects sector shot events based on strict trigger rules.

    Implements:
    - Initiator signal detection
    - Confirmation status (CONFIRMED/PARTIAL/UNCONFIRMED)
    - Sector diffusion detection
    - Cooldown logic with direction-aware grace period
    """

    def __init__(
        self,
        feature_queue: asyncio.Queue,
        event_queue: asyncio.Queue,
        rest_client: BinanceRestClient,
        storage: Storage,
        config: Config
    ):
        self.feature_queue = feature_queue
        self.event_queue = event_queue
        self.rest_client = rest_client
        self.storage = storage
        self.config = config

        # Track pending sector events (waiting for followers)
        self.pending_sector_events: Dict[str, PendingSectorEvent] = {}

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
        """Check for initiator signal and sector diffusion."""
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

            # Determine confirmation status
            status = self._determine_status(features)

            # Create event
            event = Event(
                event_id=f"{features.symbol}_{features.ts_minute}",
                ts=features.ts_minute,
                initiator_symbol=features.symbol,
                direction=features.direction,
                status=status,
                followers=[],
                metrics=self._extract_metrics(features)
            )

            # Emit initiator alert
            await self.event_queue.put(('initiator', event))

            # Write to database
            await self.storage.write_event(event)

            # Update cooldown
            await self.storage.update_cooldown(features.symbol, features.direction, features.ts_minute)

            # Track for sector diffusion
            window_end_ts = features.ts_minute + self.config.windows.sector_diffusion_window_bars * 60 * 1000
            self.pending_sector_events[features.symbol] = PendingSectorEvent(
                initiator=event,
                window_end=window_end_ts
            )

            logger.info(f"Initiator detected: {features.symbol} {features.direction.value} (status: {status.value})")

        # Check if this is a follower for any pending sector event
        await self._check_sector_diffusion(features)

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

    def _determine_status(self, features: Features) -> EventStatus:
        """
        Determine confirmation status based on available data.

        - No API key → UNCONFIRMED
        - 2+ confirmations → CONFIRMED
        - 1 confirmation → PARTIAL
        - 0 confirmations → UNCONFIRMED
        """
        if not self.rest_client.api_key:
            return EventStatus.UNCONFIRMED

        confirmations = 0
        cfg = self.config.thresholds

        # Check OI delta
        if features.oi_delta_1h is not None and features.z_oi_delta_1h is not None:
            if features.z_oi_delta_1h >= cfg.oi_delta_z_confirm:
                confirmations += 1

        # Check liquidations
        if features.liq_15m is not None and features.z_liq_15m is not None:
            if features.z_liq_15m >= cfg.liquidation_z_confirm:
                confirmations += 1

        # Check funding rate (absolute threshold)
        if features.funding_rate is not None:
            if abs(features.funding_rate) >= cfg.funding_abs_threshold:
                confirmations += 1

        if confirmations >= 2:
            return EventStatus.CONFIRMED
        elif confirmations == 1:
            return EventStatus.PARTIAL
        else:
            return EventStatus.UNCONFIRMED

    def _extract_metrics(self, features: Features) -> Dict:
        """Extract metrics for event record."""
        return {
            'z_er': features.z_er_15m,
            'z_vol': features.z_vol_15m,
            'taker_share': features.taker_buy_share_15m or 0,
            'beta': features.beta,
            'funding': features.funding_rate or 0,
            'oi_delta': features.oi_delta_1h,
            'z_oi': features.z_oi_delta_1h,
            'liq_15m': features.liq_15m,
            'z_liq': features.z_liq_15m
        }

    async def _check_sector_diffusion(self, features: Features) -> None:
        """
        Check if this feature qualifies as a follower for pending sector events.

        Simplified signal: z_er_15m >= 2.0 AND z_vol_15m >= 2.0 AND same direction.
        Only checks followers within the same sector as the initiator.
        """
        # Clean up expired pending events
        expired_symbols = [
            symbol for symbol, pending in self.pending_sector_events.items()
            if features.ts_minute > pending.window_end
        ]
        for symbol in expired_symbols:
            del self.pending_sector_events[symbol]

        # Check each pending event
        for initiator_symbol, pending in list(self.pending_sector_events.items()):
            # Skip if this is the initiator itself
            if features.symbol == initiator_symbol:
                continue

            # Get initiator's sector
            initiator_sector = self.config.universe.get_sector_for_symbol(initiator_symbol)
            if not initiator_sector:
                continue

            # Skip if follower is not in the same sector as initiator
            if features.symbol not in initiator_sector.symbols:
                continue

            # Skip if outside time window
            if features.ts_minute > pending.window_end:
                continue

            # Skip if before initiator (mode: "after_initiator")
            if features.ts_minute <= pending.initiator.ts:
                continue

            # Check simplified signal (abs(z) >= 2.0 for both er and vol, same direction)
            if (abs(features.z_er_15m) >= 2.0 and
                features.z_vol_15m >= 2.0 and
                features.direction == pending.initiator.direction):

                # Check if already added
                if features.symbol not in [f.symbol for f in pending.followers]:
                    pending.followers.append(features)
                    logger.debug(f"Follower added: {features.symbol} for initiator {initiator_symbol} (sector: {initiator_sector.name})")

                    # Check if sector event threshold met
                    # Total sector size excludes the initiator itself
                    total_sector = len(initiator_sector.symbols) - 1
                    follower_count = len(pending.followers)

                    if total_sector > 0:
                        follower_share = follower_count / total_sector
                    else:
                        follower_share = 0

                    if (follower_count >= self.config.thresholds.sector_k_min and
                        follower_share >= self.config.thresholds.sector_share_min):

                        # Emit sector event
                        sector_event = Event(
                            event_id=f"SECTOR_{initiator_symbol}_{pending.initiator.ts}",
                            ts=features.ts_minute,
                            initiator_symbol=initiator_symbol,
                            direction=pending.initiator.direction,
                            status=EventStatus.SECTOR_DIFFUSION,
                            followers=[f.symbol for f in pending.followers],
                            metrics={
                                'sector_name': initiator_sector.name,
                                'follower_count': follower_count,
                                'follower_share': follower_share,
                                'follower_metrics': [
                                    {
                                        'symbol': f.symbol,
                                        'z_er': f.z_er_15m,
                                        'z_vol': f.z_vol_15m
                                    }
                                    for f in pending.followers
                                ]
                            }
                        )

                        await self.event_queue.put(('sector', sector_event))
                        await self.storage.write_event(sector_event)

                        logger.info(f"Sector diffusion detected: {initiator_symbol} in sector '{initiator_sector.name}' with {follower_count} followers")

                        # Remove from pending
                        del self.pending_sector_events[initiator_symbol]
                        break
