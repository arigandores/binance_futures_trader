"""Virtual position manager - opens and closes positions based on alerts and exit conditions."""

import asyncio
import logging
import aiohttp
from typing import Dict, Optional
from datetime import datetime

def format_price(price: float) -> str:
    """Format price with appropriate decimal places based on magnitude."""
    if price >= 100:
        return f"${price:,.2f}"
    elif price >= 10:
        return f"${price:,.3f}"
    elif price >= 1:
        return f"${price:,.4f}"
    elif price >= 0.01:
        return f"${price:,.5f}"
    elif price >= 0.0001:
        return f"${price:,.6f}"
    else:
        return f"${price:,.8f}"
from detector.models import (
    Event, Features, Bar, Position, PositionStatus, ExitReason, Direction, EventStatus,
    PendingSignal
)
from detector.storage import Storage
from detector.features_extended import ExtendedFeatureCalculator
from detector.config import Config

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages virtual trading positions.

    Responsibilities:
    - Opens positions on initiator alerts (CONFIRMED or UNCONFIRMED)
    - Monitors open positions and updates MFE/MAE
    - Closes positions based on exit conditions:
        * Z-Score Reversal (z_er < threshold)
        * Stop Loss / Take Profit (fixed %)
        * ATR-based dynamic stops
        * Order Flow Reversal
        * Time-based exit
        * Opposite signal
    """

    def __init__(
        self,
        event_queue: asyncio.Queue,
        feature_queue: asyncio.Queue,
        bar_queue: asyncio.Queue,
        storage: Storage,
        config: Config
    ):
        self.event_queue = event_queue
        self.feature_queue = feature_queue
        self.bar_queue = bar_queue
        self.storage = storage
        self.config = config

        # Extended feature calculator
        self.extended_features = ExtendedFeatureCalculator(
            atr_period=config.position_management.atr_period
        )

        # In-memory tracking of open positions
        self.open_positions: Dict[str, Position] = {}  # position_id -> Position

        # Pending signals queue (Must-Fix architecture)
        self.pending_signals: Dict[str, PendingSignal] = {}  # signal_id -> PendingSignal

        # Must-Fix #11: Lock to prevent race conditions in dual-trigger
        self.pending_locks: Dict[str, asyncio.Lock] = {}  # symbol -> Lock

        # Latest features and bars per symbol (for exit checks)
        self.latest_features: Dict[str, Features] = {}  # symbol -> Features
        self.latest_bars: Dict[str, Bar] = {}  # symbol -> Bar

        # Telegram session
        self.telegram_session: Optional[aiohttp.ClientSession] = None

    async def init(self) -> None:
        """Initialize by loading open positions from database."""
        positions = await self.storage.get_open_positions()
        for pos in positions:
            self.open_positions[pos.position_id] = pos

        # Initialize Telegram session if enabled
        if self.config.alerts.telegram.enabled:
            self.telegram_session = aiohttp.ClientSession()
            logger.info("Telegram notifications enabled for positions")

        logger.info(f"Position manager initialized with {len(self.open_positions)} open positions")

    async def run(self) -> None:
        """Main event loop - handles alerts, features, and bars."""
        await self.init()
        logger.info("Position manager started")

        # Create tasks for different event sources
        tasks = [
            asyncio.create_task(self._handle_alerts()),
            asyncio.create_task(self._handle_features()),
            asyncio.create_task(self._handle_bars()),
            asyncio.create_task(self._cleanup_expired_pending_signals())  # NEW
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Position manager error: {e}")
            for task in tasks:
                task.cancel()

    async def _handle_alerts(self) -> None:
        """Listen for initiator alerts and create pending signals."""
        while True:
            try:
                event_type, event = await self.event_queue.get()

                # Only process initiator alerts
                if event_type == 'initiator':
                    # NEW: Create pending signal instead of opening immediately
                    await self._create_pending_signal(event)

            except Exception as e:
                logger.error(f"Error handling alert: {e}")

    async def _handle_features(self) -> None:
        """Listen for feature updates and check exit conditions."""
        while True:
            try:
                features = await self.feature_queue.get()
                self.latest_features[features.symbol] = features

                # Check exit conditions for positions in this symbol
                await self._check_exits_for_symbol(features.symbol, features)

                # Must-Fix #9: Also check pending signals for this symbol
                # (features just updated, may now be fresh enough for trigger evaluation)
                bar = self.latest_bars.get(features.symbol)
                if bar:
                    await self._check_pending_signals(features.symbol, bar)

            except Exception as e:
                logger.error(f"Error handling features: {e}")

    async def _handle_bars(self) -> None:
        """Listen for bar updates and check pending signals + position exits."""
        while True:
            try:
                bar = await self.bar_queue.get()
                self.latest_bars[bar.symbol] = bar

                # Update extended features
                self.extended_features.update(bar)

                # Update MFE/MAE for open positions
                await self._update_excursions(bar)

                # NEW: Check pending signals for this symbol
                await self._check_pending_signals(bar.symbol, bar)

            except Exception as e:
                logger.error(f"Error handling bars: {e}")

    async def _check_entry_triggers(
        self,
        event: Event,
        current_bar: Bar,
        current_features: Optional[Features]
    ) -> bool:
        """
        Check if entry triggers are met (Signal+Trigger separation).

        Entry triggers (if enabled):
        1. Z-score has cooled from peak (abs(z_ER) in range 2.0-2.5)
        2. Price has pulled back slightly from recent peak (0.5-1%)
        3. Taker flow has stabilized (< 10% change over 2-3 bars)

        Returns True if all required triggers met, False otherwise.
        """
        cfg = self.config.position_management

        # If triggers disabled, allow immediate entry
        if not cfg.use_entry_triggers:
            return True

        symbol = event.initiator_symbol
        direction = event.direction

        # Need features for z-score check
        if current_features is None:
            logger.debug(f"{symbol}: No features available for trigger validation")
            return False

        # Trigger 1: Z-score cooldown
        # Signal detected at z_ER >= 3.0, but wait for it to cool to 2.0-2.5 range
        current_z_er = abs(current_features.z_er_15m)

        if current_z_er > 3.0:
            # Still at peak, wait for cooldown
            logger.debug(f"{symbol}: Z-score still at peak ({current_z_er:.2f}), waiting for cooldown")
            return False

        if current_z_er < cfg.entry_trigger_z_cooldown:
            # Cooled too much, signal too weak
            logger.debug(f"{symbol}: Z-score too weak ({current_z_er:.2f} < {cfg.entry_trigger_z_cooldown})")
            return False

        # Z-score in acceptable range (2.0-2.5 by default)
        logger.debug(f"{symbol}: Z-score in trigger range ({current_z_er:.2f})")

        # Trigger 2: Price pullback from peak
        peak_price = self.extended_features.get_recent_price_peak(symbol, direction, lookback_bars=5)

        if peak_price is not None:
            if direction == Direction.UP:
                pullback_pct = (peak_price - current_bar.close) / peak_price * 100
            else:
                pullback_pct = (current_bar.close - peak_price) / peak_price * 100

            if pullback_pct < cfg.entry_trigger_pullback_pct:
                logger.debug(f"{symbol}: Insufficient pullback ({pullback_pct:.2f}% < {cfg.entry_trigger_pullback_pct}%)")
                return False

            logger.debug(f"{symbol}: Pullback sufficient ({pullback_pct:.2f}%)")

        # Trigger 3: Taker flow stability
        taker_stability = self.extended_features.get_taker_flow_stability(symbol, lookback_bars=3)

        if taker_stability is not None:
            if taker_stability > cfg.entry_trigger_taker_stability:
                logger.debug(f"{symbol}: Taker flow unstable (change: {taker_stability:.2f})")
                return False

            logger.debug(f"{symbol}: Taker flow stable (max change: {taker_stability:.2f})")

        # All triggers met
        logger.info(f"{symbol}: All entry triggers met - opening position")
        return True

    def _get_pending_lock(self, symbol: str) -> asyncio.Lock:
        """Get or create lock for symbol (Must-Fix #11)."""
        if symbol not in self.pending_locks:
            self.pending_locks[symbol] = asyncio.Lock()
        return self.pending_locks[symbol]

    async def _create_pending_signal(self, event: Event) -> None:
        """
        Create a pending signal that waits for entry triggers.

        If use_entry_triggers = False, opens position immediately (backward compatible).
        If use_entry_triggers = True, adds to pending queue.
        """
        cfg = self.config.position_management
        symbol = event.initiator_symbol

        # Backward compatibility: if triggers disabled, open immediately
        if not cfg.use_entry_triggers:
            await self._open_position_from_event(event)
            return

        # Must-Fix #6: Stricter pending enforcement if allow_multiple_positions=false
        if not cfg.allow_multiple_positions:
            # Check ANY pending for this symbol (regardless of direction)
            existing_pending_any = [
                s for s in self.pending_signals.values()
                if s.symbol == symbol
            ]

            if existing_pending_any:
                logger.info(
                    f"{symbol}: Pending signal already exists (any direction), "
                    f"skipping new signal (allow_multiple_positions=false prevents duplicates)"
                )
                return

            # Check if already have open position
            existing_open = [p for p in self.open_positions.values()
                            if p.symbol == symbol and p.status == PositionStatus.OPEN]

            if existing_open:
                logger.debug(f"{symbol}: Position already open, skipping new signal")
                return
        else:
            # If allow_multiple_positions=true, only block same-direction pendings
            existing_pending_same_direction = [
                s for s in self.pending_signals.values()
                if s.symbol == symbol and s.direction == event.direction
            ]

            if existing_pending_same_direction:
                logger.info(
                    f"{symbol}: Pending signal already exists for {event.direction.value} direction, "
                    f"skipping new signal"
                )
                return

        # Get current bar
        bar = self.latest_bars.get(symbol)
        if not bar:
            logger.warning(f"{symbol}: No bar data, cannot create pending signal")
            return

        # =====================================================================
        # WIN_RATE_MAX: Market regime filters (Step 4.1 integration)
        # Run BEFORE creating pending signal to filter out bad market conditions
        # =====================================================================
        if self.config.position_management.profile == "WIN_RATE_MAX":
            # Check BTC anomaly filter
            if not self._check_btc_anomaly_filter(symbol):
                logger.info(f"{symbol}: Pending signal blocked by BTC anomaly filter")
                return

            # Check symbol quality filter
            if not self._check_symbol_quality_filter(symbol):
                logger.info(f"{symbol}: Pending signal blocked by symbol quality filter")
                return

            # Check beta quality filter
            if not self._check_beta_quality_filter(symbol):
                logger.info(f"{symbol}: Pending signal blocked by beta quality filter")
                return

            logger.debug(f"{symbol}: All WIN_RATE_MAX market regime filters PASSED")

        # Create pending signal with TTL (max watch window)
        # Must-Fix #1: Use bar time scale, not event.ts or wall clock
        max_wait_ms = cfg.entry_trigger_max_wait_minutes * 60 * 1000
        signal_id = f"{symbol}_{event.event_id}_{event.direction.value}"  # Must-Fix #8: Use event.event_id

        pending = PendingSignal(
            signal_id=signal_id,
            event=event,
            created_ts=bar.ts_minute,  # Must-Fix #1: Bar time scale
            expires_ts=bar.ts_minute + max_wait_ms,  # Must-Fix #1: Bar time scale
            direction=event.direction,
            symbol=symbol,
            signal_z_er=event.metrics.get('z_er', 0),
            signal_z_vol=event.metrics.get('z_vol', 0),
            signal_price=bar.close,
            bars_since_signal=0,  # Track how many bars passed
            peak_since_signal=bar.high if event.direction == Direction.UP else bar.low  # Must-Fix #3
        )

        self.pending_signals[signal_id] = pending

        logger.info(
            f"Pending signal created: {signal_id} | {symbol} {event.direction.value} "
            f"@ {bar.close:.2f} | z_ER: {pending.signal_z_er:.2f} | "
            f"Peak since signal: {pending.peak_since_signal:.2f} | "
            f"Watch window: up to {cfg.entry_trigger_max_wait_minutes}m "
            f"(will enter AS SOON AS triggers met)"
        )

        # Send Telegram notification
        if self.config.alerts.telegram.enabled:
            message = self._format_pending_signal_created(pending, event)
            await self._send_telegram(message)

    async def _check_pending_signals(self, symbol: str, bar: Bar) -> None:
        """
        Check if any pending signals for this symbol can trigger entry.
        Called on EVERY new bar (and features update).

        IMPORTANT: This implements "watch window", not "fixed delay".
        Entry happens on FIRST bar where triggers met, not after fixed time.

        Must-Fix #11: Uses lock to prevent race conditions when called from both
        _handle_bars() and _handle_features() simultaneously.
        """
        cfg = self.config.position_management
        current_ts = bar.ts_minute
        features = self.latest_features.get(symbol)

        # CRITICAL: Acquire lock for this symbol
        async with self._get_pending_lock(symbol):
            # Get pending signals for this symbol
            pending_for_symbol = [
                (signal_id, signal)
                for signal_id, signal in list(self.pending_signals.items())
                if signal.symbol == symbol
            ]

            for signal_id, pending in pending_for_symbol:
                # Must-Fix #12: Check invalidation FIRST (before any other logic)
                if pending.invalidated:
                    removed = self.pending_signals.pop(signal_id, None)  # Must-Fix #13
                    if removed:
                        logger.info(
                            f"Removed invalidated pending: {signal_id} | "
                            f"Reason: {pending.invalidation_reason} | "
                            f"Duration: {pending.bars_since_signal} bars"
                        )
                        # Send Telegram notification
                        if self.config.alerts.telegram.enabled:
                            message = self._format_pending_signal_invalidated(pending)
                            await self._send_telegram(message)
                    continue  # Skip to next pending

                # Must-Fix #10: Only increment bars_since_signal ONCE per bar (idempotent)
                if pending.last_evaluated_bar_ts != bar.ts_minute:
                    # New bar - increment counter and update peak
                    pending.bars_since_signal += 1
                    pending.update_peak(bar)
                    pending.last_evaluated_bar_ts = bar.ts_minute
                    logger.debug(f"{signal_id}: New bar {pending.bars_since_signal}, peak updated")
                else:
                    # Same bar (features arrived after bar) - only re-evaluate triggers
                    logger.debug(f"{signal_id}: Same bar, re-evaluating triggers with fresh features")

                # Must-Fix #1: Check expiry using bar time scale
                if pending.is_expired(current_ts):
                    removed = self.pending_signals.pop(signal_id, None)  # Must-Fix #13
                    if removed:
                        logger.info(
                            f"Pending signal expired: {signal_id} "
                            f"(triggers not met within {cfg.entry_trigger_max_wait_minutes}m)"
                        )
                        # Send Telegram notification
                        if self.config.alerts.telegram.enabled:
                            message = self._format_pending_signal_expired(pending, cfg.entry_trigger_max_wait_minutes)
                            await self._send_telegram(message)
                    continue

                # Check min_wait_bars (optional filter to avoid same-bar entry)
                min_wait = getattr(cfg, 'entry_trigger_min_wait_bars', 0)
                if pending.bars_since_signal < min_wait:
                    logger.debug(f"{signal_id}: Min wait not reached ({pending.bars_since_signal}/{min_wait} bars)")
                    continue

                # Evaluate triggers
                triggers_met = await self._evaluate_pending_triggers(pending, bar, features)

                if triggers_met:
                    # All triggers met - open position IMMEDIATELY (not after fixed delay!)
                    delay_seconds = (current_ts - pending.created_ts) // 1000  # Must-Fix #1: Bar time scale
                    logger.info(
                        f"Pending signal triggered: {signal_id} "
                        f"(after {pending.bars_since_signal} bars / {delay_seconds}s)"
                    )
                    await self._open_position_from_pending(pending, bar)

                    # CRITICAL: Remove from pending queue to ensure one-signal → one-entry
                    removed = self.pending_signals.pop(signal_id, None)  # Must-Fix #13
                    if removed:
                        logger.info(f"Position opened from pending: {signal_id}")

    async def _evaluate_pending_triggers(
        self,
        pending: PendingSignal,
        current_bar: Bar,
        current_features: Optional[Features]
    ) -> bool:
        """
        Evaluate if entry triggers are met for a pending signal.

        Implements all Must-Fix issues:
        - #2: Direction-aware z-cooldown
        - #3: Pullback from peak_since_signal
        - #4: Fail-closed if data missing
        - #5: Features freshness check
        - #7: Stability + dominance

        WIN_RATE_MAX Profile additions:
        - Enhanced invalidation rules (checked FIRST, priority-ordered)
        - Re-expansion check (checked LAST, after all other triggers)

        Updates pending.z_cooldown_met, pending.pullback_met, pending.stability_met.
        Returns True if ALL triggers met.
        """
        cfg = self.config.position_management
        symbol = pending.symbol
        direction = pending.direction

        # If triggers disabled globally, always allow (shouldn't happen in pending queue)
        if not cfg.use_entry_triggers:
            return True

        # =====================================================================
        # WIN_RATE_MAX: Enhanced invalidation rules (Step 4.3 integration)
        # Check FIRST before any other trigger evaluation (highest priority)
        # =====================================================================
        if current_features is not None:
            invalidation_reason = self._check_invalidation_rules(pending, current_bar, current_features)
            if invalidation_reason:
                logger.info(
                    f"{symbol}: Pending signal invalidated - {invalidation_reason} "
                    f"(WIN_RATE_MAX enhanced invalidation)"
                )
                pending.invalidated = True
                pending.invalidation_reason = invalidation_reason
                return False

        # Need features for z-score check
        if current_features is None:
            logger.debug(f"{symbol}: No features available, waiting...")
            return False

        # Must-Fix #5: Check features freshness (avoid stale data)
        if current_features.ts_minute < current_bar.ts_minute:
            logger.debug(
                f"{symbol}: Features stale ({current_features.ts_minute} < {current_bar.ts_minute}), "
                f"waiting for fresh features"
            )
            return False

        # Trigger 1: Z-score cooldown (Must-Fix #2: Direction-aware)
        # Check if signal reversed direction first
        if direction == Direction.UP:
            if current_features.z_er_15m <= 0:
                # Signal reversed to bearish - invalidate
                pending.invalidated = True
                pending.invalidation_reason = f"Direction reversed (z_ER: {current_features.z_er_15m:.2f})"
                logger.info(f"{symbol}: {pending.invalidation_reason}")
                return False  # Signal no longer valid
            current_z_er = current_features.z_er_15m  # No abs() for UP
        else:  # DOWN
            if current_features.z_er_15m >= 0:
                # Signal reversed to bullish - invalidate
                pending.invalidated = True
                pending.invalidation_reason = f"Direction reversed (z_ER: {current_features.z_er_15m:.2f})"
                logger.info(f"{symbol}: {pending.invalidation_reason}")
                return False  # Signal no longer valid
            current_z_er = abs(current_features.z_er_15m)  # Use abs for comparison

        if current_z_er > 3.0:
            # Still at peak
            pending.z_cooldown_met = False
            logger.debug(f"{symbol}: Z-score still hot ({current_z_er:.2f})")
        elif current_z_er < cfg.entry_trigger_z_cooldown:
            # Too weak
            pending.z_cooldown_met = False
            logger.debug(f"{symbol}: Z-score too weak ({current_z_er:.2f})")
        else:
            # In range [2.0, 3.0] with correct direction
            pending.z_cooldown_met = True
            logger.debug(f"{symbol}: Z-score cooled ({current_z_er:.2f}) with correct direction ✓")

        # Trigger 2: Price pullback from peak (Must-Fix #3: Use peak_since_signal)
        peak_price = pending.peak_since_signal

        if peak_price is None:
            # Must-Fix #4: Fail-closed if data missing
            if cfg.entry_trigger_require_data:
                pending.pullback_met = False
                logger.debug(f"{symbol}: No peak data yet, waiting... (require_data=true)")
                return False
            else:
                pending.pullback_met = True  # Lenient mode
        else:
            if direction == Direction.UP:
                pullback_pct = (peak_price - current_bar.close) / peak_price * 100
            else:
                pullback_pct = (current_bar.close - peak_price) / peak_price * 100

            if pullback_pct >= cfg.entry_trigger_pullback_pct:
                pending.pullback_met = True
                logger.debug(f"{symbol}: Pullback sufficient ({pullback_pct:.2f}% from peak {peak_price:.2f}) ✓")
            else:
                pending.pullback_met = False
                logger.debug(f"{symbol}: Pullback insufficient ({pullback_pct:.2f}%)")

        # Trigger 3: Taker flow stability AND dominance (Must-Fix #7)
        taker_stability = self.extended_features.get_taker_flow_stability(symbol, lookback_bars=3)
        current_taker_share = current_bar.taker_buy_share()

        if taker_stability is None or current_taker_share is None:
            # Must-Fix #4: Fail-closed if data missing
            if cfg.entry_trigger_require_data:
                pending.stability_met = False
                logger.debug(f"{symbol}: No taker flow data yet, waiting... (require_data=true)")
                return False
            else:
                pending.stability_met = True  # Lenient mode
        else:
            # Check stability
            if taker_stability > cfg.entry_trigger_taker_stability:
                pending.stability_met = False
                logger.debug(f"{symbol}: Taker flow unstable ({taker_stability:.2f})")
            else:
                # Must-Fix #7: Check dominance (direction must still hold)
                min_dominance = cfg.entry_trigger_min_taker_dominance

                if direction == Direction.UP:
                    if current_taker_share < min_dominance:
                        pending.stability_met = False
                        logger.debug(
                            f"{symbol}: Taker buy dominance lost "
                            f"(buy share: {current_taker_share:.2f} < {min_dominance:.2f})"
                        )
                    else:
                        pending.stability_met = True
                        logger.debug(
                            f"{symbol}: Taker flow stable ({taker_stability:.2f}) "
                            f"AND dominant (buy: {current_taker_share:.2f}) ✓"
                        )
                else:  # DOWN
                    max_dominance = 1.0 - min_dominance
                    if current_taker_share > max_dominance:
                        pending.stability_met = False
                        logger.debug(
                            f"{symbol}: Taker sell dominance lost "
                            f"(sell share: {1-current_taker_share:.2f} < {min_dominance:.2f})"
                        )
                    else:
                        pending.stability_met = True
                        logger.debug(
                            f"{symbol}: Taker flow stable ({taker_stability:.2f}) "
                            f"AND dominant (sell: {1-current_taker_share:.2f}) ✓"
                        )

        # Check if all DEFAULT triggers met
        all_met = pending.all_triggers_met()

        if not all_met:
            # Default triggers not met yet, no need to check re-expansion
            return False

        # =====================================================================
        # WIN_RATE_MAX: Re-expansion check (Step 4.2 integration)
        # Check AFTER all default triggers are met, BEFORE opening position
        # This is the LAST gate before entry - ensures momentum is resuming
        # =====================================================================
        re_expansion_ok = self._check_re_expansion(pending, current_bar, current_features)
        if not re_expansion_ok:
            logger.debug(
                f"{symbol}: Re-expansion not confirmed (skipping entry this bar) - "
                f"DEFAULT triggers passed, waiting for momentum resumption"
            )
            return False  # Wait for re-expansion

        logger.info(
            f"{symbol}: ALL entry triggers met! "
            f"(z_cool: {pending.z_cooldown_met}, pullback: {pending.pullback_met}, "
            f"stability+dominance: {pending.stability_met}, re_expansion: {re_expansion_ok})"
        )

        return True

    async def _open_position_from_pending(
        self,
        pending: PendingSignal,
        bar: Bar
    ) -> None:
        """
        Open position from a pending signal that met all triggers.
        """
        event = pending.event
        symbol = pending.symbol

        # Check if position still allowed (no open position created meanwhile)
        existing_open = [
            p for p in self.open_positions.values()
            if p.symbol == symbol and p.status == PositionStatus.OPEN
        ]

        if existing_open and not self.config.position_management.allow_multiple_positions:
            logger.info(f"{symbol}: Position opened elsewhere, skipping pending signal")
            return

        # Calculate dynamic targets based on ATR
        targets = self.extended_features.calculate_dynamic_targets(
            symbol=symbol,
            entry_price=bar.close,
            direction=event.direction,
            atr_stop_mult=self.config.position_management.atr_stop_multiplier,
            atr_target_mult=self.config.position_management.atr_target_multiplier,
            min_risk_reward=self.config.position_management.min_risk_reward_ratio
        )

        # Create position
        position_id = f"{symbol}_{bar.ts_minute}_{event.direction.value}_triggered"
        metrics = event.metrics.copy()

        # Store event status
        metrics['event_status'] = event.status.value

        # Store signal → entry timing
        metrics['signal_ts'] = pending.created_ts
        metrics['entry_ts'] = bar.ts_minute
        metrics['trigger_delay_bars'] = pending.bars_since_signal  # PRIMARY (exact)
        metrics['trigger_delay_seconds_approx'] = (bar.ts_minute - pending.created_ts) // 1000  # Secondary (approximate)

        # Store dynamic targets
        if targets:
            metrics.update({
                'dynamic_stop_loss': targets['stop_loss_percent'],
                'dynamic_take_profit': targets['take_profit_percent'],
                'risk_reward_ratio': targets['risk_reward_ratio']
            })

        position = Position(
            position_id=position_id,
            event_id=event.event_id,
            symbol=symbol,
            direction=event.direction,
            status=PositionStatus.OPEN,
            open_price=bar.close,
            open_ts=bar.ts_minute,
            entry_z_er=metrics.get('z_er', 0),
            entry_z_vol=metrics.get('z_vol', 0),
            entry_taker_share=metrics.get('taker_share', 0),
            metrics=metrics
        )

        # Save to database and memory
        await self.storage.write_position(position)
        self.open_positions[position_id] = position

        logger.info(
            f"Position opened (from pending): {position_id} | "
            f"{symbol} {event.direction.value} @ {bar.close:.2f} | "
            f"Trigger delay: {metrics['trigger_delay_bars']} bars (~{metrics['trigger_delay_seconds_approx']}s)"
        )

        # Send Telegram notification
        if self.config.alerts.telegram.enabled:
            message = self._format_position_opened_from_pending(position)
            await self._send_telegram(message)

    async def _cleanup_expired_pending_signals(self) -> None:
        """
        Remove expired pending signals. Run every minute.
        Must-Fix #1: Use bar time scale, not wall clock.
        """
        while True:
            await asyncio.sleep(60)  # Every 60 seconds

            # Must-Fix #1: Get max bar timestamp (bar time scale, not wall clock)
            if not self.latest_bars:
                continue  # No bars yet

            max_bar_ts = max(bar.ts_minute for bar in self.latest_bars.values())

            expired = [
                signal_id for signal_id, signal in self.pending_signals.items()
                if signal.is_expired(max_bar_ts)
            ]

            for signal_id in expired:
                pending = self.pending_signals.get(signal_id)
                if pending:
                    removed = self.pending_signals.pop(signal_id, None)  # Must-Fix #13
                    if removed:
                        logger.info(
                            f"Cleaned up expired pending signal: {signal_id} "
                            f"(TTL: {self.config.position_management.entry_trigger_max_wait_minutes}m, "
                            f"bars evaluated: {pending.bars_since_signal})"
                        )
                        # Send Telegram notification
                        if self.config.alerts.telegram.enabled:
                            message = self._format_pending_signal_expired(pending, self.config.position_management.entry_trigger_max_wait_minutes)
                            await self._send_telegram(message)

    async def _open_position(self, event: Event) -> None:
        """
        DEPRECATED: Backward compatibility alias for _create_pending_signal.
        Kept for tests and legacy code. Use _create_pending_signal instead.
        """
        await self._create_pending_signal(event)

    async def _open_position_from_event(self, event: Event) -> None:
        """
        Open position immediately from event (when triggers disabled).
        This is the OLD behavior for backward compatibility.

        Opens for both CONFIRMED and UNCONFIRMED events.
        """
        symbol = event.initiator_symbol

        # Check if we already have an open position for this symbol
        # (allow only one position per symbol at a time)
        existing = [p for p in self.open_positions.values()
                   if p.symbol == symbol and p.status == PositionStatus.OPEN]

        if existing and not self.config.position_management.allow_multiple_positions:
            logger.info(f"Position already open for {symbol}, skipping new position")
            return

        # Get current bar
        bar = self.latest_bars.get(symbol)

        if not bar:
            logger.warning(f"No bar data available for {symbol}, cannot open position")
            return

        # Calculate dynamic targets based on ATR
        targets = self.extended_features.calculate_dynamic_targets(
            symbol=symbol,
            entry_price=bar.close,
            direction=event.direction,
            atr_stop_mult=self.config.position_management.atr_stop_multiplier,
            atr_target_mult=self.config.position_management.atr_target_multiplier,
            min_risk_reward=self.config.position_management.min_risk_reward_ratio
        )

        # Create position
        position_id = f"{symbol}_{event.ts}_{event.direction.value}"
        metrics = event.metrics.copy()

        # Store event status for Telegram notifications
        metrics['event_status'] = event.status.value

        # Store dynamic targets in metrics for later use
        if targets:
            metrics.update({
                'dynamic_stop_loss': targets['stop_loss_percent'],
                'dynamic_take_profit': targets['take_profit_percent'],
                'risk_reward_ratio': targets['risk_reward_ratio']
            })

        position = Position(
            position_id=position_id,
            event_id=event.event_id,
            symbol=symbol,
            direction=event.direction,
            status=PositionStatus.OPEN,
            open_price=bar.close,
            open_ts=event.ts,
            entry_z_er=metrics.get('z_er', 0),
            entry_z_vol=metrics.get('z_vol', 0),
            entry_taker_share=metrics.get('taker_share', 0),
            metrics=metrics
        )

        # Save to database and memory
        await self.storage.write_position(position)
        self.open_positions[position_id] = position

        logger.info(
            f"Position opened: {position_id} | {symbol} {event.direction.value} "
            f"@ {bar.close:.2f} | Status: {event.status.value}"
        )

        # Send Telegram notification
        if self.config.alerts.telegram.enabled:
            message = self._format_position_opened(position)
            await self._send_telegram(message)

    async def _update_excursions(self, bar: Bar) -> None:
        """Update MFE/MAE for open positions in this symbol."""
        for position in list(self.open_positions.values()):
            if position.symbol == bar.symbol and position.status == PositionStatus.OPEN:
                position.update_excursions(bar.close)

                # Periodically save updated excursions to DB
                if bar.ts_minute % (5 * 60 * 1000) == 0:  # Every 5 minutes
                    await self.storage.write_position(position)

    async def _check_exits_for_symbol(self, symbol: str, features: Features) -> None:
        """Check exit conditions for all open positions in this symbol."""
        bar = self.latest_bars.get(symbol)
        if not bar:
            return

        positions_to_check = [
            p for p in self.open_positions.values()
            if p.symbol == symbol and p.status == PositionStatus.OPEN
        ]

        for position in positions_to_check:
            # WIN_RATE_MAX: Execute partial profit if target reached (before exit checks)
            await self._execute_partial_profit(position, bar.close, bar.ts_minute)

            # Update trailing stop
            await self._update_trailing_stop(position, bar.close, features)

            # Then check exit conditions
            exit_reason = await self._check_exit_conditions(position, features, bar)

            if exit_reason:
                await self._close_position(position, bar, features, exit_reason)

    def _calculate_trailing_stop_price(
        self,
        position: Position,
        current_price: float,
        atr: Optional[float]
    ) -> Optional[float]:
        """
        Calculate trailing stop price based on current price and ATR.
        """
        cfg = self.config.position_management

        if not cfg.use_trailing_stop or atr is None:
            return None

        trail_distance = atr * cfg.trailing_stop_distance_atr

        if position.direction == Direction.UP:
            # For longs, stop trails below price
            return current_price - trail_distance
        else:
            # For shorts, stop trails above price
            return current_price + trail_distance

    # =========================================================================
    # WIN_RATE_MAX Profile: Market Regime Filters (Step 4.1)
    # =========================================================================

    def _check_btc_anomaly_filter(self, symbol: str) -> bool:
        """
        Check if BTC anomaly filter allows trading.
        Blocks trades during BTC volatility spikes (market chaos mode).

        BTC anomaly is defined as: abs(z_ER) >= 3.0 AND z_VOL >= 3.0
        (same rules as initiator signal detection).

        This is a NO_SECTOR filter - single-symbol only, no sector dependencies.

        Args:
            symbol: The symbol being evaluated for entry

        Returns:
            True if trading allowed (no BTC anomaly), False if blocked
        """
        # Skip for DEFAULT profile
        if self.config.position_management.profile != "WIN_RATE_MAX":
            return True

        profile = self.config.position_management.win_rate_max_profile

        if not profile.btc_anomaly_filter:
            return True  # Filter disabled

        # Get BTC features
        btc_symbol = self.config.universe.benchmark_symbol  # Usually "BTCUSDT"
        btc_features = self.latest_features.get(btc_symbol)

        if btc_features is None:
            # Fail-closed: no BTC data = block trade
            logger.debug(
                f"{symbol}: BTC anomaly filter BLOCKED - no BTC features available "
                f"(fail-closed safety)"
            )
            return False

        # Check if BTC has anomaly (same rules as initiator detection)
        btc_z_er = abs(btc_features.z_er_15m) if btc_features.z_er_15m else 0
        btc_z_vol = btc_features.z_vol_15m if btc_features.z_vol_15m else 0

        # BTC anomaly = abs(z_ER) >= 3.0 AND z_VOL >= 3.0
        if btc_z_er >= 3.0 and btc_z_vol >= 3.0:
            logger.debug(
                f"{symbol}: BTC anomaly filter BLOCKED - BTC in anomaly mode "
                f"(z_ER: {btc_z_er:.2f}, z_VOL: {btc_z_vol:.2f})"
            )
            return False  # Block: BTC in anomaly

        logger.debug(
            f"{symbol}: BTC anomaly filter PASSED "
            f"(BTC z_ER: {btc_z_er:.2f}, z_VOL: {btc_z_vol:.2f})"
        )
        return True  # Allow trading

    def _check_symbol_quality_filter(self, symbol: str) -> bool:
        """
        Check if symbol passes quality filters (volume, blacklist).
        Blocks trades on low-quality or illiquid symbols.

        Quality checks:
        1. Symbol not in blacklist
        2. Minimum volume (notional) threshold met
        3. Minimum trades per bar threshold met (if available)

        This is a NO_SECTOR filter - single-symbol only, no sector dependencies.

        Args:
            symbol: The symbol being evaluated for entry

        Returns:
            True if symbol is tradeable, False if blocked
        """
        # Skip for DEFAULT profile
        if self.config.position_management.profile != "WIN_RATE_MAX":
            return True

        profile = self.config.position_management.win_rate_max_profile

        if not profile.symbol_quality_filter:
            return True  # Filter disabled

        # Check blacklist first (fast check)
        if symbol in profile.symbol_blacklist:
            logger.debug(
                f"{symbol}: Symbol quality filter BLOCKED - symbol in blacklist"
            )
            return False  # Blocked: symbol in blacklist

        # Get recent bar for volume check
        bars = list(self.extended_features.bars_windows.get(symbol, []))

        if not bars:
            # Fail-closed: no bar data = block trade
            logger.debug(
                f"{symbol}: Symbol quality filter BLOCKED - no bar data available "
                f"(fail-closed safety)"
            )
            return False

        current_bar = bars[-1]

        # Check minimum volume (notional = price * quantity = USD volume)
        if current_bar.notional < profile.min_volume_usd:
            logger.debug(
                f"{symbol}: Symbol quality filter BLOCKED - insufficient volume "
                f"(notional: ${current_bar.notional:,.0f} < ${profile.min_volume_usd:,.0f})"
            )
            return False  # Blocked: insufficient volume

        # Check minimum number of trades (liquidity proxy)
        if current_bar.trades < profile.min_trades_per_bar:
            logger.debug(
                f"{symbol}: Symbol quality filter BLOCKED - insufficient trades "
                f"(trades: {current_bar.trades} < {profile.min_trades_per_bar})"
            )
            return False  # Blocked: insufficient trades

        logger.debug(
            f"{symbol}: Symbol quality filter PASSED "
            f"(notional: ${current_bar.notional:,.0f}, trades: {current_bar.trades})"
        )
        return True  # Allow trading

    def _check_beta_quality_filter(self, symbol: str) -> bool:
        """
        Check if beta quality is sufficient for reliable excess return calculation.
        Blocks trades when beta neutralization is unreliable.

        Quality checks:
        1. Beta is within reasonable range [beta_min_abs, beta_max_abs]
        2. R-squared check (future enhancement when metric available)

        This is a NO_SECTOR filter - single-symbol only, no sector dependencies.

        Args:
            symbol: The symbol being evaluated for entry

        Returns:
            True if beta quality acceptable, False if blocked
        """
        # Skip for DEFAULT profile
        if self.config.position_management.profile != "WIN_RATE_MAX":
            return True

        profile = self.config.position_management.win_rate_max_profile

        if not profile.beta_quality_filter:
            return True  # Filter disabled

        # Get features for beta check
        features = self.latest_features.get(symbol)

        if features is None:
            # Fail-closed: no features = block trade
            logger.debug(
                f"{symbol}: Beta quality filter BLOCKED - no features available "
                f"(fail-closed safety)"
            )
            return False

        # Check if beta is reasonable (not too low, not too high)
        beta = features.beta if features.beta else 0

        if abs(beta) < profile.beta_min_abs:
            logger.debug(
                f"{symbol}: Beta quality filter BLOCKED - beta too low "
                f"(|beta|: {abs(beta):.3f} < {profile.beta_min_abs:.3f})"
            )
            return False  # Blocked: beta too low, poor market correlation

        if abs(beta) > profile.beta_max_abs:
            logger.debug(
                f"{symbol}: Beta quality filter BLOCKED - beta too high "
                f"(|beta|: {abs(beta):.3f} > {profile.beta_max_abs:.3f})"
            )
            return False  # Blocked: beta too high, unreliable

        # TODO: Add R-squared check when metric is calculated
        # For now, we only check beta range
        # if features.r_squared is not None:
        #     if features.r_squared < profile.beta_min_r_squared:
        #         logger.debug(
        #             f"{symbol}: Beta quality filter BLOCKED - R-squared too low "
        #             f"(R2: {features.r_squared:.3f} < {profile.beta_min_r_squared:.3f})"
        #         )
        #         return False

        logger.debug(
            f"{symbol}: Beta quality filter PASSED (beta: {beta:.3f})"
        )
        return True  # Allow trading

    # =========================================================================
    # WIN_RATE_MAX Profile: Entry Validation (Step 4.2)
    # =========================================================================

    def _check_z_cooldown_declining(self, pending: PendingSignal, features: Features) -> bool:
        """
        Check if z-score is NOT declining 2 bars in a row.

        For WIN_RATE_MAX profile only. Prevents entry when momentum is fading.
        This is a "cooled and dying" protection - if z-score declined for 2 consecutive
        bars, momentum may be exhausted rather than consolidating.

        NOTE: This method requires features history (z-score for last 3 bars) which
        is not currently maintained in ExtendedFeatureCalculator. For now, this is
        a pass-through that returns True. When features window is implemented,
        this method will check: z[-2] > z[-1] AND z[-1] > z[0] (declining pattern).

        Args:
            pending: Pending signal being evaluated
            features: Current features for the symbol

        Returns:
            True if z-score not declining (safe to enter), False if declining 2 bars
        """
        # Skip for DEFAULT profile
        if self.config.position_management.profile != "WIN_RATE_MAX":
            return True

        symbol = pending.symbol

        # TODO: Implement when features history is available in ExtendedFeatureCalculator
        # This requires maintaining a rolling window of Features (similar to bars_windows)
        # and tracking z_er_15m values for the last 3 bars.
        #
        # Future implementation:
        # 1. Add features_windows: Dict[str, deque[Features]] to ExtendedFeatureCalculator
        # 2. Update features window on each features update
        # 3. Check: if z[-2] > z[-1] > z[0] (declining 2 bars), return False
        #
        # For now, return True to not block entry due to missing infrastructure.

        logger.debug(
            f"{symbol}: Z-cooldown declining check SKIPPED "
            f"(features history not yet implemented - pass-through)"
        )
        return True  # Don't block entry due to missing infrastructure

    def _check_re_expansion(self, pending: PendingSignal, bar: Bar, features: Features) -> bool:
        """
        Check if re-expansion is confirmed (1 of 3 methods).

        Re-expansion validates that momentum is resuming after pullback.
        This is critical for WIN_RATE_MAX profile to avoid entering during
        dead-cat bounces or continued decline.

        Methods (require 1 of 3):
        1. Price action: close > prev_high (for LONG) / close < prev_low (for SHORT)
        2. Micro impulse: bar return in signal direction (bar_return > 0 for LONG)
        3. Flow acceleration: taker dominance increasing 2 bars in a row

        Args:
            pending: Pending signal being evaluated
            bar: Current 1-minute bar
            features: Current features for the symbol

        Returns:
            True if re-expansion confirmed (at least 1 method passed), False otherwise
        """
        # Skip for DEFAULT profile
        if self.config.position_management.profile != "WIN_RATE_MAX":
            return True

        profile = self.config.position_management.win_rate_max_profile

        # If re-expansion not required, pass through
        if not profile.require_re_expansion:
            return True

        symbol = pending.symbol
        direction = pending.direction

        # Track which methods passed
        methods_passed = []

        # Method 1: Price action (close > prev_high for LONG / close < prev_low for SHORT)
        if profile.re_expansion_price_action:
            bars = list(self.extended_features.bars_windows.get(symbol, []))
            if len(bars) >= 2:
                prev_bar = bars[-2]
                current_bar = bars[-1]

                if direction == Direction.UP:
                    if current_bar.close > prev_bar.high:
                        methods_passed.append("price_action")
                        logger.debug(
                            f"{symbol}: Re-expansion via PRICE_ACTION "
                            f"(close {current_bar.close:.4f} > prev_high {prev_bar.high:.4f})"
                        )
                else:  # Direction.DOWN
                    if current_bar.close < prev_bar.low:
                        methods_passed.append("price_action")
                        logger.debug(
                            f"{symbol}: Re-expansion via PRICE_ACTION "
                            f"(close {current_bar.close:.4f} < prev_low {prev_bar.low:.4f})"
                        )

        # Method 2: Micro impulse (bar return in signal direction)
        if profile.re_expansion_micro_impulse:
            bar_return = self.extended_features.get_bar_return(symbol)
            if bar_return is not None:
                if direction == Direction.UP and bar_return > 0:
                    methods_passed.append("micro_impulse")
                    logger.debug(
                        f"{symbol}: Re-expansion via MICRO_IMPULSE "
                        f"(bar_return {bar_return:.4f} > 0 for LONG)"
                    )
                elif direction == Direction.DOWN and bar_return < 0:
                    methods_passed.append("micro_impulse")
                    logger.debug(
                        f"{symbol}: Re-expansion via MICRO_IMPULSE "
                        f"(bar_return {bar_return:.4f} < 0 for SHORT)"
                    )

        # Method 3: Flow acceleration (taker dominance increasing/decreasing 2 bars)
        if profile.re_expansion_flow_acceleration:
            flow_bars = self.extended_features.get_flow_acceleration_bars(symbol, lookback=2)
            if flow_bars is not None and len(flow_bars) == 3:
                # flow_bars = [oldest, middle, newest] (3 bars for 2-bar comparison)
                # For LONG: want buy dominance increasing (flow[0] < flow[1] < flow[2])
                # For SHORT: want sell dominance increasing (buy share decreasing)

                if direction == Direction.UP:
                    # Buy dominance increasing = bullish flow acceleration
                    if flow_bars[0] < flow_bars[1] < flow_bars[2]:
                        methods_passed.append("flow_acceleration")
                        logger.debug(
                            f"{symbol}: Re-expansion via FLOW_ACCELERATION "
                            f"(buy dominance increasing: {flow_bars[0]:.3f} < {flow_bars[1]:.3f} < {flow_bars[2]:.3f})"
                        )
                else:  # Direction.DOWN
                    # Sell dominance increasing = buy share decreasing
                    if flow_bars[0] > flow_bars[1] > flow_bars[2]:
                        methods_passed.append("flow_acceleration")
                        logger.debug(
                            f"{symbol}: Re-expansion via FLOW_ACCELERATION "
                            f"(sell dominance increasing: buy share {flow_bars[0]:.3f} > {flow_bars[1]:.3f} > {flow_bars[2]:.3f})"
                        )

        # Result: at least 1 method passed?
        re_expansion_confirmed = len(methods_passed) > 0

        if re_expansion_confirmed:
            logger.debug(
                f"{symbol}: Re-expansion CONFIRMED via methods: {methods_passed}"
            )
        else:
            logger.debug(
                f"{symbol}: Re-expansion NOT confirmed (no methods passed) - "
                f"checked: price_action={profile.re_expansion_price_action}, "
                f"micro_impulse={profile.re_expansion_micro_impulse}, "
                f"flow_acceleration={profile.re_expansion_flow_acceleration}"
            )

        return re_expansion_confirmed

    # =========================================================================
    # WIN_RATE_MAX Profile: Enhanced Invalidation (Step 4.3)
    # =========================================================================

    def _check_invalidation_rules(
        self,
        pending: PendingSignal,
        bar: Bar,
        features: Features
    ) -> Optional[str]:
        """
        Check if pending signal should be invalidated (WIN_RATE_MAX profile only).

        Checks invalidation rules in STRICT priority order (first match wins):
        1. Direction flip (highest priority) - z_ER sign changed
        2. Momentum died - abs(z_ER) < invalidate_z_er_min (1.8)
        3. Flow died - dominance < threshold for N consecutive bars
        4. Structure broken - pullback exceeded max (latched)
        5. TTL expiry (lowest priority) - watch window exceeded

        Latching behavior:
        - pullback_exceeded_max: Once set, stays set (latched)
        - flow_death_bar_count: Counter that accumulates, resets on recovery
        - All other checks are live (current bar only)

        Args:
            pending: Pending signal to check
            bar: Current 1-minute bar
            features: Current features for the symbol

        Returns:
            Invalidation reason (str) if should invalidate, None otherwise
        """
        # Skip for DEFAULT profile (use existing basic invalidation only)
        if self.config.position_management.profile != "WIN_RATE_MAX":
            logger.debug(
                f"{pending.symbol}: Enhanced invalidation SKIPPED (profile != WIN_RATE_MAX)"
            )
            return None

        profile = self.config.position_management.win_rate_max_profile
        symbol = pending.symbol
        direction = pending.direction

        # =====================================================================
        # PRIORITY 1: Direction flip (HIGHEST PRIORITY)
        # If z_ER sign changed, signal is invalid - momentum reversed
        # =====================================================================
        z_er = features.z_er_15m if features.z_er_15m is not None else 0.0

        if direction == Direction.UP and z_er < 0:
            logger.debug(
                f"{symbol}: Invalidation TRIGGERED (priority 1) - direction flip "
                f"(was UP, z_ER now {z_er:.2f} < 0)"
            )
            return "direction_flip"
        elif direction == Direction.DOWN and z_er > 0:
            logger.debug(
                f"{symbol}: Invalidation TRIGGERED (priority 1) - direction flip "
                f"(was DOWN, z_ER now {z_er:.2f} > 0)"
            )
            return "direction_flip"

        logger.debug(
            f"{symbol}: Priority 1 (direction flip) PASSED "
            f"(direction={direction.value}, z_ER={z_er:.2f})"
        )

        # =====================================================================
        # PRIORITY 2: Momentum died
        # If abs(z_ER) < invalidate_z_er_min, momentum too weak to continue
        # =====================================================================
        abs_z_er = abs(z_er)

        if abs_z_er < profile.invalidate_z_er_min:
            logger.debug(
                f"{symbol}: Invalidation TRIGGERED (priority 2) - momentum died "
                f"(|z_ER|={abs_z_er:.2f} < {profile.invalidate_z_er_min})"
            )
            return "momentum_died"

        logger.debug(
            f"{symbol}: Priority 2 (momentum died) PASSED "
            f"(|z_ER|={abs_z_er:.2f} >= {profile.invalidate_z_er_min})"
        )

        # =====================================================================
        # PRIORITY 3: Flow died
        # If taker dominance < threshold for N consecutive bars, flow exhausted
        # Uses counter that accumulates but resets on recovery
        # =====================================================================
        taker_share = bar.taker_buy_share()

        if taker_share is not None:
            # Check dominance in signal direction
            if direction == Direction.UP:
                # For LONG: need buy dominance >= threshold
                dominance_ok = taker_share >= profile.invalidate_taker_dominance_min
                current_dominance = taker_share
                dominance_label = "buy"
            else:  # Direction.DOWN
                # For SHORT: need sell dominance >= threshold (1 - buy_share)
                sell_dominance = 1.0 - taker_share
                dominance_ok = sell_dominance >= profile.invalidate_taker_dominance_min
                current_dominance = sell_dominance
                dominance_label = "sell"

            # Update flow death counter
            if not dominance_ok:
                pending.flow_death_bar_count += 1
                logger.debug(
                    f"{symbol}: Low {dominance_label} dominance bar "
                    f"{pending.flow_death_bar_count}/{profile.invalidate_taker_dominance_bars} "
                    f"(current: {current_dominance:.3f} < {profile.invalidate_taker_dominance_min})"
                )
            else:
                # Reset counter if dominance recovers
                if pending.flow_death_bar_count > 0:
                    logger.debug(
                        f"{symbol}: {dominance_label.capitalize()} dominance recovered "
                        f"({current_dominance:.3f} >= {profile.invalidate_taker_dominance_min}), "
                        f"resetting counter from {pending.flow_death_bar_count} to 0"
                    )
                pending.flow_death_bar_count = 0

            # Invalidate if threshold reached
            if pending.flow_death_bar_count >= profile.invalidate_taker_dominance_bars:
                logger.debug(
                    f"{symbol}: Invalidation TRIGGERED (priority 3) - flow died "
                    f"({pending.flow_death_bar_count} consecutive bars with "
                    f"{dominance_label} dominance < {profile.invalidate_taker_dominance_min})"
                )
                return "flow_died"

            logger.debug(
                f"{symbol}: Priority 3 (flow died) PASSED "
                f"(flow_death_bar_count={pending.flow_death_bar_count} "
                f"< {profile.invalidate_taker_dominance_bars})"
            )
        else:
            # No taker data available - log but don't block (fail-open for this check)
            logger.debug(
                f"{symbol}: Priority 3 (flow died) SKIPPED - no taker data available"
            )

        # =====================================================================
        # PRIORITY 4: Structure broken
        # If pullback_exceeded_max was set (LATCHED), structure is broken
        # This flag is set elsewhere when pullback > max_pullback_pct/atr
        # =====================================================================
        if pending.pullback_exceeded_max:
            logger.debug(
                f"{symbol}: Invalidation TRIGGERED (priority 4) - structure broken "
                f"(pullback exceeded max, flag was latched)"
            )
            return "structure_broken"

        logger.debug(
            f"{symbol}: Priority 4 (structure broken) PASSED "
            f"(pullback_exceeded_max={pending.pullback_exceeded_max})"
        )

        # =====================================================================
        # PRIORITY 5: TTL expiry (LOWEST PRIORITY)
        # Already checked in _check_pending_signals, but include for completeness
        # =====================================================================
        if pending.is_expired(bar.ts_minute):
            logger.debug(
                f"{symbol}: Invalidation TRIGGERED (priority 5) - TTL expired "
                f"(current_ts={bar.ts_minute} >= expires_ts={pending.expires_ts})"
            )
            return "ttl_expired"

        logger.debug(
            f"{symbol}: Priority 5 (TTL expiry) PASSED "
            f"(current_ts={bar.ts_minute} < expires_ts={pending.expires_ts})"
        )

        # =====================================================================
        # All checks passed - no invalidation
        # =====================================================================
        logger.debug(
            f"{symbol}: All invalidation rules PASSED (no invalidation)"
        )
        return None

    # =========================================================================
    # WIN_RATE_MAX Profile: Exit Enhancements (Step 4.4)
    # =========================================================================

    def _calculate_pnl_percent(self, position: Position, current_price: float) -> float:
        """
        Calculate position PnL as percentage.

        Args:
            position: Position to calculate PnL for
            current_price: Current market price

        Returns:
            PnL as percentage (positive = profit, negative = loss)
        """
        if position.direction == Direction.UP:
            return ((current_price - position.open_price) / position.open_price) * 100
        else:  # Direction.DOWN
            return ((position.open_price - current_price) / position.open_price) * 100

    async def _execute_partial_profit(
        self,
        position: Position,
        current_price: float,
        bar_ts: int
    ) -> bool:
        """
        Execute partial profit taking if target reached (WIN_RATE_MAX only).

        Closes 50% of position at +1.0xATR, moves stop loss to breakeven.
        Can only execute once per position.

        Args:
            position: Position to check
            current_price: Current market price
            bar_ts: Current bar timestamp (ms)

        Returns:
            True if partial profit executed, False otherwise
        """
        # Skip for DEFAULT profile
        if self.config.position_management.profile != "WIN_RATE_MAX":
            return False

        profile = self.config.position_management.win_rate_max_profile

        # Check if enabled
        if not profile.use_partial_profit:
            logger.debug(f"{position.symbol}: Partial profit SKIPPED (disabled)")
            return False

        # Check if already executed (one-time only)
        if position.partial_profit_executed:
            logger.debug(f"{position.symbol}: Partial profit SKIPPED (already executed)")
            return False

        # Calculate partial profit target (+1.0xATR)
        atr = self.extended_features.get_atr(position.symbol)
        if atr is None:
            logger.debug(f"{position.symbol}: Partial profit SKIPPED (ATR unavailable)")
            return False  # Can't calculate without ATR

        # Calculate target distance as percentage
        target_distance_pct = (atr / position.open_price) * profile.partial_profit_target_atr * 100

        # Check if target reached
        current_pnl_pct = self._calculate_pnl_percent(position, current_price)

        if current_pnl_pct >= target_distance_pct:
            # Execute partial profit
            position.partial_profit_executed = True
            position.partial_profit_price = current_price
            position.partial_profit_pnl_percent = current_pnl_pct
            position.partial_profit_ts = bar_ts

            logger.info(
                f"{position.symbol}: Partial profit EXECUTED "
                f"(50% at +{current_pnl_pct:.2f}%, target was +{target_distance_pct:.2f}%)"
            )

            # Move stop loss to breakeven if configured
            if profile.partial_profit_move_sl_breakeven:
                # Store breakeven stop in position metrics for exit logic to use
                position.metrics['breakeven_stop_active'] = True
                position.metrics['breakeven_stop_price'] = position.open_price
                logger.info(
                    f"{position.symbol}: Stop loss moved to breakeven ({position.open_price:.6f})"
                )

            # Send Telegram notification
            if self.config.alerts.telegram.enabled:
                message = self._format_partial_profit_executed(position, current_price, current_pnl_pct)
                await self._send_telegram(message)

            return True

        logger.debug(
            f"{position.symbol}: Partial profit NOT triggered "
            f"(PnL: {current_pnl_pct:.2f}% < target: {target_distance_pct:.2f}%)"
        )
        return False

    def _check_time_exit(
        self,
        position: Position,
        current_price: float,
        bar_ts: int
    ) -> bool:
        """
        Check if position should exit due to time limit (WIN_RATE_MAX only).

        Exits if position not at minimum profit target after time_exit_minutes.
        Prevents holding stagnant positions that eat into win rate.

        Args:
            position: Position to check
            current_price: Current market price
            bar_ts: Current bar timestamp (ms)

        Returns:
            True if should exit due to time, False otherwise
        """
        # Skip for DEFAULT profile
        if self.config.position_management.profile != "WIN_RATE_MAX":
            return False

        profile = self.config.position_management.win_rate_max_profile

        # Check if enabled
        if not profile.time_exit_enabled:
            logger.debug(f"{position.symbol}: Time exit SKIPPED (disabled)")
            return False

        # Calculate position duration (minutes)
        duration_ms = bar_ts - position.open_ts
        duration_minutes = duration_ms / (1000 * 60)

        if duration_minutes < profile.time_exit_minutes:
            logger.debug(
                f"{position.symbol}: Time exit NOT triggered "
                f"(held {duration_minutes:.1f}m < {profile.time_exit_minutes}m)"
            )
            return False  # Not enough time elapsed

        # Check if position is profitable enough
        atr = self.extended_features.get_atr(position.symbol)
        current_pnl_pct = self._calculate_pnl_percent(position, current_price)

        if atr is None:
            # Can't calculate threshold without ATR, use fallback logic
            # If held > 2x time_exit_minutes and not positive, exit
            if duration_minutes > profile.time_exit_minutes * 2:
                if current_pnl_pct <= 0:
                    logger.info(
                        f"{position.symbol}: Time exit TRIGGERED (fallback) "
                        f"(held {duration_minutes:.1f}m, PnL {current_pnl_pct:.2f}% <= 0, no ATR)"
                    )
                    return True
            logger.debug(
                f"{position.symbol}: Time exit SKIPPED (no ATR, PnL: {current_pnl_pct:.2f}%)"
            )
            return False

        # Calculate minimum profit requirement (+0.5xATR by default)
        min_profit_pct = (atr / position.open_price) * profile.time_exit_min_pnl_atr_mult * 100

        if current_pnl_pct < min_profit_pct:
            logger.info(
                f"{position.symbol}: Time exit TRIGGERED "
                f"(held {duration_minutes:.1f}m, PnL {current_pnl_pct:.2f}% < min {min_profit_pct:.2f}%)"
            )
            return True

        logger.debug(
            f"{position.symbol}: Time exit NOT triggered "
            f"(PnL {current_pnl_pct:.2f}% >= min {min_profit_pct:.2f}%)"
        )
        return False

    async def _update_trailing_stop(
        self,
        position: Position,
        current_price: float,
        features: Features
    ) -> None:
        """
        Update trailing stop if activated and price has moved favorably.
        """
        cfg = self.config.position_management

        if not cfg.use_trailing_stop:
            return

        # Calculate current PnL
        direction_mult = 1 if position.direction == Direction.UP else -1
        pnl_pct = ((current_price - position.open_price) / position.open_price * 100) * direction_mult

        # Get dynamic TP if available, otherwise use configured TP
        dynamic_tp = position.metrics.get('dynamic_take_profit')
        take_profit_pct = dynamic_tp if dynamic_tp else cfg.take_profit_percent

        activation_threshold = take_profit_pct * cfg.trailing_stop_activation

        # Check if trailing stop should activate
        trailing_active = position.metrics.get('trailing_stop_active', False)

        if not trailing_active and pnl_pct >= activation_threshold:
            # Activate trailing stop
            position.metrics['trailing_stop_active'] = True
            logger.info(f"{position.symbol}: Trailing stop activated at PnL {pnl_pct:.2f}%")

        # If active, update trailing stop price
        if position.metrics.get('trailing_stop_active'):
            atr = self.extended_features._calculate_atr(position.symbol)
            new_stop = self._calculate_trailing_stop_price(position, current_price, atr)

            if new_stop is not None:
                current_stop = position.metrics.get('trailing_stop_price')

                # Only update if new stop is more favorable
                if current_stop is None:
                    position.metrics['trailing_stop_price'] = new_stop
                    logger.debug(f"{position.symbol}: Trailing stop set at {new_stop:.2f}")
                else:
                    if position.direction == Direction.UP and new_stop > current_stop:
                        position.metrics['trailing_stop_price'] = new_stop
                        logger.debug(f"{position.symbol}: Trailing stop raised to {new_stop:.2f}")
                    elif position.direction == Direction.DOWN and new_stop < current_stop:
                        position.metrics['trailing_stop_price'] = new_stop
                        logger.debug(f"{position.symbol}: Trailing stop lowered to {new_stop:.2f}")

    async def _check_exit_conditions(
        self,
        position: Position,
        features: Features,
        bar: Bar
    ) -> Optional[ExitReason]:
        """
        Check all exit conditions with new priority order.

        Exit priority (revised):
        1. Trailing Stop (if activated and hit)
        2. Breakeven Stop (WIN_RATE_MAX: after partial profit, SL at entry)
        3. Fixed Stop Loss (protect capital)
        4. Take Profit (lock in gains)
        5. Z-Score Reversal (signal weakened - now more lenient at 0.5)
        6. Order Flow Reversal
        7. WIN_RATE_MAX Time Exit (stricter: 25m, must be +0.5xATR)
        8. Default Time Exit (120 minutes)
        """
        cfg = self.config.position_management
        direction_multiplier = 1 if position.direction == Direction.UP else -1

        current_price = bar.close
        pnl_pct = ((current_price - position.open_price) / position.open_price * 100) * direction_multiplier

        # 1. Trailing Stop check (highest priority if activated)
        if position.metrics.get('trailing_stop_active'):
            trailing_stop_price = position.metrics.get('trailing_stop_price')

            if trailing_stop_price is not None:
                if position.direction == Direction.UP and current_price <= trailing_stop_price:
                    return ExitReason.TRAILING_STOP
                elif position.direction == Direction.DOWN and current_price >= trailing_stop_price:
                    return ExitReason.TRAILING_STOP

        # 2. Breakeven Stop check (WIN_RATE_MAX: after partial profit executed)
        if position.metrics.get('breakeven_stop_active'):
            breakeven_price = position.metrics.get('breakeven_stop_price')
            if breakeven_price is not None:
                if position.direction == Direction.UP and current_price <= breakeven_price:
                    logger.info(
                        f"{position.symbol}: Breakeven stop triggered "
                        f"(price {current_price:.6f} <= breakeven {breakeven_price:.6f})"
                    )
                    return ExitReason.STOP_LOSS  # Exit at breakeven (0% loss)
                elif position.direction == Direction.DOWN and current_price >= breakeven_price:
                    logger.info(
                        f"{position.symbol}: Breakeven stop triggered "
                        f"(price {current_price:.6f} >= breakeven {breakeven_price:.6f})"
                    )
                    return ExitReason.STOP_LOSS  # Exit at breakeven (0% loss)

        # 3. Fixed Stop Loss check
        stop_loss_pct = position.metrics.get('dynamic_stop_loss', cfg.stop_loss_percent)

        if cfg.use_atr_stops and 'dynamic_stop_loss' not in position.metrics:
            atr_stop = self.extended_features.get_atr_multiple(
                position.symbol, position.open_price, cfg.atr_stop_multiplier
            )
            if atr_stop:
                stop_loss_pct = max(stop_loss_pct, atr_stop)

        if pnl_pct <= -stop_loss_pct:
            return ExitReason.STOP_LOSS

        # 4. Take Profit check
        take_profit_pct = position.metrics.get('dynamic_take_profit', cfg.take_profit_percent)

        if pnl_pct >= take_profit_pct:
            return ExitReason.TAKE_PROFIT

        # 5. Z-Score Reversal check (RELAXED from 1.0 to 0.5)
        if abs(features.z_er_15m) < cfg.z_score_exit_threshold:
            return ExitReason.Z_SCORE_REVERSAL

        # 6. Opposite Signal check (strong signal in opposite direction)
        if cfg.exit_on_opposite_signal:
            opposite_direction = Direction.DOWN if position.direction == Direction.UP else Direction.UP
            if features.direction == opposite_direction and abs(features.z_er_15m) >= cfg.opposite_signal_threshold:
                return ExitReason.OPPOSITE_SIGNAL

        # 7. Order Flow Reversal check
        if cfg.exit_on_order_flow_reversal:
            extended = self.extended_features.update(bar)
            taker_delta = extended.get('taker_flow_delta')

            if taker_delta is not None:
                # For long positions, exit if taker flow becomes very negative
                # For short positions, exit if taker flow becomes very positive
                threshold = cfg.order_flow_reversal_threshold
                if position.direction == Direction.UP and taker_delta < -threshold:
                    return ExitReason.ORDER_FLOW_REVERSAL
                elif position.direction == Direction.DOWN and taker_delta > threshold:
                    return ExitReason.ORDER_FLOW_REVERSAL

        # 8. WIN_RATE_MAX Time Exit (stricter: must be profitable after N minutes)
        if self._check_time_exit(position, current_price, bar.ts_minute):
            return ExitReason.TIME_EXIT

        # 9. Default Time Exit check (max hold time)
        if cfg.max_hold_minutes > 0:
            duration_minutes = (bar.ts_minute - position.open_ts) // (60 * 1000)
            if duration_minutes >= cfg.max_hold_minutes:
                return ExitReason.TIME_EXIT

        return None

    async def _close_position(
        self,
        position: Position,
        bar: Bar,
        features: Features,
        exit_reason: ExitReason
    ) -> None:
        """Close position and save to database."""
        position.close_position(
            close_price=bar.close,
            close_ts=bar.ts_minute,
            exit_reason=exit_reason,
            exit_z_er=features.z_er_15m,
            exit_z_vol=features.z_vol_15m
        )

        # Save to database
        await self.storage.write_position(position)

        # Remove from open positions
        if position.position_id in self.open_positions:
            del self.open_positions[position.position_id]

        logger.info(
            f"Position closed: {position.position_id} | {position.symbol} "
            f"@ {bar.close:.2f} | PnL: {position.pnl_percent:+.2f}% | "
            f"Reason: {exit_reason.value} | Duration: {position.duration_minutes}m"
        )

        # Send Telegram notification
        if self.config.alerts.telegram.enabled:
            message = self._format_position_closed(position)
            await self._send_telegram(message)

    async def _send_telegram(self, message: str) -> None:
        """Send Telegram notification."""
        if not self.config.alerts.telegram.enabled or not self.telegram_session:
            return

        bot_token = self.config.alerts.telegram.bot_token
        chat_id = self.config.alerts.telegram.chat_id

        if not bot_token or not chat_id:
            return

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }

        try:
            async with self.telegram_session.post(url, json=payload, timeout=10) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Telegram API error: {resp.status} - {text}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

    def _format_position_opened(self, position: Position) -> str:
        """Format message for position opened (immediate entry, no triggers)."""
        timestamp = datetime.fromtimestamp(position.open_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

        direction_emoji = "🟢" if position.direction == Direction.UP else "🔴"
        status_emoji = "✅" if position.metrics.get('event_status') == 'CONFIRMED' else "⚠️"

        message = f"""📊 <b>POSITION OPENED</b>

{direction_emoji} <b>{position.symbol} {position.direction.value}</b>
{status_emoji} Status: {position.metrics.get('event_status', 'UNCONFIRMED')}

💰 Entry Price: {format_price(position.open_price)}
📈 Entry Z-scores:
   • ER: {position.entry_z_er:.2f}σ
   • VOL: {position.entry_z_vol:.2f}σ

🎯 Taker Buy Share: {position.entry_taker_share:.1%}
📊 Beta: {position.metrics.get('beta', 0):.2f}
💸 Funding: {position.metrics.get('funding', 0):.4%}

🕐 Time: {timestamp}
🆔 ID: {position.position_id}

⚙️ <b>Exit Settings:</b>
   • Stop Loss: {self.config.position_management.stop_loss_percent}%
   • Take Profit: {self.config.position_management.take_profit_percent}%
   • Max Hold: {self.config.position_management.max_hold_minutes}m
"""

        if self.config.position_management.use_atr_stops:
            message += f"   • ATR Stop: {self.config.position_management.atr_stop_multiplier}x ATR\n"

        return message

    def _format_position_opened_from_pending(self, position: Position) -> str:
        """Format message for position opened from pending signal (with triggers)."""
        timestamp = datetime.fromtimestamp(position.open_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

        direction_emoji = "🟢" if position.direction == Direction.UP else "🔴"
        status_emoji = "✅" if position.metrics.get('event_status') == 'CONFIRMED' else "⚠️"

        trigger_delay_bars = position.metrics.get('trigger_delay_bars', 0)
        trigger_delay_seconds = position.metrics.get('trigger_delay_seconds_approx', 0)

        message = f"""📊 <b>POSITION OPENED</b> (from pending signal)

{direction_emoji} <b>{position.symbol} {position.direction.value}</b>
{status_emoji} Status: {position.metrics.get('event_status', 'UNCONFIRMED')}

⏱️ <b>Entry Timing:</b>
   • Signal → Entry: {trigger_delay_bars} bars
   • Approx time: ~{trigger_delay_seconds}s
   • Triggers met: z-cooldown ✓, pullback ✓, dominance ✓

💰 Entry Price: {format_price(position.open_price)}
📈 Entry Z-scores:
   • ER: {position.entry_z_er:.2f}σ
   • VOL: {position.entry_z_vol:.2f}σ

🎯 Taker Buy Share: {position.entry_taker_share:.1%}
📊 Beta: {position.metrics.get('beta', 0):.2f}
💸 Funding: {position.metrics.get('funding', 0):.4%}

🕐 Time: {timestamp}
🆔 ID: {position.position_id}

⚙️ <b>Exit Settings:</b>
   • Stop Loss: {self.config.position_management.stop_loss_percent}%
   • Take Profit: {self.config.position_management.take_profit_percent}%
   • Max Hold: {self.config.position_management.max_hold_minutes}m
"""

        if self.config.position_management.use_atr_stops:
            message += f"   • ATR Stop: {self.config.position_management.atr_stop_multiplier}x ATR\n"

        return message

    def _format_position_closed(self, position: Position) -> str:
        """Format message for position closed."""
        open_time = datetime.fromtimestamp(position.open_ts / 1000).strftime('%H:%M:%S')
        close_time = datetime.fromtimestamp(position.close_ts / 1000).strftime('%H:%M:%S')

        # Determine emoji based on PnL
        if position.pnl_percent > 0:
            pnl_emoji = "✅ WIN"
            color = "🟢"
        else:
            pnl_emoji = "❌ LOSS"
            color = "🔴"

        direction_emoji = "🟢" if position.direction == Direction.UP else "🔴"

        # Exit reason emoji
        exit_emoji_map = {
            ExitReason.TAKE_PROFIT: "🎯",
            ExitReason.STOP_LOSS: "🛑",
            ExitReason.TRAILING_STOP: "📈",
            ExitReason.Z_SCORE_REVERSAL: "📉",
            ExitReason.ORDER_FLOW_REVERSAL: "🔄",
            ExitReason.TIME_EXIT: "⏱️",
            ExitReason.OPPOSITE_SIGNAL: "⚡"
        }
        exit_emoji = exit_emoji_map.get(position.exit_reason, "🚪")

        message = f"""💼 <b>POSITION CLOSED</b> {pnl_emoji}

{direction_emoji} <b>{position.symbol} {position.direction.value}</b>

💰 <b>PnL: {position.pnl_percent:+.2f}%</b>
💵 Price: {format_price(position.open_price)} → {format_price(position.close_price)}

📊 <b>Performance:</b>
   • MFE (Best): {position.max_favorable_excursion:+.2f}%
   • MAE (Worst): {position.max_adverse_excursion:+.2f}%
   • Duration: {position.duration_minutes}m

{exit_emoji} <b>Exit Reason:</b> {position.exit_reason.value if position.exit_reason else 'N/A'}

📈 Exit Z-scores:
   • ER: {position.exit_z_er:.2f}σ
   • VOL: {position.exit_z_vol:.2f}σ

⏰ {open_time} → {close_time}
🆔 {position.position_id}
"""

        return message

    def _format_pending_signal_created(self, pending: PendingSignal, event: Event) -> str:
        """Format message for pending signal created."""
        timestamp = datetime.fromtimestamp(pending.created_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

        direction_emoji = "🟢" if pending.direction == Direction.UP else "🔴"
        status_emoji = "✅" if event.status == EventStatus.CONFIRMED else "⚠️"

        message = f"""⏳ <b>PENDING SIGNAL CREATED</b>

{direction_emoji} <b>{pending.symbol} {pending.direction.value}</b>
{status_emoji} Status: {event.status.value}

📊 <b>Signal Metrics:</b>
   • Z-Score (ER): {pending.signal_z_er:.2f}σ
   • Z-Score (VOL): {pending.signal_z_vol:.2f}σ
   • Price: {format_price(pending.signal_price)}
   • Peak: {format_price(pending.peak_since_signal)}

⏱️ <b>Watch Window:</b>
   • Max wait: {self.config.position_management.entry_trigger_max_wait_minutes}m
   • Will enter AS SOON AS triggers met:
      - Z-score cooldown ✓
      - Price pullback ✓
      - Taker flow stable + dominant ✓

🕐 Time: {timestamp}
🆔 ID: {pending.signal_id}
"""

        return message

    def _format_pending_signal_invalidated(self, pending: PendingSignal) -> str:
        """Format message for pending signal invalidated."""
        timestamp = datetime.fromtimestamp(pending.created_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

        direction_emoji = "🟢" if pending.direction == Direction.UP else "🔴"

        message = f"""❌ <b>PENDING SIGNAL INVALIDATED</b>

{direction_emoji} <b>{pending.symbol} {pending.direction.value}</b>

⚠️ <b>Invalidation Reason:</b> {pending.invalidation_reason}

📊 <b>Signal Metrics (at creation):</b>
   • Z-Score (ER): {pending.signal_z_er:.2f}σ
   • Price: {format_price(pending.signal_price)}
   • Peak: {format_price(pending.peak_since_signal)}

⏱️ <b>Duration:</b>
   • Bars evaluated: {pending.bars_since_signal}
   • Created: {timestamp}

🆔 ID: {pending.signal_id}

💡 <b>Result:</b> No position opened - signal no longer valid
"""

        return message

    def _format_pending_signal_expired(self, pending: PendingSignal, max_wait_minutes: int) -> str:
        """Format message for pending signal expired."""
        created_time = datetime.fromtimestamp(pending.created_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

        direction_emoji = "🟢" if pending.direction == Direction.UP else "🔴"

        message = f"""⏰ <b>PENDING SIGNAL EXPIRED</b>

{direction_emoji} <b>{pending.symbol} {pending.direction.value}</b>

⌛ <b>Watch Window Exceeded:</b>
   • Max wait: {max_wait_minutes}m
   • Bars evaluated: {pending.bars_since_signal}

📊 <b>Signal Metrics (at creation):</b>
   • Z-Score (ER): {pending.signal_z_er:.2f}σ
   • Price: {format_price(pending.signal_price)}
   • Peak: {format_price(pending.peak_since_signal)}

🕐 Created: {created_time}
🆔 ID: {pending.signal_id}

💡 <b>Result:</b> No position opened - triggers never met within watch window
"""

        return message

    def _format_partial_profit_executed(self, position: Position, price: float, pnl_pct: float) -> str:
        """Format message for partial profit execution (WIN_RATE_MAX only)."""
        timestamp = datetime.fromtimestamp(position.partial_profit_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

        direction_emoji = "🟢" if position.direction == Direction.UP else "🔴"

        # Calculate duration
        duration_ms = position.partial_profit_ts - position.open_ts
        duration_minutes = duration_ms / (1000 * 60)

        breakeven_active = position.metrics.get('breakeven_stop_active', False)

        message = f"""💰 <b>PARTIAL PROFIT EXECUTED</b>

{direction_emoji} <b>{position.symbol} {position.direction.value}</b>

📊 <b>Profit Details:</b>
   • Position size: 50% closed ✓
   • Exit price: {format_price(price)}
   • PnL: <b>+{pnl_pct:.2f}%</b>
   • Entry price: {format_price(position.open_price)}

⏱️ <b>Duration:</b> {duration_minutes:.1f}m

🛡️ <b>Risk Management:</b>
   • Remaining: 50% position size
"""

        if breakeven_active:
            message += f"   • Stop loss moved to: {format_price(position.open_price)} (BREAKEVEN)\n"

        message += f"""
🕐 Time: {timestamp}
🆔 ID: {position.position_id}
"""

        return message

    async def close(self) -> None:
        """Cleanup on shutdown."""
        # Save all open positions
        for position in self.open_positions.values():
            await self.storage.write_position(position)

        # Close Telegram session
        if self.telegram_session:
            await self.telegram_session.close()

        logger.info("Position manager closed")
