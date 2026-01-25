"""Virtual position manager - opens and closes positions based on alerts and exit conditions."""

import asyncio
import logging
import aiohttp
from typing import Dict, Optional
from datetime import datetime

from detector.utils import format_price
from detector.models import (
    Event, Features, Bar, Position, PositionStatus, ExitReason, Direction,
    PendingSignal, SignalClass, TradingMode
)
from detector.storage import Storage
from detector.features_extended import ExtendedFeatureCalculator
from detector.config import Config
from detector.trading_improvements import TradingImprovements

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages virtual trading positions.

    Responsibilities:
    - Opens positions on initiator alerts
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

        # Trading improvements module (7 optimizations)
        self.trading_improvements = TradingImprovements(config, self.extended_features)

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
        # HYBRID STRATEGY: Determine signal class first (needed for class-aware filters)
        # =====================================================================
        hs_cfg = self.config.hybrid_strategy

        if hs_cfg.enabled and event.signal_class is not None:
            # Hybrid mode: set up based on signal class
            signal_class = event.signal_class
            trading_mode, trade_direction = self._get_trading_mode_and_direction(
                signal_class, event.direction
            )

            # Mode-specific TTL
            if signal_class == SignalClass.EXTREME_SPIKE:
                max_wait_bars = hs_cfg.mean_reversion.max_bars_before_expiry
            elif signal_class == SignalClass.STRONG_SIGNAL:
                max_wait_bars = hs_cfg.conditional_momentum.max_wait_bars
            else:  # EARLY_SIGNAL
                max_wait_bars = hs_cfg.early_momentum.max_wait_bars

            max_wait_ms = max_wait_bars * 60 * 1000  # 1 bar = 1 minute
        else:
            # Legacy mode
            signal_class = None
            trading_mode = None
            trade_direction = event.direction
            max_wait_ms = cfg.entry_trigger_max_wait_minutes * 60 * 1000

        # =====================================================================
        # FILTERS: Class-aware (hybrid) OR WIN_RATE_MAX (legacy)
        # Run BEFORE creating pending signal to filter out bad market conditions
        # =====================================================================

        # Option 1: Class-aware filters (hybrid strategy)
        if hs_cfg.enabled and hs_cfg.class_aware_filters.enabled and signal_class is not None:
            passed, block_reason = self._apply_class_aware_filters(symbol, signal_class, bar)
            if not passed:
                # Logging already done in _apply_class_aware_filters
                return
            logger.debug(f"{symbol}: Class-aware filters PASSED for {signal_class.value}")

        # Option 2: Legacy WIN_RATE_MAX filters (if class-aware disabled)
        elif self.config.position_management.profile == "WIN_RATE_MAX":
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

        # =====================================================================
        # CRITICAL FIXES V3: Entry Filters (Fix 5)
        # Check R:R ratio and minimum ATR before creating pending signal
        # =====================================================================

        # Fix 5a: Entry R:R Filter
        rr_passed, rr_reason = self.trading_improvements.check_entry_rr_filter(
            symbol=symbol,
            signal_class=signal_class,
            direction=trade_direction,
            entry_price=bar.close
        )
        if not rr_passed:
            logger.info(f"{symbol}: Pending signal blocked by R:R filter: {rr_reason}")
            return

        # Fix 5b: Minimum ATR Filter
        atr_passed, atr_reason = self.trading_improvements.check_min_atr_filter(
            symbol=symbol,
            entry_price=bar.close
        )
        if not atr_passed:
            logger.info(f"{symbol}: Pending signal blocked by min ATR filter: {atr_reason}")
            return

        logger.debug(f"{symbol}: Critical Fixes v3 entry filters PASSED")

        # Create pending signal with TTL (max watch window)
        # Must-Fix #1: Use bar time scale, not event.ts or wall clock
        signal_id = f"{symbol}_{event.event_id}_{event.direction.value}"  # Must-Fix #8: Use event.event_id

        # NOTE: signal_class, trading_mode, trade_direction, max_wait_ms already determined above
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
            peak_since_signal=bar.high if event.direction == Direction.UP else bar.low,  # Must-Fix #3
            # HYBRID STRATEGY fields
            signal_class=signal_class,
            trading_mode=trading_mode,
            trade_direction=trade_direction,
            original_direction=event.direction,
        )

        # Set original_price in event if not set
        if event.original_price is None:
            event.original_price = bar.close

        # Initialize history arrays with initial values (Critical for helper methods)
        pending.z_history.append(pending.signal_z_er)
        pending.vol_history.append(pending.signal_z_vol)
        if event.original_taker_share is not None:
            pending.taker_history.append(event.original_taker_share)
        pending.price_history.append(bar.close)

        self.pending_signals[signal_id] = pending

        # Log message based on mode
        if hs_cfg.enabled and signal_class:
            mode_str = trading_mode.value if trading_mode else "UNKNOWN"
            trade_dir_str = trade_direction.value if trade_direction else event.direction.value
            logger.info(
                f"HYBRID pending signal created: {signal_id} | {symbol} "
                f"class={signal_class.value} mode={mode_str} | "
                f"Original direction: {event.direction.value}, Trade direction: {trade_dir_str} | "
                f"z_ER: {pending.signal_z_er:.2f} @ {bar.close:.2f} | "
                f"Watch window: {max_wait_ms // 60000}m"
            )
        else:
            logger.info(
                f"Pending signal created: {signal_id} | {symbol} {event.direction.value} "
                f"@ {bar.close:.2f} | z_ER: {pending.signal_z_er:.2f} | "
                f"Peak since signal: {pending.peak_since_signal:.2f} | "
                f"Watch window: up to {cfg.entry_trigger_max_wait_minutes}m "
                f"(will enter AS SOON AS triggers met)"
            )

        # Send Telegram notification and save to audit log
        message = self._format_pending_signal_created(pending, event)

        # Build comprehensive metadata for audit analysis
        market_ctx = self._build_market_context(symbol, bar)
        metadata = {
            # Signal data
            'signal_z_er': pending.signal_z_er,
            'signal_z_vol': pending.signal_z_vol,
            'signal_price': pending.signal_price,
            'peak_since_signal': pending.peak_since_signal,
            'event_id': event.event_id,
            # Entry trigger settings
            'max_wait_minutes': cfg.entry_trigger_max_wait_minutes,
            'z_cooldown_threshold': cfg.entry_trigger_z_cooldown,
            'pullback_threshold_pct': cfg.entry_trigger_pullback_pct,
            'taker_stability_threshold': cfg.entry_trigger_taker_stability,
            'min_taker_dominance': cfg.entry_trigger_min_taker_dominance,
            # Event confirmation
            'confirmation_status': event.metrics.get('confirmation_status', 'UNKNOWN'),
            'confirmations': event.metrics.get('confirmations', []),
            # HYBRID STRATEGY metadata
            'hybrid_strategy_enabled': hs_cfg.enabled,
            'signal_class': pending.signal_class.value if pending.signal_class else None,
            'trading_mode': pending.trading_mode.value if pending.trading_mode else None,
            'original_direction': pending.original_direction.value if pending.original_direction else None,
            'trade_direction': pending.trade_direction.value if pending.trade_direction else None,
            # Market context
            **market_ctx
        }

        await self._send_and_save_alert(
            alert_type="PENDING_SIGNAL_CREATED",
            symbol=pending.symbol,
            direction=pending.direction.value,
            message=message,
            alert_id=f"pending_created_{pending.signal_id}",
            ts=pending.created_ts,
            metadata=metadata
        )

        # Update cooldown AFTER signal is accepted (not blocked by filters)
        # This ensures cooldown only applies when user actually sees a signal
        await self.storage.update_cooldown(symbol, event.direction, bar.ts_minute)
        logger.debug(f"{symbol}: Cooldown updated after pending signal created")

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
                        # Send Telegram notification and save to audit log
                        message = self._format_pending_signal_invalidated(pending)

                        # Build comprehensive metadata for audit analysis
                        market_ctx = self._build_market_context(pending.symbol, bar)
                        metadata = {
                            # Invalidation details
                            'invalidation_reason': pending.invalidation_reason,
                            'invalidation_details': pending.invalidation_details,
                            # Signal data at creation
                            'signal_z_er': pending.signal_z_er,
                            'signal_z_vol': pending.signal_z_vol,
                            'signal_price': pending.signal_price,
                            'peak_since_signal': pending.peak_since_signal,
                            # Timing
                            'bars_since_signal': pending.bars_since_signal,
                            'created_ts': pending.created_ts,
                            'duration_seconds': (bar.ts_minute - pending.created_ts) // 1000,
                            # Current state at invalidation
                            'current_price': bar.close,
                            'price_change_pct': ((bar.close - pending.signal_price) / pending.signal_price * 100) if pending.signal_price else None,
                            # HYBRID STRATEGY metadata
                            'signal_class': pending.signal_class.value if pending.signal_class else None,
                            'trading_mode': pending.trading_mode.value if pending.trading_mode else None,
                            'original_direction': pending.original_direction.value if pending.original_direction else None,
                            'trade_direction': pending.trade_direction.value if pending.trade_direction else None,
                            'mode_switched': pending.mode_switched,
                            # Market context
                            **market_ctx
                        }

                        await self._send_and_save_alert(
                            alert_type="PENDING_SIGNAL_INVALIDATED",
                            symbol=pending.symbol,
                            direction=pending.direction.value,
                            message=message,
                            alert_id=f"pending_invalidated_{pending.signal_id}_{bar.ts_minute}",
                            ts=bar.ts_minute,
                            metadata=metadata
                        )

                        # Clear cooldown since no position was opened
                        # This allows new signals to be accepted immediately
                        if await self.storage.clear_cooldown(pending.symbol):
                            logger.info(f"{pending.symbol}: Cooldown cleared after signal invalidation")
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
                        # Send Telegram notification and save to audit log
                        message = self._format_pending_signal_expired(pending, cfg.entry_trigger_max_wait_minutes)

                        # Build comprehensive metadata for audit analysis
                        market_ctx = self._build_market_context(pending.symbol, bar)
                        metadata = {
                            # Expiry details
                            'max_wait_minutes': cfg.entry_trigger_max_wait_minutes,
                            'expiry_reason': 'TTL_EXCEEDED',
                            # Signal data at creation
                            'signal_z_er': pending.signal_z_er,
                            'signal_z_vol': pending.signal_z_vol,
                            'signal_price': pending.signal_price,
                            'peak_since_signal': pending.peak_since_signal,
                            # Timing
                            'bars_since_signal': pending.bars_since_signal,
                            'created_ts': pending.created_ts,
                            'duration_seconds': (bar.ts_minute - pending.created_ts) // 1000,
                            # Current state at expiry
                            'current_price': bar.close,
                            'price_change_pct': ((bar.close - pending.signal_price) / pending.signal_price * 100) if pending.signal_price else None,
                            'pullback_from_peak_pct': self._calculate_pullback_pct(pending, bar.close),
                            # Why triggers not met (for analysis)
                            'last_taker_share': bar.taker_buy_share(),
                            # HYBRID STRATEGY metadata
                            'signal_class': pending.signal_class.value if pending.signal_class else None,
                            'trading_mode': pending.trading_mode.value if pending.trading_mode else None,
                            'original_direction': pending.original_direction.value if pending.original_direction else None,
                            'trade_direction': pending.trade_direction.value if pending.trade_direction else None,
                            'mode_switched': pending.mode_switched,
                            # Market context
                            **market_ctx
                        }

                        await self._send_and_save_alert(
                            alert_type="PENDING_SIGNAL_EXPIRED",
                            symbol=pending.symbol,
                            direction=pending.direction.value,
                            message=message,
                            alert_id=f"pending_expired_{pending.signal_id}_{bar.ts_minute}",
                            ts=bar.ts_minute,
                            metadata=metadata
                        )

                        # Clear cooldown since no position was opened
                        # This allows new signals to be accepted immediately
                        if await self.storage.clear_cooldown(pending.symbol):
                            logger.info(f"{pending.symbol}: Cooldown cleared after signal expiry")
                    continue

                # Check min_wait_bars (optional filter to avoid same-bar entry)
                min_wait = getattr(cfg, 'entry_trigger_min_wait_bars', 0)
                if pending.bars_since_signal < min_wait:
                    logger.debug(f"{signal_id}: Min wait not reached ({pending.bars_since_signal}/{min_wait} bars)")
                    continue

                # Evaluate triggers - route based on hybrid strategy mode
                if self.config.hybrid_strategy.enabled and pending.signal_class is not None:
                    triggers_met = await self._evaluate_hybrid_triggers(pending, bar, features)
                else:
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
                pending.invalidation_reason = "direction_flip"
                pending.invalidation_details = (
                    f"Направление развернулось: сигнал был UP, но z_ER стал {current_features.z_er_15m:.2f}σ (≤ 0). "
                    f"Рынок перешёл в медвежью фазу."
                )
                logger.info(f"{symbol}: {pending.invalidation_reason} - {pending.invalidation_details}")
                return False  # Signal no longer valid
            current_z_er = current_features.z_er_15m  # No abs() for UP
        else:  # DOWN
            if current_features.z_er_15m >= 0:
                # Signal reversed to bullish - invalidate
                pending.invalidated = True
                pending.invalidation_reason = "direction_flip"
                pending.invalidation_details = (
                    f"Направление развернулось: сигнал был DOWN, но z_ER стал +{current_features.z_er_15m:.2f}σ (≥ 0). "
                    f"Рынок перешёл в бычью фазу."
                )
                logger.info(f"{symbol}: {pending.invalidation_reason} - {pending.invalidation_details}")
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

        # =====================================================================
        # HYBRID STRATEGY: Use trade_direction and mode-specific parameters
        # =====================================================================
        hs_cfg = self.config.hybrid_strategy
        cfg = self.config.position_management

        # Determine trade direction (may differ from signal direction for mean-reversion)
        if hs_cfg.enabled and pending.trade_direction is not None:
            trade_direction = pending.trade_direction
            trading_mode = pending.trading_mode

            # Get mode-specific exit parameters
            if trading_mode == TradingMode.MEAN_REVERSION:
                mode_cfg = hs_cfg.mean_reversion
                atr_stop_mult = mode_cfg.atr_stop_multiplier
                atr_target_mult = mode_cfg.atr_target_multiplier
            elif trading_mode == TradingMode.CONDITIONAL_MOMENTUM:
                mode_cfg = hs_cfg.conditional_momentum
                atr_stop_mult = mode_cfg.atr_stop_multiplier
                atr_target_mult = mode_cfg.atr_target_multiplier
            else:  # EARLY_MOMENTUM
                mode_cfg = hs_cfg.early_momentum
                atr_stop_mult = mode_cfg.atr_stop_multiplier
                atr_target_mult = mode_cfg.atr_target_multiplier
        else:
            # Legacy mode
            trade_direction = event.direction
            trading_mode = None
            atr_stop_mult = cfg.atr_stop_multiplier
            atr_target_mult = cfg.atr_target_multiplier

        # Calculate dynamic targets based on ATR
        targets = self.extended_features.calculate_dynamic_targets(
            symbol=symbol,
            entry_price=bar.close,
            direction=trade_direction,
            atr_stop_mult=atr_stop_mult,
            atr_target_mult=atr_target_mult,
            min_risk_reward=cfg.min_risk_reward_ratio
        )

        # =====================================================================
        # TRADING IMPROVEMENTS: Pre-entry checks and calculations
        # =====================================================================
        signal_class = pending.signal_class

        # Improvement 4: Direction Filters (check before opening)
        features = self.latest_features.get(symbol)
        btc_features = self.latest_features.get(self.config.universe.benchmark_symbol)
        btc_z_er = btc_features.z_er_15m if btc_features else None
        z_er = features.z_er_15m if features else 0.0
        vol_z = features.z_vol_15m if features else None

        dir_filter_passed, dir_filter_reason = self.trading_improvements.check_direction_filters(
            symbol=symbol,
            signal_class=signal_class,
            direction=trade_direction,
            z_er=z_er,
            btc_z_er=btc_z_er,
            volume_z=vol_z
        )
        if not dir_filter_passed:
            logger.info(f"{symbol}: Position BLOCKED by direction filter: {dir_filter_reason}")
            return

        # Improvement 7: Min Profit Filter
        profit_filter_passed, profit_filter_reason = self.trading_improvements.check_min_profit_filter(
            symbol=symbol,
            signal_class=signal_class,
            direction=trade_direction,
            entry_price=bar.close
        )
        if not profit_filter_passed:
            logger.info(f"{symbol}: Position BLOCKED by min profit filter: {profit_filter_reason}")
            return

        # Improvement 1: Calculate Adaptive Stop-Loss
        adaptive_stop_price, adaptive_stop_mult = self.trading_improvements.calculate_adaptive_stop_loss(
            symbol=symbol,
            signal_class=signal_class,
            direction=trade_direction,
            entry_price=bar.close
        )

        # Improvement 2: Calculate Tiered Take-Profit levels
        tiered_tp = self.trading_improvements.calculate_tiered_tp_levels(
            symbol=symbol,
            signal_class=signal_class,
            direction=trade_direction,
            entry_price=bar.close
        )

        # Create position with trade direction
        position_id = f"{symbol}_{bar.ts_minute}_{trade_direction.value}_triggered"
        metrics = event.metrics.copy()

        # Store signal → entry timing
        metrics['signal_ts'] = pending.created_ts
        metrics['entry_ts'] = bar.ts_minute
        metrics['trigger_delay_bars'] = pending.bars_since_signal  # PRIMARY (exact)
        metrics['trigger_delay_seconds_approx'] = (bar.ts_minute - pending.created_ts) // 1000  # Secondary (approximate)

        # Store hybrid strategy metadata
        if hs_cfg.enabled and pending.signal_class is not None:
            metrics['signal_class'] = pending.signal_class.value
            metrics['trading_mode'] = trading_mode.value if trading_mode else None
            metrics['original_direction'] = pending.original_direction.value if pending.original_direction else None
            metrics['trade_direction'] = trade_direction.value
            metrics['mode_switched'] = pending.mode_switched

        # Store dynamic targets
        if targets:
            metrics.update({
                'dynamic_stop_loss': targets['stop_loss_percent'],
                'dynamic_take_profit': targets['take_profit_percent'],
                'risk_reward_ratio': targets['risk_reward_ratio']
            })

        # Generate detailed entry explanation
        entry_details = self._generate_entry_reason_details(pending, bar)

        # Prepare signal class string for Position
        signal_class_str = signal_class.value if signal_class else None

        # Get trailing stop params for this class (Improvement 5)
        trail_profit_threshold, trail_distance_atr = self.trading_improvements.get_trailing_params_for_class(
            signal_class
        )

        position = Position(
            position_id=position_id,
            event_id=event.event_id,
            symbol=symbol,
            direction=trade_direction,  # Use trade_direction for hybrid mode
            status=PositionStatus.OPEN,
            open_price=bar.close,
            open_ts=bar.ts_minute,
            entry_z_er=metrics.get('z_er', 0),
            entry_z_vol=metrics.get('z_vol', 0),
            entry_taker_share=metrics.get('taker_share', 0),
            entry_reason_details=entry_details,
            # Trading Improvements: Adaptive Stop-Loss
            adaptive_stop_price=adaptive_stop_price,
            adaptive_stop_multiplier=adaptive_stop_mult,
            # Trading Improvements: Tiered Take-Profit
            tp1_price=tiered_tp.tp1_price if tiered_tp else None,
            tp2_price=tiered_tp.tp2_price if tiered_tp else None,
            tp3_price=tiered_tp.tp3_price if tiered_tp else None,
            # Trading Improvements: Trailing Stop by Class
            trailing_distance_atr=trail_distance_atr,
            # Signal class for exit logic
            signal_class=signal_class_str,
            metrics=metrics
        )

        # Save to database and memory
        await self.storage.write_position(position)
        self.open_positions[position_id] = position

        # Log message based on mode
        if hs_cfg.enabled and trading_mode:
            logger.info(
                f"HYBRID position opened: {position_id} | "
                f"{symbol} {trade_direction.value} ({trading_mode.value}) @ {bar.close:.2f} | "
                f"Signal class: {pending.signal_class.value if pending.signal_class else 'N/A'} | "
                f"Trigger delay: {metrics['trigger_delay_bars']} bars"
            )
        else:
            logger.info(
                f"Position opened (from pending): {position_id} | "
                f"{symbol} {trade_direction.value} @ {bar.close:.2f} | "
                f"Trigger delay: {metrics['trigger_delay_bars']} bars (~{metrics['trigger_delay_seconds_approx']}s)"
            )

        # Send Telegram notification and save to audit log
        message = self._format_position_opened_from_pending(position)

        # Calculate stop/target prices for audit (use mode-specific params if hybrid)
        atr = self.extended_features.get_atr(symbol)
        stop_price = None
        target_price = None

        if cfg.use_atr_stops and atr:
            if trade_direction == Direction.UP:
                stop_price = bar.close - (atr * atr_stop_mult)
                target_price = bar.close + (atr * atr_target_mult)
            else:
                stop_price = bar.close + (atr * atr_stop_mult)
                target_price = bar.close - (atr * atr_target_mult)
        else:
            # Fixed percentage stops
            if trade_direction == Direction.UP:
                stop_price = bar.close * (1 - cfg.stop_loss_percent / 100)
                target_price = bar.close * (1 + cfg.take_profit_percent / 100)
            else:
                stop_price = bar.close * (1 + cfg.stop_loss_percent / 100)
                target_price = bar.close * (1 - cfg.take_profit_percent / 100)

        # Build comprehensive metadata for audit analysis
        market_ctx = self._build_market_context(symbol, bar)
        audit_metadata = {
            # Position identification
            'position_id': position.position_id,
            'event_id': event.event_id,
            'from_pending': True,
            # Entry prices and metrics
            'open_price': position.open_price,
            'entry_z_er': position.entry_z_er,
            'entry_z_vol': position.entry_z_vol,
            'entry_taker_share': position.entry_taker_share,
            # Signal data (from pending)
            'signal_price': pending.signal_price,
            'signal_z_er': pending.signal_z_er,
            'signal_z_vol': pending.signal_z_vol,
            'peak_since_signal': pending.peak_since_signal,
            # Trigger timing
            'trigger_delay_bars': metrics.get('trigger_delay_bars', 0),
            'trigger_delay_seconds_approx': metrics.get('trigger_delay_seconds_approx', 0),
            # Entry quality analysis
            'entry_pullback_pct': self._calculate_pullback_pct(pending, bar.close),
            'entry_reason_details': entry_details,
            # Exit settings (for comparison with actual exit)
            'stop_loss_price': stop_price,
            'take_profit_price': target_price,
            'atr': atr,
            'atr_stop_multiplier': cfg.atr_stop_multiplier if cfg.use_atr_stops else None,
            'atr_target_multiplier': cfg.atr_target_multiplier if cfg.use_atr_stops else None,
            'stop_loss_percent': cfg.stop_loss_percent,
            'take_profit_percent': cfg.take_profit_percent,
            'max_hold_minutes': cfg.max_hold_minutes,
            'use_trailing_stop': cfg.use_trailing_stop,
            # Confirmation
            'confirmation_status': event.metrics.get('confirmation_status', 'UNKNOWN'),
            'confirmations': event.metrics.get('confirmations', []),
            # HYBRID STRATEGY metadata
            'signal_class': pending.signal_class.value if pending.signal_class else None,
            'trading_mode': pending.trading_mode.value if pending.trading_mode else None,
            'original_direction': pending.original_direction.value if pending.original_direction else None,
            'trade_direction': pending.trade_direction.value if pending.trade_direction else None,
            'mode_switched': pending.mode_switched,
            'z_history_length': len(pending.z_history),
            'price_history_length': len(pending.price_history),
            # TRADING IMPROVEMENTS metadata
            'adaptive_stop_price': position.adaptive_stop_price,
            'adaptive_stop_multiplier': position.adaptive_stop_multiplier,
            'tp1_price': position.tp1_price,
            'tp2_price': position.tp2_price,
            'tp3_price': position.tp3_price,
            'trailing_distance_atr': position.trailing_distance_atr,
            # Market context
            **market_ctx
        }

        await self._send_and_save_alert(
            alert_type="POSITION_OPENED",
            symbol=position.symbol,
            direction=position.direction.value,
            message=message,
            alert_id=f"position_opened_{position.position_id}",
            ts=position.open_ts,
            metadata=audit_metadata
        )

    def _generate_entry_reason_details(self, pending: PendingSignal, bar: Bar) -> str:
        """
        Generate human-readable detailed explanation for position entry.

        Mode-specific messages:
        - MEAN_REVERSION: reversal, price confirmation, volume fade
        - CONDITIONAL_MOMENTUM: z-stability, flow confirmation
        - EARLY_MOMENTUM: continuation, z-growth
        - DEFAULT (standard): z-cooldown, pullback, dominance
        """
        cfg = self.config.position_management
        hs_cfg = self.config.hybrid_strategy

        # Determine actual trade direction (may differ for fade strategies)
        trade_direction = pending.trade_direction if pending.trade_direction else pending.direction
        trade_dir_str = "SHORT" if trade_direction == Direction.DOWN else "LONG"
        signal_dir_str = "UP" if pending.direction == Direction.UP else "DOWN"

        # Wait time (common for all modes)
        wait_bars = pending.bars_since_signal
        wait_seconds = (bar.ts_minute - pending.created_ts) // 1000

        details_parts = []

        # 1. Signal detection (common)
        details_parts.append(
            f"Сигнал: z_ER = {pending.signal_z_er:.2f}σ при цене {format_price(pending.signal_price)}"
        )

        # Route based on trading mode
        if pending.trading_mode == TradingMode.MEAN_REVERSION:
            # MEAN_REVERSION (EXTREME_SPIKE fade) - different triggers
            mr_cfg = hs_cfg.mean_reversion

            # Show that this is a fade trade
            details_parts.append(
                f"Fade-сделка: сигнал {signal_dir_str} → позиция {trade_dir_str}"
            )

            # Z-score drop (reversal)
            z_drop = pending.get_z_drop_pct()
            z_drop_str = f"{z_drop:.0%}" if z_drop else "N/A"
            details_parts.append(
                f"Разворот z-score: -{z_drop_str} (требуется: ≥{mr_cfg.reversal_z_drop_pct:.0%}) ✓"
            )

            # Price confirmation
            if mr_cfg.require_price_confirmation:
                price_move = "вниз" if trade_direction == Direction.DOWN else "вверх"
                details_parts.append(f"Цена подтвердила движение {price_move} ✓")

            # Volume fade
            if mr_cfg.require_volume_fade and pending.vol_history:
                current_vol = pending.vol_history[-1] if pending.vol_history else 0
                details_parts.append(
                    f"Объём снизился: {current_vol:.2f}σ < {pending.signal_z_vol:.2f}σ ✓"
                )

        elif pending.trading_mode == TradingMode.CONDITIONAL_MOMENTUM:
            # CONDITIONAL_MOMENTUM (STRONG_SIGNAL) - stability triggers
            cm_cfg = hs_cfg.conditional_momentum

            # Z-score stability
            z_var = pending.get_z_variance(cm_cfg.z_stability_bars)
            z_var_str = f"{z_var:.4f}" if z_var else "N/A"
            details_parts.append(
                f"Z-score стабилен: variance={z_var_str} (max: {cm_cfg.z_stability_variance_max}) ✓"
            )

            # Flow confirmation (use trade_direction for dominance)
            taker_share = bar.taker_buy_share()
            dominance_pct = taker_share * 100 if taker_share else 0
            if trade_direction == Direction.DOWN:
                dominance_pct = 100 - dominance_pct
            flow_type = "Buy" if trade_direction == Direction.UP else "Sell"
            details_parts.append(
                f"{flow_type}-доминирование: {dominance_pct:.1f}% (требуется: ≥{cm_cfg.min_taker_dominance * 100:.0f}%) ✓"
            )

        elif pending.trading_mode == TradingMode.EARLY_MOMENTUM:
            # EARLY_MOMENTUM (EARLY_SIGNAL) - continuation triggers
            em_cfg = hs_cfg.early_momentum

            # Wait completed
            details_parts.append(
                f"Минимум баров: {pending.bars_since_signal} (требуется: ≥{em_cfg.min_wait_bars}) ✓"
            )

            # Z-growth
            z_growth = pending.get_z_growth_pct()
            z_growth_str = f"{z_growth:.0%}" if z_growth else "N/A"
            details_parts.append(
                f"Z-score рост: +{z_growth_str} (требуется: ≥{em_cfg.z_growth_threshold:.0%}) ✓"
            )

            # Flow continuation (use trade_direction)
            taker_share = bar.taker_buy_share()
            dominance_pct = taker_share * 100 if taker_share else 0
            if trade_direction == Direction.DOWN:
                dominance_pct = 100 - dominance_pct
            flow_type = "Buy" if trade_direction == Direction.UP else "Sell"
            details_parts.append(
                f"{flow_type}-доминирование: {dominance_pct:.1f}% ✓"
            )

        else:
            # DEFAULT mode (standard triggers)
            # Calculate pullback from peak
            pullback_pct = 0.0
            if pending.peak_since_signal:
                if pending.direction == Direction.UP:
                    pullback_pct = (pending.peak_since_signal - bar.close) / pending.peak_since_signal * 100
                else:
                    pullback_pct = (bar.close - pending.peak_since_signal) / pending.peak_since_signal * 100

            # Get taker share (use trade_direction for correct dominance)
            taker_share = bar.taker_buy_share()
            dominance_pct = taker_share * 100 if taker_share else 0
            if trade_direction == Direction.DOWN:
                dominance_pct = 100 - dominance_pct

            # Z-score cooldown
            details_parts.append(
                f"Z-score остыл: диапазон [{cfg.entry_trigger_z_cooldown:.1f}, 3.0]σ ✓"
            )

            # Pullback
            details_parts.append(
                f"Откат: {pullback_pct:.2f}% от пика {format_price(pending.peak_since_signal)} "
                f"(требуется: ≥{cfg.entry_trigger_pullback_pct:.1f}%) ✓"
            )

            # Flow dominance (use trade_direction)
            flow_type = "Buy" if trade_direction == Direction.UP else "Sell"
            details_parts.append(
                f"{flow_type}-доминирование: {dominance_pct:.1f}% "
                f"(требуется: ≥{cfg.entry_trigger_min_taker_dominance * 100:.0f}%) ✓"
            )

        # Wait time (common for all modes)
        details_parts.append(
            f"Ожидание: {wait_bars} баров (~{wait_seconds}s)"
        )

        return " | ".join(details_parts)

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
                        # Send Telegram notification and save to audit log
                        message = self._format_pending_signal_expired(pending, self.config.position_management.entry_trigger_max_wait_minutes)

                        # Get bar for this symbol if available
                        bar = self.latest_bars.get(pending.symbol)
                        current_price = bar.close if bar else None

                        # Build comprehensive metadata for audit analysis
                        market_ctx = self._build_market_context(pending.symbol, bar)
                        metadata = {
                            # Expiry details
                            'max_wait_minutes': self.config.position_management.entry_trigger_max_wait_minutes,
                            'expiry_reason': 'CLEANUP_TTL_EXCEEDED',
                            # Signal data at creation
                            'signal_z_er': pending.signal_z_er,
                            'signal_z_vol': pending.signal_z_vol,
                            'signal_price': pending.signal_price,
                            'peak_since_signal': pending.peak_since_signal,
                            # Timing
                            'bars_since_signal': pending.bars_since_signal,
                            'created_ts': pending.created_ts,
                            'duration_seconds': (max_bar_ts - pending.created_ts) // 1000,
                            # Current state at expiry
                            'current_price': current_price,
                            'price_change_pct': ((current_price - pending.signal_price) / pending.signal_price * 100) if (current_price and pending.signal_price) else None,
                            'pullback_from_peak_pct': self._calculate_pullback_pct(pending, current_price) if current_price else None,
                            # Why triggers not met (for analysis)
                            'last_taker_share': bar.taker_buy_share() if bar else None,
                            # Market context
                            **market_ctx
                        }

                        await self._send_and_save_alert(
                            alert_type="PENDING_SIGNAL_EXPIRED",
                            symbol=pending.symbol,
                            direction=pending.direction.value,
                            message=message,
                            alert_id=f"pending_expired_cleanup_{pending.signal_id}_{max_bar_ts}",
                            ts=max_bar_ts,
                            metadata=metadata
                        )

                        # Clear cooldown since no position was opened
                        # This allows new signals to be accepted immediately
                        if await self.storage.clear_cooldown(pending.symbol):
                            logger.info(f"{pending.symbol}: Cooldown cleared after signal expiry (cleanup)")

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
            f"@ {bar.close:.2f}"
        )

        # Send Telegram notification and save to audit log
        message = self._format_position_opened(position)

        # Calculate stop/target prices for audit
        cfg = self.config.position_management
        atr = self.extended_features.get_atr(symbol)
        stop_price = None
        target_price = None

        if cfg.use_atr_stops and atr:
            if event.direction == Direction.UP:
                stop_price = bar.close - (atr * cfg.atr_stop_multiplier)
                target_price = bar.close + (atr * cfg.atr_target_multiplier)
            else:
                stop_price = bar.close + (atr * cfg.atr_stop_multiplier)
                target_price = bar.close - (atr * cfg.atr_target_multiplier)
        else:
            # Fixed percentage stops
            if event.direction == Direction.UP:
                stop_price = bar.close * (1 - cfg.stop_loss_percent / 100)
                target_price = bar.close * (1 + cfg.take_profit_percent / 100)
            else:
                stop_price = bar.close * (1 + cfg.stop_loss_percent / 100)
                target_price = bar.close * (1 - cfg.take_profit_percent / 100)

        # Build comprehensive metadata for audit analysis
        market_ctx = self._build_market_context(symbol, bar)
        audit_metadata = {
            # Position identification
            'position_id': position.position_id,
            'event_id': event.event_id,
            'from_pending': False,
            # Entry prices and metrics
            'open_price': position.open_price,
            'entry_z_er': position.entry_z_er,
            'entry_z_vol': position.entry_z_vol,
            'entry_taker_share': position.entry_taker_share,
            # Exit settings (for comparison with actual exit)
            'stop_loss_price': stop_price,
            'take_profit_price': target_price,
            'atr': atr,
            'atr_stop_multiplier': cfg.atr_stop_multiplier if cfg.use_atr_stops else None,
            'atr_target_multiplier': cfg.atr_target_multiplier if cfg.use_atr_stops else None,
            'stop_loss_percent': cfg.stop_loss_percent,
            'take_profit_percent': cfg.take_profit_percent,
            'max_hold_minutes': cfg.max_hold_minutes,
            'use_trailing_stop': cfg.use_trailing_stop,
            # Confirmation
            'confirmation_status': event.metrics.get('confirmation_status', 'UNKNOWN'),
            'confirmations': event.metrics.get('confirmations', []),
            # Dynamic targets calculated
            'dynamic_stop_loss': targets.get('stop_loss_percent') if targets else None,
            'dynamic_take_profit': targets.get('take_profit_percent') if targets else None,
            'risk_reward_ratio': targets.get('risk_reward_ratio') if targets else None,
            # Market context
            **market_ctx
        }

        await self._send_and_save_alert(
            alert_type="POSITION_OPENED",
            symbol=position.symbol,
            direction=position.direction.value,
            message=message,
            alert_id=f"position_opened_{position.position_id}",
            ts=position.open_ts,
            metadata=audit_metadata
        )

        # Update cooldown AFTER position is actually opened
        # This ensures cooldown only applies when user sees a position
        await self.storage.update_cooldown(symbol, event.direction, bar.ts_minute)
        logger.debug(f"{symbol}: Cooldown updated after position opened directly")

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
            # Update ATR history for volatility percentile calculation
            self.trading_improvements.update_atr_history(symbol)

            # WIN_RATE_MAX: Execute partial profit if target reached (before exit checks)
            await self._execute_partial_profit(position, bar.close, bar.ts_minute)

            # =====================================================================
            # TRADING IMPROVEMENTS: Tiered Take-Profit check (Improvement 2)
            # Check BEFORE regular exit conditions (high priority)
            # =====================================================================
            tiered_tp_result = self.trading_improvements.check_tiered_tp(
                position, bar.close, bar.ts_minute
            )
            if tiered_tp_result:
                tp_exit_reason, close_pct = tiered_tp_result
                # Log tiered TP hit
                logger.info(
                    f"{symbol}: Tiered TP {tp_exit_reason.value} hit | "
                    f"Closing {close_pct}% @ {bar.close:.6f}"
                )
                # For TP1/TP2 we continue (partial close), TP3 closes everything
                if tp_exit_reason == ExitReason.TAKE_PROFIT_TP3:
                    await self._close_position(position, bar, features, tp_exit_reason)
                    continue

            # =====================================================================
            # TRADING IMPROVEMENTS: Trailing Stop by Class (Improvement 5)
            # Check activation and update, then check if hit
            # =====================================================================
            self.trading_improvements.check_trailing_stop_activation(
                position, bar.close, bar.ts_minute
            )
            trailing_exit = self.trading_improvements.update_and_check_trailing_stop(
                position, bar.close
            )
            if trailing_exit:
                await self._close_position(position, bar, features, trailing_exit)
                continue

            # Update legacy trailing stop (for backward compatibility)
            await self._update_trailing_stop(position, bar.close, features)

            # =====================================================================
            # TRADING IMPROVEMENTS: Intelligent Time Exit (Improvement 6)
            # Check BEFORE standard time exit
            # =====================================================================
            time_exit_reason = self.trading_improvements.check_time_exit(
                position, bar.close, bar.ts_minute
            )
            if time_exit_reason:
                await self._close_position(position, bar, features, time_exit_reason)
                continue

            # =====================================================================
            # TRADING IMPROVEMENTS: Delayed Z-Exit (Improvement 3)
            # Replaces standard Z-exit with conditions
            # =====================================================================
            z_exit_result = self.trading_improvements.check_delayed_z_exit(
                position,
                current_z_er=features.z_er_15m if features.z_er_15m else 0.0,
                current_price=bar.close,
                current_ts=bar.ts_minute
            )
            if z_exit_result:
                z_exit_reason, close_pct = z_exit_result
                await self._close_position(position, bar, features, z_exit_reason)
                continue

            # Then check remaining exit conditions (SL, legacy TP, order flow, etc.)
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

        This is a single-symbol filter - no cross-symbol dependencies.

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

        This is a single-symbol filter - no cross-symbol dependencies.

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

        This is a single-symbol filter - no cross-symbol dependencies.

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
    # CLASS-AWARE FILTERING (Hybrid Strategy)
    # Replaces WIN_RATE_MAX filters when hybrid_strategy.class_aware_filters.enabled
    # Different signal classes have different filter strictness
    # =========================================================================

    def _apply_class_aware_filters(
        self,
        symbol: str,
        signal_class: SignalClass,
        bar: Bar
    ) -> tuple[bool, Optional[str]]:
        """
        Apply class-aware filters based on signal strength.

        Different signal classes have different filter strictness:
        - EXTREME_SPIKE: Relaxed filters (signal is reliable itself)
        - STRONG_SIGNAL: Standard filters
        - EARLY_SIGNAL: Strict filters (weak signals need extra validation)

        This replaces WIN_RATE_MAX filters when class_aware_filters.enabled.

        Args:
            symbol: The symbol being evaluated for entry
            signal_class: The signal classification (EXTREME_SPIKE, STRONG_SIGNAL, EARLY_SIGNAL)
            bar: Current bar data for volume/trades check

        Returns:
            (True, None) if filters passed
            (False, reason) if blocked, reason contains the block reason
        """
        caf_cfg = self.config.hybrid_strategy.class_aware_filters

        if not caf_cfg.enabled:
            return (True, None)  # Class-aware filters disabled

        # Get filter config for this signal class
        if signal_class == SignalClass.EXTREME_SPIKE:
            filter_cfg = caf_cfg.extreme_spike
        elif signal_class == SignalClass.STRONG_SIGNAL:
            filter_cfg = caf_cfg.strong_signal
        elif signal_class == SignalClass.EARLY_SIGNAL:
            filter_cfg = caf_cfg.early_signal
        else:
            # Unknown class - use STRONG_SIGNAL defaults
            filter_cfg = caf_cfg.strong_signal

        # 1. Check global blacklist
        if filter_cfg.use_global_blacklist:
            profile = self.config.position_management.win_rate_max_profile
            if symbol in profile.symbol_blacklist:
                logger.info(
                    f"[FILTER] {symbol}: blocked for {signal_class.value}, "
                    f"reason=global_blacklist"
                )
                return (False, "global_blacklist")

        # 2. Check class-specific additional blacklist
        if symbol in filter_cfg.additional_blacklist:
            logger.info(
                f"[FILTER] {symbol}: blocked for {signal_class.value}, "
                f"reason=class_blacklist"
            )
            return (False, "class_blacklist")

        # 3. Check liquidity - volume
        volume_usd = self._get_recent_volume_usd(symbol, bar)
        if volume_usd < filter_cfg.min_volume_usd:
            logger.info(
                f"[FILTER] {symbol}: blocked for {signal_class.value}, "
                f"reason=low_volume, metrics={{volume=${volume_usd:,.0f}, "
                f"required=${filter_cfg.min_volume_usd:,.0f}}}"
            )
            return (False, "low_volume")

        # 4. Check liquidity - trades count
        trades_count = self._get_recent_trades_count(symbol, bar)
        if trades_count < filter_cfg.min_trades_per_bar:
            logger.info(
                f"[FILTER] {symbol}: blocked for {signal_class.value}, "
                f"reason=low_trades, metrics={{trades={trades_count}, "
                f"required={filter_cfg.min_trades_per_bar}}}"
            )
            return (False, "low_trades")

        # 5. Check BTC anomaly filter (if enabled for this class)
        if filter_cfg.apply_btc_anomaly_filter:
            if not self._check_btc_anomaly_filter_internal():
                logger.info(
                    f"[FILTER] {symbol}: blocked for {signal_class.value}, "
                    f"reason=btc_anomaly"
                )
                return (False, "btc_anomaly")

        # 6. Check beta quality filter (if enabled for this class)
        if filter_cfg.apply_beta_quality_filter:
            beta_result = self._check_beta_quality_filter_internal(
                symbol, filter_cfg.beta_min_abs, filter_cfg.beta_max_abs
            )
            if beta_result is not None:
                logger.info(
                    f"[FILTER] {symbol}: blocked for {signal_class.value}, "
                    f"reason={beta_result}"
                )
                return (False, beta_result)

        # 7. Additional check for EARLY_SIGNAL: require volume spike
        if signal_class == SignalClass.EARLY_SIGNAL and filter_cfg.require_recent_volume_spike:
            avg_volume = self._get_average_volume(symbol, lookback=60)
            if avg_volume > 0:
                volume_ratio = volume_usd / avg_volume
                if volume_ratio < filter_cfg.recent_volume_spike_threshold:
                    logger.info(
                        f"[FILTER] {symbol}: blocked for {signal_class.value}, "
                        f"reason=no_volume_spike, metrics={{current_vol=${volume_usd:,.0f}, "
                        f"avg_vol=${avg_volume:,.0f}, ratio={volume_ratio:.2f}x, "
                        f"required={filter_cfg.recent_volume_spike_threshold:.1f}x}}"
                    )
                    return (False, "no_volume_spike")

        # All filters passed
        filters_applied = []
        if filter_cfg.use_global_blacklist:
            filters_applied.append("blacklist")
        filters_applied.append("liquidity")
        if filter_cfg.apply_btc_anomaly_filter:
            filters_applied.append("btc_anomaly")
        if filter_cfg.apply_beta_quality_filter:
            filters_applied.append("beta_quality")
        if signal_class == SignalClass.EARLY_SIGNAL and filter_cfg.require_recent_volume_spike:
            filters_applied.append("volume_spike")

        logger.debug(
            f"[FILTER] {symbol}: passed for {signal_class.value}, "
            f"filters_applied={{{', '.join(filters_applied)}}}"
        )
        return (True, None)

    def _get_recent_volume_usd(self, symbol: str, bar: Bar) -> float:
        """
        Get recent volume in USD for a symbol.

        For USDT pairs, notional volume IS USD volume.

        Args:
            symbol: The symbol
            bar: Current bar data

        Returns:
            Volume in USD (notional value from bar)
        """
        # notional = price * quantity = USD value for USDT pairs
        return bar.notional if bar.notional else 0.0

    def _get_recent_trades_count(self, symbol: str, bar: Bar) -> int:
        """
        Get recent trades count for a symbol.

        Args:
            symbol: The symbol
            bar: Current bar data

        Returns:
            Number of trades in the bar
        """
        return bar.trades if bar.trades else 0

    def _get_average_volume(self, symbol: str, lookback: int = 60) -> float:
        """
        Get average volume over lookback period.

        Args:
            symbol: The symbol
            lookback: Number of bars to average (default 60 = 1 hour)

        Returns:
            Average volume in USD, or 0 if not enough data
        """
        bars = list(self.extended_features.bars_windows.get(symbol, []))

        if not bars or len(bars) < 2:
            return 0.0

        # Use available bars up to lookback limit
        bars_to_use = bars[-min(len(bars), lookback):]

        if not bars_to_use:
            return 0.0

        total_volume = sum(b.notional for b in bars_to_use if b.notional)
        return total_volume / len(bars_to_use)

    def _check_btc_anomaly_filter_internal(self) -> bool:
        """
        Internal BTC anomaly check without profile dependency.

        Returns:
            True if trading allowed (no BTC anomaly), False if blocked
        """
        btc_symbol = self.config.universe.benchmark_symbol
        btc_features = self.latest_features.get(btc_symbol)

        if btc_features is None:
            return False  # Fail-closed: no BTC data = block trade

        btc_z_er = abs(btc_features.z_er_15m) if btc_features.z_er_15m else 0
        btc_z_vol = btc_features.z_vol_15m if btc_features.z_vol_15m else 0

        # BTC anomaly = abs(z_ER) >= 3.0 AND z_VOL >= 3.0
        return not (btc_z_er >= 3.0 and btc_z_vol >= 3.0)

    def _check_beta_quality_filter_internal(
        self,
        symbol: str,
        beta_min_abs: float,
        beta_max_abs: float
    ) -> Optional[str]:
        """
        Internal beta quality check with custom thresholds.

        Args:
            symbol: The symbol
            beta_min_abs: Minimum absolute beta value
            beta_max_abs: Maximum absolute beta value

        Returns:
            None if beta quality OK, reason string if blocked
        """
        features = self.latest_features.get(symbol)

        if features is None:
            return "beta_no_data"

        beta = features.beta if features.beta else 0

        if abs(beta) < beta_min_abs:
            return "beta_too_low"

        if abs(beta) > beta_max_abs:
            return "beta_too_high"

        return None  # Beta quality OK

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
            details = (
                f"Направление развернулось: сигнал был UP, но z_ER стал {z_er:.2f}σ (< 0). "
                f"Рынок перешёл в медвежью фазу."
            )
            logger.debug(
                f"{symbol}: Invalidation TRIGGERED (priority 1) - direction flip "
                f"(was UP, z_ER now {z_er:.2f} < 0)"
            )
            pending.invalidation_details = details
            return "direction_flip"
        elif direction == Direction.DOWN and z_er > 0:
            details = (
                f"Направление развернулось: сигнал был DOWN, но z_ER стал +{z_er:.2f}σ (> 0). "
                f"Рынок перешёл в бычью фазу."
            )
            logger.debug(
                f"{symbol}: Invalidation TRIGGERED (priority 1) - direction flip "
                f"(was DOWN, z_ER now {z_er:.2f} > 0)"
            )
            pending.invalidation_details = details
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
            details = (
                f"Импульс угас: |z_ER| = {abs_z_er:.2f}σ (было {pending.signal_z_er:.2f}σ). "
                f"Минимальный порог: {profile.invalidate_z_er_min}σ. "
                f"Сигнал потерял силу."
            )
            logger.debug(
                f"{symbol}: Invalidation TRIGGERED (priority 2) - momentum died "
                f"(|z_ER|={abs_z_er:.2f} < {profile.invalidate_z_er_min})"
            )
            pending.invalidation_details = details
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
                details = (
                    f"Поток ордеров иссяк: {dominance_label}-доминирование = {current_dominance:.1%} "
                    f"(порог: {profile.invalidate_taker_dominance_min:.0%}). "
                    f"Подряд {pending.flow_death_bar_count} баров без поддержки направления."
                )
                logger.debug(
                    f"{symbol}: Invalidation TRIGGERED (priority 3) - flow died "
                    f"({pending.flow_death_bar_count} consecutive bars with "
                    f"{dominance_label} dominance < {profile.invalidate_taker_dominance_min})"
                )
                pending.invalidation_details = details
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
            pullback_pct = 0.0
            peak_str = format_price(pending.peak_since_signal) if pending.peak_since_signal else "N/A"
            close_str = format_price(bar.close) if bar.close else "N/A"
            if pending.peak_since_signal and pending.signal_price:
                if direction == Direction.UP:
                    pullback_pct = (pending.peak_since_signal - bar.close) / pending.peak_since_signal * 100
                else:
                    pullback_pct = (bar.close - pending.peak_since_signal) / pending.peak_since_signal * 100
            details = (
                f"Структура сломана: откат {pullback_pct:.2f}% превысил максимум. "
                f"Пик: {peak_str}, текущая: {close_str}. "
                f"Слишком глубокая коррекция."
            )
            logger.debug(
                f"{symbol}: Invalidation TRIGGERED (priority 4) - structure broken "
                f"(pullback exceeded max, flag was latched)"
            )
            pending.invalidation_details = details
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
            elapsed_minutes = (bar.ts_minute - pending.created_ts) // (60 * 1000)
            max_wait = self.config.position_management.entry_trigger_max_wait_minutes
            details = (
                f"Время ожидания истекло: прошло {elapsed_minutes}m (лимит: {max_wait}m). "
                f"Триггеры не сработали за {pending.bars_since_signal} баров. "
                f"Сигнал устарел."
            )
            logger.debug(
                f"{symbol}: Invalidation TRIGGERED (priority 5) - TTL expired "
                f"(current_ts={bar.ts_minute} >= expires_ts={pending.expires_ts})"
            )
            pending.invalidation_details = details
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

            # Send Telegram notification and save to audit log
            message = self._format_partial_profit_executed(position, current_price, current_pnl_pct)

            # Get bar for market context
            bar = self.latest_bars.get(position.symbol)

            # Build comprehensive metadata for audit analysis
            market_ctx = self._build_market_context(position.symbol, bar)
            audit_metadata = {
                # Position identification
                'position_id': position.position_id,
                'event_id': position.event_id,
                # Entry data
                'open_price': position.open_price,
                'open_ts': position.open_ts,
                'entry_z_er': position.entry_z_er,
                'entry_z_vol': position.entry_z_vol,
                # Partial profit execution
                'partial_profit_price': current_price,
                'partial_profit_pnl_percent': current_pnl_pct,
                'target_pnl_percent': target_distance_pct,
                # Time to partial profit
                'time_to_partial_profit_ms': bar_ts - position.open_ts,
                'time_to_partial_profit_minutes': (bar_ts - position.open_ts) // 60000,
                # ATR data
                'atr': atr,
                'partial_profit_target_atr': profile.partial_profit_target_atr,
                # Stop loss management
                'breakeven_stop_active': position.metrics.get('breakeven_stop_active', False),
                'breakeven_stop_price': position.metrics.get('breakeven_stop_price'),
                'partial_profit_move_sl_breakeven': profile.partial_profit_move_sl_breakeven,
                # Current position state
                'max_favorable_excursion': position.max_favorable_excursion,
                'max_adverse_excursion': position.max_adverse_excursion,
                # Market context
                **market_ctx
            }

            await self._send_and_save_alert(
                alert_type="PARTIAL_PROFIT_EXECUTED",
                symbol=position.symbol,
                direction=position.direction.value,
                message=message,
                alert_id=f"partial_profit_{position.position_id}_{bar_ts}",
                ts=bar_ts,
                metadata=audit_metadata
            )

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
        Check all exit conditions with CORRECTED priority order.

        CRITICAL FIX v3: Exit priority with MAX_LOSS_CAP as ABSOLUTE HIGHEST:
        0. MAX_LOSS_CAP - ABSOLUTE HIGHEST (hard cap, never exceed -1.2%)
        1. Trailing Stop v3 (if activated and hit) - protect profits
        2. Trailing Stop (legacy) - protect profits
        3. Breakeven Stop - protect capital at entry
        4. Fixed Stop Loss - protect from big losses
        5. TAKE_PROFIT - lock in target gains
        6. Z-Score Reversal (with PnL conditions) - signal ended
        7. Order Flow Reversal
        8. Opposite Signal
        9. Max Hold Time Exit
        10. TIME_EXIT_LOSING - LAST (only if nothing else worked)

        CRITICAL: Grace period check ensures aggressive exits don't fire too early.
        """
        cfg = self.config.position_management
        direction_multiplier = 1 if position.direction == Direction.UP else -1

        current_price = bar.close
        pnl_pct = ((current_price - position.open_price) / position.open_price * 100) * direction_multiplier
        duration_minutes = (bar.ts_minute - position.open_ts) // (60 * 1000)

        # CRITICAL FIX v2: Check grace period
        # During grace period, only SL/TP are allowed (no aggressive time exits)
        in_grace_period = duration_minutes < cfg.grace_period_minutes

        # =====================================================================
        # 0. MAX LOSS CAP - ABSOLUTE HIGHEST (Critical Fix v3 - Fix 6)
        # Hard cap on maximum loss per position, fires before ANY other exit
        # =====================================================================
        max_loss_exit = self.trading_improvements.check_max_loss_cap(position, current_price)
        if max_loss_exit:
            return max_loss_exit

        # =====================================================================
        # 0.5. TRAILING STOP V3 - Update and check (Critical Fix v3 - Fix 3)
        # Earlier and tighter trailing with accelerated mode
        # =====================================================================
        self.trading_improvements.update_trailing_stop_v3(position, current_price)
        trailing_v3_exit = self.trading_improvements.check_trailing_stop_v3_hit(position, current_price)
        if trailing_v3_exit:
            return trailing_v3_exit

        # =====================================================================
        # 1. TRAILING STOP (legacy) - protect unrealized gains
        # =====================================================================
        if position.metrics.get('trailing_stop_active'):
            trailing_stop_price = position.metrics.get('trailing_stop_price')

            if trailing_stop_price is not None:
                if position.direction == Direction.UP and current_price <= trailing_stop_price:
                    return ExitReason.TRAILING_STOP
                elif position.direction == Direction.DOWN and current_price >= trailing_stop_price:
                    return ExitReason.TRAILING_STOP

        # =====================================================================
        # 2. BREAKEVEN STOP - SECOND (WIN_RATE_MAX: after partial profit)
        # =====================================================================
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

        # =====================================================================
        # 3. STOP LOSS - THIRD (protect capital from big losses)
        # =====================================================================
        # Priority: adaptive_stop_price > sl_moved_to_breakeven > dynamic_stop_loss > fixed %
        if position.adaptive_stop_price is not None:
            # Use price-based stop (most accurate)
            if position.direction == Direction.UP and current_price <= position.adaptive_stop_price:
                logger.info(
                    f"{position.symbol}: Adaptive SL triggered "
                    f"(price {current_price:.6f} <= stop {position.adaptive_stop_price:.6f})"
                )
                return ExitReason.STOP_LOSS
            elif position.direction == Direction.DOWN and current_price >= position.adaptive_stop_price:
                logger.info(
                    f"{position.symbol}: Adaptive SL triggered "
                    f"(price {current_price:.6f} >= stop {position.adaptive_stop_price:.6f})"
                )
                return ExitReason.STOP_LOSS
        else:
            # Fallback to percentage-based stop
            stop_loss_pct = position.metrics.get('dynamic_stop_loss', cfg.stop_loss_percent)

            if cfg.use_atr_stops and 'dynamic_stop_loss' not in position.metrics:
                atr_stop = self.extended_features.get_atr_multiple(
                    position.symbol, position.open_price, cfg.atr_stop_multiplier
                )
                if atr_stop:
                    stop_loss_pct = max(stop_loss_pct, atr_stop)

            if pnl_pct <= -stop_loss_pct:
                return ExitReason.STOP_LOSS

        # =====================================================================
        # 4. TAKE PROFIT - FOURTH (lock in target gains)
        # =====================================================================
        take_profit_pct = position.metrics.get('dynamic_take_profit', cfg.take_profit_percent)
        if pnl_pct >= take_profit_pct:
            return ExitReason.TAKE_PROFIT

        # =====================================================================
        # 5. Z-SCORE REVERSAL - FIFTH (signal weakened)
        # =====================================================================
        trading_mode = position.metrics.get('trading_mode')
        original_direction = position.metrics.get('original_direction')

        # =====================================================================
        # CRITICAL FIX V3: Z-Exit with PnL Conditions (Fix 4)
        # Only allow z-exit when position is profitable
        # =====================================================================
        z_exit_result = self.trading_improvements.check_z_exit_with_pnl_conditions(
            position=position,
            current_z_er=features.z_er_15m if features.z_er_15m else 0.0,
            current_price=current_price,
            current_ts=bar.ts_minute
        )
        if z_exit_result:
            z_exit_reason, close_pct = z_exit_result
            return z_exit_reason

        # Fallback to existing z-exit logic for trading modes if v3 didn't trigger
        if trading_mode == 'MEAN_REVERSION':
            # Mean-Reversion: use mode-specific threshold and direction-aware check
            hs_cfg = self.config.hybrid_strategy.mean_reversion
            z_threshold = hs_cfg.z_score_exit_threshold

            if hs_cfg.use_z_exit:
                # For SHORT (fading up-spike): exit when z fell to threshold (z <= threshold)
                # For LONG (fading down-spike): exit when z rose to threshold (z >= -threshold)
                if position.direction == Direction.DOWN:  # Shorting to fade up-spike
                    if features.z_er_15m <= z_threshold:
                        # Check PnL condition before exiting
                        if pnl_pct >= cfg.z_score_exit.min_pnl_for_full_exit if cfg.z_score_exit.enabled else True:
                            return ExitReason.Z_SCORE_REVERSAL
                else:  # Long to fade down-spike
                    if features.z_er_15m >= -z_threshold:
                        if pnl_pct >= cfg.z_score_exit.min_pnl_for_full_exit if cfg.z_score_exit.enabled else True:
                            return ExitReason.Z_SCORE_REVERSAL
        elif trading_mode == 'CONDITIONAL_MOMENTUM':
            hs_cfg = self.config.hybrid_strategy.conditional_momentum
            if hs_cfg.use_z_exit and abs(features.z_er_15m) < hs_cfg.z_score_exit_threshold:
                if pnl_pct >= cfg.z_score_exit.min_pnl_for_full_exit if cfg.z_score_exit.enabled else True:
                    return ExitReason.Z_SCORE_REVERSAL
        elif trading_mode == 'EARLY_MOMENTUM':
            # Early Momentum: use_z_exit = False by default, skip z-score exit
            hs_cfg = self.config.hybrid_strategy.early_momentum
            if hs_cfg.use_z_exit and abs(features.z_er_15m) < cfg.z_score_exit_threshold:
                if pnl_pct >= cfg.z_score_exit.min_pnl_for_full_exit if cfg.z_score_exit.enabled else True:
                    return ExitReason.Z_SCORE_REVERSAL
        # Note: Legacy mode z-exit is handled by check_z_exit_with_pnl_conditions above

        # =====================================================================
        # 6. ORDER FLOW REVERSAL - SIXTH
        # =====================================================================
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

        # =====================================================================
        # 7. OPPOSITE SIGNAL - SEVENTH
        # =====================================================================
        if cfg.exit_on_opposite_signal:
            opposite_direction = Direction.DOWN if position.direction == Direction.UP else Direction.UP
            if features.direction == opposite_direction and abs(features.z_er_15m) >= cfg.opposite_signal_threshold:
                return ExitReason.OPPOSITE_SIGNAL

        # =====================================================================
        # 8. MAX HOLD TIME EXIT - EIGHTH (time limit reached)
        # =====================================================================
        if cfg.max_hold_minutes > 0:
            if duration_minutes >= cfg.max_hold_minutes:
                return ExitReason.TIME_EXIT

        # =====================================================================
        # 9. TIME_EXIT_LOSING - LAST (aggressive exit, only if nothing else worked)
        # CRITICAL FIX v2: Check AFTER all other exits, respects grace period
        # =====================================================================
        if not in_grace_period:
            # WIN_RATE_MAX Time Exit (stricter: must be profitable after N minutes)
            if self._check_time_exit(position, current_price, bar.ts_minute):
                return ExitReason.TIME_EXIT

        return None

    def _generate_exit_reason_details(
        self,
        position: Position,
        bar: Bar,
        features: Features,
        exit_reason: ExitReason
    ) -> str:
        """Generate human-readable detailed explanation for exit reason."""
        cfg = self.config.position_management
        direction_multiplier = 1 if position.direction == Direction.UP else -1
        current_price = bar.close
        pnl_pct = ((current_price - position.open_price) / position.open_price * 100) * direction_multiplier
        duration_minutes = (bar.ts_minute - position.open_ts) // (60 * 1000)

        if exit_reason == ExitReason.TRAILING_STOP:
            # Check both storage locations: position attribute (trading_improvements)
            # and metrics dict (legacy position_manager)
            trailing_price = (
                position.trailing_price
                or position.metrics.get('trailing_stop_price')
                or 0
            )
            return (
                f"Трейлинг-стоп сработал: цена {format_price(current_price)} достигла "
                f"трейлинг-уровня {format_price(trailing_price)}. "
                f"MFE: {position.max_favorable_excursion:+.2f}%, финальный PnL: {pnl_pct:+.2f}%."
            )

        elif exit_reason == ExitReason.STOP_LOSS:
            # Check if it's a breakeven stop
            if position.metrics.get('breakeven_stop_active'):
                breakeven_price = position.metrics.get('breakeven_stop_price', position.open_price)
                return (
                    f"Безубыточный стоп сработал: цена {format_price(current_price)} вернулась к "
                    f"уровню входа {format_price(breakeven_price)}. "
                    f"Прибыль зафиксирована частичным закрытием ранее."
                )
            else:
                stop_pct = position.metrics.get('dynamic_stop_loss', cfg.stop_loss_percent)
                return (
                    f"Стоп-лосс сработал: PnL достиг -{stop_pct:.1f}% "
                    f"(вход: {format_price(position.open_price)}, текущая: {format_price(current_price)}). "
                    f"MAE: {position.max_adverse_excursion:.2f}%."
                )

        elif exit_reason == ExitReason.TAKE_PROFIT:
            tp_pct = position.metrics.get('dynamic_take_profit', cfg.take_profit_percent)
            return (
                f"Тейк-профит достигнут: PnL = +{pnl_pct:.2f}% (цель: +{tp_pct:.1f}%). "
                f"Вход: {format_price(position.open_price)}, выход: {format_price(current_price)}. "
                f"Длительность: {duration_minutes}m."
            )

        elif exit_reason == ExitReason.Z_SCORE_REVERSAL:
            return (
                f"Z-score ослаб: |z_ER| = {abs(features.z_er_15m):.2f}σ "
                f"(порог выхода: {cfg.z_score_exit_threshold}σ). "
                f"Был: {position.entry_z_er:.2f}σ. Импульс исчерпан."
            )

        elif exit_reason == ExitReason.ORDER_FLOW_REVERSAL:
            taker_share = bar.taker_buy_share()
            flow_direction = "продажи" if position.direction == Direction.UP else "покупки"
            return (
                f"Поток ордеров развернулся: доминируют {flow_direction}. "
                f"Taker buy: {taker_share:.1%} (порог: {cfg.order_flow_reversal_threshold:.0%}). "
                f"Противоположная сторона захватила контроль."
            )

        elif exit_reason == ExitReason.TIME_EXIT:
            max_minutes = cfg.max_hold_minutes
            return (
                f"Выход по времени: позиция держалась {duration_minutes}m (лимит: {max_minutes}m). "
                f"PnL: {pnl_pct:+.2f}%, MFE: {position.max_favorable_excursion:+.2f}%. "
                f"Время истекло без достижения цели."
            )

        elif exit_reason == ExitReason.OPPOSITE_SIGNAL:
            opposite = "DOWN" if position.direction == Direction.UP else "UP"
            return (
                f"Противоположный сигнал: z_ER = {features.z_er_15m:+.2f}σ указывает на {opposite}. "
                f"Рынок развернулся против позиции. "
                f"Порог противосигнала: {cfg.opposite_signal_threshold}σ."
            )

        elif exit_reason == ExitReason.MAX_LOSS_CAP:
            max_loss = cfg.risk_management.max_loss_per_position.max_loss_pct
            return (
                f"КРИТИЧЕСКИЙ СТОП: достигнут лимит убытка -{max_loss:.1f}%. "
                f"PnL: {pnl_pct:+.2f}% (вход: {format_price(position.open_price)}, "
                f"выход: {format_price(current_price)}). "
                f"Защита от катастрофических потерь активирована."
            )

        elif exit_reason == ExitReason.TRAILING_STOP_V3:
            trailing_price = position.metrics.get('trailing_stop_v3_price', 0)
            return (
                f"Улучшенный трейлинг-стоп V3 сработал: цена {format_price(current_price)} "
                f"достигла уровня {format_price(trailing_price)}. "
                f"MFE: {position.max_favorable_excursion:+.2f}%, финальный PnL: {pnl_pct:+.2f}%."
            )

        else:
            return f"Выход по причине: {exit_reason.value}"

    async def _close_position(
        self,
        position: Position,
        bar: Bar,
        features: Features,
        exit_reason: ExitReason
    ) -> None:
        """Close position and save to database."""
        # Generate detailed exit reason explanation
        exit_details = self._generate_exit_reason_details(
            position, bar, features, exit_reason
        )

        position.close_position(
            close_price=bar.close,
            close_ts=bar.ts_minute,
            exit_reason=exit_reason,
            exit_z_er=features.z_er_15m,
            exit_z_vol=features.z_vol_15m
        )
        position.exit_reason_details = exit_details

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

        # Send Telegram notification and save to audit log
        message = self._format_position_closed(position)

        # Get ATR at exit for comparison with entry
        atr_at_exit = self.extended_features.get_atr(position.symbol)

        # Calculate BTC performance during position hold time for comparison
        btc_pnl_percent = None
        btc_bar = self.latest_bars.get('BTCUSDT')
        # We'd need to store BTC price at entry to calculate this properly,
        # but for now we can use features if available

        # Build comprehensive metadata for audit analysis
        market_ctx = self._build_market_context(position.symbol, bar)
        audit_metadata = {
            # Position identification
            'position_id': position.position_id,
            'event_id': position.event_id,
            # Entry data (from position)
            'open_price': position.open_price,
            'open_ts': position.open_ts,
            'entry_z_er': position.entry_z_er,
            'entry_z_vol': position.entry_z_vol,
            'entry_taker_share': position.entry_taker_share,
            # Exit data
            'close_price': position.close_price,
            'close_ts': position.close_ts,
            'exit_z_er': position.exit_z_er,
            'exit_z_vol': position.exit_z_vol,
            'exit_taker_share': bar.taker_buy_share(),
            # Performance metrics
            'pnl_percent': position.pnl_percent,
            'max_favorable_excursion': position.max_favorable_excursion,
            'max_adverse_excursion': position.max_adverse_excursion,
            'mfe_to_pnl_ratio': (position.max_favorable_excursion / position.pnl_percent) if position.pnl_percent and position.pnl_percent != 0 else None,
            # Timing
            'duration_minutes': position.duration_minutes,
            'bars_held': position.duration_minutes,  # Approximate
            # Exit analysis
            'exit_reason': exit_reason.value,
            'exit_reason_details': exit_details,
            # ATR comparison
            'atr_at_exit': atr_at_exit,
            'atr_at_entry': position.metrics.get('atr'),
            # Trailing stop data (if used)
            'trailing_stop_active': position.metrics.get('trailing_stop_active', False),
            'trailing_stop_price': position.metrics.get('trailing_stop_price'),
            'partial_profit_taken': position.metrics.get('partial_profit_taken', False),
            'breakeven_stop_active': position.metrics.get('breakeven_stop_active', False),
            # Original entry settings (for comparison)
            'original_stop_loss_price': position.metrics.get('stop_loss_price'),
            'original_take_profit_price': position.metrics.get('take_profit_price'),
            # HYBRID STRATEGY metadata
            'signal_class': position.metrics.get('signal_class'),
            'trading_mode': position.metrics.get('trading_mode'),
            'original_direction': position.metrics.get('original_direction'),
            'mode_switched': position.metrics.get('mode_switched', False),
            # TRADING IMPROVEMENTS metadata
            'adaptive_stop_price': position.adaptive_stop_price,
            'adaptive_stop_multiplier': position.adaptive_stop_multiplier,
            'tp1_price': position.tp1_price,
            'tp2_price': position.tp2_price,
            'tp3_price': position.tp3_price,
            'tp1_hit': position.tp1_hit,
            'tp2_hit': position.tp2_hit,
            'tp3_hit': position.tp3_hit,
            'remaining_quantity_pct': position.remaining_quantity_pct,
            'sl_moved_to_breakeven': position.sl_moved_to_breakeven,
            'trailing_active': position.trailing_active,
            'trailing_price': position.trailing_price,
            'trailing_distance_atr': position.trailing_distance_atr,
            # Market context at exit
            **market_ctx
        }

        await self._send_and_save_alert(
            alert_type="POSITION_CLOSED",
            symbol=position.symbol,
            direction=position.direction.value,
            message=message,
            alert_id=f"position_closed_{position.position_id}",
            ts=position.close_ts,
            metadata=audit_metadata
        )

        # Clear cooldown after position closes
        # This allows new signals to be accepted immediately for this symbol
        if await self.storage.clear_cooldown(position.symbol):
            logger.info(f"{position.symbol}: Cooldown cleared after position closed")

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

    async def _send_and_save_alert(
        self,
        alert_type: str,
        symbol: str,
        direction: str,
        message: str,
        alert_id: str,
        ts: int,
        metadata: dict = None
    ) -> None:
        """
        Send alert to Telegram and save to database for audit.

        Args:
            alert_type: Type of alert (PENDING_SIGNAL_CREATED, POSITION_OPENED, etc.)
            symbol: Trading symbol
            direction: UP or DOWN
            message: Formatted alert message
            alert_id: Unique identifier for the alert
            ts: Timestamp in milliseconds
            metadata: Additional data to store
        """
        # Send to Telegram if enabled
        if self.config.alerts.telegram.enabled:
            await self._send_telegram(message)

        # Always save to database for audit
        await self.storage.write_alert(
            alert_id=alert_id,
            ts=ts,
            alert_type=alert_type,
            symbol=symbol,
            direction=direction,
            message_text=message,
            metadata=metadata
        )

    def _build_market_context(self, symbol: str, bar: Optional[Bar] = None) -> dict:
        """
        Build market context dict for audit logging.

        Collects BTC data, ATR, profile and other contextual information
        needed for comprehensive analysis of trading decisions.

        Args:
            symbol: Trading symbol
            bar: Current bar (optional, will use latest_bars if not provided)

        Returns:
            Dict with market context data
        """
        context = {
            'profile': self.config.position_management.profile,
        }

        # Get bar if not provided
        if bar is None:
            bar = self.latest_bars.get(symbol)

        # Symbol data
        if bar:
            context['price'] = bar.close
            context['volume'] = bar.volume
            context['notional'] = bar.notional
            context['trades'] = bar.trades
            context['taker_buy_share'] = bar.taker_buy_share()
            context['funding'] = bar.funding
            context['oi'] = bar.oi

        # Symbol features
        features = self.latest_features.get(symbol)
        if features:
            context['z_er'] = features.z_er_15m
            context['z_vol'] = features.z_vol_15m
            context['beta'] = features.beta
            context['er_15m'] = features.er_15m

        # ATR
        atr = self.extended_features.get_atr(symbol)
        if atr:
            context['atr'] = atr
            if bar and bar.close > 0:
                context['atr_percent'] = (atr / bar.close) * 100

        # BTC context (benchmark)
        btc_bar = self.latest_bars.get('BTCUSDT')
        btc_features = self.latest_features.get('BTCUSDT')

        if btc_bar:
            context['btc_price'] = btc_bar.close
            context['btc_taker_buy_share'] = btc_bar.taker_buy_share()

        if btc_features:
            context['btc_z_er'] = btc_features.z_er_15m
            context['btc_z_vol'] = btc_features.z_vol_15m
            # Check if BTC is in anomaly state
            btc_in_anomaly = (
                abs(btc_features.z_er_15m or 0) >= 3.0 and
                (btc_features.z_vol_15m or 0) >= 3.0
            )
            context['btc_in_anomaly'] = btc_in_anomaly

        # BTC ATR
        btc_atr = self.extended_features.get_atr('BTCUSDT')
        if btc_atr:
            context['btc_atr'] = btc_atr

        return context

    def _calculate_pullback_pct(self, pending: PendingSignal, current_price: float) -> Optional[float]:
        """
        Calculate pullback percentage from peak since signal.

        For LONG: pullback = (peak - current) / peak * 100
        For SHORT: pullback = (current - peak) / peak * 100

        Args:
            pending: PendingSignal with peak_since_signal
            current_price: Current price

        Returns:
            Pullback percentage or None if peak not available
        """
        if not pending.peak_since_signal or pending.peak_since_signal == 0:
            return None

        if pending.direction == Direction.UP:
            return (pending.peak_since_signal - current_price) / pending.peak_since_signal * 100
        else:
            return (current_price - pending.peak_since_signal) / pending.peak_since_signal * 100

    def _format_position_opened(self, position: Position) -> str:
        """Format message for position opened (immediate entry, no triggers)."""
        timestamp = datetime.fromtimestamp(position.open_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

        direction_emoji = "🟢" if position.direction == Direction.UP else "🔴"

        message = f"""📊 <b>POSITION OPENED</b>

{direction_emoji} <b>{position.symbol} {position.direction.value}</b>

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

        trigger_delay_bars = position.metrics.get('trigger_delay_bars', 0)
        trigger_delay_seconds = position.metrics.get('trigger_delay_seconds_approx', 0)

        # =====================================================================
        # HYBRID STRATEGY: Show signal class and trading mode
        # =====================================================================
        hs_cfg = self.config.hybrid_strategy
        hybrid_section = ""

        signal_class_val = position.metrics.get('signal_class')
        trading_mode_val = position.metrics.get('trading_mode')
        original_direction_val = position.metrics.get('original_direction')

        if hs_cfg.enabled and signal_class_val:
            # Signal class descriptions
            signal_class_names = {
                "EXTREME_SPIKE": "🔥 EXTREME SPIKE",
                "STRONG_SIGNAL": "💪 STRONG SIGNAL",
                "EARLY_SIGNAL": "🌱 EARLY SIGNAL"
            }
            signal_class_display = signal_class_names.get(signal_class_val, signal_class_val)

            # Trading mode descriptions
            trading_mode_names = {
                "MEAN_REVERSION": "📉 MEAN-REVERSION",
                "CONDITIONAL_MOMENTUM": "📈 CONDITIONAL MOMENTUM",
                "EARLY_MOMENTUM": "🚀 EARLY MOMENTUM"
            }
            trading_mode_display = trading_mode_names.get(trading_mode_val, trading_mode_val or "N/A")

            # Show direction change for mean-reversion
            direction_note = ""
            if original_direction_val and original_direction_val != position.direction.value:
                orig_emoji = "🟢" if original_direction_val == "UP" else "🔴"
                direction_note = f"\n   • ⚠️ Signal was {orig_emoji} {original_direction_val}, trading opposite (fade)"

            hybrid_section = f"""
🧬 <b>HYBRID STRATEGY:</b>
   • Класс: {signal_class_display}
   • Режим: {trading_mode_display}{direction_note}
"""

        message = f"""📊 <b>POSITION OPENED</b> (from pending signal)

{direction_emoji} <b>{position.symbol} {position.direction.value}</b>
{hybrid_section}
⏱️ <b>Entry Timing:</b>
   • Signal → Entry: {trigger_delay_bars} bars (~{trigger_delay_seconds}s)

📝 <b>Почему открыта:</b>
{position.entry_reason_details if position.entry_reason_details else 'Все триггеры выполнены ✓'}

💰 Entry Price: {format_price(position.open_price)}
📈 Entry Z-scores:
   • ER: {position.entry_z_er:.2f}σ
   • VOL: {position.entry_z_vol:.2f}σ

🎯 Taker Buy Share: {position.entry_taker_share:.1%}
📊 Beta: {position.metrics.get('beta', 0):.2f}
💸 Funding: {position.metrics.get('funding', 0):.4%}

🕐 Time: {timestamp}
🆔 ID: {position.position_id}

⚙️ <b>Exit Settings (Trading Improvements):</b>
"""

        # Add Adaptive Stop-Loss info
        if position.adaptive_stop_price:
            message += f"   • 🎯 Adaptive SL: {format_price(position.adaptive_stop_price)}"
            if position.adaptive_stop_multiplier:
                message += f" ({position.adaptive_stop_multiplier:.2f}x ATR)\n"
            else:
                message += "\n"
        else:
            message += f"   • Stop Loss: {self.config.position_management.stop_loss_percent}%\n"

        # Add Tiered Take-Profit levels
        if position.tp1_price:
            message += f"   • 📊 TP1: {format_price(position.tp1_price)} (30%)\n"
        if position.tp2_price:
            message += f"   • 📊 TP2: {format_price(position.tp2_price)} (30%)\n"
        if position.tp3_price:
            message += f"   • 📊 TP3: {format_price(position.tp3_price)} (40%)\n"

        if not position.tp1_price:
            message += f"   • Take Profit: {self.config.position_management.take_profit_percent}%\n"

        message += f"   • Max Hold: {self.config.position_management.max_hold_minutes}m\n"

        if self.config.position_management.use_atr_stops:
            message += f"   • ATR Stop: {self.config.position_management.atr_stop_multiplier}x ATR\n"

        # Add trailing stop class info
        if position.trailing_distance_atr:
            message += f"   • 📈 Trail Distance: {position.trailing_distance_atr:.1f}x ATR\n"

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
            ExitReason.TAKE_PROFIT_TP1: "🎯",
            ExitReason.TAKE_PROFIT_TP2: "🎯",
            ExitReason.TAKE_PROFIT_TP3: "🎯",
            ExitReason.STOP_LOSS: "🛑",
            ExitReason.TRAILING_STOP: "📈",
            ExitReason.Z_SCORE_REVERSAL: "📉",
            ExitReason.Z_SCORE_REVERSAL_PARTIAL: "📉",
            ExitReason.ORDER_FLOW_REVERSAL: "🔄",
            ExitReason.TIME_EXIT: "⏱️",
            ExitReason.TIME_EXIT_LOSING: "⏱️",
            ExitReason.TIME_EXIT_FLAT: "⏱️",
            ExitReason.OPPOSITE_SIGNAL: "⚡"
        }
        exit_emoji = exit_emoji_map.get(position.exit_reason, "🚪")

        # Human-readable exit reason names (Trading Improvements)
        exit_reason_names = {
            ExitReason.TAKE_PROFIT: "🎯 Тейк-профит",
            ExitReason.TAKE_PROFIT_TP1: "🎯 TP1 (30%)",
            ExitReason.TAKE_PROFIT_TP2: "🎯 TP2 (60%)",
            ExitReason.TAKE_PROFIT_TP3: "🎯 TP3 (100%)",
            ExitReason.STOP_LOSS: "🛑 Стоп-лосс",
            ExitReason.TRAILING_STOP: "📈 Трейлинг-стоп",
            ExitReason.Z_SCORE_REVERSAL: "📉 Z-score ослаб",
            ExitReason.Z_SCORE_REVERSAL_PARTIAL: "📉 Z-score (partial)",
            ExitReason.ORDER_FLOW_REVERSAL: "🔄 Разворот потока",
            ExitReason.TIME_EXIT: "⏱️ Выход по времени",
            ExitReason.TIME_EXIT_LOSING: "⏱️ Time Exit (убыток)",
            ExitReason.TIME_EXIT_FLAT: "⏱️ Time Exit (flat)",
            ExitReason.OPPOSITE_SIGNAL: "⚡ Противосигнал"
        }
        exit_reason_display = exit_reason_names.get(position.exit_reason, position.exit_reason.value if position.exit_reason else 'N/A')

        # =====================================================================
        # HYBRID STRATEGY: Show signal class and trading mode in close message
        # =====================================================================
        hs_cfg = self.config.hybrid_strategy
        hybrid_section = ""

        signal_class_val = position.metrics.get('signal_class')
        trading_mode_val = position.metrics.get('trading_mode')

        if hs_cfg.enabled and signal_class_val:
            signal_class_names = {
                "EXTREME_SPIKE": "🔥 EXTREME",
                "STRONG_SIGNAL": "💪 STRONG",
                "EARLY_SIGNAL": "🌱 EARLY"
            }
            trading_mode_names = {
                "MEAN_REVERSION": "📉 MR",
                "CONDITIONAL_MOMENTUM": "📈 CM",
                "EARLY_MOMENTUM": "🚀 EM"
            }
            signal_class_display = signal_class_names.get(signal_class_val, signal_class_val)
            trading_mode_display = trading_mode_names.get(trading_mode_val, trading_mode_val or "N/A")
            hybrid_section = f"\n🧬 Strategy: {signal_class_display} | {trading_mode_display}"

        message = f"""💼 <b>POSITION CLOSED</b> {pnl_emoji}

{direction_emoji} <b>{position.symbol} {position.direction.value}</b>{hybrid_section}

💰 <b>PnL: {position.pnl_percent:+.2f}%</b>
💵 Price: {format_price(position.open_price)} → {format_price(position.close_price)}

📊 <b>Performance:</b>
   • MFE (Best): {position.max_favorable_excursion:+.2f}%
   • MAE (Worst): {position.max_adverse_excursion:+.2f}%
   • Duration: {position.duration_minutes}m

{exit_emoji} <b>Причина выхода:</b> {exit_reason_display}

📝 <b>Детали:</b>
{position.exit_reason_details if position.exit_reason_details else 'N/A'}

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
        direction_ru = "ЛОНГ" if pending.direction == Direction.UP else "ШОРТ"

        # Get metrics from event
        metrics = event.metrics
        z_er = metrics.get('z_er', pending.signal_z_er)
        z_vol = metrics.get('z_vol', pending.signal_z_vol)
        taker_share = metrics.get('taker_share', 0)
        beta = metrics.get('beta', 0)
        funding = metrics.get('funding', 0)

        # Generate detailed reason for signal creation
        cfg = self.config.position_management
        hs_cfg = self.config.hybrid_strategy
        taker_threshold_high = 0.65
        taker_threshold_low = 0.35

        # Determine flow direction
        if taker_share >= taker_threshold_high:
            flow_description = f"Агрессивные покупки: taker buy = {taker_share:.1%} (≥ {taker_threshold_high:.0%})"
        elif taker_share <= taker_threshold_low:
            flow_description = f"Агрессивные продажи: taker buy = {taker_share:.1%} (≤ {taker_threshold_low:.0%})"
        else:
            flow_description = f"Taker buy share: {taker_share:.1%}"

        # Build detection reason
        detection_details = (
            f"Обнаружен аномальный импульс {direction_ru}:\n"
            f"   • Z-score ER: {z_er:.2f}σ (порог: ≥3.0σ) ✓\n"
            f"   • Z-score VOL: {z_vol:.2f}σ (порог: ≥3.0σ) ✓\n"
            f"   • {flow_description} ✓"
        )

        # =====================================================================
        # HYBRID STRATEGY: Add signal classification and trading mode info
        # =====================================================================
        hybrid_section = ""
        if hs_cfg.enabled and pending.signal_class is not None:
            # Signal class descriptions
            signal_class_names = {
                SignalClass.EXTREME_SPIKE: "🔥 EXTREME SPIKE (z ≥ 5.0σ)",
                SignalClass.STRONG_SIGNAL: "💪 STRONG SIGNAL (3.0-5.0σ)",
                SignalClass.EARLY_SIGNAL: "🌱 EARLY SIGNAL (1.5-3.0σ)"
            }
            signal_class_display = signal_class_names.get(pending.signal_class, pending.signal_class.value)

            # Trading mode descriptions
            trading_mode_names = {
                TradingMode.MEAN_REVERSION: "📉 MEAN-REVERSION (fade the move)",
                TradingMode.CONDITIONAL_MOMENTUM: "📈 CONDITIONAL MOMENTUM (follow if confirmed)",
                TradingMode.EARLY_MOMENTUM: "🚀 EARLY MOMENTUM (wait for continuation)"
            }
            trading_mode_display = trading_mode_names.get(pending.trading_mode, pending.trading_mode.value if pending.trading_mode else "N/A")

            # Trade direction vs original direction
            trade_dir_emoji = "🟢" if pending.trade_direction == Direction.UP else "🔴"
            trade_dir_ru = "ЛОНГ" if pending.trade_direction == Direction.UP else "ШОРТ"

            # Build hybrid section
            hybrid_section = f"""
🧬 <b>HYBRID STRATEGY:</b>
   • Класс: {signal_class_display}
   • Режим: {trading_mode_display}
   • Направление сигнала: {pending.direction.value} ({direction_ru})
   • Направление сделки: {trade_dir_emoji} {pending.trade_direction.value} ({trade_dir_ru})
"""
            # Add mode-specific entry requirements
            if pending.signal_class == SignalClass.EXTREME_SPIKE:
                mr_cfg = hs_cfg.mean_reversion
                hybrid_section += f"""
🎯 <b>Entry Requirements (Mean-Reversion):</b>
   • Z-score drop ≥ {mr_cfg.reversal_z_drop_pct:.0%} от сигнала
   • Price confirmation: движение в направлении сделки
   • Volume fade: снижение объёма
   • Min bars: {mr_cfg.min_bars_before_entry} | Max bars: {mr_cfg.max_bars_before_expiry}
"""
            elif pending.signal_class == SignalClass.STRONG_SIGNAL:
                cm_cfg = hs_cfg.conditional_momentum
                hybrid_section += f"""
🎯 <b>Entry Requirements (Conditional Momentum):</b>
   • No divergence: цена и z-score вместе
   • Volume retention ≥ {cm_cfg.min_volume_retention:.0%}
   • Taker dominance ≥ {cm_cfg.min_taker_dominance:.0%}
   • Z-score stability: variance < {cm_cfg.max_z_variance_pct:.0%}
   • Mode switch: если momentum fails → mean-reversion
"""
            else:  # EARLY_SIGNAL
                em_cfg = hs_cfg.early_momentum
                hybrid_section += f"""
🎯 <b>Entry Requirements (Early Momentum):</b>
   • Z-score growth ≥ {em_cfg.min_z_growth_pct:.0%}
   • Price follow-through ≥ {em_cfg.min_price_follow_through_pct:.1%}
   • Volume sustained
   • Taker persistence ≥ {em_cfg.min_taker_persistence:.0%}
   • Min wait: {em_cfg.min_wait_bars} bars | Max: {em_cfg.max_wait_bars} bars
"""
        else:
            # Legacy mode entry requirements
            entry_requirements = (
                f"Для входа в позицию требуется:\n"
                f"   • Z-score остынет до [{cfg.entry_trigger_z_cooldown:.1f}, 3.0]σ\n"
                f"   • Откат от пика ≥ {cfg.entry_trigger_pullback_pct:.1f}%\n"
                f"   • Flow-доминирование ≥ {cfg.entry_trigger_min_taker_dominance:.0%}"
            )
            hybrid_section = f"""
🎯 <b>Условия входа:</b>
{entry_requirements}
"""

        # Determine max wait based on mode
        if hs_cfg.enabled and pending.signal_class is not None:
            if pending.signal_class == SignalClass.EXTREME_SPIKE:
                max_wait_bars = hs_cfg.mean_reversion.max_bars_before_expiry
            elif pending.signal_class == SignalClass.STRONG_SIGNAL:
                max_wait_bars = hs_cfg.conditional_momentum.max_wait_bars
            else:
                max_wait_bars = hs_cfg.early_momentum.max_wait_bars
            watch_window_info = f"   • Макс. ожидание: {max_wait_bars} bars (~{max_wait_bars}m)"
        else:
            watch_window_info = f"   • Макс. ожидание: {cfg.entry_trigger_max_wait_minutes}m"

        message = f"""⏳ <b>PENDING SIGNAL CREATED</b>

{direction_emoji} <b>{pending.symbol} {pending.direction.value}</b>

📊 <b>Метрики сигнала:</b>
   • Z-Score (ER): {pending.signal_z_er:.2f}σ
   • Z-Score (VOL): {pending.signal_z_vol:.2f}σ
   • Цена: {format_price(pending.signal_price)}
   • Пик: {format_price(pending.peak_since_signal)}
   • Beta vs BTC: {beta:.2f}
   • Funding: {funding:.4%}

📝 <b>Почему создан сигнал:</b>
{detection_details}
{hybrid_section}
⏱️ <b>Watch Window:</b>
{watch_window_info}
   • Вход: как только все триггеры сработают

🕐 Время: {timestamp}
🆔 ID: {pending.signal_id}
"""

        return message

    def _format_pending_signal_invalidated(self, pending: PendingSignal) -> str:
        """Format message for pending signal invalidated."""
        timestamp = datetime.fromtimestamp(pending.created_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

        direction_emoji = "🟢" if pending.direction == Direction.UP else "🔴"

        # Human-readable reason mapping
        reason_names = {
            "direction_flip": "🔄 Разворот направления",
            "momentum_died": "📉 Импульс угас",
            "flow_died": "💧 Поток ордеров иссяк",
            "structure_broken": "🔨 Структура сломана",
            "ttl_expired": "⏰ Время истекло"
        }
        reason_display = reason_names.get(pending.invalidation_reason, pending.invalidation_reason)

        message = f"""❌ <b>PENDING SIGNAL INVALIDATED</b>

{direction_emoji} <b>{pending.symbol} {pending.direction.value}</b>

⚠️ <b>Причина:</b> {reason_display}

📝 <b>Детали:</b>
{pending.invalidation_details if pending.invalidation_details else 'N/A'}

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

    # =========================================================================
    # HYBRID STRATEGY METHODS
    # =========================================================================

    def _get_trading_mode_and_direction(
        self,
        signal_class: SignalClass,
        original_direction: Direction
    ) -> tuple:
        """
        Determine trading mode and trade direction based on signal class.

        For EXTREME_SPIKE (mean-reversion): trade AGAINST the move (fade)
        For STRONG_SIGNAL / EARLY_SIGNAL (momentum): trade WITH the move

        Returns:
            (TradingMode, Direction) tuple
        """
        if signal_class == SignalClass.EXTREME_SPIKE:
            # Mean-reversion: fade the extreme move
            trade_direction = Direction.DOWN if original_direction == Direction.UP else Direction.UP
            return (TradingMode.MEAN_REVERSION, trade_direction)

        elif signal_class == SignalClass.STRONG_SIGNAL:
            # Conditional momentum: follow if confirmed
            return (TradingMode.CONDITIONAL_MOMENTUM, original_direction)

        else:  # EARLY_SIGNAL
            # Early momentum: follow if continuation confirmed
            return (TradingMode.EARLY_MOMENTUM, original_direction)

    async def _evaluate_hybrid_triggers(
        self,
        pending: PendingSignal,
        current_bar: Bar,
        current_features: Optional['Features']
    ) -> bool:
        """
        Evaluate hybrid strategy entry triggers based on signal class.

        Routes to mode-specific evaluator:
        - EXTREME_SPIKE -> _evaluate_mean_reversion_triggers
        - STRONG_SIGNAL -> _evaluate_strong_signal_triggers (with mode switch)
        - EARLY_SIGNAL -> _evaluate_early_signal_triggers

        Returns True if position should be opened.
        """
        # Update history tracking
        if current_features is not None:
            pending.update_history(current_features, current_bar)

        signal_class = pending.signal_class

        if signal_class == SignalClass.EXTREME_SPIKE:
            return await self._evaluate_mean_reversion_triggers(
                pending, current_bar, current_features
            )

        elif signal_class == SignalClass.STRONG_SIGNAL:
            result = await self._evaluate_strong_signal_triggers(
                pending, current_bar, current_features
            )

            # Check for mode switch
            if not result and not pending.invalidated and pending.mode_switched:
                # Switched to mean-reversion, re-evaluate with MR triggers
                logger.info(
                    f"{pending.symbol}: Strong signal switched to MEAN_REVERSION mode, "
                    f"re-evaluating with MR triggers"
                )
                return await self._evaluate_mean_reversion_triggers(
                    pending, current_bar, current_features
                )

            return result

        else:  # EARLY_SIGNAL
            return await self._evaluate_early_signal_triggers(
                pending, current_bar, current_features
            )

    async def _evaluate_mean_reversion_triggers(
        self,
        pending: PendingSignal,
        current_bar: Bar,
        current_features: Optional['Features']
    ) -> bool:
        """
        Evaluate entry triggers for MEAN_REVERSION mode (EXTREME_SPIKE signals).

        Entry conditions (ALL must be met):
        1. Reversal started: Z-score dropped >= 20% from signal
        2. Price confirmation: Price moving in trade_direction
        3. Volume fade: Current volume z-score < signal volume z-score
        4. Time filter: bars >= min_bars AND bars <= max_bars

        Returns True if all triggers met.
        """
        symbol = pending.symbol
        hs_cfg = self.config.hybrid_strategy.mean_reversion

        # Need features
        if current_features is None:
            logger.debug(f"{symbol}: MR - No features available, waiting...")
            return False

        # Time filter: minimum bars before entry
        if pending.bars_since_signal < hs_cfg.min_bars_before_entry:
            logger.debug(
                f"{symbol}: MR - Too early, bars={pending.bars_since_signal} < {hs_cfg.min_bars_before_entry}"
            )
            return False

        # Invalidation: Z-score grew (move continuing, not reversing)
        if hs_cfg.invalidate_on_z_growth:
            current_z_abs = abs(current_features.z_er_15m)
            signal_z_abs = abs(pending.signal_z_er)
            growth_ratio = current_z_abs / signal_z_abs if signal_z_abs > 0 else 1.0

            if growth_ratio >= hs_cfg.z_growth_invalidate_threshold:
                pending.invalidated = True
                pending.invalidation_reason = "z_growth"
                pending.invalidation_details = (
                    f"Z-score grew from {signal_z_abs:.2f} to {current_z_abs:.2f} "
                    f"(ratio {growth_ratio:.2f} >= {hs_cfg.z_growth_invalidate_threshold}). "
                    f"Move continuing, not reversing."
                )
                logger.info(f"{symbol}: MR invalidated - {pending.invalidation_details}")
                return False

        # Invalidation: Volume still growing (momentum not fading)
        if hs_cfg.require_volume_fade and len(pending.vol_history) >= 2:
            if pending.vol_history[-1] > pending.vol_history[-2]:
                pending.volume_growth_streak += 1
            else:
                pending.volume_growth_streak = 0

            if pending.volume_growth_streak >= hs_cfg.max_volume_growth_bars:
                pending.invalidated = True
                pending.invalidation_reason = "volume_growth"
                pending.invalidation_details = (
                    f"Volume grew for {pending.volume_growth_streak} consecutive bars. "
                    f"Momentum not fading."
                )
                logger.info(f"{symbol}: MR invalidated - {pending.invalidation_details}")
                return False

        # Trigger 1: Reversal started (Z-score dropped sufficiently)
        z_drop = pending.get_z_drop_pct()
        if z_drop is None or z_drop < hs_cfg.reversal_z_drop_pct:
            logger.debug(
                f"{symbol}: MR - Z-score not dropped enough "
                f"(drop={z_drop:.1%} if z_drop else 'N/A', need {hs_cfg.reversal_z_drop_pct:.0%})"
            )
            pending.reversal_started = False
        else:
            pending.reversal_started = True
            logger.debug(f"{symbol}: MR - Reversal detected (z dropped {z_drop:.1%}) ✓")

        # Trigger 2: Price confirmation
        price_confirmed = False
        if hs_cfg.require_price_confirmation:
            trade_dir = pending.trade_direction
            if trade_dir == Direction.UP:
                # For long (fading a down move): price should be rising
                price_confirmed = current_bar.close > pending.signal_price
            else:  # DOWN
                # For short (fading an up move): price should be falling
                price_confirmed = current_bar.close < pending.signal_price

            if price_confirmed:
                logger.debug(f"{symbol}: MR - Price confirming reversal ✓")
            else:
                logger.debug(f"{symbol}: MR - Price not confirming yet")
        else:
            price_confirmed = True

        # Trigger 3: Volume fade
        volume_fading = False
        if hs_cfg.require_volume_fade and len(pending.vol_history) >= 1:
            current_vol_z = current_features.z_vol_15m
            signal_vol_z = pending.signal_z_vol
            volume_fading = current_vol_z < signal_vol_z
            if volume_fading:
                logger.debug(f"{symbol}: MR - Volume fading ({current_vol_z:.2f} < {signal_vol_z:.2f}) ✓")
            else:
                logger.debug(f"{symbol}: MR - Volume still elevated")
        else:
            volume_fading = True  # If disabled or no data

        # Check if all triggers met
        all_met = (
            pending.reversal_started and
            price_confirmed and
            volume_fading
        )

        if all_met:
            logger.info(
                f"{symbol}: MR - ALL entry triggers met! "
                f"(reversal: {pending.reversal_started}, price: {price_confirmed}, vol_fade: {volume_fading})"
            )

        return all_met

    async def _evaluate_strong_signal_triggers(
        self,
        pending: PendingSignal,
        current_bar: Bar,
        current_features: Optional['Features']
    ) -> bool:
        """
        Evaluate entry triggers for CONDITIONAL_MOMENTUM mode (STRONG_SIGNAL signals).

        Entry conditions (ALL must be met):
        1. No divergence: Price and z-score moving in same direction
        2. Volume maintained: Current vol_z >= signal_vol_z * retention_pct
        3. Taker dominance: Strong (>= 70% for long, <= 30% for short)
        4. Z-score stability: Variance over last N bars within threshold

        Mode switch to MEAN_REVERSION if:
        - Z dropped > 30%
        - Taker flow reversed (crossed 50%)
        - Price reversed past signal price

        Returns True if all triggers met.
        """
        symbol = pending.symbol
        hs_cfg = self.config.hybrid_strategy.conditional_momentum
        common_cfg = self.config.hybrid_strategy.common

        # Need features
        if current_features is None:
            logger.debug(f"{symbol}: CM - No features available, waiting...")
            return False

        # Need minimum bars for stability check
        if pending.bars_since_signal < hs_cfg.confirmation_bars:
            logger.debug(
                f"{symbol}: CM - Need more bars for stability check "
                f"({pending.bars_since_signal}/{hs_cfg.confirmation_bars})"
            )
            return False

        direction = pending.direction
        current_z = current_features.z_er_15m
        signal_z = pending.signal_z_er

        # Check for mode switch conditions
        if hs_cfg.enable_mode_switch and not pending.mode_switched:
            switch_reason = None

            # Check Z-score drop
            z_drop = pending.get_z_drop_pct()
            if z_drop is not None and z_drop >= hs_cfg.mode_switch_z_drop_pct:
                switch_reason = f"z_drop={z_drop:.1%}"

            # Check taker reversal
            current_taker = current_features.taker_buy_share_15m
            if current_taker is not None:
                if direction == Direction.UP and current_taker < hs_cfg.mode_switch_taker_reversal:
                    switch_reason = f"taker_reversal={current_taker:.2f}"
                elif direction == Direction.DOWN and current_taker > hs_cfg.mode_switch_taker_reversal:
                    switch_reason = f"taker_reversal={current_taker:.2f}"

            # Check price reversal
            if hs_cfg.mode_switch_price_reversal:
                if direction == Direction.UP and current_bar.close < pending.signal_price:
                    switch_reason = f"price_reversal={current_bar.close:.2f} < {pending.signal_price:.2f}"
                elif direction == Direction.DOWN and current_bar.close > pending.signal_price:
                    switch_reason = f"price_reversal={current_bar.close:.2f} > {pending.signal_price:.2f}"

            if switch_reason:
                # Switch to mean-reversion mode
                pending.trading_mode = TradingMode.MEAN_REVERSION
                pending.trade_direction = Direction.DOWN if direction == Direction.UP else Direction.UP
                pending.mode_switched = True
                logger.info(
                    f"{symbol}: CM -> MR mode switch triggered: {switch_reason}. "
                    f"New trade direction: {pending.trade_direction.value}"
                )
                return False  # Will re-evaluate with MR triggers

        # Trigger 1: No divergence (price and z moving together)
        no_divergence = True
        if hs_cfg.require_no_divergence:
            price_moving_up = current_bar.close > pending.signal_price
            z_stable_or_better = abs(current_z) >= abs(signal_z) * 0.75

            if direction == Direction.UP:
                no_divergence = price_moving_up and z_stable_or_better
            else:
                no_divergence = (not price_moving_up) and z_stable_or_better

            if no_divergence:
                logger.debug(f"{symbol}: CM - No divergence ✓")
            else:
                logger.debug(f"{symbol}: CM - Divergence detected")

        # Trigger 2: Volume maintained
        current_vol_z = current_features.z_vol_15m
        signal_vol_z = pending.signal_z_vol
        min_vol = signal_vol_z * hs_cfg.min_volume_retention
        volume_maintained = current_vol_z >= min_vol

        if volume_maintained:
            logger.debug(f"{symbol}: CM - Volume maintained ({current_vol_z:.2f} >= {min_vol:.2f}) ✓")
        else:
            logger.debug(f"{symbol}: CM - Volume dropped ({current_vol_z:.2f} < {min_vol:.2f})")

        # Trigger 3: Taker dominance (strong)
        current_taker = current_features.taker_buy_share_15m
        taker_dominant = False
        if current_taker is not None:
            if direction == Direction.UP:
                taker_dominant = current_taker >= hs_cfg.min_taker_dominance
            else:
                taker_dominant = current_taker <= (1.0 - hs_cfg.min_taker_dominance)

            if taker_dominant:
                logger.debug(f"{symbol}: CM - Taker dominant ({current_taker:.2f}) ✓")
            else:
                logger.debug(f"{symbol}: CM - Taker not dominant ({current_taker:.2f})")
        else:
            logger.debug(f"{symbol}: CM - No taker data")

        # Trigger 4: Z-score stability
        z_variance = pending.get_z_variance(last_n_bars=hs_cfg.confirmation_bars)
        z_stable = False
        if z_variance is not None:
            max_variance = (abs(signal_z) * hs_cfg.max_z_variance_pct) ** 2
            z_stable = z_variance <= max_variance
            if z_stable:
                pending.z_stable_bar_count += 1
                logger.debug(f"{symbol}: CM - Z-score stable (var={z_variance:.4f} <= {max_variance:.4f}) ✓")
            else:
                pending.z_stable_bar_count = 0
                logger.debug(f"{symbol}: CM - Z-score unstable (var={z_variance:.4f} > {max_variance:.4f})")
        else:
            logger.debug(f"{symbol}: CM - Not enough history for variance check")

        # Check if all triggers met
        all_met = no_divergence and volume_maintained and taker_dominant and z_stable

        if all_met:
            logger.info(
                f"{symbol}: CM - ALL entry triggers met! "
                f"(no_div: {no_divergence}, vol: {volume_maintained}, "
                f"taker: {taker_dominant}, z_stable: {z_stable})"
            )

        return all_met

    async def _evaluate_early_signal_triggers(
        self,
        pending: PendingSignal,
        current_bar: Bar,
        current_features: Optional['Features']
    ) -> bool:
        """
        Evaluate entry triggers for EARLY_MOMENTUM mode (EARLY_SIGNAL signals).

        Must wait min_wait_bars (3) before evaluating.

        Entry conditions (ALL must be met):
        1. Z-score growth: Z grew >= 30% from signal
        2. Price follow-through: Price moved >= 0.15% in direction
        3. Volume sustained: Current volume >= early volume
        4. Taker persistence: Min taker over last 3 bars >= threshold

        Invalidation:
        - Z declined > 20% (momentum died)
        - No continuation by max_wait_bars

        Returns True if all triggers met.
        """
        symbol = pending.symbol
        hs_cfg = self.config.hybrid_strategy.early_momentum
        direction = pending.direction

        # Need features
        if current_features is None:
            logger.debug(f"{symbol}: EM - No features available, waiting...")
            return False

        # Must wait minimum bars
        if pending.bars_since_signal < hs_cfg.min_wait_bars:
            logger.debug(
                f"{symbol}: EM - Waiting for min bars "
                f"({pending.bars_since_signal}/{hs_cfg.min_wait_bars})"
            )
            return False

        pending.early_wait_complete = True

        # Check invalidation: Z declined
        z_growth = pending.get_z_growth_pct()
        if z_growth is not None and z_growth < -hs_cfg.z_decline_invalidate_pct:
            pending.invalidated = True
            pending.invalidation_reason = "z_decline"
            pending.invalidation_details = (
                f"Z-score declined {abs(z_growth):.1%} (threshold: {hs_cfg.z_decline_invalidate_pct:.0%}). "
                f"Momentum died."
            )
            logger.info(f"{symbol}: EM invalidated - {pending.invalidation_details}")
            return False

        # Trigger 1: Z-score growth
        z_growing = False
        if z_growth is not None:
            z_growing = z_growth >= hs_cfg.min_z_growth_pct
            if z_growing:
                logger.debug(f"{symbol}: EM - Z growing ({z_growth:.1%} >= {hs_cfg.min_z_growth_pct:.0%}) ✓")
            else:
                logger.debug(f"{symbol}: EM - Z not growing enough ({z_growth:.1%})")
        else:
            logger.debug(f"{symbol}: EM - Cannot calculate z growth")

        # Trigger 2: Price follow-through
        price_change = pending.get_price_change_pct()
        price_following = False
        if price_change is not None:
            price_following = price_change >= hs_cfg.min_price_follow_through_pct
            if price_following:
                logger.debug(
                    f"{symbol}: EM - Price following "
                    f"({price_change:.2f}% >= {hs_cfg.min_price_follow_through_pct}%) ✓"
                )
            else:
                logger.debug(f"{symbol}: EM - Price not following ({price_change:.2f}%)")
        else:
            logger.debug(f"{symbol}: EM - Cannot calculate price change")

        # Trigger 3: Volume sustained
        volume_sustained = True
        if hs_cfg.volume_must_sustain:
            early_vol = pending.get_avg_volume_early()
            current_vol = pending.get_avg_volume_current()
            if early_vol is not None and current_vol is not None:
                volume_sustained = current_vol >= early_vol
                if volume_sustained:
                    logger.debug(
                        f"{symbol}: EM - Volume sustained ({current_vol:.2f} >= {early_vol:.2f}) ✓"
                    )
                else:
                    logger.debug(
                        f"{symbol}: EM - Volume declined ({current_vol:.2f} < {early_vol:.2f})"
                    )
            else:
                logger.debug(f"{symbol}: EM - Not enough volume history")

        # Trigger 4: Taker persistence
        min_taker = pending.get_min_taker_recent(last_n_bars=3)
        taker_persistent = False
        if min_taker is not None:
            if direction == Direction.UP:
                taker_persistent = min_taker >= hs_cfg.min_taker_persistence
            else:
                taker_persistent = min_taker <= (1.0 - hs_cfg.min_taker_persistence)

            if taker_persistent:
                logger.debug(f"{symbol}: EM - Taker persistent (min={min_taker:.2f}) ✓")
            else:
                logger.debug(f"{symbol}: EM - Taker not persistent (min={min_taker:.2f})")
        else:
            logger.debug(f"{symbol}: EM - Not enough taker history")

        # Check if all triggers met
        all_met = z_growing and price_following and volume_sustained and taker_persistent

        if all_met:
            pending.continuation_confirmed = True
            logger.info(
                f"{symbol}: EM - ALL entry triggers met (continuation confirmed)! "
                f"(z_grow: {z_growing}, price: {price_following}, "
                f"vol: {volume_sustained}, taker: {taker_persistent})"
            )

        return all_met

    async def close(self) -> None:
        """Cleanup on shutdown."""
        # Save all open positions
        for position in self.open_positions.values():
            await self.storage.write_position(position)

        # Close Telegram session
        if self.telegram_session:
            await self.telegram_session.close()

        logger.info("Position manager closed")
