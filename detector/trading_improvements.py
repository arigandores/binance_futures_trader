"""
Trading Improvements Module - Implementation of 7 trading optimizations.

Based on trading_improvements_specification.md analysis of 62 trades:
- Win Rate: 51.6% -> Target 57-60%
- R:R Ratio: 0.89 -> Target 1.6
- Net PnL: -4.96% (after fees) -> Target +5-10%

Improvements:
1. Adaptive Stop-Loss: SL based on signal class + volatility + direction
2. Tiered Take-Profit: 3 levels (TP1, TP2, TP3) with partial closes
3. Delayed Z-Exit: Minimum profit/time conditions before z-exit
4. Direction Filters: Stricter requirements for LONG positions
5. Trailing Stop by Class: Different activation thresholds per class
6. Intelligent Time Exit: Aggressive exits for losing/flat positions
7. Min Profit Filter: Skip trades with insufficient expected profit
"""

import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from detector.models import (
    Position, Direction, SignalClass, Bar, Features, ExitReason
)
from detector.config import (
    Config, AdaptiveStopLossConfig, TieredTakeProfitConfig,
    DelayedZExitConfig, DirectionFiltersConfig, TrailingStopByClassConfig,
    TimeExitConfig, MinProfitFilterConfig
)
from detector.features_extended import ExtendedFeatureCalculator

logger = logging.getLogger(__name__)


@dataclass
class TieredTPLevels:
    """Calculated tiered take-profit levels."""
    tp1_price: float
    tp1_close_pct: int
    tp2_price: float
    tp2_close_pct: int
    tp3_price: float
    tp3_close_pct: int


class TradingImprovements:
    """
    Implementation of 7 trading improvements for position management.

    This class is initialized with config and extended_features calculator,
    and provides methods for each improvement that can be called from
    PositionManager.
    """

    def __init__(self, config: Config, extended_features: ExtendedFeatureCalculator):
        self.config = config
        self.extended_features = extended_features
        self.pm_cfg = config.position_management

        # ATR history for volatility percentile calculation
        self.atr_history: Dict[str, list] = {}  # symbol -> list of recent ATR values

    # =========================================================================
    # IMPROVEMENT 1: Adaptive Stop-Loss
    # =========================================================================

    def calculate_adaptive_stop_loss(
        self,
        symbol: str,
        signal_class: Optional[SignalClass],
        direction: Direction,
        entry_price: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate adaptive stop-loss price based on signal class, volatility, and direction.

        Returns:
            Tuple of (stop_price, final_multiplier) or (None, None) if disabled/no ATR
        """
        asl_cfg = self.pm_cfg.adaptive_stop_loss

        if not asl_cfg.enabled:
            return None, None

        # Get current ATR
        atr = self.extended_features.get_atr(symbol)
        if atr is None or atr <= 0:
            logger.debug(f"{symbol}: Adaptive SL skipped - no ATR available")
            return None, None

        # Step 1: Base multiplier by signal class
        if signal_class == SignalClass.EXTREME_SPIKE:
            base_mult = asl_cfg.base_multiplier_extreme_spike
        elif signal_class == SignalClass.STRONG_SIGNAL:
            base_mult = asl_cfg.base_multiplier_strong_signal
        elif signal_class == SignalClass.EARLY_SIGNAL:
            base_mult = asl_cfg.base_multiplier_early_signal
        else:
            # Fallback for legacy positions without signal class
            base_mult = asl_cfg.base_multiplier_strong_signal

        # Step 2: Volatility adjustment
        vol_mult = 1.0
        if asl_cfg.volatility_adjustment_enabled:
            vol_mult = self._get_volatility_adjustment(symbol, asl_cfg)

        # Step 3: Direction adjustment (longs are riskier)
        dir_mult = 1.0
        if asl_cfg.direction_adjustment_enabled:
            if direction == Direction.UP:  # LONG
                dir_mult = asl_cfg.long_additional_multiplier
            else:  # SHORT
                dir_mult = asl_cfg.short_additional_multiplier

        # Calculate final multiplier
        final_mult = base_mult * vol_mult * dir_mult

        # Calculate stop distance in price
        stop_distance = atr * final_mult
        stop_distance_pct = (stop_distance / entry_price) * 100

        # Apply safety limits
        min_distance = entry_price * (asl_cfg.min_stop_distance_pct / 100)
        max_distance = entry_price * (asl_cfg.max_stop_distance_pct / 100)

        stop_distance = max(stop_distance, min_distance)
        stop_distance = min(stop_distance, max_distance)

        # Calculate stop price
        if direction == Direction.UP:  # LONG: stop below entry
            stop_price = entry_price - stop_distance
        else:  # SHORT: stop above entry
            stop_price = entry_price + stop_distance

        logger.debug(
            f"{symbol}: Adaptive SL calculated | "
            f"base={base_mult:.2f} × vol={vol_mult:.2f} × dir={dir_mult:.2f} = {final_mult:.2f} | "
            f"ATR={atr:.6f} | stop_dist={stop_distance_pct:.2f}% | "
            f"stop_price={stop_price:.6f}"
        )

        return stop_price, final_mult

    def _get_volatility_adjustment(
        self,
        symbol: str,
        asl_cfg: AdaptiveStopLossConfig
    ) -> float:
        """
        Calculate volatility adjustment based on ATR percentile.

        Returns:
            Multiplier: 1.5 for high vol, 1.0 for normal, 0.75 for low vol
        """
        # Get ATR history for symbol
        atr_history = self.atr_history.get(symbol, [])

        if len(atr_history) < 10:
            # Not enough history, return neutral
            return 1.0

        current_atr = self.extended_features.get_atr(symbol)
        if current_atr is None:
            return 1.0

        # Calculate percentile
        sorted_history = sorted(atr_history)
        percentile_rank = sum(1 for x in sorted_history if x <= current_atr) / len(sorted_history) * 100

        # Determine regime
        if percentile_rank > asl_cfg.high_volatility_percentile:
            return asl_cfg.high_volatility_multiplier  # 1.5
        elif percentile_rank < asl_cfg.low_volatility_percentile:
            return asl_cfg.low_volatility_multiplier   # 0.75
        else:
            return 1.0  # Normal

    def update_atr_history(self, symbol: str) -> None:
        """Update ATR history for volatility percentile calculation."""
        atr = self.extended_features.get_atr(symbol)
        if atr is None:
            return

        if symbol not in self.atr_history:
            self.atr_history[symbol] = []

        self.atr_history[symbol].append(atr)

        # Keep only lookback_bars history
        max_history = self.pm_cfg.adaptive_stop_loss.volatility_lookback_bars
        if len(self.atr_history[symbol]) > max_history:
            self.atr_history[symbol] = self.atr_history[symbol][-max_history:]

    # =========================================================================
    # IMPROVEMENT 2: Tiered Take-Profit
    # =========================================================================

    def calculate_tiered_tp_levels(
        self,
        symbol: str,
        signal_class: Optional[SignalClass],
        direction: Direction,
        entry_price: float
    ) -> Optional[TieredTPLevels]:
        """
        Calculate three tiered take-profit levels based on signal class and ATR.

        Returns:
            TieredTPLevels dataclass or None if disabled/no ATR
        """
        ttp_cfg = self.pm_cfg.tiered_take_profit

        if not ttp_cfg.enabled:
            return None

        atr = self.extended_features.get_atr(symbol)
        if atr is None or atr <= 0:
            logger.debug(f"{symbol}: Tiered TP skipped - no ATR available")
            return None

        # Get ATR multipliers for signal class
        if signal_class == SignalClass.EXTREME_SPIKE:
            tp1_atr = ttp_cfg.extreme_spike_tp1_atr
            tp1_pct = ttp_cfg.extreme_spike_tp1_close_pct
            tp2_atr = ttp_cfg.extreme_spike_tp2_atr
            tp2_pct = ttp_cfg.extreme_spike_tp2_close_pct
            tp3_atr = ttp_cfg.extreme_spike_tp3_atr
            tp3_pct = ttp_cfg.extreme_spike_tp3_close_pct
        elif signal_class == SignalClass.STRONG_SIGNAL:
            tp1_atr = ttp_cfg.strong_signal_tp1_atr
            tp1_pct = ttp_cfg.strong_signal_tp1_close_pct
            tp2_atr = ttp_cfg.strong_signal_tp2_atr
            tp2_pct = ttp_cfg.strong_signal_tp2_close_pct
            tp3_atr = ttp_cfg.strong_signal_tp3_atr
            tp3_pct = ttp_cfg.strong_signal_tp3_close_pct
        elif signal_class == SignalClass.EARLY_SIGNAL:
            tp1_atr = ttp_cfg.early_signal_tp1_atr
            tp1_pct = ttp_cfg.early_signal_tp1_close_pct
            tp2_atr = ttp_cfg.early_signal_tp2_atr
            tp2_pct = ttp_cfg.early_signal_tp2_close_pct
            tp3_atr = ttp_cfg.early_signal_tp3_atr
            tp3_pct = ttp_cfg.early_signal_tp3_close_pct
        else:
            # Fallback: use STRONG_SIGNAL params
            tp1_atr = ttp_cfg.strong_signal_tp1_atr
            tp1_pct = ttp_cfg.strong_signal_tp1_close_pct
            tp2_atr = ttp_cfg.strong_signal_tp2_atr
            tp2_pct = ttp_cfg.strong_signal_tp2_close_pct
            tp3_atr = ttp_cfg.strong_signal_tp3_atr
            tp3_pct = ttp_cfg.strong_signal_tp3_close_pct

        # Calculate prices
        if direction == Direction.UP:  # LONG: TP above entry
            tp1_price = entry_price + (atr * tp1_atr)
            tp2_price = entry_price + (atr * tp2_atr)
            tp3_price = entry_price + (atr * tp3_atr)
        else:  # SHORT: TP below entry
            tp1_price = entry_price - (atr * tp1_atr)
            tp2_price = entry_price - (atr * tp2_atr)
            tp3_price = entry_price - (atr * tp3_atr)

        logger.debug(
            f"{symbol}: Tiered TP levels | class={signal_class} | "
            f"TP1={tp1_price:.6f} ({tp1_pct}%) | "
            f"TP2={tp2_price:.6f} ({tp2_pct}%) | "
            f"TP3={tp3_price:.6f} ({tp3_pct}%)"
        )

        return TieredTPLevels(
            tp1_price=tp1_price,
            tp1_close_pct=tp1_pct,
            tp2_price=tp2_price,
            tp2_close_pct=tp2_pct,
            tp3_price=tp3_price,
            tp3_close_pct=tp3_pct
        )

    def check_tiered_tp(
        self,
        position: Position,
        current_price: float,
        current_ts: int
    ) -> Optional[Tuple[ExitReason, int]]:
        """
        Check if any tiered TP level is hit.

        Returns:
            Tuple of (ExitReason, close_percent) or None if no TP hit
        """
        ttp_cfg = self.pm_cfg.tiered_take_profit

        if not ttp_cfg.enabled:
            return None

        # Check TP1
        if not position.tp1_hit and position.tp1_price is not None:
            hit = self._price_reached_target(
                position.direction, current_price, position.tp1_price
            )
            if hit:
                position.tp1_hit = True
                position.remaining_quantity_pct -= 30  # Close 30%

                # Move SL to breakeven if enabled
                if ttp_cfg.move_sl_breakeven_on_tp1:
                    position.sl_moved_to_breakeven = True
                    position.adaptive_stop_price = position.open_price
                    logger.info(
                        f"{position.symbol}: TP1 hit @ {current_price:.6f} | "
                        f"Closed 30%, SL moved to breakeven"
                    )

                return (ExitReason.TAKE_PROFIT_TP1, 30)

        # Check TP2
        if not position.tp2_hit and position.tp2_price is not None:
            hit = self._price_reached_target(
                position.direction, current_price, position.tp2_price
            )
            if hit:
                position.tp2_hit = True
                position.remaining_quantity_pct -= 30  # Close 30%

                # Activate trailing stop if enabled
                if ttp_cfg.activate_trailing_on_tp2:
                    position.trailing_active = True
                    logger.info(
                        f"{position.symbol}: TP2 hit @ {current_price:.6f} | "
                        f"Closed 30%, trailing stop activated"
                    )

                return (ExitReason.TAKE_PROFIT_TP2, 30)

        # Check TP3 (final)
        if not position.tp3_hit and position.tp3_price is not None:
            hit = self._price_reached_target(
                position.direction, current_price, position.tp3_price
            )
            if hit:
                position.tp3_hit = True
                close_pct = int(position.remaining_quantity_pct)
                position.remaining_quantity_pct = 0
                logger.info(
                    f"{position.symbol}: TP3 hit @ {current_price:.6f} | "
                    f"Closed remaining {close_pct}%"
                )
                return (ExitReason.TAKE_PROFIT_TP3, close_pct)

        return None

    def _price_reached_target(
        self,
        direction: Direction,
        current_price: float,
        target_price: float
    ) -> bool:
        """Check if price reached target level."""
        if direction == Direction.UP:  # LONG
            return current_price >= target_price
        else:  # SHORT
            return current_price <= target_price

    # =========================================================================
    # IMPROVEMENT 3: Delayed Z-Exit
    # =========================================================================

    def check_delayed_z_exit(
        self,
        position: Position,
        current_z_er: float,
        current_price: float,
        current_ts: int
    ) -> Optional[Tuple[ExitReason, int]]:
        """
        Check Z-exit with delay conditions.

        CRITICAL FIX v2: Added require_min_profit flag to disable the profit
        requirement. When require_min_profit=False, Z-exit triggers immediately
        when z-score threshold is reached (standard behavior).

        Returns:
            Tuple of (ExitReason, close_percent) or None if z-exit not allowed
        """
        dze_cfg = self.pm_cfg.delayed_z_exit
        z_threshold = self.pm_cfg.z_score_exit_threshold

        if not dze_cfg.enabled:
            # Use standard z-exit
            if abs(current_z_er) <= z_threshold:
                return (ExitReason.Z_SCORE_REVERSAL, 100)
            return None

        # Check if z-score is below threshold
        if abs(current_z_er) > z_threshold:
            return None  # Z not reversed yet

        # CRITICAL FIX v2: If require_min_profit is disabled, exit immediately on z-reversal
        if not dze_cfg.require_min_profit:
            logger.info(
                f"{position.symbol}: Z-exit triggered (require_min_profit=false) | "
                f"|z_ER|={abs(current_z_er):.2f} <= {z_threshold:.2f}"
            )
            return (ExitReason.Z_SCORE_REVERSAL, 100)

        # Get signal class parameters
        signal_class_str = position.signal_class
        if signal_class_str == "EXTREME_SPIKE":
            min_profit = dze_cfg.extreme_spike_min_profit_pct
            min_hold = dze_cfg.extreme_spike_min_hold_minutes
            partial_pct = dze_cfg.extreme_spike_partial_close_pct
        elif signal_class_str == "STRONG_SIGNAL":
            min_profit = dze_cfg.strong_signal_min_profit_pct
            min_hold = dze_cfg.strong_signal_min_hold_minutes
            partial_pct = dze_cfg.strong_signal_partial_close_pct
        elif signal_class_str == "EARLY_SIGNAL":
            min_profit = dze_cfg.early_signal_min_profit_pct
            min_hold = dze_cfg.early_signal_min_hold_minutes
            partial_pct = dze_cfg.early_signal_partial_close_pct
        else:
            # Fallback
            min_profit = dze_cfg.strong_signal_min_profit_pct
            min_hold = dze_cfg.strong_signal_min_hold_minutes
            partial_pct = dze_cfg.strong_signal_partial_close_pct

        # Calculate current PnL and hold time
        direction_mult = 1 if position.direction == Direction.UP else -1
        current_pnl = ((current_price - position.open_price) / position.open_price * 100) * direction_mult
        hold_minutes = (current_ts - position.open_ts) // (60 * 1000)

        # Check delay conditions
        profit_ok = current_pnl >= min_profit
        time_ok = hold_minutes >= min_hold
        already_partial = position.tp1_hit  # TP1 counts as partial

        if profit_ok or time_ok or already_partial:
            # Z-exit allowed

            # Decide full or partial close
            if not already_partial and dze_cfg.partial_close_enabled:
                # Partial close
                if dze_cfg.skip_partial_if_tp1_hit and position.tp1_hit:
                    return (ExitReason.Z_SCORE_REVERSAL, 100)

                logger.info(
                    f"{position.symbol}: Delayed z-exit (partial) | "
                    f"PnL={current_pnl:.2f}% | hold={hold_minutes}m | "
                    f"closing {partial_pct}%"
                )
                return (ExitReason.Z_SCORE_REVERSAL_PARTIAL, partial_pct)
            else:
                # Full close
                logger.info(
                    f"{position.symbol}: Delayed z-exit (full) | "
                    f"PnL={current_pnl:.2f}% | hold={hold_minutes}m"
                )
                return (ExitReason.Z_SCORE_REVERSAL, 100)
        else:
            # Delay z-exit
            logger.debug(
                f"{position.symbol}: Z-exit delayed | "
                f"PnL={current_pnl:.2f}%/{min_profit:.2f}% | "
                f"hold={hold_minutes}m/{min_hold}m"
            )
            return None

    # =========================================================================
    # IMPROVEMENT 4: Direction Filters
    # =========================================================================

    def check_direction_filters(
        self,
        symbol: str,
        signal_class: Optional[SignalClass],
        direction: Direction,
        z_er: float,
        btc_z_er: Optional[float],
        volume_z: Optional[float]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if trade passes direction-specific filters.

        Args:
            symbol: Trading symbol
            signal_class: Signal classification
            direction: Trade direction (UP=LONG, DOWN=SHORT)
            z_er: Z-score of excess return
            btc_z_er: BTC z-score (for LONG filter)
            volume_z: Volume z-score

        Returns:
            Tuple of (passed, block_reason) - passed=True means trade allowed
        """
        df_cfg = self.pm_cfg.direction_filters

        if not df_cfg.enabled:
            return True, None

        # SHORT positions: use standard thresholds
        if direction == Direction.DOWN:
            if signal_class == SignalClass.EXTREME_SPIKE:
                min_z = df_cfg.short_min_extreme_spike_z
            else:
                min_z = df_cfg.short_min_strong_signal_z

            if abs(z_er) < min_z:
                return False, f"SHORT z-score too low ({abs(z_er):.2f} < {min_z})"

            return True, None

        # LONG positions: stricter requirements

        # Check if EARLY_SIGNAL longs are disabled
        if df_cfg.long_disable_for_early_signal and signal_class == SignalClass.EARLY_SIGNAL:
            return False, "LONG disabled for EARLY_SIGNAL class"

        # Check minimum z-score (higher for longs)
        if signal_class == SignalClass.EXTREME_SPIKE:
            min_z = df_cfg.long_min_extreme_spike_z
        else:
            min_z = df_cfg.long_min_strong_signal_z

        if abs(z_er) < min_z:
            return False, f"LONG z-score too low ({abs(z_er):.2f} < {min_z})"

        # BTC filter for LONG
        if df_cfg.long_btc_filter_enabled and btc_z_er is not None:
            if btc_z_er < df_cfg.long_btc_block_threshold:
                return False, f"LONG blocked: BTC z={btc_z_er:.2f} < {df_cfg.long_btc_block_threshold}"

            if btc_z_er < df_cfg.long_btc_restrict_threshold:
                # Restrict to EXTREME_SPIKE only
                if signal_class != SignalClass.EXTREME_SPIKE:
                    return False, f"LONG restricted: BTC z={btc_z_er:.2f}, only EXTREME_SPIKE allowed"

        return True, None

    # =========================================================================
    # IMPROVEMENT 5: Trailing Stop by Class
    # =========================================================================

    def get_trailing_params_for_class(
        self,
        signal_class: Optional[SignalClass]
    ) -> Tuple[float, float]:
        """
        Get trailing stop parameters based on signal class.

        Returns:
            Tuple of (profit_threshold_pct, distance_atr)
        """
        tsc_cfg = self.pm_cfg.trailing_stop_by_class

        if not tsc_cfg.enabled:
            # Use legacy params
            return (
                self.pm_cfg.trailing_stop_activation * 100,  # Convert to %
                self.pm_cfg.trailing_stop_distance_atr
            )

        if signal_class == SignalClass.EXTREME_SPIKE:
            return (
                tsc_cfg.extreme_spike_profit_threshold_pct,
                tsc_cfg.extreme_spike_distance_atr
            )
        elif signal_class == SignalClass.STRONG_SIGNAL:
            return (
                tsc_cfg.strong_signal_profit_threshold_pct,
                tsc_cfg.strong_signal_distance_atr
            )
        elif signal_class == SignalClass.EARLY_SIGNAL:
            return (
                tsc_cfg.early_signal_profit_threshold_pct,
                tsc_cfg.early_signal_distance_atr
            )
        else:
            # Fallback
            return (
                tsc_cfg.strong_signal_profit_threshold_pct,
                tsc_cfg.strong_signal_distance_atr
            )

    def check_trailing_stop_activation(
        self,
        position: Position,
        current_price: float,
        current_ts: int
    ) -> bool:
        """
        Check if trailing stop should be activated.

        Returns:
            True if trailing was just activated
        """
        tsc_cfg = self.pm_cfg.trailing_stop_by_class

        if not tsc_cfg.enabled:
            return False

        if position.trailing_active:
            return False  # Already active

        # Get activation threshold for class
        signal_class = None
        if position.signal_class:
            try:
                signal_class = SignalClass(position.signal_class)
            except ValueError:
                pass

        profit_threshold, distance_atr = self.get_trailing_params_for_class(signal_class)

        # Calculate current profit %
        direction_mult = 1 if position.direction == Direction.UP else -1
        current_pnl = ((current_price - position.open_price) / position.open_price * 100) * direction_mult

        # Check if profit threshold reached
        if current_pnl >= profit_threshold:
            position.trailing_active = True
            position.trailing_activation_profit = current_pnl
            position.trailing_distance_atr = distance_atr

            # Initialize trailing price
            atr = self.extended_features.get_atr(position.symbol)
            if atr:
                if position.direction == Direction.UP:
                    position.trailing_price = current_price - (atr * distance_atr)
                else:
                    position.trailing_price = current_price + (atr * distance_atr)

            logger.info(
                f"{position.symbol}: Trailing stop activated | "
                f"PnL={current_pnl:.2f}% >= {profit_threshold:.2f}% | "
                f"trail_price={position.trailing_price:.6f}"
            )
            return True

        return False

    def update_and_check_trailing_stop(
        self,
        position: Position,
        current_price: float
    ) -> Optional[ExitReason]:
        """
        Update trailing stop and check if triggered.

        Returns:
            ExitReason.TRAILING_STOP if hit, None otherwise
        """
        if not position.trailing_active or position.trailing_price is None:
            return None

        atr = self.extended_features.get_atr(position.symbol)
        if atr is None:
            return None

        distance_atr = position.trailing_distance_atr or self.pm_cfg.trailing_stop_distance_atr

        if position.direction == Direction.UP:  # LONG
            # Update trail higher
            new_trail = current_price - (atr * distance_atr)
            if new_trail > position.trailing_price:
                position.trailing_price = new_trail

            # Check if hit
            if current_price <= position.trailing_price:
                logger.info(
                    f"{position.symbol}: Trailing stop HIT | "
                    f"price={current_price:.6f} <= trail={position.trailing_price:.6f}"
                )
                return ExitReason.TRAILING_STOP
        else:  # SHORT
            # Update trail lower
            new_trail = current_price + (atr * distance_atr)
            if new_trail < position.trailing_price:
                position.trailing_price = new_trail

            # Check if hit
            if current_price >= position.trailing_price:
                logger.info(
                    f"{position.symbol}: Trailing stop HIT | "
                    f"price={current_price:.6f} >= trail={position.trailing_price:.6f}"
                )
                return ExitReason.TRAILING_STOP

        return None

    # =========================================================================
    # IMPROVEMENT 6: Intelligent Time Exit
    # =========================================================================

    def check_time_exit(
        self,
        position: Position,
        current_price: float,
        current_ts: int,
        features: Optional['Features'] = None
    ) -> Optional[ExitReason]:
        """
        Check intelligent time-based exits.

        CRITICAL FIX v2: Added controls to disable aggressive exits (LOSING, FLAT)
        which were causing 38.5% of positions to close in loss.

        Checks in order (respecting grace period and enabled flags):
        1. Grace period check - skip aggressive exits if too early
        2. Losing position timeout (if aggressive_exits_enabled)
        3. Flat position timeout (if flat_position_exit_enabled)
        4. Max hold timeout (always checked)

        Args:
            position: Position to check
            current_price: Current market price
            current_ts: Current timestamp in ms
            features: Optional Features for conditional mode (z-score, flow checks)

        Returns:
            ExitReason if time exit triggered, None otherwise
        """
        te_cfg = self.pm_cfg.time_exit

        if not te_cfg.enabled:
            return None

        # Calculate current PnL and hold time
        direction_mult = 1 if position.direction == Direction.UP else -1
        current_pnl = ((current_price - position.open_price) / position.open_price * 100) * direction_mult
        hold_minutes = (current_ts - position.open_ts) // (60 * 1000)

        # Get thresholds for signal class
        signal_class_str = position.signal_class
        if signal_class_str == "EXTREME_SPIKE":
            losing_threshold = te_cfg.extreme_spike_losing_threshold_pct
            losing_max_min = te_cfg.extreme_spike_losing_max_minutes
            flat_threshold = te_cfg.extreme_spike_flat_threshold_pct
            flat_max_min = te_cfg.extreme_spike_flat_max_minutes
            max_hold = te_cfg.extreme_spike_max_hold_minutes
        elif signal_class_str == "STRONG_SIGNAL":
            losing_threshold = te_cfg.strong_signal_losing_threshold_pct
            losing_max_min = te_cfg.strong_signal_losing_max_minutes
            flat_threshold = te_cfg.strong_signal_flat_threshold_pct
            flat_max_min = te_cfg.strong_signal_flat_max_minutes
            max_hold = te_cfg.strong_signal_max_hold_minutes
        elif signal_class_str == "EARLY_SIGNAL":
            losing_threshold = te_cfg.early_signal_losing_threshold_pct
            losing_max_min = te_cfg.early_signal_losing_max_minutes
            flat_threshold = te_cfg.early_signal_flat_threshold_pct
            flat_max_min = te_cfg.early_signal_flat_max_minutes
            max_hold = te_cfg.early_signal_max_hold_minutes
        else:
            # Fallback: use STRONG_SIGNAL
            losing_threshold = te_cfg.strong_signal_losing_threshold_pct
            losing_max_min = te_cfg.strong_signal_losing_max_minutes
            flat_threshold = te_cfg.strong_signal_flat_threshold_pct
            flat_max_min = te_cfg.strong_signal_flat_max_minutes
            max_hold = te_cfg.strong_signal_max_hold_minutes

        # CRITICAL FIX v2: Grace period check
        # Don't apply aggressive exits during grace period
        grace_period = te_cfg.grace_period_minutes
        in_grace_period = hold_minutes < grace_period

        # CRITICAL FIX v2: Check aggressive_exits_enabled flag
        aggressive_exits_allowed = te_cfg.aggressive_exits_enabled and not in_grace_period

        # Check 1: Losing position timeout (only if aggressive exits enabled)
        if aggressive_exits_allowed:
            # CRITICAL FIX v2: Conditional mode - require additional confirmations
            if te_cfg.conditional_mode_enabled:
                # Only trigger if ALL conditions met
                loss_deep_enough = current_pnl < te_cfg.conditional_min_loss_pct
                time_long_enough = hold_minutes >= te_cfg.conditional_min_time_minutes

                # Check z-reversal if required and features available
                z_reversed = True  # Default to true if not checking
                if te_cfg.conditional_require_z_reversal and features is not None:
                    # For LONG: z should be negative (against us)
                    # For SHORT: z should be positive (against us)
                    if position.direction == Direction.UP:
                        z_reversed = features.z_er_15m < 0
                    else:
                        z_reversed = features.z_er_15m > 0

                # Check flow reversal if required
                flow_reversed = True  # Default to true if not checking
                if te_cfg.conditional_require_flow_reversal and features is not None:
                    # For LONG: taker buy share < 0.5 (sellers dominant)
                    # For SHORT: taker buy share > 0.5 (buyers dominant)
                    # Note: We'd need bar data for taker_buy_share, using features if available
                    pass  # Flow check requires bar data, skip for now

                if loss_deep_enough and time_long_enough and z_reversed and flow_reversed:
                    logger.info(
                        f"{position.symbol}: TIME_EXIT_LOSING (conditional) | "
                        f"PnL={current_pnl:.2f}% | z_reversed={z_reversed} | "
                        f"hold={hold_minutes}m"
                    )
                    return ExitReason.TIME_EXIT_LOSING
            else:
                # Standard aggressive losing exit
                if hold_minutes >= losing_max_min and current_pnl < losing_threshold:
                    logger.info(
                        f"{position.symbol}: TIME_EXIT_LOSING | "
                        f"PnL={current_pnl:.2f}% < {losing_threshold:.2f}% after {hold_minutes}m"
                    )
                    return ExitReason.TIME_EXIT_LOSING
        elif not te_cfg.aggressive_exits_enabled:
            logger.debug(
                f"{position.symbol}: TIME_EXIT_LOSING SKIPPED (aggressive_exits disabled)"
            )

        # Check 2: Flat position timeout (only if enabled)
        if te_cfg.flat_position_exit_enabled and not in_grace_period:
            if hold_minutes >= flat_max_min and abs(current_pnl) < flat_threshold:
                logger.info(
                    f"{position.symbol}: TIME_EXIT_FLAT | "
                    f"|PnL|={abs(current_pnl):.2f}% < {flat_threshold:.2f}% after {hold_minutes}m"
                )
                return ExitReason.TIME_EXIT_FLAT
        elif not te_cfg.flat_position_exit_enabled:
            logger.debug(
                f"{position.symbol}: TIME_EXIT_FLAT SKIPPED (flat_position_exit disabled)"
            )

        # Check 3: Max hold timeout (always checked, not affected by grace period)
        if hold_minutes >= max_hold:
            logger.info(
                f"{position.symbol}: TIME_EXIT (max hold) | "
                f"held for {hold_minutes}m >= {max_hold}m | PnL={current_pnl:.2f}%"
            )
            return ExitReason.TIME_EXIT

        return None

    # =========================================================================
    # IMPROVEMENT 7: Min Profit Filter
    # =========================================================================

    def check_min_profit_filter(
        self,
        symbol: str,
        signal_class: Optional[SignalClass],
        direction: Direction,
        entry_price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if expected profit is sufficient to cover fees.

        Returns:
            Tuple of (passed, block_reason) - passed=True means trade allowed
        """
        mpf_cfg = self.pm_cfg.min_profit_filter

        if not mpf_cfg.enabled:
            return True, None

        atr = self.extended_features.get_atr(symbol)
        if atr is None or atr <= 0:
            # Can't calculate expected profit without ATR
            logger.debug(f"{symbol}: Min profit filter skipped - no ATR")
            return True, None

        # Get TP1 level for expected profit calculation
        ttp_cfg = self.pm_cfg.tiered_take_profit

        if signal_class == SignalClass.EXTREME_SPIKE:
            tp1_atr = ttp_cfg.extreme_spike_tp1_atr
            min_profit = mpf_cfg.extreme_spike_min_profit_pct
        elif signal_class == SignalClass.STRONG_SIGNAL:
            tp1_atr = ttp_cfg.strong_signal_tp1_atr
            min_profit = mpf_cfg.strong_signal_min_profit_pct
        elif signal_class == SignalClass.EARLY_SIGNAL:
            tp1_atr = ttp_cfg.early_signal_tp1_atr
            min_profit = mpf_cfg.early_signal_min_profit_pct
        else:
            tp1_atr = ttp_cfg.strong_signal_tp1_atr
            min_profit = mpf_cfg.min_expected_profit_pct

        # Calculate expected TP1 profit
        expected_profit_pct = (atr * tp1_atr / entry_price) * 100

        # Subtract fees
        net_expected = expected_profit_pct - mpf_cfg.estimated_fees_pct

        if net_expected < min_profit:
            reason = (
                f"Expected profit too low: {expected_profit_pct:.2f}% - {mpf_cfg.estimated_fees_pct:.2f}% fees = "
                f"{net_expected:.2f}% < {min_profit:.2f}%"
            )
            logger.info(f"{symbol}: Min profit filter BLOCKED | {reason}")
            return False, reason

        logger.debug(
            f"{symbol}: Min profit filter PASSED | "
            f"expected={expected_profit_pct:.2f}% - fees={mpf_cfg.estimated_fees_pct:.2f}% = "
            f"{net_expected:.2f}% >= {min_profit:.2f}%"
        )
        return True, None

    # =========================================================================
    # CRITICAL FIXES V3: New methods for profitability improvements
    # =========================================================================

    def check_entry_rr_filter(
        self,
        symbol: str,
        signal_class: Optional[SignalClass],
        direction: Direction,
        entry_price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Critical Fix 5: Entry R:R Filter.
        Check if risk/reward ratio is sufficient to allow entry.

        Uses the configured expected_tp level for R:R calculation:
        - "tp1": TP1 only (partial close target) - too strict with tiered TP
        - "tp3": TP3 (full profit target) - recommended with tiered TP
        - "weighted": Weighted average across all TP levels

        Returns:
            Tuple of (passed, block_reason) - passed=True means trade allowed
        """
        ef_cfg = self.pm_cfg.entry_filters.min_rr_filter

        if not ef_cfg.enabled:
            return True, None

        atr = self.extended_features.get_atr(symbol)
        if atr is None or atr <= 0:
            logger.debug(f"{symbol}: Entry R:R filter skipped - no ATR available")
            return True, None

        ttp_cfg = self.pm_cfg.tiered_take_profit
        asl_cfg = self.pm_cfg.adaptive_stop_loss

        # Determine TP and SL ATR multipliers by signal class
        if signal_class == SignalClass.EXTREME_SPIKE:
            tp1_atr = ttp_cfg.extreme_spike_tp1_atr
            tp2_atr = ttp_cfg.extreme_spike_tp2_atr
            tp3_atr = ttp_cfg.extreme_spike_tp3_atr
            tp1_pct = ttp_cfg.extreme_spike_tp1_close_pct
            tp2_pct = ttp_cfg.extreme_spike_tp2_close_pct
            tp3_pct = ttp_cfg.extreme_spike_tp3_close_pct
            sl_atr = asl_cfg.base_multiplier_extreme_spike
            min_rr = ef_cfg.extreme_spike_min_rr
        elif signal_class == SignalClass.STRONG_SIGNAL:
            tp1_atr = ttp_cfg.strong_signal_tp1_atr
            tp2_atr = ttp_cfg.strong_signal_tp2_atr
            tp3_atr = ttp_cfg.strong_signal_tp3_atr
            tp1_pct = ttp_cfg.strong_signal_tp1_close_pct
            tp2_pct = ttp_cfg.strong_signal_tp2_close_pct
            tp3_pct = ttp_cfg.strong_signal_tp3_close_pct
            sl_atr = asl_cfg.base_multiplier_strong_signal
            min_rr = ef_cfg.strong_signal_min_rr
        elif signal_class == SignalClass.EARLY_SIGNAL:
            tp1_atr = ttp_cfg.early_signal_tp1_atr
            tp2_atr = ttp_cfg.early_signal_tp2_atr
            tp3_atr = ttp_cfg.early_signal_tp3_atr
            tp1_pct = ttp_cfg.early_signal_tp1_close_pct
            tp2_pct = ttp_cfg.early_signal_tp2_close_pct
            tp3_pct = ttp_cfg.early_signal_tp3_close_pct
            sl_atr = asl_cfg.base_multiplier_early_signal
            min_rr = ef_cfg.early_signal_min_rr
        else:
            tp1_atr = ttp_cfg.strong_signal_tp1_atr
            tp2_atr = ttp_cfg.strong_signal_tp2_atr
            tp3_atr = ttp_cfg.strong_signal_tp3_atr
            tp1_pct = ttp_cfg.strong_signal_tp1_close_pct
            tp2_pct = ttp_cfg.strong_signal_tp2_close_pct
            tp3_pct = ttp_cfg.strong_signal_tp3_close_pct
            sl_atr = asl_cfg.base_multiplier_strong_signal
            min_rr = ef_cfg.min_rr_ratio

        if sl_atr <= 0:
            logger.debug(f"{symbol}: Entry R:R filter skipped - invalid SL multiplier")
            return True, None

        # Select TP level for R:R calculation based on config
        tp_mode = ef_cfg.expected_tp
        if tp_mode == "tp3":
            # Use full profit target (recommended with tiered TP)
            tp_atr = tp3_atr
            tp_label = "TP3"
        elif tp_mode == "weighted":
            # Weighted average across all TP levels
            total_pct = tp1_pct + tp2_pct + tp3_pct
            if total_pct > 0:
                tp_atr = (tp1_atr * tp1_pct + tp2_atr * tp2_pct + tp3_atr * tp3_pct) / total_pct
            else:
                tp_atr = tp3_atr
            tp_label = "TP_weighted"
        else:
            # Default: tp1 (original behavior)
            tp_atr = tp1_atr
            tp_label = "TP1"

        rr_ratio = tp_atr / sl_atr

        if rr_ratio < min_rr:
            reason = (
                f"R:R too low: {tp_label}={tp_atr:.2f}xATR / SL={sl_atr:.2f}xATR = "
                f"{rr_ratio:.2f} < min {min_rr:.2f}"
            )
            logger.info(f"{symbol}: Entry R:R filter BLOCKED | {reason}")
            return False, reason

        logger.debug(
            f"{symbol}: Entry R:R filter PASSED | "
            f"{tp_label}={tp_atr:.2f}xATR / SL={sl_atr:.2f}xATR = {rr_ratio:.2f} >= {min_rr:.2f}"
        )
        return True, None

    def check_min_atr_filter(
        self,
        symbol: str,
        entry_price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Critical Fix 5b: Minimum ATR Filter.
        Check if volatility is sufficient for profitable trades.

        Returns:
            Tuple of (passed, block_reason) - passed=True means trade allowed
        """
        atr_cfg = self.pm_cfg.entry_filters.min_atr_filter

        if not atr_cfg.enabled:
            return True, None

        atr = self.extended_features.get_atr(symbol)
        if atr is None:
            logger.debug(f"{symbol}: Min ATR filter skipped - no ATR available")
            return True, None

        atr_pct = (atr / entry_price) * 100

        if atr_pct < atr_cfg.min_atr_pct:
            reason = (
                f"ATR too low: {atr_pct:.3f}% < min {atr_cfg.min_atr_pct:.2f}%"
            )
            logger.info(f"{symbol}: Min ATR filter BLOCKED | {reason}")
            return False, reason

        logger.debug(
            f"{symbol}: Min ATR filter PASSED | ATR={atr_pct:.3f}% >= {atr_cfg.min_atr_pct:.2f}%"
        )
        return True, None

    def check_max_loss_cap(
        self,
        position: Position,
        current_price: float
    ) -> Optional[ExitReason]:
        """
        Critical Fix 6: Max Loss Cap per Position.
        Check if position has hit maximum loss cap.

        This is the HIGHEST PRIORITY exit check - fires before any other exit.

        Returns:
            ExitReason.MAX_LOSS_CAP if cap hit, None otherwise
        """
        rm_cfg = self.pm_cfg.risk_management.max_loss_per_position

        if not rm_cfg.enabled:
            return None

        # Calculate current PnL
        direction_multiplier = 1 if position.direction == Direction.UP else -1
        pnl_pct = ((current_price - position.open_price) / position.open_price * 100) * direction_multiplier

        # Check if max loss cap hit
        if pnl_pct <= -rm_cfg.max_loss_pct:
            logger.info(
                f"{position.symbol}: MAX LOSS CAP triggered | "
                f"PnL={pnl_pct:.2f}% <= -{rm_cfg.max_loss_pct:.2f}% | "
                f"Action: {rm_cfg.action}"
            )
            return ExitReason.MAX_LOSS_CAP

        return None

    def check_z_exit_with_pnl_conditions(
        self,
        position: Position,
        current_z_er: float,
        current_price: float,
        current_ts: int
    ) -> Optional[Tuple[ExitReason, int]]:
        """
        Critical Fix 4: Z-Score Exit with PnL Conditions.
        Only allow z-exit when position is profitable.

        Returns:
            Tuple of (ExitReason, close_percent) if exit should happen, None otherwise
        """
        ze_cfg = self.pm_cfg.z_score_exit

        if not ze_cfg.enabled:
            return None

        # Get signal class threshold
        signal_class_str = position.signal_class
        if signal_class_str == 'EXTREME_SPIKE':
            z_threshold = ze_cfg.extreme_spike_threshold
        elif signal_class_str == 'STRONG_SIGNAL':
            z_threshold = ze_cfg.strong_signal_threshold
        elif signal_class_str == 'EARLY_SIGNAL':
            z_threshold = ze_cfg.early_signal_threshold
        else:
            z_threshold = ze_cfg.strong_signal_threshold

        # Check if z-score has reversed (below threshold)
        z_reversed = abs(current_z_er) < z_threshold

        if not z_reversed:
            # Z-score still strong, no exit
            return None

        # Z-score reversed - now check PnL conditions
        direction_multiplier = 1 if position.direction == Direction.UP else -1
        pnl_pct = ((current_price - position.open_price) / position.open_price * 100) * direction_multiplier

        # Check if position is profitable enough for full exit
        if pnl_pct >= ze_cfg.min_pnl_for_full_exit:
            close_pct = 100
            logger.info(
                f"{position.symbol}: Z-exit FULL | z={current_z_er:.2f} < {z_threshold:.2f} | "
                f"PnL={pnl_pct:.2f}% >= {ze_cfg.min_pnl_for_full_exit:.2f}%"
            )
            return (ExitReason.Z_SCORE_REVERSAL, close_pct)

        # Check if profitable enough for partial exit
        if ze_cfg.partial_close_enabled and pnl_pct >= ze_cfg.min_pnl_for_partial_exit:
            close_pct = ze_cfg.partial_close_percent
            logger.info(
                f"{position.symbol}: Z-exit PARTIAL ({close_pct}%) | z={current_z_er:.2f} < {z_threshold:.2f} | "
                f"PnL={pnl_pct:.2f}% >= {ze_cfg.min_pnl_for_partial_exit:.2f}%"
            )
            return (ExitReason.Z_SCORE_REVERSAL, close_pct)

        # Position is losing - check behavior configuration
        if ze_cfg.require_positive_pnl:
            # Track when z-reversal started (for max_additional_hold_minutes)
            z_reversal_start = position.metrics.get('z_reversal_start_ts')

            if z_reversal_start is None:
                # First time z-reversed while losing - record timestamp
                position.metrics['z_reversal_start_ts'] = current_ts
                logger.debug(
                    f"{position.symbol}: Z-reversal detected while losing | "
                    f"PnL={pnl_pct:.2f}% | Holding for max {ze_cfg.max_additional_hold_minutes}m"
                )
                return None  # Hold position

            # Check if we've exceeded max additional hold time
            hold_minutes = (current_ts - z_reversal_start) // (60 * 1000)

            if hold_minutes >= ze_cfg.max_additional_hold_minutes:
                # Max hold time exceeded - fall back to trailing or SL
                logger.info(
                    f"{position.symbol}: Z-exit hold period expired | "
                    f"held {hold_minutes}m >= max {ze_cfg.max_additional_hold_minutes}m | "
                    f"Falling back to {ze_cfg.fallback_exit}"
                )
                # Don't return z-exit, let other exit conditions handle it
                return None

            logger.debug(
                f"{position.symbol}: Z-reversal hold | "
                f"PnL={pnl_pct:.2f}% | held {hold_minutes}m / {ze_cfg.max_additional_hold_minutes}m"
            )
            return None  # Continue holding

        # Not requiring positive PnL - just use standard behavior
        return None

    def calculate_trailing_stop_v3(
        self,
        position: Position,
        current_price: float
    ) -> Optional[float]:
        """
        Critical Fix 3: Earlier and Tighter Trailing Stop.
        Calculate class-aware trailing stop with accelerated mode.

        Returns:
            New trailing stop price, or None if not activated
        """
        ts_cfg = self.pm_cfg.trailing_stop_v3

        if not ts_cfg.enabled:
            return None

        # Get ATR for distance calculation
        atr = self.extended_features.get_atr(position.symbol)
        if atr is None or atr <= 0:
            return None

        # Get current PnL for activation check
        direction_multiplier = 1 if position.direction == Direction.UP else -1
        pnl_pct = ((current_price - position.open_price) / position.open_price * 100) * direction_multiplier

        # Get class-specific settings
        signal_class_str = position.signal_class
        if signal_class_str == 'EXTREME_SPIKE':
            profit_threshold = ts_cfg.extreme_spike_profit_threshold_pct
            distance_atr = ts_cfg.extreme_spike_distance_atr
        elif signal_class_str == 'STRONG_SIGNAL':
            profit_threshold = ts_cfg.strong_signal_profit_threshold_pct
            distance_atr = ts_cfg.strong_signal_distance_atr
        elif signal_class_str == 'EARLY_SIGNAL':
            profit_threshold = ts_cfg.early_signal_profit_threshold_pct
            distance_atr = ts_cfg.early_signal_distance_atr
        else:
            profit_threshold = ts_cfg.strong_signal_profit_threshold_pct
            distance_atr = ts_cfg.strong_signal_distance_atr

        # Check if position is profitable enough to activate trailing
        if pnl_pct < profit_threshold:
            return None

        # Check accelerated trailing
        accel_cfg = ts_cfg.accelerated_trailing
        if accel_cfg.enabled and pnl_pct >= accel_cfg.trigger_profit_pct:
            # Apply tighter distance
            distance_atr *= accel_cfg.tighter_distance_multiplier
            logger.debug(
                f"{position.symbol}: Accelerated trailing active | "
                f"PnL={pnl_pct:.2f}% >= {accel_cfg.trigger_profit_pct:.2f}% | "
                f"distance reduced by {(1-accel_cfg.tighter_distance_multiplier)*100:.0f}%"
            )

        # Calculate trailing stop price
        trail_distance = atr * distance_atr

        if position.direction == Direction.UP:
            # For longs, stop trails below price
            new_stop = current_price - trail_distance
            # Only update if higher than current stop
            current_stop = position.metrics.get('trailing_stop_v3_price')
            if current_stop is not None and new_stop <= current_stop:
                return None  # Don't lower the stop
        else:
            # For shorts, stop trails above price
            new_stop = current_price + trail_distance
            # Only update if lower than current stop
            current_stop = position.metrics.get('trailing_stop_v3_price')
            if current_stop is not None and new_stop >= current_stop:
                return None  # Don't raise the stop

        logger.debug(
            f"{position.symbol}: Trailing stop v3 updated | "
            f"PnL={pnl_pct:.2f}% | distance={distance_atr:.2f}xATR | "
            f"new_stop={new_stop:.6f}"
        )

        return new_stop

    def check_trailing_stop_v3_hit(
        self,
        position: Position,
        current_price: float
    ) -> Optional[ExitReason]:
        """
        Check if trailing stop v3 has been hit.

        Returns:
            ExitReason.TRAILING_STOP_V3 if hit, None otherwise
        """
        ts_cfg = self.pm_cfg.trailing_stop_v3

        if not ts_cfg.enabled:
            return None

        # Get current trailing stop price
        trailing_stop_price = position.metrics.get('trailing_stop_v3_price')
        if trailing_stop_price is None:
            return None

        # Check if hit
        if position.direction == Direction.UP:
            if current_price <= trailing_stop_price:
                logger.info(
                    f"{position.symbol}: Trailing stop v3 HIT | "
                    f"price={current_price:.6f} <= stop={trailing_stop_price:.6f}"
                )
                return ExitReason.TRAILING_STOP_V3
        else:
            if current_price >= trailing_stop_price:
                logger.info(
                    f"{position.symbol}: Trailing stop v3 HIT | "
                    f"price={current_price:.6f} >= stop={trailing_stop_price:.6f}"
                )
                return ExitReason.TRAILING_STOP_V3

        return None

    def update_trailing_stop_v3(
        self,
        position: Position,
        current_price: float
    ) -> None:
        """
        Update trailing stop v3 price for position.
        Called on every bar.
        """
        new_stop = self.calculate_trailing_stop_v3(position, current_price)
        if new_stop is not None:
            position.metrics['trailing_stop_v3_price'] = new_stop
            position.metrics['trailing_stop_v3_active'] = True
