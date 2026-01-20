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
        current_ts: int
    ) -> Optional[ExitReason]:
        """
        Check intelligent time-based exits.

        Checks in order:
        1. Losing position timeout
        2. Flat position timeout
        3. Max hold timeout

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

        # Check 1: Losing position timeout
        if hold_minutes >= losing_max_min and current_pnl < losing_threshold:
            logger.info(
                f"{position.symbol}: TIME_EXIT_LOSING | "
                f"PnL={current_pnl:.2f}% < {losing_threshold:.2f}% after {hold_minutes}m"
            )
            return ExitReason.TIME_EXIT_LOSING

        # Check 2: Flat position timeout
        if hold_minutes >= flat_max_min and abs(current_pnl) < flat_threshold:
            logger.info(
                f"{position.symbol}: TIME_EXIT_FLAT | "
                f"|PnL|={abs(current_pnl):.2f}% < {flat_threshold:.2f}% after {hold_minutes}m"
            )
            return ExitReason.TIME_EXIT_FLAT

        # Check 3: Max hold timeout
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
