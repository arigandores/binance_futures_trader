"""Extended features for position management - ATR and Order Flow metrics."""

import numpy as np
import logging
from typing import List, Optional
from collections import deque
from detector.models import Bar

logger = logging.getLogger(__name__)


class ExtendedFeatureCalculator:
    """
    Calculates extended features for position exit decisions:
    - ATR (Average True Range) for dynamic stops
    - Order Flow changes (taker buy/sell ratio shifts)
    """

    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period

        # Rolling windows per symbol
        self.bars_windows = {}  # symbol -> deque of bars

    def update(self, bar: Bar) -> dict:
        """
        Update rolling windows and calculate extended features.

        Returns dict with:
        - atr: Average True Range
        - taker_buy_share: Current taker buy share
        - taker_flow_delta: Change in taker flow vs previous bar
        """
        symbol = bar.symbol

        # Initialize window if needed
        if symbol not in self.bars_windows:
            self.bars_windows[symbol] = deque(maxlen=self.atr_period)

        # Add current bar
        self.bars_windows[symbol].append(bar)

        # Calculate features
        atr = self._calculate_atr(symbol)
        taker_share = bar.taker_buy_share()
        taker_delta = self._calculate_taker_flow_delta(symbol)

        return {
            'atr': atr,
            'taker_buy_share': taker_share,
            'taker_flow_delta': taker_delta
        }

    def _calculate_atr(self, symbol: str) -> Optional[float]:
        """
        Calculate Average True Range.

        ATR = Average of True Range over period
        True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        """
        bars = list(self.bars_windows.get(symbol, []))

        if len(bars) < 2:
            return None

        true_ranges = []

        for i in range(1, len(bars)):
            current = bars[i]
            previous = bars[i - 1]

            high_low = current.high - current.low
            high_close = abs(current.high - previous.close)
            low_close = abs(current.low - previous.close)

            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)

        if not true_ranges:
            return None

        # Calculate ATR as simple moving average of TR
        atr = np.mean(true_ranges)
        return float(atr)

    def _calculate_taker_flow_delta(self, symbol: str) -> Optional[float]:
        """
        Calculate change in taker buy share compared to previous bar.

        Positive delta = increasing buying pressure
        Negative delta = increasing selling pressure
        """
        bars = list(self.bars_windows.get(symbol, []))

        if len(bars) < 2:
            return None

        current_share = bars[-1].taker_buy_share()
        previous_share = bars[-2].taker_buy_share()

        if current_share is None or previous_share is None:
            return None

        return current_share - previous_share

    def get_atr(self, symbol: str) -> Optional[float]:
        """
        Get current ATR (Average True Range) for a symbol.

        This is a public wrapper for _calculate_atr().

        Args:
            symbol: Trading symbol

        Returns:
            ATR value or None if insufficient data
        """
        return self._calculate_atr(symbol)

    def get_atr_multiple(self, symbol: str, price: float, multiplier: float = 1.0) -> Optional[float]:
        """
        Get ATR as a multiple of current price (for percentage-based stops).

        Returns ATR * multiplier as percentage of price.
        """
        atr = self._calculate_atr(symbol)
        if atr is None or price == 0:
            return None

        return (atr * multiplier / price) * 100

    def get_recent_price_peak(self, symbol: str, direction, lookback_bars: int = 5) -> Optional[float]:
        """
        Get highest (for UP) or lowest (for DOWN) price in recent bars.
        Used to calculate pullback from peak.

        Args:
            symbol: Trading symbol
            direction: Direction.UP or Direction.DOWN
            lookback_bars: Number of bars to look back

        Returns:
            Peak price (high for UP, low for DOWN) or None if insufficient data
        """
        from detector.models import Direction

        bars = list(self.bars_windows.get(symbol, []))

        if len(bars) < 2:
            return None

        recent_bars = bars[-lookback_bars:] if len(bars) >= lookback_bars else bars

        if direction == Direction.UP:
            return max(bar.high for bar in recent_bars)
        else:
            return min(bar.low for bar in recent_bars)

    def get_taker_flow_stability(self, symbol: str, lookback_bars: int = 3) -> Optional[float]:
        """
        Calculate max absolute change in taker buy share over last N bars.
        Returns max change - lower values indicate more stable flow.

        Args:
            symbol: Trading symbol
            lookback_bars: Number of bars to analyze

        Returns:
            Maximum absolute change in taker buy share, or None if insufficient data
        """
        bars = list(self.bars_windows.get(symbol, []))

        if len(bars) < lookback_bars + 1:
            return None

        recent_bars = bars[-(lookback_bars + 1):]
        shares = [bar.taker_buy_share() for bar in recent_bars]

        if None in shares:
            return None

        changes = [abs(shares[i] - shares[i-1]) for i in range(1, len(shares))]
        return max(changes) if changes else None

    def get_atr_percentile(self, symbol: str, current_atr: float, percentile_window: int = 100) -> Optional[float]:
        """
        Calculate what percentile the current ATR is at (for volatility regime detection).
        Returns value 0-100.

        Note: Simplified implementation for now - returns None.
        Can be enhanced later with historical ATR tracking.
        """
        # Would need to track historical ATR values
        # Placeholder for future enhancement
        return None

    def calculate_dynamic_targets(
        self,
        symbol: str,
        entry_price: float,
        direction,
        atr_stop_mult: float = 1.5,
        atr_target_mult: float = 3.0,
        min_risk_reward: float = 2.0
    ) -> Optional[dict]:
        """
        Calculate stop loss and take profit based on ATR with minimum R:R ratio.

        Args:
            symbol: Trading symbol
            entry_price: Position entry price
            direction: Direction.UP or Direction.DOWN
            atr_stop_mult: ATR multiplier for stop loss (default 1.5)
            atr_target_mult: ATR multiplier for take profit (default 3.0)
            min_risk_reward: Minimum risk/reward ratio (default 2.0)

        Returns:
            Dict with stop_loss_price, take_profit_price, stop_loss_percent,
            take_profit_percent, risk_reward_ratio, or None if ATR unavailable
        """
        from detector.models import Direction

        atr = self._calculate_atr(symbol)

        if atr is None or entry_price == 0:
            # Fallback to fixed percentages not available here
            return None

        # Calculate stop and target distances
        stop_distance = atr * atr_stop_mult
        target_distance = atr * atr_target_mult

        # Ensure minimum risk/reward ratio
        if target_distance / stop_distance < min_risk_reward:
            target_distance = stop_distance * min_risk_reward

        # Calculate prices and percentages based on direction
        if direction == Direction.UP:
            stop_loss_price = entry_price - stop_distance
            take_profit_price = entry_price + target_distance
        else:
            stop_loss_price = entry_price + stop_distance
            take_profit_price = entry_price - target_distance

        stop_loss_pct = abs(stop_distance / entry_price) * 100
        take_profit_pct = abs(target_distance / entry_price) * 100
        risk_reward = target_distance / stop_distance

        return {
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'stop_loss_percent': stop_loss_pct,
            'take_profit_percent': take_profit_pct,
            'risk_reward_ratio': risk_reward
        }

    def get_bar_return(self, symbol: str) -> Optional[float]:
        """
        Calculate return of current bar (close - open) / open.
        Used for micro impulse detection in WIN_RATE_MAX profile.

        Returns:
            Bar return as decimal (e.g., 0.005 = +0.5%), or None if no data
        """
        bars = list(self.bars_windows.get(symbol, []))
        if not bars:
            return None

        current = bars[-1]
        if current.open == 0:
            return None

        return (current.close - current.open) / current.open

    def get_flow_acceleration_bars(self, symbol: str, lookback: int = 2) -> Optional[List[float]]:
        """
        Get taker buy share for last N bars (for flow acceleration check).

        Args:
            symbol: Trading symbol
            lookback: Number of bars to retrieve

        Returns:
            List of taker_buy_share values (oldest to newest), or None if insufficient data
        """
        bars = list(self.bars_windows.get(symbol, []))
        if len(bars) < lookback + 1:
            return None

        recent_bars = bars[-(lookback + 1):]
        shares = [bar.taker_buy_share() for bar in recent_bars]

        if None in shares:
            return None

        return shares
