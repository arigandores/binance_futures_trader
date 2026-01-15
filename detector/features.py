"""Feature calculator with rolling windows and z-scores."""

import asyncio
import logging
import math
import numpy as np
from collections import deque
from typing import Dict, List, Optional
from detector.models import Bar, Features, Direction
from detector.storage import Storage
from detector.config import Config
from detector.binance_rest import BinanceRestClient

logger = logging.getLogger(__name__)


class RollingWindow:
    """Manages rolling data for one symbol."""

    def __init__(self, maxlen: int = 720):
        self.maxlen = maxlen

        # OHLCV
        self.closes = deque(maxlen=maxlen)
        self.volumes = deque(maxlen=maxlen)

        # Taker buy/sell
        self.taker_buys = deque(maxlen=maxlen)
        self.taker_sells = deque(maxlen=maxlen)

        # Derived features
        self.returns_1m = deque(maxlen=maxlen)
        self.excess_returns = deque(maxlen=maxlen)

    def append(self, bar: Bar) -> None:
        """Append new bar to window."""
        self.closes.append(bar.close)
        self.volumes.append(bar.volume)
        self.taker_buys.append(bar.taker_buy)
        self.taker_sells.append(bar.taker_sell)

    def is_ready(self, min_bars: int = 15) -> bool:
        """Check if window has enough data."""
        return len(self.closes) >= min_bars


class FeatureCalculator:
    """
    Calculates rolling features and z-scores.

    Implements:
    - Returns (1m, 15m)
    - Beta vs BTC (on 5m aggregated returns)
    - Excess returns
    - Robust z-scores (MAD-based)
    - Taker buy share
    - OI delta and liquidation features (if available)
    """

    def __init__(
        self,
        bar_queue: asyncio.Queue,
        feature_queue: asyncio.Queue,
        storage: Storage,
        rest_client: BinanceRestClient,
        config: Config,
        extra_feature_queues: list = None
    ):
        self.bar_queue = bar_queue
        self.feature_queue = feature_queue
        self.extra_feature_queues = extra_feature_queues or []
        self.storage = storage
        self.rest_client = rest_client
        self.config = config

        # Rolling windows per symbol
        self.windows: Dict[str, RollingWindow] = {}

    async def backfill(self) -> None:
        """Load ALL bars from DB, populate windows, and write features to database."""
        logger.info("Backfilling rolling windows from database...")

        # First pass: Load ALL bars into windows (not just 720)
        # This is needed to calculate features for all historical data
        all_bars_by_symbol = {}

        for symbol in self.config.universe.all_symbols:
            # Load ALL bars from database (no limit)
            bars = await self.storage.get_recent_bars(symbol, limit=10000)  # Large limit to get all data
            all_bars_by_symbol[symbol] = bars

            window = RollingWindow(maxlen=720)
            # Only keep last 720 in rolling window for memory efficiency
            for bar in bars[-720:]:
                window.append(bar)

            self.windows[symbol] = window
            logger.debug(f"Loaded {len(bars)} bars for {symbol} (keeping {len(bars[-720:])} in rolling window)")

        # Second pass: Calculate and WRITE features for ALL historical bars
        logger.info("Calculating features for all historical data...")

        btc_bars = all_bars_by_symbol.get(self.config.universe.benchmark_symbol, [])
        if len(btc_bars) < 15:
            logger.warning("Insufficient BTC data for feature calculation")
            return

        # Create temporary extended windows that hold ALL bars (for backfill calculation)
        extended_windows = {}
        for symbol in self.config.universe.all_symbols:
            bars = all_bars_by_symbol.get(symbol, [])
            if len(bars) < 15:
                continue

            extended_window = RollingWindow(maxlen=len(bars) + 100)  # Large enough for all bars
            for bar in bars:
                extended_window.append(bar)
            extended_windows[symbol] = extended_window

        # BTC extended window
        btc_extended_window = extended_windows.get(self.config.universe.benchmark_symbol)
        if not btc_extended_window:
            logger.warning("No BTC extended window available")
            return

        # Calculate features for each symbol
        total_features_written = 0

        for symbol in self.config.universe.all_symbols:
            bars = all_bars_by_symbol.get(symbol, [])
            extended_window = extended_windows.get(symbol)

            if not extended_window or len(bars) < 15:
                continue

            features_batch = []

            # Calculate features for each historical bar (starting from bar 15)
            for i in range(15, len(bars)):
                bar = bars[i]

                # Calculate returns
                r_1m = self._calculate_return_at_index(extended_window.closes, i, lookback=1)
                r_15m = self._calculate_return_at_index(extended_window.closes, i, lookback=15)

                # Calculate beta
                beta = self._calculate_beta_at_index_extended(symbol, i, extended_windows)

                # Calculate BTC 15m return
                btc_r_15m = self._calculate_return_at_index(btc_extended_window.closes, i, lookback=15)

                # Calculate excess return
                er_15m = r_15m - beta * btc_r_15m

                # Store excess return in extended window for z-score calculation
                extended_window.excess_returns.append(er_15m)

                # Calculate z-scores
                z_er_15m = self._robust_zscore(list(extended_window.excess_returns))
                z_vol_15m = self._robust_zscore(list(extended_window.volumes)[:i+1])

                # Volume sum over last 15 bars
                vol_15m = sum(list(extended_window.volumes)[max(0, i-14):i+1])

                # Taker buy share
                taker_buy_share_15m = self._calculate_taker_share_at_index(extended_window, i, lookback=15)

                # Create features object
                features = Features(
                    symbol=symbol,
                    ts_minute=bar.ts_minute,
                    r_1m=r_1m,
                    r_15m=r_15m,
                    beta=beta,
                    er_15m=er_15m,
                    z_er_15m=z_er_15m,
                    vol_15m=vol_15m,
                    z_vol_15m=z_vol_15m,
                    taker_buy_share_15m=taker_buy_share_15m,
                    funding_rate=bar.funding
                )

                features.direction = features.determine_direction()
                features_batch.append(features)

            # Write features to database in batches
            if features_batch:
                await self.storage.batch_write_features(features_batch)
                await self.storage.flush_all()
                total_features_written += len(features_batch)
                logger.info(f"Wrote {len(features_batch)} features for {symbol}")

        # Now populate the actual rolling windows with pre-calculated excess returns
        logger.info("Populating rolling windows with pre-calculated data...")

        for symbol in self.config.universe.all_symbols:
            extended_window = extended_windows.get(symbol)
            window = self.windows.get(symbol)

            if not extended_window or not window:
                continue

            # Copy last 720 excess returns to the actual rolling window
            if len(extended_window.excess_returns) > 0:
                for er in list(extended_window.excess_returns)[-720:]:
                    window.excess_returns.append(er)

        logger.info(f"Backfill complete: {total_features_written} features written to database for {len(self.windows)} symbols")

    async def run(self) -> None:
        """Consume bars and calculate features."""
        logger.info("Feature calculator started")

        bar_count = 0

        while True:
            try:
                bar = await self.bar_queue.get()

                # Write bar to storage buffer
                await self.storage.batch_write_bars([bar])

                features = await self.calculate_features(bar)

                if features:
                    # Write features to storage buffer
                    await self.storage.batch_write_features([features])
                    await self.feature_queue.put(features)

                    # Broadcast to extra queues (e.g. position manager)
                    for queue in self.extra_feature_queues:
                        try:
                            await queue.put(features)
                        except Exception as e:
                            logger.error(f"Error broadcasting features to extra queue: {e}")

                    bar_count += 1

                    # Log every 10 features calculated
                    if bar_count % 10 == 0:
                        logger.info(f"Features calculated: {bar_count} bars processed, last: {bar.symbol} (z_er={features.z_er_15m:.2f}, z_vol={features.z_vol_15m:.2f})")

            except Exception as e:
                logger.error(f"Error calculating features: {e}")

    async def calculate_features(self, bar: Bar) -> Optional[Features]:
        """Calculate all features for this bar."""
        symbol = bar.symbol

        # Get or create window
        window = self.windows.get(symbol)
        if not window:
            window = RollingWindow(maxlen=720)
            self.windows[symbol] = window

        # Append bar to window
        window.append(bar)

        # Need minimum bars for calculations
        if not window.is_ready(min_bars=15):
            return None

        # Calculate returns
        r_1m = self._calculate_return(window.closes, lookback=1)
        r_15m = self._calculate_return(window.closes, lookback=15)

        # Calculate beta (vs BTC)
        beta = self._calculate_beta(symbol)

        # Excess return
        btc_window = self.windows.get(self.config.universe.benchmark_symbol)
        if btc_window and btc_window.is_ready():
            btc_r_15m = self._calculate_return(btc_window.closes, lookback=15)
            er_15m = r_15m - beta * btc_r_15m
        else:
            er_15m = r_15m  # Fallback if BTC not ready

        # Store excess return in window for z-score calculation
        window.excess_returns.append(er_15m)

        # Z-scores
        z_er_15m = self._robust_zscore(list(window.excess_returns))

        # Volume sum over last 15 bars
        vol_15m = sum(list(window.volumes)[-15:]) if len(window.volumes) >= 15 else 0
        z_vol_15m = self._robust_zscore(list(window.volumes))

        # Taker buy share (over last 15 bars)
        taker_buy_share_15m = self._calculate_taker_share(window, lookback=15)

        # Funding rate (from bar)
        funding_rate = bar.funding

        # Create features object
        features = Features(
            symbol=symbol,
            ts_minute=bar.ts_minute,
            r_1m=r_1m,
            r_15m=r_15m,
            beta=beta,
            er_15m=er_15m,
            z_er_15m=z_er_15m,
            vol_15m=vol_15m,
            z_vol_15m=z_vol_15m,
            taker_buy_share_15m=taker_buy_share_15m,
            funding_rate=funding_rate
        )

        # Determine direction
        features.direction = features.determine_direction()

        return features

    def _calculate_return(self, closes: deque, lookback: int) -> float:
        """Calculate log return over lookback periods."""
        if len(closes) < lookback + 1:
            return 0.0

        close_now = closes[-1]
        close_past = closes[-(lookback + 1)]

        if close_past <= 0 or close_now <= 0:
            return 0.0

        return np.log(close_now / close_past)

    def _calculate_return_at_index(self, closes: deque, index: int, lookback: int) -> float:
        """Calculate log return at a specific historical index."""
        closes_list = list(closes)

        if index < lookback or index >= len(closes_list):
            return 0.0

        close_now = closes_list[index]
        close_past = closes_list[index - lookback]

        if close_past <= 0 or close_now <= 0:
            return 0.0

        return np.log(close_now / close_past)

    def _calculate_beta(self, symbol: str) -> float:
        """
        Calculate beta vs BTC on 5m aggregated returns.

        Uses last beta_lookback_bars / 5 periods (240 / 5 = 48 x 5m = 4h).
        """
        if symbol == self.config.universe.benchmark_symbol:
            return 1.0  # BTC has beta of 1 vs itself

        window = self.windows.get(symbol)
        btc_window = self.windows.get(self.config.universe.benchmark_symbol)

        if not window or not btc_window:
            return 0.0

        # Need at least beta_lookback_bars for calculation
        if len(window.closes) < self.config.windows.beta_lookback_bars:
            return 0.0

        if len(btc_window.closes) < self.config.windows.beta_lookback_bars:
            return 0.0

        # Aggregate 1m bars to 5m bars
        agg_minutes = self.config.windows.beta_aggregation_minutes
        num_periods = self.config.windows.beta_lookback_bars // agg_minutes

        if num_periods < 10:
            return 0.0

        # Calculate 5m returns
        symbol_5m_returns = []
        btc_5m_returns = []

        closes_list = list(window.closes)
        btc_closes_list = list(btc_window.closes)

        for i in range(num_periods):
            idx_end = len(closes_list) - i * agg_minutes
            idx_start = idx_end - agg_minutes

            if idx_start < 0:
                break

            # 5m return = ln(close_end / close_start)
            # Check denominator before division to avoid divide by zero warnings
            if closes_list[idx_start - 1] > 0 and closes_list[idx_end - 1] > 0:
                symbol_ret = np.log(closes_list[idx_end - 1] / closes_list[idx_start - 1])
            else:
                symbol_ret = 0.0

            if btc_closes_list[idx_start - 1] > 0 and btc_closes_list[idx_end - 1] > 0:
                btc_ret = np.log(btc_closes_list[idx_end - 1] / btc_closes_list[idx_start - 1])
            else:
                btc_ret = 0.0

            symbol_5m_returns.append(symbol_ret)
            btc_5m_returns.append(btc_ret)

        # Reverse to chronological order
        symbol_5m_returns.reverse()
        btc_5m_returns.reverse()

        # Calculate beta via OLS: beta = cov(asset, btc) / var(btc)
        if len(symbol_5m_returns) < 10:
            return 0.0

        symbol_arr = np.array(symbol_5m_returns)
        btc_arr = np.array(btc_5m_returns)

        var_btc = np.var(btc_arr)

        if var_btc < 1e-12:
            return 0.0

        cov = np.cov(symbol_arr, btc_arr)[0, 1]
        beta = cov / var_btc

        return beta

    def _calculate_beta_at_index(self, symbol: str, index: int) -> float:
        """
        Calculate beta at a specific historical index.

        For backfill, we use a simplified approach: calculate beta from all available data up to this point.
        This is an approximation but sufficient for initializing historical z-scores.
        """
        if symbol == self.config.universe.benchmark_symbol:
            return 1.0  # BTC has beta of 1 vs itself

        window = self.windows.get(symbol)
        btc_window = self.windows.get(self.config.universe.benchmark_symbol)

        if not window or not btc_window:
            return 0.0

        # Use simplified beta: just use current full-window beta for all historical points
        # This is an approximation but avoids complex index-based calculations
        return self._calculate_beta(symbol)

    def _robust_zscore(self, series: List[float]) -> float:
        """
        Calculate robust z-score using MAD (Median Absolute Deviation).

        z = (x - median) / (1.4826 * MAD)
        """
        if len(series) < 10:
            return 0.0

        # Filter out None and NaN values
        filtered = [x for x in series if x is not None and not np.isnan(x)]

        if len(filtered) < 10:
            return 0.0

        # Get the last valid value
        last_value = filtered[-1]

        arr = np.array(filtered)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))

        robust_sigma = 1.4826 * mad

        if robust_sigma < 1e-9:
            return 0.0

        z = (last_value - median) / robust_sigma
        return z

    def _calculate_taker_share(self, window: RollingWindow, lookback: int = 15) -> Optional[float]:
        """Calculate taker buy share over last N bars."""
        if len(window.taker_buys) < lookback or len(window.taker_sells) < lookback:
            return None

        taker_buys_list = list(window.taker_buys)
        taker_sells_list = list(window.taker_sells)

        total_buy = sum(taker_buys_list[-lookback:])
        total_sell = sum(taker_sells_list[-lookback:])

        total = total_buy + total_sell

        if total == 0:
            return None

        return total_buy / total

    def _calculate_taker_share_at_index(self, window: RollingWindow, index: int, lookback: int = 15) -> Optional[float]:
        """Calculate taker buy share at a specific index for backfill."""
        if index < lookback - 1 or len(window.taker_buys) <= index or len(window.taker_sells) <= index:
            return None

        taker_buys_list = list(window.taker_buys)
        taker_sells_list = list(window.taker_sells)

        start_idx = max(0, index - lookback + 1)
        end_idx = index + 1

        total_buy = sum(taker_buys_list[start_idx:end_idx])
        total_sell = sum(taker_sells_list[start_idx:end_idx])

        total = total_buy + total_sell

        if total == 0:
            return None

        return total_buy / total

    def _calculate_beta_at_index_extended(self, symbol: str, index: int, extended_windows: dict) -> float:
        """
        Calculate beta at a specific historical index using extended windows.

        For backfill, we use a simplified approach: calculate beta from all available data up to this point.
        This is an approximation but sufficient for initializing historical z-scores.
        """
        if symbol == self.config.universe.benchmark_symbol:
            return 1.0  # BTC has beta of 1 vs itself

        window = extended_windows.get(symbol)
        btc_window = extended_windows.get(self.config.universe.benchmark_symbol)

        if not window or not btc_window:
            return 0.0

        # Use simplified beta: calculate from available data up to current index
        # For simplicity, use the last 240 bars (or all available if less)
        lookback = min(240, index + 1)

        if lookback < 2:
            return 0.0

        closes_list = list(window.closes)[:index + 1]
        btc_closes_list = list(btc_window.closes)[:index + 1]

        if len(closes_list) < lookback or len(btc_closes_list) < lookback:
            return 0.0

        # Calculate log returns for the lookback period
        symbol_returns = []
        btc_returns = []

        for i in range(-lookback + 1, 0):
            if closes_list[i - 1] > 0 and btc_closes_list[i - 1] > 0:
                symbol_ret = math.log(closes_list[i] / closes_list[i - 1])
                btc_ret = math.log(btc_closes_list[i] / btc_closes_list[i - 1])
                symbol_returns.append(symbol_ret)
                btc_returns.append(btc_ret)

        if len(symbol_returns) < 2:
            return 0.0

        # Simple OLS regression: beta = cov(symbol, btc) / var(btc)
        import statistics

        if len(btc_returns) < 2:
            return 0.0

        try:
            btc_var = statistics.variance(btc_returns)
            if btc_var < 1e-10:
                return 0.0

            btc_mean = statistics.mean(btc_returns)
            symbol_mean = statistics.mean(symbol_returns)

            cov = sum((btc_returns[i] - btc_mean) * (symbol_returns[i] - symbol_mean) for i in range(len(btc_returns))) / len(btc_returns)

            beta = cov / btc_var
            return beta

        except Exception:
            return 0.0
