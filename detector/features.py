"""Feature calculator with rolling windows and z-scores."""

import asyncio
import logging
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

        # Liquidations (if available)
        self.liq_notionals = deque(maxlen=maxlen)

        # Derived features
        self.returns_1m = deque(maxlen=maxlen)
        self.excess_returns = deque(maxlen=maxlen)

    def append(self, bar: Bar) -> None:
        """Append new bar to window."""
        self.closes.append(bar.close)
        self.volumes.append(bar.volume)
        self.taker_buys.append(bar.taker_buy)
        self.taker_sells.append(bar.taker_sell)
        self.liq_notionals.append(bar.liq_notional)

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
        config: Config
    ):
        self.bar_queue = bar_queue
        self.feature_queue = feature_queue
        self.storage = storage
        self.rest_client = rest_client
        self.config = config

        # Rolling windows per symbol
        self.windows: Dict[str, RollingWindow] = {}

    async def backfill(self) -> None:
        """Load last 720 bars from DB into memory and pre-calculate excess returns."""
        logger.info("Backfilling rolling windows from database...")

        # First pass: Load all bars into windows
        for symbol in self.config.universe.all_symbols:
            bars = await self.storage.get_recent_bars(symbol, limit=720)

            window = RollingWindow(maxlen=720)
            for bar in bars:
                window.append(bar)

            self.windows[symbol] = window

            logger.debug(f"Backfilled {len(bars)} bars for {symbol}")

        # Second pass: Calculate excess returns for historical bars
        logger.info("Pre-calculating excess returns for historical data...")

        btc_window = self.windows.get(self.config.universe.benchmark_symbol)
        if not btc_window or len(btc_window.closes) < 15:
            logger.warning("Insufficient BTC data for excess return calculation")
            return

        for symbol in self.config.universe.all_symbols:
            window = self.windows.get(symbol)
            if not window or len(window.closes) < 15:
                continue

            # Calculate excess returns for each historical bar
            for i in range(15, len(window.closes)):
                # Calculate 15m return for this bar
                r_15m = self._calculate_return_at_index(window.closes, i, lookback=15)

                # Calculate beta (using available data up to this point)
                beta = self._calculate_beta_at_index(symbol, i)

                # Calculate BTC 15m return
                btc_r_15m = self._calculate_return_at_index(btc_window.closes, i, lookback=15)

                # Calculate excess return
                er_15m = r_15m - beta * btc_r_15m

                # Store in window
                window.excess_returns.append(er_15m)

            logger.debug(f"Pre-calculated {len(window.excess_returns)} excess returns for {symbol}")

        logger.info(f"Backfilled rolling windows for {len(self.windows)} symbols with excess returns")

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

        # OI delta (from REST client)
        oi_delta_1h = self.rest_client.get_oi_delta_1h(symbol)
        z_oi_delta_1h = None  # Would need historical OI deltas to calculate z-score

        # Liquidation sum over last 15 bars
        liq_15m = sum(list(window.liq_notionals)[-15:]) if len(window.liq_notionals) >= 15 else 0
        z_liq_15m = self._robust_zscore(list(window.liq_notionals)) if liq_15m > 0 else 0

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
            oi_delta_1h=oi_delta_1h,
            z_oi_delta_1h=z_oi_delta_1h,
            liq_15m=liq_15m,
            z_liq_15m=z_liq_15m,
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

        if close_past <= 0:
            return 0.0

        return np.log(close_now / close_past)

    def _calculate_return_at_index(self, closes: deque, index: int, lookback: int) -> float:
        """Calculate log return at a specific historical index."""
        closes_list = list(closes)

        if index < lookback or index >= len(closes_list):
            return 0.0

        close_now = closes_list[index]
        close_past = closes_list[index - lookback]

        if close_past <= 0:
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
            symbol_ret = np.log(closes_list[idx_end - 1] / closes_list[idx_start - 1]) if closes_list[idx_start - 1] > 0 else 0
            btc_ret = np.log(btc_closes_list[idx_end - 1] / btc_closes_list[idx_start - 1]) if btc_closes_list[idx_start - 1] > 0 else 0

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

        arr = np.array(series)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))

        robust_sigma = 1.4826 * mad

        if robust_sigma < 1e-9:
            return 0.0

        z = (arr[-1] - median) / robust_sigma
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
