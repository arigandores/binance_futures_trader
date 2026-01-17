"""Feature calculator with rolling windows and z-scores."""

import asyncio
import logging
import math
import numpy as np
from typing import Dict, List, Optional
from detector.models import Bar, Features, Direction
from detector.storage import Storage
from detector.config import Config
from detector.binance_rest import BinanceRestClient

logger = logging.getLogger(__name__)


class RollingWindow:
    """
    Manages rolling data for one symbol using pre-allocated numpy arrays.

    Performance optimization for 500+ symbols:
    - Uses circular buffers (no memory allocation on append)
    - Pre-allocated numpy arrays (2-3x faster than deque)
    - O(1) append operations
    - Derived fields (excess_returns) use deque for correctness (appended separately)
    """

    def __init__(self, maxlen: int = 720):
        self.maxlen = maxlen
        self._index = 0  # Current write position (circular)
        self._count = 0  # Number of elements stored

        # Pre-allocated numpy arrays (circular buffers) for bar data
        self._closes = np.zeros(maxlen, dtype=np.float64)
        self._volumes = np.zeros(maxlen, dtype=np.float64)
        self._taker_buys = np.zeros(maxlen, dtype=np.float64)
        self._taker_sells = np.zeros(maxlen, dtype=np.float64)

        # Derived fields use simple list for correctness
        # (appended at different times than bar data, so can't share circular index)
        self._excess_returns: List[float] = []

    def append(self, bar: Bar) -> None:
        """Append new bar to circular buffer - O(1), no allocation."""
        idx = self._index % self.maxlen
        self._closes[idx] = bar.close
        self._volumes[idx] = bar.volume
        self._taker_buys[idx] = bar.taker_buy
        self._taker_sells[idx] = bar.taker_sell
        self._index += 1
        self._count = min(self._count + 1, self.maxlen)

    def is_ready(self, min_bars: int = 15) -> bool:
        """Check if window has enough data."""
        return self._count >= min_bars

    def _get_ordered(self, arr: np.ndarray) -> np.ndarray:
        """Get chronologically ordered view of circular buffer."""
        if self._count < self.maxlen:
            return arr[:self._count].copy()
        start = self._index % self.maxlen
        return np.concatenate([arr[start:], arr[:start]])

    # Property accessors for compatibility with existing code
    @property
    def closes(self) -> np.ndarray:
        """Get chronologically ordered closes."""
        return self._get_ordered(self._closes)

    @property
    def volumes(self) -> np.ndarray:
        """Get chronologically ordered volumes."""
        return self._get_ordered(self._volumes)

    @property
    def taker_buys(self) -> np.ndarray:
        """Get chronologically ordered taker buys."""
        return self._get_ordered(self._taker_buys)

    @property
    def taker_sells(self) -> np.ndarray:
        """Get chronologically ordered taker sells."""
        return self._get_ordered(self._taker_sells)

    @property
    def excess_returns(self) -> 'ExcessReturnsBuffer':
        """Return buffer for excess_returns with append support and maxlen enforcement."""
        return ExcessReturnsBuffer(self)


class ExcessReturnsBuffer:
    """
    Simple wrapper for excess_returns list that enforces maxlen.
    Appended separately from bar data, so needs independent storage.
    """

    def __init__(self, window: RollingWindow):
        self._window = window

    def append(self, value: float) -> None:
        """Append value, maintaining maxlen."""
        self._window._excess_returns.append(value)
        # Trim to maxlen
        if len(self._window._excess_returns) > self._window.maxlen:
            self._window._excess_returns = self._window._excess_returns[-self._window.maxlen:]

    def __len__(self) -> int:
        return len(self._window._excess_returns)

    def __iter__(self):
        return iter(self._window._excess_returns)

    def __getitem__(self, idx):
        return self._window._excess_returns[idx]


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
        import time
        start_time = time.time()
        print("Backfilling rolling windows from database...")

        # First pass: Load ALL bars into windows
        # BTC (benchmark) is loaded FIRST to ensure it's available for calculations
        all_bars_by_symbol = {}
        benchmark = self.config.universe.benchmark_symbol

        # Helper function for parallel loading
        async def load_bars_for_symbol(symbol: str):
            bars = await self.storage.get_recent_bars(symbol, limit=10000)
            return symbol, bars

        # STEP 1: Load BTC first (required for beta/excess return calculations)
        print(f"Loading benchmark {benchmark} first...")
        btc_symbol, btc_bars = await load_bars_for_symbol(benchmark)
        all_bars_by_symbol[btc_symbol] = btc_bars

        btc_window = RollingWindow(maxlen=720)
        for bar in btc_bars[-720:]:
            btc_window.append(bar)
        self.windows[btc_symbol] = btc_window
        print(f"Benchmark {benchmark} loaded: {len(btc_bars)} bars")

        # STEP 2: Load remaining symbols in parallel
        other_symbols = [s for s in self.config.universe.all_symbols if s != benchmark]

        tasks = [load_bars_for_symbol(s) for s in other_symbols]
        results = await asyncio.gather(*tasks)

        for symbol, bars in results:
            all_bars_by_symbol[symbol] = bars

            window = RollingWindow(maxlen=720)
            # Only keep last 720 in rolling window for memory efficiency
            for bar in bars[-720:]:
                window.append(bar)

            self.windows[symbol] = window

        load_time = time.time() - start_time
        print(f"Loaded bars for {len(self.windows)} symbols in {load_time:.1f}s")

        # Second pass: Calculate features using VECTORIZED numpy operations
        print("Calculating features for all historical data (vectorized)...")

        btc_bars = all_bars_by_symbol.get(benchmark, [])
        if len(btc_bars) < 15:
            logger.warning(f"Insufficient {benchmark} data for feature calculation ({len(btc_bars)} bars, need 15)")
            return

        # Pre-convert ALL data to numpy arrays ONCE (major optimization)
        print("Converting data to numpy arrays...")
        numpy_data = {}
        for symbol in self.config.universe.all_symbols:
            bars = all_bars_by_symbol.get(symbol, [])
            if len(bars) < 15:
                continue

            # Extract all data into numpy arrays at once
            numpy_data[symbol] = {
                'closes': np.array([b.close for b in bars], dtype=np.float64),
                'volumes': np.array([b.volume for b in bars], dtype=np.float64),
                'taker_buys': np.array([b.taker_buy for b in bars], dtype=np.float64),
                'taker_sells': np.array([b.taker_sell for b in bars], dtype=np.float64),
                'ts_minutes': np.array([b.ts_minute for b in bars], dtype=np.int64),
                'fundings': [b.funding for b in bars],  # Keep as list (may have None)
                'bars': bars
            }

        if benchmark not in numpy_data:
            logger.warning("No BTC numpy data available")
            return

        btc_closes = numpy_data[benchmark]['closes']

        # Calculate features for each symbol with progress tracking
        total_features_written = 0
        symbols_processed = 0
        symbols_to_process = [s for s in self.config.universe.all_symbols if s in numpy_data]
        total_symbols = len(symbols_to_process)
        calc_start_time = time.time()

        # Process symbols in batches for better DB performance
        BATCH_SIZE = 10  # Flush DB every N symbols

        for symbol in symbols_to_process:
            data = numpy_data[symbol]
            closes = data['closes']
            volumes = data['volumes']
            taker_buys = data['taker_buys']
            taker_sells = data['taker_sells']
            ts_minutes = data['ts_minutes']
            fundings = data['fundings']
            bars = data['bars']
            n = len(closes)

            # Align BTC data to same length as current symbol
            # Different symbols may have different amounts of historical data
            n_btc = len(btc_closes)
            if n_btc < n:
                # If BTC has fewer bars than this symbol, use available BTC data length
                # Take the last n_btc bars from the symbol to align
                logger.warning(f"{symbol}: BTC has fewer bars ({n_btc}) than symbol ({n}), using aligned subset")
                closes = closes[-n_btc:]
                volumes = volumes[-n_btc:]
                taker_buys = taker_buys[-n_btc:]
                taker_sells = taker_sells[-n_btc:]
                ts_minutes = ts_minutes[-n_btc:]
                fundings = fundings[-n_btc:]
                bars = bars[-n_btc:]
                n = n_btc
                btc_closes_aligned = btc_closes
            else:
                # Normal case: BTC has same or more bars, use last n bars from BTC
                btc_closes_aligned = btc_closes[-n:]

            # Pre-calculate beta once per symbol
            symbol_beta = self._calculate_beta_vectorized(closes, btc_closes_aligned)

            # VECTORIZED calculations for all bars at once
            # 1. Calculate all returns vectorized
            r_1m_all = np.zeros(n)
            r_15m_all = np.zeros(n)
            btc_r_15m_all = np.zeros(n)

            # Log returns: ln(close[i] / close[i-lookback])
            r_1m_all[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-10))
            r_15m_all[15:] = np.log(closes[15:] / np.maximum(closes[:-15], 1e-10))
            btc_r_15m_all[15:] = np.log(btc_closes_aligned[15:] / np.maximum(btc_closes_aligned[:-15], 1e-10))

            # 2. Calculate excess returns
            er_15m_all = r_15m_all - symbol_beta * btc_r_15m_all

            # 3. Calculate rolling 15-bar volume sums
            vol_15m_all = np.convolve(volumes, np.ones(15), mode='full')[:n]
            # Fix first 14 values (partial sums)
            for i in range(14):
                vol_15m_all[i] = np.sum(volumes[:i+1])

            # 4. Calculate rolling taker buy share
            taker_total = taker_buys + taker_sells
            # Rolling 15-bar sums
            taker_buy_sum = np.convolve(taker_buys, np.ones(15), mode='full')[:n]
            taker_total_sum = np.convolve(taker_total, np.ones(15), mode='full')[:n]
            # Use np.divide with out/where to avoid division by zero warning
            # (np.where evaluates both branches before selecting, causing warnings)
            taker_buy_share_all = np.divide(
                taker_buy_sum,
                taker_total_sum,
                out=np.full_like(taker_buy_sum, 0.5),
                where=taker_total_sum > 0
            )

            # 5. Calculate rolling z-scores (this is the most expensive part)
            z_er_15m_all = self._rolling_robust_zscore_vectorized(er_15m_all)
            z_vol_15m_all = self._rolling_robust_zscore_vectorized(volumes)

            # Build features batch
            features_batch = []
            for i in range(15, n):
                features = Features(
                    symbol=symbol,
                    ts_minute=int(ts_minutes[i]),  # Convert numpy.int64 to Python int for SQLite
                    r_1m=float(r_1m_all[i]),
                    r_15m=float(r_15m_all[i]),
                    beta=symbol_beta,
                    er_15m=float(er_15m_all[i]),
                    z_er_15m=float(z_er_15m_all[i]),
                    vol_15m=float(vol_15m_all[i]),
                    z_vol_15m=float(z_vol_15m_all[i]),
                    taker_buy_share_15m=float(taker_buy_share_all[i]),
                    funding_rate=fundings[i]
                )
                features.direction = features.determine_direction()
                features_batch.append(features)

            # Write features to database
            if features_batch:
                await self.storage.batch_write_features(features_batch)
                total_features_written += len(features_batch)
                symbols_processed += 1

                # Store excess returns for rolling window
                if symbol in self.windows:
                    window = self.windows[symbol]
                    for er in er_15m_all[-720:]:
                        window.excess_returns.append(er)

                # Flush DB periodically and show progress
                if symbols_processed % BATCH_SIZE == 0:
                    await self.storage.flush_all()
                    elapsed = time.time() - calc_start_time
                    rate = symbols_processed / elapsed if elapsed > 0 else 0
                    remaining = (total_symbols - symbols_processed) / rate if rate > 0 else 0
                    print(f"Features: {symbols_processed}/{total_symbols} symbols ({total_features_written} features) - {remaining:.0f}s remaining")

        # Final flush
        await self.storage.flush_all()

        total_time = time.time() - start_time
        print(f"Backfill complete: {total_features_written} features written to database for {len(self.windows)} symbols in {total_time:.1f}s")

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
        excess_returns_arr = np.array(list(window.excess_returns))
        z_er_15m = self._robust_zscore(list(excess_returns_arr))

        # Volume sum over last 15 bars - use numpy for efficiency
        volumes = window.volumes
        if len(volumes) >= 15:
            vol_15m = float(np.sum(volumes[-15:])) if isinstance(volumes, np.ndarray) else sum(list(volumes)[-15:])
        else:
            vol_15m = 0.0
        z_vol_15m = self._robust_zscore(list(volumes) if isinstance(volumes, np.ndarray) else list(volumes))

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

    def _calculate_return(self, closes, lookback: int) -> float:
        """Calculate log return over lookback periods.

        Args:
            closes: numpy array or deque of close prices
            lookback: number of periods to look back
        """
        # Handle both numpy array and deque
        if hasattr(closes, '__len__'):
            length = len(closes)
        else:
            length = len(list(closes))

        if length < lookback + 1:
            return 0.0

        close_now = closes[-1]
        close_past = closes[-(lookback + 1)]

        if close_past <= 0 or close_now <= 0:
            return 0.0

        return float(np.log(close_now / close_past))

    def _calculate_return_at_index(self, closes, index: int, lookback: int) -> float:
        """Calculate log return at a specific historical index."""
        closes_list = list(closes) if not isinstance(closes, list) else closes

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

        # Get closes as arrays
        closes = window.closes
        btc_closes = btc_window.closes

        # Need at least beta_lookback_bars for calculation
        if len(closes) < self.config.windows.beta_lookback_bars:
            return 0.0

        if len(btc_closes) < self.config.windows.beta_lookback_bars:
            return 0.0

        # Aggregate 1m bars to 5m bars
        agg_minutes = self.config.windows.beta_aggregation_minutes
        num_periods = self.config.windows.beta_lookback_bars // agg_minutes

        if num_periods < 10:
            return 0.0

        # Calculate 5m returns
        symbol_5m_returns = []
        btc_5m_returns = []

        # Convert to list if numpy array for indexing compatibility
        closes_list = list(closes) if isinstance(closes, np.ndarray) else list(closes)
        btc_closes_list = list(btc_closes) if isinstance(btc_closes, np.ndarray) else list(btc_closes)

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

    def _rolling_robust_zscore_vectorized(self, series: np.ndarray, window_size: int = 720) -> np.ndarray:
        """
        Calculate rolling robust z-scores for entire series using optimized vectorized operations.

        Uses MAD (Median Absolute Deviation) for robustness against outliers.
        z = (x - median) / (1.4826 * MAD)

        Performance optimization for 500+ symbols:
        - Uses numpy stride_tricks for sliding windows (no copy)
        - Vectorized median and MAD calculations
        - 10-50x faster than loop-based approach
        """
        n = len(series)
        result = np.zeros(n, dtype=np.float64)

        if n < 10:
            return result

        # For small arrays or when n < window_size, use stride_tricks for true vectorization
        # For the warm-up period (first window_size elements), we still need incremental calculation

        # Determine effective window for each position
        # Optimization: use numpy sliding_window_view for large enough arrays
        if n >= window_size:
            from numpy.lib.stride_tricks import sliding_window_view

            # Create sliding windows view (no memory copy!)
            windows = sliding_window_view(series, window_size)

            # Vectorized calculation for the full-window portion
            # Calculate medians for all windows at once
            medians = np.median(windows, axis=1)

            # Calculate MADs for all windows
            # This is the expensive operation - compute abs differences
            abs_deviations = np.abs(windows - medians[:, np.newaxis])
            mads = np.median(abs_deviations, axis=1)

            # Calculate robust sigma
            robust_sigmas = 1.4826 * mads

            # Calculate z-scores (handling division by zero)
            valid_mask = robust_sigmas > 1e-9
            # Get the corresponding series values (last value in each window)
            values = series[window_size - 1:]

            # Apply z-score formula where valid
            result[window_size - 1:][valid_mask] = (values[valid_mask] - medians[valid_mask]) / robust_sigmas[valid_mask]

            # Handle warm-up period (indices 10 to window_size-1)
            for i in range(10, window_size - 1):
                window_data = series[:i + 1]
                valid_data = window_data[~np.isnan(window_data)]

                if len(valid_data) >= 10:
                    median = np.median(valid_data)
                    mad = np.median(np.abs(valid_data - median))
                    robust_sigma = 1.4826 * mad

                    if robust_sigma > 1e-9:
                        result[i] = (series[i] - median) / robust_sigma
        else:
            # For arrays smaller than window_size, use the original loop
            # (this is rare in production - only during initial startup)
            for i in range(10, n):
                window_data = series[:i + 1]
                valid_data = window_data[~np.isnan(window_data)]

                if len(valid_data) < 10:
                    continue

                median = np.median(valid_data)
                mad = np.median(np.abs(valid_data - median))
                robust_sigma = 1.4826 * mad

                if robust_sigma > 1e-9:
                    result[i] = (series[i] - median) / robust_sigma

        return result

    def _calculate_beta_vectorized(self, closes: np.ndarray, btc_closes: np.ndarray) -> float:
        """
        Calculate beta vs BTC using vectorized numpy operations.

        Uses 5-minute aggregated returns over 240 1-minute bars (48 x 5m = 4h).
        beta = cov(asset, btc) / var(btc)
        """
        n = min(len(closes), len(btc_closes))

        if n < 240:  # Need enough data for beta calculation
            return 0.0

        # Use last 240 bars
        closes = closes[-240:]
        btc_closes = btc_closes[-240:]

        # Aggregate to 5-minute returns
        agg_minutes = 5
        num_periods = 240 // agg_minutes  # 48 periods

        symbol_5m_returns = []
        btc_5m_returns = []

        for i in range(num_periods):
            idx_start = i * agg_minutes
            idx_end = (i + 1) * agg_minutes

            if idx_end > len(closes):
                break

            # 5m return = ln(close_end / close_start)
            if closes[idx_start] > 0 and btc_closes[idx_start] > 0:
                symbol_ret = np.log(closes[idx_end - 1] / closes[idx_start])
                btc_ret = np.log(btc_closes[idx_end - 1] / btc_closes[idx_start])
                symbol_5m_returns.append(symbol_ret)
                btc_5m_returns.append(btc_ret)

        if len(symbol_5m_returns) < 10:
            return 0.0

        symbol_arr = np.array(symbol_5m_returns)
        btc_arr = np.array(btc_5m_returns)

        var_btc = np.var(btc_arr)

        if var_btc < 1e-12:
            return 0.0

        cov = np.cov(symbol_arr, btc_arr)[0, 1]
        beta = cov / var_btc

        return float(beta)

    def _calculate_taker_share(self, window: RollingWindow, lookback: int = 15) -> Optional[float]:
        """Calculate taker buy share over last N bars."""
        taker_buys = window.taker_buys
        taker_sells = window.taker_sells

        if len(taker_buys) < lookback or len(taker_sells) < lookback:
            return None

        # Use numpy sum for efficiency if arrays
        if isinstance(taker_buys, np.ndarray):
            total_buy = float(np.sum(taker_buys[-lookback:]))
            total_sell = float(np.sum(taker_sells[-lookback:]))
        else:
            total_buy = sum(list(taker_buys)[-lookback:])
            total_sell = sum(list(taker_sells)[-lookback:])

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
