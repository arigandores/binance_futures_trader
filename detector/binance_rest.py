"""Binance REST API client with graceful degradation."""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)


class BinanceRestClient:
    """
    REST API client for Binance USD-M Futures.

    Handles:
    - Open Interest history
    - Taker buy/sell ratio
    - Klines (for backfill)
    - Graceful degradation (returns None if no API key)
    - Rate limiting and exponential backoff
    """

    BASE_URL = "https://fapi.binance.com"
    FUTURES_DATA_URL = "https://fapi.binance.com/futures/data"

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session: Optional[aiohttp.ClientSession] = None

        # Cache for REST data (updated every poll_sec)
        self.oi_cache: Dict[str, List[Dict]] = {}
        self.taker_ratio_cache: Dict[str, List[Dict]] = {}

        if not api_key:
            logger.warning(
                "No API key provided - advanced features (OI, taker ratio) disabled. "
                "Events will be marked UNCONFIRMED."
            )

    async def init_session(self) -> None:
        """Initialize aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self) -> None:
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def get_oi_history(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 30
    ) -> Optional[List[Dict]]:
        """
        Get Open Interest history.

        Returns None if API key not provided.
        """
        if not self.api_key:
            return None

        endpoint = f"{self.FUTURES_DATA_URL}/openInterestHist"
        params = {
            'symbol': symbol,
            'period': period,
            'limit': limit
        }

        return await self._get_with_retry(endpoint, params)

    async def get_taker_ratio(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 30
    ) -> Optional[List[Dict]]:
        """
        Get taker long/short ratio.

        Returns None if API key not provided.
        """
        if not self.api_key:
            return None

        endpoint = f"{self.FUTURES_DATA_URL}/takerlongshortRatio"
        params = {
            'symbol': symbol,
            'period': period,
            'limit': limit
        }

        return await self._get_with_retry(endpoint, params)

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        start_time: Optional[int] = None
    ) -> Optional[List[Dict]]:
        """
        Get klines (OHLCV) data for backfill.

        Note: Can work without API key for basic access.
        """
        endpoint = f"{self.BASE_URL}/fapi/v1/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time

        return await self._get_with_retry(endpoint, params, require_key=False)

    async def _get_with_retry(
        self,
        endpoint: str,
        params: Dict,
        max_retries: int = 3,
        require_key: bool = True
    ) -> Optional[List[Dict]]:
        """
        HTTP GET with exponential backoff retry.

        Returns None on failure.
        """
        if require_key and not self.api_key:
            return None

        if not self.session:
            await self.init_session()

        headers = {}
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key

        for attempt in range(max_retries):
            try:
                async with self.session.get(endpoint, params=params, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data

                    elif resp.status == 429:
                        # Rate limit hit
                        retry_after = int(resp.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limit hit, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)

                    elif resp.status >= 500:
                        # Server error, retry with backoff
                        delay = 2 ** attempt
                        logger.warning(f"Server error {resp.status}, retrying in {delay}s")
                        await asyncio.sleep(delay)

                    else:
                        # Client error, don't retry
                        text = await resp.text()
                        logger.error(f"HTTP {resp.status} for {endpoint}: {text}")
                        return None

            except asyncio.TimeoutError:
                delay = 2 ** attempt
                logger.warning(f"Request timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"Error fetching {endpoint}: {e}")
                return None

        logger.error(f"Max retries exceeded for {endpoint}")
        return None

    async def poll_loop(self, symbols: List[str], interval_sec: int = 60) -> None:
        """
        Polling loop for OI and taker ratio data.

        Updates internal cache every interval_sec seconds.
        """
        if not self.api_key:
            logger.info("Skipping REST poll loop (no API key)")
            return

        logger.info(f"Starting REST poll loop (interval: {interval_sec}s)")

        while True:
            try:
                await self._poll_all_symbols(symbols)
            except Exception as e:
                logger.error(f"Error in poll loop: {e}")

            await asyncio.sleep(interval_sec)

    async def _poll_all_symbols(self, symbols: List[str]) -> None:
        """Poll OI and taker ratio for all symbols."""
        oi_count = 0
        taker_count = 0

        for symbol in symbols:
            # Fetch OI
            oi_data = await self.get_oi_history(symbol, period="5m", limit=30)
            if oi_data:
                self.oi_cache[symbol] = oi_data
                oi_count += 1

            # Fetch taker ratio
            taker_data = await self.get_taker_ratio(symbol, period="5m", limit=30)
            if taker_data:
                self.taker_ratio_cache[symbol] = taker_data
                taker_count += 1

            # Small delay to avoid burst rate limiting
            await asyncio.sleep(0.5)

        logger.info(f"REST poll complete: OI data for {oi_count}/{len(symbols)} symbols, Taker ratio for {taker_count}/{len(symbols)} symbols")

    def get_latest_oi(self, symbol: str) -> Optional[float]:
        """Get latest OI value from cache."""
        oi_data = self.oi_cache.get(symbol)
        if not oi_data or len(oi_data) == 0:
            return None

        # Latest entry
        latest = oi_data[-1]
        return float(latest.get('sumOpenInterest', 0))

    def get_oi_delta_1h(self, symbol: str) -> Optional[float]:
        """
        Calculate OI delta over last 1 hour.

        Returns None if insufficient data.
        """
        oi_data = self.oi_cache.get(symbol)
        if not oi_data or len(oi_data) < 12:  # Need at least 12 x 5m = 1h
            return None

        # Compare latest to 1 hour ago (12 periods back)
        latest_oi = float(oi_data[-1].get('sumOpenInterest', 0))
        hour_ago_oi = float(oi_data[-12].get('sumOpenInterest', 0))

        if hour_ago_oi == 0:
            return None

        delta = latest_oi - hour_ago_oi
        return delta

    def get_latest_taker_ratio(self, symbol: str) -> Optional[float]:
        """
        Get latest taker buy/sell ratio from cache.

        Returns buy_ratio (buy / (buy + sell)).
        """
        taker_data = self.taker_ratio_cache.get(symbol)
        if not taker_data or len(taker_data) == 0:
            return None

        latest = taker_data[-1]
        buy_vol = float(latest.get('buySellRatio', 1.0))

        # Binance returns buySellRatio = buyVol / sellVol
        # Convert to buy_share = buy / (buy + sell)
        # buy_share = buy / (buy + sell) = (buy/sell) / (buy/sell + 1)
        if buy_vol <= 0:
            return 0.0

        buy_share = buy_vol / (buy_vol + 1.0)
        return buy_share
