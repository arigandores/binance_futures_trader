"""Utility functions for logging, time handling, and parsing."""

import logging
import structlog
import sys
from datetime import datetime, timedelta
from typing import Tuple


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structured logging with human-readable output.

    Uses structlog for structured logging with colored console output.
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True)
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    logger = structlog.get_logger()
    logger.info("Logging initialized", level=log_level)


def parse_timerange(timerange: str) -> Tuple[int, int]:
    """
    Parse timerange string (e.g., "24h", "7d") to timestamp range.

    Args:
        timerange: String like "24h", "7d", "30d"

    Returns:
        Tuple of (start_ts_ms, end_ts_ms)
    """
    now = datetime.utcnow()
    end_ts = int(now.timestamp() * 1000)

    # Parse unit
    if timerange.endswith('h'):
        hours = int(timerange[:-1])
        start = now - timedelta(hours=hours)
    elif timerange.endswith('d'):
        days = int(timerange[:-1])
        start = now - timedelta(days=days)
    elif timerange.endswith('w'):
        weeks = int(timerange[:-1])
        start = now - timedelta(weeks=weeks)
    else:
        raise ValueError(f"Invalid timerange format: {timerange}. Use formats like '24h', '7d', '4w'")

    start_ts = int(start.timestamp() * 1000)

    return start_ts, end_ts


def format_timestamp(timestamp_ms: int) -> str:
    """
    Format timestamp as human-readable string.

    Args:
        timestamp_ms: Timestamp in milliseconds since epoch

    Returns:
        Formatted string (YYYY-MM-DD HH:MM:SS UTC)
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime('%Y-%m-%d %H:%M:%S UTC')


def format_price(price: float) -> str:
    """
    Format price with appropriate decimal places based on magnitude.

    Args:
        price: Price value to format

    Returns:
        Formatted price string with $ prefix
    """
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
