"""Report generator - creates JSON reports of detected events."""

import json
import logging
from collections import Counter
from typing import Dict, Any, List
from detector.storage import Storage
from detector.utils import parse_timerange, format_timestamp

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates JSON reports of detected events.

    Includes:
    - Summary statistics
    - Top symbols by event count
    - Full event list with metrics
    """

    def __init__(self, storage: Storage):
        self.storage = storage

    async def generate_report(self, since: str = "24h", output_file: str = "report.json") -> Dict[str, Any]:
        """
        Generate JSON report for events since timerange.

        Args:
            since: Timerange string (e.g., "24h", "7d")
            output_file: Output file path

        Returns:
            Report dictionary
        """
        logger.info(f"Generating report for events since {since}")

        # Parse timerange
        start_ts, end_ts = parse_timerange(since)

        # Query events
        events = await self.storage.query_events(since_ts=start_ts)

        # Generate report
        report = self._build_report(events, start_ts, end_ts)

        # Write to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report written to {output_file}")

        return report

    def _build_report(self, events: List, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """Build report dictionary from events."""
        # Count all events (simplified - no status filtering)
        total_events = len(events)

        # Count by symbol
        symbol_counts = Counter(e.initiator_symbol for e in events)
        top_symbols = [
            {'symbol': symbol, 'event_count': count}
            for symbol, count in symbol_counts.most_common(10)
        ]

        # Convert events to dicts
        events_list = [e.to_dict() for e in events]

        # Build report
        report = {
            'summary': {
                'total_events': total_events,
                'time_range': {
                    'start': format_timestamp(start_ts),
                    'end': format_timestamp(end_ts),
                    'start_ts': start_ts,
                    'end_ts': end_ts
                }
            },
            'top_symbols': top_symbols,
            'events': events_list
        }

        return report

    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print report summary to stdout."""
        summary = report['summary']

        print("\n" + "=" * 60)
        print("ANOMALY DETECTOR - REPORT SUMMARY")
        print("=" * 60)
        print(f"Time Range: {summary['time_range']['start']} to {summary['time_range']['end']}")
        print(f"Total Events: {summary['total_events']}")
        print("\nTop Symbols:")

        for entry in report['top_symbols'][:5]:
            print(f"  {entry['symbol']}: {entry['event_count']} events")

        print("=" * 60 + "\n")
