"""Alert dispatcher - formats and sends alerts to stdout and Telegram."""

import asyncio
import aiohttp
import logging
from datetime import datetime
from detector.models import Event
from detector.config import Config

logger = logging.getLogger(__name__)


class AlertDispatcher:
    """
    Dispatches alerts to configured channels.

    Handles:
    - Stdout (always)
    - Telegram (if enabled)
    - Alert formatting (compact blocks)
    """

    def __init__(self, event_queue: asyncio.Queue, config: Config):
        self.event_queue = event_queue
        self.config = config
        self.session: aiohttp.ClientSession = None

    async def init_session(self) -> None:
        """Initialize HTTP session for Telegram."""
        if self.config.alerts.telegram.enabled and not self.session:
            self.session = aiohttp.ClientSession()

    async def run(self) -> None:
        """Consume events and send alerts."""
        await self.init_session()
        logger.info("Alert dispatcher started")

        while True:
            try:
                event_type, event = await self.event_queue.get()

                if event_type == 'initiator':
                    await self._send_initiator_alert(event)
                elif event_type == 'sector':
                    await self._send_sector_alert(event)

            except Exception as e:
                logger.error(f"Error dispatching alert: {e}")

    async def _send_initiator_alert(self, event: Event) -> None:
        """Send initiator alert to stdout and Telegram."""
        message = self._format_initiator(event)

        # Always print to stdout
        print("\n" + message + "\n")

        # Send to Telegram if enabled
        if self.config.alerts.telegram.enabled:
            await self._send_telegram(message)

    async def _send_sector_alert(self, event: Event) -> None:
        """Send sector diffusion alert to stdout and Telegram."""
        message = self._format_sector(event)

        # Always print to stdout
        print("\n" + message + "\n")

        # Send to Telegram if enabled
        if self.config.alerts.telegram.enabled:
            await self._send_telegram(message)

    def _format_initiator(self, event: Event) -> str:
        """Format compact initiator alert."""
        metrics = event.metrics
        timestamp = datetime.fromtimestamp(event.ts / 1000).strftime('%Y-%m-%d %H:%M:%S UTC')

        # Format confirmations
        confirmations = []
        if metrics.get('z_oi') and metrics['z_oi'] >= self.config.thresholds.oi_delta_z_confirm:
            confirmations.append(f"OI_Δ={metrics['z_oi']:.1f}σ")
        if metrics.get('z_liq') and metrics['z_liq'] >= self.config.thresholds.liquidation_z_confirm:
            confirmations.append(f"Liq={metrics['z_liq']:.1f}σ")
        if metrics.get('funding') and abs(metrics['funding']) >= self.config.thresholds.funding_abs_threshold:
            confirmations.append(f"Funding={metrics['funding']:.2%}")

        confirmations_str = ", ".join(confirmations) if confirmations else "None"

        message = f"""🚨 SECTOR SHOT - INITIATOR
Symbol: {event.initiator_symbol} | Direction: {event.direction.value} | Status: {event.status.value}
Time: {timestamp}
Z-Scores: ER={metrics['z_er']:.1f}σ, VOL={metrics['z_vol']:.1f}σ
Taker Buy Share: {metrics['taker_share']:.1%}
Beta: {metrics['beta']:.2f} | Funding: {metrics['funding']:.2%}
Confirmations: {confirmations_str}"""

        return message

    def _format_sector(self, event: Event) -> str:
        """Format sector diffusion alert."""
        timestamp = datetime.fromtimestamp(event.ts / 1000).strftime('%Y-%m-%d %H:%M:%S UTC')
        initiator_ts = datetime.fromtimestamp(event.initiator_symbol / 1000).strftime('%H:%M')

        follower_count = event.metrics.get('follower_count', 0)
        total_sector = len(self.config.universe.sector_symbols) - 1
        follower_share = event.metrics.get('follower_share', 0)

        # Format follower list
        follower_lines = []
        follower_metrics = event.metrics.get('follower_metrics', [])
        for fm in follower_metrics:
            follower_lines.append(f"  • {fm['symbol']}: ER_z={fm['z_er']:.1f}σ, VOL_z={fm['z_vol']:.1f}σ")

        followers_str = "\n".join(follower_lines)

        message = f"""🎯 SECTOR DIFFUSION DETECTED
Initiator: {event.initiator_symbol} ({event.direction.value}) at {timestamp}
Followers ({follower_count}/{total_sector} = {follower_share:.0%}):
{followers_str}"""

        return message

    async def _send_telegram(self, message: str) -> None:
        """Send message to Telegram."""
        if not self.session:
            return

        bot_token = self.config.alerts.telegram.bot_token
        chat_id = self.config.alerts.telegram.chat_id

        if not bot_token or not chat_id:
            logger.warning("Telegram enabled but token/chat_id not configured")
            return

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }

        try:
            async with self.session.post(url, json=payload, timeout=10) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Telegram API error: {resp.status} - {text}")
                else:
                    logger.debug("Telegram message sent successfully")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

    async def close(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
