# Binance Sector Shot Detector

Real-time anomaly detection for coordinated sector movements on Binance USD-M Perpetual Futures.

## Overview

The Sector Shot Detector monitors Binance futures markets for "sector shot" anomalies - coordinated price movements across correlated assets (e.g., privacy coins: ZEC, DASH, XMR). It ingests WebSocket streams, aggregates to 1-minute bars, calculates rolling z-scores and beta vs BTC, and alerts when strict trigger rules are met.

### Key Features

- **Real-time Detection**: WebSocket streaming with automatic reconnection
- **Statistical Rigor**: Robust z-scores (MAD-based), rolling beta calculation
- **Sector Diffusion**: Detects coordinated movements across multiple assets
- **Graceful Degradation**: Works without API key (WS-only mode)
- **Cooldown Logic**: Direction-aware cooldown to prevent alert spam
- **Persistent Storage**: SQLite with WAL mode and batched writes
- **Alert Channels**: Stdout + optional Telegram integration

## Architecture

```
WS Streams → Aggregator (1m bars) → Features (z-scores, beta) → Detector (rules) → Alerts
     ↓            ↓                      ↓                          ↓
  Storage      Storage                Storage                   Storage
```

## Requirements

- Python 3.12+
- Poetry (dependency management)
- Optional: Binance API key for advanced features (OI, taker ratio)

## Installation

### 1. Install Poetry

```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### 2. Clone Repository

```bash
cd BinanceAlertManager
```

### 3. Install Dependencies

```bash
poetry install
```

### 4. Configure

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

**Optional**: Add Binance API credentials to `config.yaml` for advanced features:

```yaml
api:
  key: "your_api_key_here"
  secret: "your_api_secret_here"
```

**Without API key**: Service will continue with WebSocket-only features, marking events as `UNCONFIRMED`.

### 5. Initialize Database

```bash
poetry run python -m detector db-migrate
```

## Usage

### Run Detector

```bash
poetry run python -m detector run --config config.yaml
```

The service will:
- **Automatically backfill historical data if needed** (first run only, takes 1-2 minutes)
- Connect to Binance WebSocket streams
- Aggregate ticks into 1-minute bars
- Calculate rolling features and z-scores
- Detect anomaly events
- Print alerts to stdout (and Telegram if configured)
- Persist data to SQLite

Press `Ctrl+C` to stop.

**Note**: On first run, the detector automatically checks if you have enough historical data (720 bars). If not, it backfills ~13 hours of data from Binance before starting. This takes 1-2 minutes and ensures alerts work immediately.

To skip automatic backfill:
```bash
poetry run python -m detector run --config config.yaml --skip-backfill
```

### Generate Report

```bash
poetry run python -m detector report --since 24h --output report.json
```

Options:
- `--since`: Time range (e.g., `24h`, `7d`, `30d`)
- `--output`: Output file path (default: `report.json`)

### Backfill Historical Data (Optional)

**Good news**: The detector now **automatically backfills** historical data on first run! You don't need to run this command separately.

However, if you want to manually backfill (e.g., to refresh data or backfill more than the default):

```bash
poetry run python -m detector backfill --hours 13 --config config.yaml
```

This fetches the last 13 hours of 1-minute klines (780+ bars) for all configured symbols, providing sufficient data for z-score calculations immediately.

Options:
- `--hours`: Number of hours to backfill (default: 13, minimum: 12 for stable z-scores)
- `--config`: Path to config file (default: config.yaml)

**Note**: Backfill takes 1-2 minutes and provides OHLCV + taker buy/sell volume splits.

## Configuration

### Key Parameters

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `universe` | `benchmark_symbol` | `BTCUSDT` | Benchmark for beta calculation |
| `universe` | `sector_symbols` | `[ZECUSDT, DASHUSDT, XMRUSDT]` | Sector coins to monitor |
| `windows` | `zscore_lookback_bars` | `720` | Rolling window for z-scores (12h) |
| `windows` | `beta_lookback_bars` | `240` | Rolling window for beta (4h) |
| `thresholds` | `excess_return_z_initiator` | `3.0` | Z-score threshold for excess return |
| `thresholds` | `volume_z_initiator` | `3.0` | Z-score threshold for volume |
| `thresholds` | `taker_dominance_min` | `0.65` | Taker buy share threshold |
| `thresholds` | `sector_k_min` | `2` | Minimum followers for sector event |
| `alerts` | `cooldown_minutes_per_symbol` | `60` | Cooldown period (minutes) |
| `alerts` | `direction_swap_grace_minutes` | `15` | Grace period for opposite direction |

See `config.example.yaml` for full configuration.

## Detection Logic

### Initiator Signal

An initiator event triggers when **ALL** conditions met:

1. **Excess Return Z-Score** >= 3.0
2. **Volume Z-Score** >= 3.0
3. **Taker Buy Share** >= 0.65 OR <= 0.35 (bidirectional)

Direction: Determined by sign of excess return (UP if positive, DOWN if negative)

### Confirmation Status

- **CONFIRMED**: 2+ confirmations (OI delta, liquidations, funding rate)
- **PARTIAL**: 1 confirmation
- **UNCONFIRMED**: No confirmations (or no API key)

### Sector Diffusion

After an initiator, a sector event triggers when:

1. **Followers**: >= 2 additional coins with simplified signal (z >= 2.0 for both ER and VOL)
2. **Sector Share**: >= 40% of sector coins giving signal
3. **Time Window**: Within 2 hours **after** initiator
4. **Same Direction**: Followers must match initiator direction

### Cooldown Logic

- **Same Direction**: Blocked for 60 minutes
- **Opposite Direction**: Allowed after 15 minutes (grace period)

Example:
- UP alert at 10:00
- UP alert at 10:30 → BLOCKED (same direction, < 60m)
- DOWN alert at 10:20 → ALLOWED (opposite direction, >= 15m)
- DOWN alert at 10:25 → BLOCKED (same direction as 10:20, < 60m)

## Alert Examples

### Initiator Alert

```
🚨 SECTOR SHOT - INITIATOR
Symbol: XMRUSDT | Direction: UP | Status: CONFIRMED
Time: 2026-01-13 14:23:00 UTC
Z-Scores: ER=3.2σ, VOL=2.8σ
Taker Buy Share: 68.5%
Beta: 0.85 | Funding: +0.12%
Confirmations: OI_Δ=2.3σ, Liq=1.9σ
```

### Sector Diffusion Alert

```
🎯 SECTOR DIFFUSION DETECTED
Initiator: XMRUSDT (UP) at 2026-01-13 14:23:00 UTC
Followers (3/3 = 100%):
  • ZECUSDT: ER_z=2.4σ, VOL_z=2.1σ
  • DASHUSDT: ER_z=2.8σ, VOL_z=2.5σ
  • LPTUSDT: ER_z=2.3σ, VOL_z=2.2σ
```

## Telegram Setup (Optional)

1. Create a Telegram bot via @BotFather
2. Get your chat ID (use @userinfobot)
3. Update `config.yaml`:

```yaml
alerts:
  telegram:
    enabled: true
    bot_token: "YOUR_BOT_TOKEN"
    chat_id: "YOUR_CHAT_ID"
```

## Testing

```bash
poetry run pytest tests/ -v
```

Tests include:
- Bar aggregation from synthetic ticks
- Robust z-score calculation
- Initiator trigger rules
- Cooldown logic
- Sector diffusion detection

## Limitations

- **aggTrades History**: Backfill limited to last 1-3 days
- **forceOrder Stream**: Optional and undocumented; may not be available
- **API Key Required**: Advanced features (OI, taker ratio) require Binance API key
- **No GUI**: Command-line only (no dashboard or visualization)

## Project Structure

```
BinanceAlertManager/
├── pyproject.toml           # Poetry dependencies
├── config.example.yaml      # Configuration template
├── README.md                # This file
├── detector/
│   ├── __init__.py
│   ├── main.py              # CLI orchestrator
│   ├── config.py            # Config loader
│   ├── models.py            # Data structures
│   ├── binance_ws.py        # WebSocket client
│   ├── binance_rest.py      # REST API client
│   ├── aggregator.py        # Tick-to-bar aggregation
│   ├── features.py          # Feature calculation
│   ├── detector.py          # Anomaly detection
│   ├── storage.py           # SQLite persistence
│   ├── alerts.py            # Alert dispatching
│   ├── report.py            # Report generation
│   └── utils.py             # Utilities
├── tests/
│   ├── test_aggregator.py
│   ├── test_features.py
│   ├── test_detector_rules.py
│   ├── test_cooldown.py
│   └── test_sector_diffusion.py
└── data/
    └── market.db            # SQLite database (created on first run)
```

## Troubleshooting

### WebSocket Connection Issues

- Check firewall settings
- Verify internet connection
- Service will auto-reconnect with exponential backoff

### No Events Detected

- Market may not be exhibiting anomalies
- Try lowering thresholds in `config.yaml`
- Check logs for errors

### Database Locked

- Ensure only one instance running
- WAL mode should prevent most lock issues
- Check `data/market.db-wal` and `data/market.db-shm` files

### API Rate Limits

- REST polling limited to 1200 req/min (Binance limit)
- Service implements exponential backoff
- Reduce `rest_poll_sec` if hitting limits

## License

This project is provided as-is for educational purposes. Use at your own risk. Always test on paper accounts before trading with real funds.

## Contributing

Pull requests welcome! Please ensure:
- Code follows existing style
- Tests pass (`pytest tests/`)
- Documentation updated

## Support

For issues, please open a GitHub issue with:
- Python version
- Poetry version
- Config (sanitized, no API keys)
- Full error traceback
- Steps to reproduce

## Acknowledgments

- Binance API documentation
- Python async/await ecosystem
- Open source community
