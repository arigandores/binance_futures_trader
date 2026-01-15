# Quick Start Guide - BinanceAlertManager

## Get Started in 2 Minutes

### Step 1: Initialize Database
```bash
poetry run python -m detector db-migrate --config config.yaml
```

This creates the SQLite database schema.

---

### Step 2: Run the Detector
```bash
poetry run python -m detector run --config config.yaml
```

That's it! The detector will:
- **Automatically check and backfill historical data** (if needed, takes 1-2 minutes on first run)
- Connect to Binance WebSocket streams
- Start processing real-time data
- Calculate z-scores using the historical data
- **Generate alerts immediately** when conditions are met (no waiting!)

Press `Ctrl+C` to stop.

---

## What Happens on First Run?

On first startup, you'll see:

```
[INFO] Starting Sector Shot Detector...
[INFO] Checking database for historical data...
[INFO] Found 0 bars in database (need 720 for stable z-scores)
[WARNING] Insufficient data in database (0/720 bars)
[INFO] Starting automatic backfill of 13 hours of historical data...
[INFO] This will take 1-2 minutes. Please wait...
[INFO] Backfilled 780 bars for BTCUSDT (total: 780)
[INFO] Backfilled 780 bars for ZECUSDT (total: 1560)
[INFO] Backfilled 780 bars for DASHUSDT (total: 2340)
[INFO] Backfilled 780 bars for XMRUSDT (total: 3120)
[INFO] Automatic backfill complete!
[INFO] Loading rolling windows from database...
[INFO] Backfilled rolling windows for 4 symbols
[INFO] All components started. Press Ctrl+C to stop.
```

**Total time: 1-2 minutes** on first run, then instant on subsequent runs!

---

## What to Expect

### First Run (1-2 minutes)
- Database check → Automatic backfill → System starts
- WebSocket connects and streams real-time ticks
- Bars are aggregated every minute
- Features calculated using historical + new data
- **Alerts can be generated immediately** if market conditions are volatile

### Subsequent Runs (instant)
- Database has data → Skips backfill → System starts immediately
- Continues processing from where it left off

### Logs You'll See
```
[INFO] Starting Sector Shot Detector...
[INFO] Backfilled rolling windows for 4 symbols
[INFO] All components started. Press Ctrl+C to stop.
[INFO] Processing ticks: 100 received, 4 symbols active
[INFO] Closed 4 bars: BTCUSDT(v:123.45, t:89), ZECUSDT(v:12.34, t:15), ...
[INFO] Features calculated: 10 bars processed, last: XMRUSDT (z_er=1.23, z_vol=0.89)
[INFO] DB flush: 4 bars, 4 features written
```

### When You'll See Alerts
Alerts trigger when **ALL** conditions are met:
- **abs(z_er_15m) >= 3.0** - Strong excess return vs BTC (bidirectional)
- **z_vol_15m >= 3.0** - Unusual volume spike
- **Taker buy share >= 0.65 OR <= 0.35** - Extreme directional pressure

Example alert:
```
🚨 SECTOR SHOT - INITIATOR
Symbol: XMRUSDT | Direction: UP | Status: CONFIRMED
Time: 2026-01-14 15:23:00 UTC
Z-Scores: ER=3.2σ, VOL=2.8σ
Taker Buy Share: 68.5%
Beta: 0.85 | Funding: +0.12%
Confirmations: OI_Δ=2.3σ, Liq=1.9σ
```

---

## Troubleshooting

### "No bars found in database"
Run the backfill command first (Step 2).

### "Database not initialized"
Run db-migrate first (Step 1).

### "No alerts after 1 hour"
This is normal if the market isn't volatile. The thresholds (z >= 3.0) are strict to avoid false positives.

To verify the system is working:
```bash
python check_database.py
```

This shows:
- How many bars are stored
- Recent z-scores for each symbol
- Whether any values are close to triggering thresholds

### Want More Alerts?
If you're confident the system is working but want more frequent alerts, lower the thresholds in `config.yaml`:

```yaml
thresholds:
  excess_return_z_initiator: 2.5  # Was 3.0
  volume_z_initiator: 2.5  # Was 3.0
  taker_dominance_min: 0.60  # Was 0.65
```

**Warning**: Lower thresholds = more alerts but potentially more false positives.

---

## Optional: Verify Installation

Check that everything is working:

```bash
# Check database after backfill
python check_database.py

# Expected output:
# [OK] Database found at data\market.db
# Total bars: 3120 (780 per symbol x 4 symbols)
# Total features: 3120
# Recent features show z-scores being calculated
```

---

## Summary

✓ **Step 1**: `db-migrate` - Create database schema
✓ **Step 2**: `run` - Start detector (auto-backfills on first run, then works immediately!)

**Just 2 commands!** The system automatically handles everything else, including historical data backfill on first run.

**No more waiting 12+ hours!** The system is ready to detect anomalies from the moment it starts.
