# Auto-Backfill Feature - Even Simpler!

## What Changed?

Based on your feedback, I've made the system even easier to use. The backfill now happens **automatically** when you start the detector!

---

## How It Works Now

### The Old Way (3 commands):
```bash
# Step 1
poetry run python -m detector db-migrate

# Step 2
poetry run python -m detector backfill --hours 13

# Step 3
poetry run python -m detector run
```

### The New Way (2 commands):
```bash
# Step 1
poetry run python -m detector db-migrate

# Step 2
poetry run python -m detector run
# ↑ This now automatically backfills if needed!
```

---

## What Happens on Startup?

When you run `poetry run python -m detector run`, the system:

1. **Checks the database** for historical data
2. **If less than 720 bars found** → Automatically backfills 13 hours
3. **If 720+ bars found** → Skips backfill, starts immediately
4. **Continues normally** → Connects to WebSocket and starts detecting

---

## Example: First Run

```bash
$ poetry run python -m detector run --config config.yaml

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

**Time: 1-2 minutes**, then fully operational!

---

## Example: Subsequent Runs

```bash
$ poetry run python -m detector run --config config.yaml

[INFO] Starting Sector Shot Detector...
[INFO] Checking database for historical data...
[INFO] Found 1250 bars in database (need 720 for stable z-scores)
[INFO] Sufficient historical data found, skipping backfill
[INFO] Loading rolling windows from database...
[INFO] Backfilled rolling windows for 4 symbols
[INFO] All components started. Press Ctrl+C to stop.
```

**Time: Instant!** No backfill needed.

---

## Advanced Options

### Skip Auto-Backfill

If you want to skip the automatic backfill (e.g., you know the DB is empty and want to collect data naturally):

```bash
poetry run python -m detector run --config config.yaml --skip-backfill
```

### Manual Backfill

You can still manually backfill if needed:

```bash
poetry run python -m detector backfill --hours 13 --config config.yaml
```

**When to use manual backfill:**
- You want to refresh old data
- You want to backfill more than 13 hours
- You want to pre-populate data before starting the detector

---

## Technical Details

### How It Detects If Backfill Is Needed

The detector:
1. Queries the database for the benchmark symbol (BTCUSDT)
2. Counts how many bars exist
3. Compares to `zscore_lookback_bars` config (default: 720)
4. If `bars_count < 720` → Triggers backfill

### What Gets Backfilled

Same as manual backfill:
- Last 13 hours of 1-minute klines
- OHLCV data
- Taker buy/sell volume splits
- Trade counts and notional volume

### Performance

- **First run**: 1-2 minutes (includes backfill)
- **Subsequent runs**: Instant (skips backfill)
- **API calls**: 4 requests (one per symbol)
- **Database writes**: Batched for efficiency

---

## Benefits

✓ **Simpler workflow** - Just 2 commands instead of 3
✓ **No manual intervention** - System handles data management
✓ **Idempotent** - Safe to restart, won't re-backfill unnecessarily
✓ **Smart** - Only backfills when needed
✓ **Fast** - Subsequent runs start instantly

---

## Migration Guide

If you previously ran the manual backfill:

**Nothing changes!** Your data is already there, so:
- Run `poetry run python -m detector run`
- System detects 720+ bars exist
- Skips backfill
- Starts immediately

**No action required!**

---

## Summary

You asked: "Can we do backfill at the start of main script without running another script?"

**Answer: YES! ✓**

The detector now automatically:
1. Checks for historical data on startup
2. Backfills if needed (first run only)
3. Skips backfill if data already exists
4. Starts detecting immediately

**Result:**
- Before: 3 commands to get started
- **Now: 2 commands to get started**
- First run: 1-2 minutes (auto-backfill)
- Subsequent runs: Instant

The system is now truly plug-and-play! 🚀
