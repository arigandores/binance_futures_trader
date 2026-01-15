# Quick Reference Card

## 🚀 Getting Started (2 Commands)

```bash
# Step 1: Create database
poetry run python -m detector db-migrate

# Step 2: Run (auto-backfills on first run)
poetry run python -m detector run
```

**That's it!** System is ready in 1-2 minutes.

---

## 📋 Common Commands

```bash
# Start detector (production)
poetry run python -m detector run --config config.yaml

# Skip auto-backfill (if you know DB is empty)
poetry run python -m detector run --skip-backfill

# Manual backfill (optional)
poetry run python -m detector backfill --hours 13

# Check database status
python check_database.py

# Generate report
poetry run python -m detector report --since 24h --output report.json

# Run tests
poetry run pytest tests/ -v
```

---

## 🐛 Bugs Fixed

1. ✅ Database not being populated → **FIXED**
2. ✅ Only detecting upward moves → **FIXED** (now bidirectional)
3. ✅ Sector diffusion broken → **FIXED**

---

## ✨ Features Added

1. ✅ Auto-backfill on startup (1-2 min)
2. ✅ Smart data validation (freshness check)
3. ✅ Stale data clearing

---

## 🎯 Alert Conditions

Alerts trigger when **ALL** met:
- `abs(z_er_15m) >= 3.0` - Strong move (up or down)
- `z_vol_15m >= 3.0` - Volume spike
- `taker_share >= 0.65 or <= 0.35` - Directional pressure

---

## ⚙️ Configuration (config.yaml)

### Lower thresholds for more alerts:
```yaml
thresholds:
  excess_return_z_initiator: 2.5  # Default: 3.0
  volume_z_initiator: 2.5         # Default: 3.0
  taker_dominance_min: 0.60       # Default: 0.65
```

### Adjust symbols:
```yaml
universe:
  benchmark_symbol: "BTCUSDT"
  sector_symbols:
    - "ZECUSDT"
    - "DASHUSDT"
    - "XMRUSDT"
    # Add more symbols here
```

---

## 🔍 Troubleshooting

### No alerts after 1 hour?
✓ Normal if market isn't volatile
✓ Run: `python check_database.py`
✓ Look for z-scores near 2.5-3.0

### Database errors?
✓ Only run one detector instance
✓ Check file permissions on `data/` folder

### Backfill fails?
✓ Check internet connection
✓ Verify Binance API is accessible
✓ System continues with existing data

---

## 📊 What to Expect

### First Run
- Auto-backfill: 1-2 minutes
- Logs: "Backfilled 780 bars for BTCUSDT..."
- Status: Alert-ready immediately

### Subsequent Runs
- Start time: < 1 second
- Logs: "Sufficient data, skipping backfill"
- Status: Continues from where it left off

### After 3+ Hour Downtime
- Auto-clear: Old stale data removed
- Auto-backfill: Fresh 13 hours loaded
- Status: Based on current market conditions

---

## 📝 Important Files

- `config.yaml` - Your configuration
- `data/market.db` - SQLite database
- `check_database.py` - Diagnostic tool
- `FINAL_SUMMARY.md` - Complete documentation

---

## 🎉 Success Indicators

✅ Logs show: "DB flush: X bars, Y features written"
✅ `check_database.py` shows 720+ bars
✅ Features show non-zero z-scores
✅ No errors in logs

---

## 💡 Pro Tips

1. **Monitor logs** - Look for z-scores approaching 3.0
2. **Run diagnostics** - Use `check_database.py` after 1 hour
3. **Test configuration** - Lower thresholds to verify system works
4. **Check cooldowns** - Alerts blocked for 60 min (same direction)

---

## ⏱️ Timeline

- **Setup**: 30 seconds (db-migrate)
- **First run**: 1-2 minutes (auto-backfill)
- **Subsequent runs**: Instant
- **Alert capability**: Immediate after backfill

**Total time to production: ~2 minutes!**

---

## 🆘 Quick Diagnostics

```bash
# Check if database has data
python check_database.py

# Check if system is calculating features
# Look in logs for: "Features calculated: X bars processed"

# Check if WebSocket is connected
# Look in logs for: "Processing ticks: X received"

# Check for high z-scores
# Run: python check_database.py
# Look for z_er or z_vol near 3.0
```

---

## 📞 Getting Help

1. Read `FINAL_SUMMARY.md` for complete details
2. Check logs for error messages
3. Run `python check_database.py` for diagnostics
4. Verify `config.yaml` settings

---

**Everything is fixed and ready to go! 🚀**
