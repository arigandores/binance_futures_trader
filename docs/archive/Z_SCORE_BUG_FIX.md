# Z-Score Bug Fix - Complete Resolution

## Your Question
**"z=0.00 is correct value?"**

**Answer**: NO - it was a bug! Now fixed.

---

## Bug Details

### The Problem
After backfilling 840 bars, z-scores were all showing **0.00** instead of realistic values based on historical data.

### Root Cause
The `features.backfill()` method loaded bars into memory but **didn't pre-calculate excess returns**.

When the first real-time bar arrived:
- The `excess_returns` deque had only **1 value**
- Z-score calculation requires minimum **10 data points**
- Triggered fallback: `return 0.0`

```python
def _robust_zscore(self, series: List[float]) -> float:
    if len(series) < 10:
        return 0.0  # Fallback - NOT ENOUGH DATA!
```

---

## The Fix

### What Changed
Modified `detector/features.py` backfill method to:

1. Load 840 bars into memory ✓
2. **NEW**: Pre-calculate excess returns for all historical bars ✓
3. Populate `excess_returns` deque with ~825 values ✓
4. Z-scores now calculated from full historical context ✓

### Added Methods
- `_calculate_return_at_index()` - Calculate returns at specific historical points
- `_calculate_beta_at_index()` - Calculate beta for historical data
- Enhanced `backfill()` - Two-pass approach (bars first, then excess returns)

---

## Verification Results

### Before Fix (WRONG)
```
BTCUSDT    z_ER: 0.00  ❌ (fallback)
DASHUSDT   z_ER: 0.00  ❌ (fallback)
ZECUSDT    z_ER: 0.00  ❌ (fallback)
XMRUSDT    z_ER: 0.00  ❌ (fallback)
```

**Problem**: All zeros → Not using historical data

### After Fix (CORRECT)
```
BTCUSDT    z_ER: 0.000  ✓ (correct - it's the benchmark)
DASHUSDT   z_ER: 0.881  ✓ (calculated from 840 bars)
ZECUSDT    z_ER: 0.438  ✓ (calculated from 840 bars)
XMRUSDT    z_ER: 1.190  ✓ (calculated from 840 bars)
```

**Result**: Realistic values based on 840-bar history!

---

## Why BTCUSDT z_ER = 0.000 is Actually Correct

BTCUSDT is the **benchmark symbol**, so mathematically:
- Beta vs itself = 1.000
- Excess Return = r_BTC - (beta × r_BTC) = r_BTC - (1.0 × r_BTC) = **0**
- Z-score of a constant series of zeros = **0.000**

This is **correct by design**!

---

## Test Results

### Test 1: Data Pre-Processing
```
[INFO] Backfilling rolling windows from database...
[INFO] Pre-calculating excess returns for historical data...  ✓ NEW
[INFO] Backfilled rolling windows for 4 symbols with excess returns  ✓ NEW
```

### Test 2: Z-Score Calculations
| Symbol | Beta | z_ER | z_VOL | Status |
|--------|------|------|-------|--------|
| BTCUSDT | 1.000 | 0.000 | 0.100 | ✓ Correct (benchmark) |
| DASHUSDT | 0.809 | 0.881 | -0.308 | ✓ Using 840 bars |
| ZECUSDT | 0.806 | 0.438 | -0.453 | ✓ Using 840 bars |
| XMRUSDT | -0.249 | 1.190 | -1.089 | ✓ Using 840 bars |

### Test 3: Real-Time Operation
- Processed 9,000+ ticks ✓
- Closed bars at minute boundaries ✓
- Features calculated and stored ✓
- Database writes working ✓

---

## Impact on Alert Detection

### Before Fix
- Z-scores always 0.00 → **Never trigger alerts**
- System could never detect anomalies
- Even with extreme market moves, no alerts

### After Fix
- Z-scores calculated correctly → **Will trigger alerts when conditions met**
- System can detect real anomalies
- Thresholds: abs(z_ER) >= 3.0 AND z_VOL >= 3.0 AND extreme taker share

### Current Market (Test Period)
- z_ER values: 0.00 to 1.19 (below threshold of 3.0)
- z_VOL values: -1.09 to 0.10 (below threshold of 3.0)
- **Result**: 0 alerts (correct - market not volatile enough)

---

## Technical Details

### Excess Return Calculation
For each historical bar (after bar 15):
```python
r_15m = log(price[i] / price[i-15])  # 15-minute return
beta = calculate_beta(symbol)         # vs BTC
btc_r_15m = log(btc_price[i] / btc_price[i-15])
er_15m = r_15m - beta * btc_r_15m    # Excess return

window.excess_returns.append(er_15m)  # Store for z-score
```

### Z-Score Calculation
```python
# Uses ALL historical excess returns (825 values)
median = np.median(excess_returns)
mad = np.median(|excess_returns - median|)
robust_sigma = 1.4826 * mad

z = (current_er - median) / robust_sigma  # MAD-based z-score
```

---

## Files Modified

**detector/features.py**:
- Enhanced `backfill()` - Added second pass for excess returns
- Added `_calculate_return_at_index()` - Historical return calculation
- Added `_calculate_beta_at_index()` - Historical beta calculation

---

## Verification Steps

Run these commands to verify the fix:

```bash
# 1. Check z-scores in database
python -c "
import sqlite3
conn = sqlite3.connect('data/market.db')
cursor = conn.cursor()
cursor.execute('SELECT symbol, z_er_15m, z_vol_15m FROM features ORDER BY ts_minute DESC LIMIT 4')
for row in cursor.fetchall():
    print(f'{row[0]}: z_ER={row[1]:.3f}, z_VOL={row[2]:.3f}')
conn.close()
"

# Expected output: Non-zero z-scores for DASH, ZEC, XMR
# BTCUSDT should still be 0.000 (it's the benchmark)
```

---

## Summary

### Question
"z=0.00 is correct value?"

### Answer
**It depends:**

1. **For BTCUSDT (benchmark)**: YES
   - Always 0.000 by mathematical definition
   - This is correct

2. **For other symbols (DASH, ZEC, XMR)**: NO
   - Should vary based on price movements vs BTC
   - **WAS 0.00 (bug)** → **NOW 0.4-1.2 (correct)**

### Status
✅ **BUG FIXED** - Z-scores now properly calculated from 840-bar historical context

The system is now fully operational and will correctly detect anomalies when market conditions meet the thresholds!

---

## All Bugs Fixed Summary

| Bug # | Issue | Status |
|-------|-------|--------|
| 1 | Database not being populated | ✅ FIXED |
| 2 | Only detecting upward moves | ✅ FIXED |
| 3 | Sector diffusion broken | ✅ FIXED |
| 4 | Z-scores always 0.00 | ✅ FIXED |

**All systems operational!** 🎉
