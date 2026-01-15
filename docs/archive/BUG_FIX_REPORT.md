# Bug Fix Report - BinanceAlertManager

## Investigation Date
2026-01-14

## Problem Statement
Script ran for 3 hours with 0 alerts generated. Investigation revealed critical bugs preventing data storage and alert detection.

---

## Bugs Found and Fixed

### BUG #1: Database Not Being Populated (CRITICAL)
**Severity**: CRITICAL
**File**: `detector/features.py`
**Lines**: 97-117 (FeatureCalculator.run method)

**Issue**:
The FeatureCalculator was receiving bars from the aggregator and calculating features, but never writing them to the database. The Storage class has `batch_write_bars()` and `batch_write_features()` methods, but they were never called.

**Impact**:
- Database remained empty (0 bars, 0 features)
- No historical data for z-score calculations
- No basis for anomaly detection
- **This is the primary reason why you got 0 alerts**

**Root Cause**:
Missing storage calls in the feature processing loop.

**Fix Applied**:
```python
async def run(self) -> None:
    """Consume bars and calculate features."""
    logger.info("Feature calculator started")

    bar_count = 0

    while True:
        try:
            bar = await self.bar_queue.get()

            # ADDED: Write bar to storage buffer
            await self.storage.batch_write_bars([bar])

            features = await self.calculate_features(bar)

            if features:
                # ADDED: Write features to storage buffer
                await self.storage.batch_write_features([features])
                await self.feature_queue.put(features)
                bar_count += 1

                # Log every 10 features calculated
                if bar_count % 10 == 0:
                    logger.info(f"Features calculated: {bar_count} bars processed, last: {bar.symbol} (z_er={features.z_er_15m:.2f}, z_vol={features.z_vol_15m:.2f})")

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
```

---

### BUG #2: Only Detecting Upward Moves (CRITICAL)
**Severity**: CRITICAL
**File**: `detector/detector.py`
**Lines**: 121 (_check_initiator_trigger method)

**Issue**:
The initiator trigger was only detecting upward price movements (positive z-scores), completely ignoring downward movements (negative z-scores).

**Code Before**:
```python
# Rule A: Excess return z-score
if features.z_er_15m < cfg.excess_return_z_initiator:
    return False
```

**Problem**:
- If z_er_15m = 3.5 (strong upward move) → PASSES ✓
- If z_er_15m = -3.5 (strong downward move) → FAILS ✗ (because -3.5 < 3.0)

The system was designed to be bidirectional (the taker share check allows both directions), but the z-score check was only catching positive excess returns.

**Impact**:
- 50% of potential alerts were being ignored (all bearish/downward moves)
- Even with data in the database, downward anomalies would never trigger alerts

**Fix Applied**:
```python
# Rule A: Excess return z-score (bidirectional - use absolute value)
if abs(features.z_er_15m) < cfg.excess_return_z_initiator:
    return False
```

Now both positive and negative z-scores are properly detected:
- If abs(z_er_15m) = abs(3.5) = 3.5 → PASSES ✓
- If abs(z_er_15m) = abs(-3.5) = 3.5 → PASSES ✓

---

### BUG #3: Sector Diffusion Only Detecting Upward Followers
**Severity**: HIGH
**File**: `detector/detector.py`
**Lines**: 223 (_check_sector_diffusion method)

**Issue**:
Similar to Bug #2, the sector diffusion logic was only detecting followers with positive z-scores, ignoring downward moves.

**Code Before**:
```python
# Check simplified signal (z >= 2.0 for both er and vol, same direction)
if (features.z_er_15m >= 2.0 and
    features.z_vol_15m >= 2.0 and
    features.direction == pending.initiator.direction):
```

**Impact**:
- Sector diffusion events would not be detected for bearish coordinated moves
- Even if a bearish initiator was detected (after Bug #2 fix), follower detection would fail

**Fix Applied**:
```python
# Check simplified signal (abs(z) >= 2.0 for both er and vol, same direction)
if (abs(features.z_er_15m) >= 2.0 and
    features.z_vol_15m >= 2.0 and
    features.direction == pending.initiator.direction):
```

---

## Testing Recommendations

### 1. Verify Database Population
After restarting the script, check that data is being written:
```bash
python check_database.py
```

You should see:
- Non-zero bars count
- Non-zero features count
- Recent timestamps in both tables

### 2. Monitor Logs
Watch for these log messages:
- "Features calculated: X bars processed..." (every 10 bars)
- "DB flush: X bars, Y features written" (every 5 seconds)

### 3. Wait for Sufficient Data
The system needs:
- Minimum 15 bars per symbol to start calculating features
- Minimum 720 bars (12 hours) for stable z-score calculations

### 4. Check Alert Generation
With the fixes applied and proper data accumulation, you should start seeing alerts when market conditions meet the criteria:
- abs(z_er_15m) >= 3.0
- z_vol_15m >= 3.0
- taker_buy_share >= 0.65 OR <= 0.35

---

## Configuration Tuning (Optional)

If alerts are still rare after 12+ hours of data collection, consider lowering thresholds in `config.yaml`:

```yaml
thresholds:
  excess_return_z_initiator: 2.5  # Was 3.0
  volume_z_initiator: 2.5  # Was 3.0
  taker_dominance_min: 0.60  # Was 0.65
```

**Note**: Only lower thresholds if you're confident the system is working correctly but market volatility is low.

---

## Summary

Three critical bugs were preventing the alert system from working:

1. **Data not being saved** → Database was empty → No basis for calculations
2. **Only detecting upward moves** → Half of all potential alerts were missed
3. **Sector diffusion broken for downward moves** → Follow-on alerts would fail

All bugs have been fixed. The system should now:
- ✓ Store all bars and features in the database
- ✓ Detect both upward and downward anomalies
- ✓ Track sector diffusion in both directions
- ✓ Generate alerts when market conditions meet the criteria

---

## Next Steps

1. **Restart the script** with the fixed code
2. **Monitor logs** to verify data is being written
3. **Run diagnostics** after 1 hour: `python check_database.py`
4. **Wait for sufficient data** (at least 12 hours for stable z-scores)
5. **Verify alerts** are being generated when market moves occur

The system should now work as designed!
