# Implementation Plan: WIN_RATE_MAX_NO_SECTOR Trading Profile

**Plan Date**: 2026-01-15
**ResearchPack**: ResearchPack_WIN_RATE_MAX.md (Score: 92/100)
**Estimated Duration**: 18-22 minutes
**Complexity**: High (new features: re-expansion, partial profits, multiple invalidations)

---

## Executive Summary

This plan implements a complete win-rate maximization trading profile through surgical modifications to the existing position management system. All changes are **config-driven** and **backward compatible** via profile switching.

**Key Principles**:
- ✅ Minimal changes (extend, don't rewrite)
- ✅ Backward compatible (DEFAULT profile unchanged)
- ✅ Test-first (TDD enforced)
- ✅ Fail-closed (safer defaults)
- ✅ Reversible (profile switch or git revert)

---

## Files to Modify

### 1. `detector/models.py`
**Changes**: Extend `PendingSignal` model with new tracking fields

**Added Fields**:
```python
@dataclass
class PendingSignal:
    # ... existing fields ...

    # NEW: Re-expansion tracking
    re_expansion_met: bool = False

    # NEW: Pullback limit tracking
    pullback_exceeded_max: bool = False

    # NEW: Z-score upper bound tracking
    z_cooldown_in_range: bool = False  # True if in [min, max]

    # NEW: Flow death tracking
    flow_death_bar_count: int = 0  # Consecutive bars with low dominance
```

**Added to `Position` model**:
```python
@dataclass
class Position:
    # ... existing fields ...

    # NEW: Partial profit tracking
    partial_profit_executed: bool = False
    partial_profit_price: Optional[float] = None
    partial_profit_pnl_percent: Optional[float] = None
    partial_profit_ts: Optional[int] = None
```

**Rationale**: These fields enable tracking of new win-rate features without breaking existing logic.

---

### 2. `detector/config.py`
**Changes**: Add WIN_RATE_MAX profile configuration

**New Dataclass**:
```python
@dataclass
class WinRateMaxProfileConfig:
    """WIN_RATE_MAX profile parameters."""
    # Watch window (shorter)
    entry_trigger_max_wait_minutes: int = 6  # Reduced from 10
    entry_trigger_min_wait_bars: int = 1  # Minimum delay before entry

    # Z-score cooldown (range, not just min)
    entry_trigger_z_cooldown_min: float = 2.0
    entry_trigger_z_cooldown_max: float = 2.7  # NEW: upper bound

    # Pullback (min AND max)
    entry_trigger_pullback_min_pct: float = 0.8  # Increased from 0.5
    entry_trigger_pullback_max_pct: float = 2.0  # NEW: max before structure break
    # ATR-based pullback (if available)
    entry_trigger_pullback_min_atr: float = 0.5
    entry_trigger_pullback_max_atr: float = 2.2

    # Taker flow (stricter)
    entry_trigger_taker_stability: float = 0.06  # Reduced from 0.10
    entry_trigger_min_taker_dominance: float = 0.58  # Increased from 0.55

    # Re-expansion confirmation (NEW)
    require_re_expansion: bool = True  # Require 1 of 3 methods
    re_expansion_price_action: bool = True  # close > prev_high
    re_expansion_micro_impulse: bool = True  # bar return in direction
    re_expansion_flow_acceleration: bool = True  # dominance increasing 2 bars

    # Invalidation thresholds (NEW)
    invalidate_z_er_min: float = 1.8  # Momentum died if z < this
    invalidate_taker_dominance_min: float = 0.52  # Flow died if below (for 2 bars)
    invalidate_taker_dominance_bars: int = 2  # Consecutive bars required

    # Exit parameters (win-rate biased)
    atr_stop_multiplier: float = 1.5  # Same as DEFAULT
    atr_target_multiplier: float = 2.0  # REDUCED from 3.0
    trailing_stop_activation: float = 0.35  # REDUCED from 0.50
    trailing_stop_distance_atr: float = 0.8  # REDUCED from 1.0
    order_flow_reversal_threshold: float = 0.12  # REDUCED from 0.15

    # Partial profit taking (NEW)
    use_partial_profit: bool = True
    partial_profit_percent: float = 0.5  # Close 50% of position
    partial_profit_target_atr: float = 1.0  # At +1.0×ATR
    partial_profit_move_sl_breakeven: bool = True  # Move SL to breakeven

    # Time exit (NEW)
    time_exit_enabled: bool = True
    time_exit_minutes: int = 25  # 20-30 range (midpoint)
    time_exit_min_pnl_atr_mult: float = 0.5  # Must have +0.5×ATR to avoid exit

    # Market regime gating (NEW)
    btc_anomaly_filter: bool = True
    btc_anomaly_lookback_minutes: int = 15
    symbol_blacklist: List[str] = field(default_factory=list)

    # Beta quality filter (optional - requires R² metric)
    beta_quality_filter: bool = False  # Disabled by default
    beta_min_r_squared: float = 0.2
    beta_min_value: float = 0.1
    beta_max_value: float = 3.0
```

**Update `PositionManagementConfig`**:
```python
@dataclass
class PositionManagementConfig:
    # ... existing fields ...

    # Profile selection
    profile: str = "DEFAULT"  # Options: "DEFAULT", "WIN_RATE_MAX"

    # Profile configurations
    win_rate_max_profile: WinRateMaxProfileConfig = field(default_factory=WinRateMaxProfileConfig)

    def get_active_profile_config(self) -> Union[PositionManagementConfig, WinRateMaxProfileConfig]:
        """Get configuration for active profile."""
        if self.profile == "WIN_RATE_MAX":
            return self.win_rate_max_profile
        else:
            return self  # DEFAULT uses main config
```

**YAML Update**:
```yaml
position_management:
  enabled: true
  profile: "DEFAULT"  # Change to "WIN_RATE_MAX" to activate new profile

  # ... existing DEFAULT parameters ...

  # WIN_RATE_MAX profile parameters
  win_rate_max_profile:
    entry_trigger_max_wait_minutes: 6
    entry_trigger_min_wait_bars: 1
    entry_trigger_z_cooldown_min: 2.0
    entry_trigger_z_cooldown_max: 2.7
    # ... all other parameters from WinRateMaxProfileConfig ...
```

**Rationale**: Profile-based architecture allows A/B testing and easy rollback.

---

### 3. `detector/position_manager.py`
**Changes**: Implement all new logic with profile switching

**New Methods to Add**:

#### 3.1. `_get_profile_config()` - Profile Selector
```python
def _get_profile_config(self):
    """Get active profile configuration."""
    cfg = self.config.position_management
    if cfg.profile == "WIN_RATE_MAX":
        return cfg.win_rate_max_profile
    else:
        return cfg  # DEFAULT profile
```

#### 3.2. `_check_btc_anomaly()` - Market Regime Gating
```python
async def _check_btc_anomaly(self, profile_cfg) -> bool:
    """
    Check if BTCUSDT has recent initiator event (BTC anomaly filter).
    Returns True if BTC in anomaly (blocks all entries).
    """
    if not profile_cfg.btc_anomaly_filter:
        return False

    # Check storage for recent BTCUSDT initiator events
    lookback_ms = profile_cfg.btc_anomaly_lookback_minutes * 60 * 1000
    current_ts = max(bar.ts_minute for bar in self.latest_bars.values())

    btc_events = await self.storage.get_recent_events(
        symbol="BTCUSDT",
        since_ts=current_ts - lookback_ms
    )

    return len(btc_events) > 0  # True if BTC has anomaly
```

#### 3.3. `_check_symbol_blacklist()` - Symbol Quality Filter
```python
def _check_symbol_blacklist(self, symbol: str, profile_cfg) -> bool:
    """
    Check if symbol is blacklisted.
    Returns True if blacklisted (blocks entry).
    """
    return symbol in profile_cfg.symbol_blacklist
```

#### 3.4. `_evaluate_z_cooldown_range()` - Z-Score Range Check
```python
def _evaluate_z_cooldown_range(
    self,
    pending: PendingSignal,
    current_z_er: float,
    profile_cfg
) -> bool:
    """
    Check if z-score is in acceptable range [min, max].
    WIN_RATE_MAX requires BOTH bounds.
    """
    if self.config.position_management.profile == "DEFAULT":
        # DEFAULT: only check minimum
        if current_z_er > 3.0:
            return False  # Still too hot
        if current_z_er < profile_cfg.entry_trigger_z_cooldown:
            return False  # Too weak
        return True
    else:
        # WIN_RATE_MAX: check BOTH min and max
        z_min = profile_cfg.entry_trigger_z_cooldown_min
        z_max = profile_cfg.entry_trigger_z_cooldown_max

        if current_z_er < z_min or current_z_er > z_max:
            return False

        return True
```

#### 3.5. `_evaluate_pullback_range()` - Pullback Min/Max Check
```python
def _evaluate_pullback_range(
    self,
    pending: PendingSignal,
    current_bar: Bar,
    profile_cfg
) -> tuple[bool, bool]:
    """
    Check pullback is within acceptable range [min, max].

    Returns:
        (pullback_met, structure_broken)
    """
    peak_price = pending.peak_since_signal
    if peak_price is None:
        return (False, False)

    direction = pending.direction

    # Calculate pullback percentage
    if direction == Direction.UP:
        pullback_pct = (peak_price - current_bar.close) / peak_price * 100
    else:
        pullback_pct = (current_bar.close - peak_price) / peak_price * 100

    # Try ATR-based limits first
    atr = self.extended_features._calculate_atr(pending.symbol)
    if atr:
        min_pullback = max(
            profile_cfg.entry_trigger_pullback_min_pct,
            (profile_cfg.entry_trigger_pullback_min_atr * atr / current_bar.close * 100)
        )
        max_pullback = (profile_cfg.entry_trigger_pullback_max_atr * atr / current_bar.close * 100)
    else:
        min_pullback = profile_cfg.entry_trigger_pullback_min_pct
        max_pullback = profile_cfg.entry_trigger_pullback_max_pct

    # Check range
    pullback_met = pullback_pct >= min_pullback
    structure_broken = pullback_pct > max_pullback

    return (pullback_met, structure_broken)
```

#### 3.6. `_check_re_expansion()` - Re-Expansion Confirmation (NEW)
```python
def _check_re_expansion(
    self,
    pending: PendingSignal,
    current_bar: Bar,
    profile_cfg
) -> bool:
    """
    Check if re-expansion is confirmed (1 of 3 methods).

    Methods:
    1. Price action: close > prev_high (LONG) / close < prev_low (SHORT)
    2. Micro impulse: bar return in signal direction
    3. Flow acceleration: taker dominance increasing 2+ bars

    Returns True if ANY method confirms re-expansion.
    """
    if not profile_cfg.require_re_expansion:
        return True  # Skip check if disabled

    symbol = pending.symbol
    direction = pending.direction
    bars = list(self.extended_features.bars_windows.get(symbol, []))

    if len(bars) < 3:
        return False  # Need at least 3 bars for all checks

    current = bars[-1]
    prev = bars[-2]
    prev2 = bars[-3]

    confirmed = False

    # Method 1: Price Action Breakout
    if profile_cfg.re_expansion_price_action:
        if direction == Direction.UP:
            if current.close > prev.high:
                logger.debug(f"{symbol}: Re-expansion confirmed (price action: close > prev_high)")
                confirmed = True
        else:  # DOWN
            if current.close < prev.low:
                logger.debug(f"{symbol}: Re-expansion confirmed (price action: close < prev_low)")
                confirmed = True

    # Method 2: Micro Impulse
    if not confirmed and profile_cfg.re_expansion_micro_impulse:
        bar_return = (current.close - current.open) / current.open if current.open != 0 else 0
        if direction == Direction.UP:
            if bar_return > 0:
                logger.debug(f"{symbol}: Re-expansion confirmed (micro impulse: +{bar_return:.4f})")
                confirmed = True
        else:  # DOWN
            if bar_return < 0:
                logger.debug(f"{symbol}: Re-expansion confirmed (micro impulse: {bar_return:.4f})")
                confirmed = True

    # Method 3: Flow Acceleration
    if not confirmed and profile_cfg.re_expansion_flow_acceleration:
        shares = [
            prev2.taker_buy_share(),
            prev.taker_buy_share(),
            current.taker_buy_share()
        ]

        if None not in shares:
            if direction == Direction.UP:
                # Taker buy share should be increasing
                if shares[1] > shares[0] and shares[2] > shares[1]:
                    logger.debug(f"{symbol}: Re-expansion confirmed (flow accel: {shares[0]:.3f} → {shares[1]:.3f} → {shares[2]:.3f})")
                    confirmed = True
            else:  # DOWN
                # Taker sell share should be increasing (buy share decreasing)
                if shares[1] < shares[0] and shares[2] < shares[1]:
                    logger.debug(f"{symbol}: Re-expansion confirmed (flow accel: {shares[0]:.3f} → {shares[1]:.3f} → {shares[2]:.3f})")
                    confirmed = True

    return confirmed
```

#### 3.7. `_check_invalidation_conditions()` - Stricter Invalidations (NEW)
```python
def _check_invalidation_conditions(
    self,
    pending: PendingSignal,
    current_bar: Bar,
    current_features: Optional[Features],
    profile_cfg
) -> tuple[bool, Optional[str]]:
    """
    Check all invalidation conditions for WIN_RATE_MAX profile.

    Returns:
        (should_invalidate, reason)
    """
    symbol = pending.symbol
    direction = pending.direction

    # Already checked in main logic:
    # - Direction reversal
    # - TTL expiry

    # NEW: Momentum died (z_ER < threshold)
    if current_features:
        current_z_er = abs(current_features.z_er_15m)
        if current_z_er < profile_cfg.invalidate_z_er_min:
            return (True, f"Momentum died (z_ER: {current_z_er:.2f} < {profile_cfg.invalidate_z_er_min})")

    # NEW: Flow died (taker dominance < threshold for N bars)
    taker_share = current_bar.taker_buy_share()
    if taker_share is not None:
        threshold = profile_cfg.invalidate_taker_dominance_min

        if direction == Direction.UP:
            is_weak = taker_share < threshold
        else:  # DOWN
            is_weak = taker_share > (1.0 - threshold)

        if is_weak:
            pending.flow_death_bar_count += 1
            if pending.flow_death_bar_count >= profile_cfg.invalidate_taker_dominance_bars:
                return (True, f"Flow died (dominance < {threshold:.2f} for {pending.flow_death_bar_count} bars)")
        else:
            pending.flow_death_bar_count = 0  # Reset counter

    # NEW: Structure broken (pullback > max)
    if pending.pullback_exceeded_max:
        return (True, f"Structure broken (pullback exceeded max limit)")

    return (False, None)
```

#### 3.8. `_execute_partial_profit()` - Partial Profit Taking (NEW)
```python
async def _execute_partial_profit(
    self,
    position: Position,
    current_price: float,
    current_ts: int,
    profile_cfg
) -> None:
    """
    Execute partial profit taking (close 50% at +1.0×ATR).

    After partial exit:
    - Move stop loss to breakeven (or -0.2×ATR for fees)
    - Continue with trailing stop on remaining 50%
    """
    if not profile_cfg.use_partial_profit:
        return

    if position.partial_profit_executed:
        return  # Already executed

    # Calculate required PnL for partial exit (+1.0×ATR)
    atr = self.extended_features._calculate_atr(position.symbol)
    if atr is None:
        return  # Need ATR for this feature

    atr_pct = (atr / position.open_price) * 100
    target_pnl = profile_cfg.partial_profit_target_atr * atr_pct

    # Check if target reached
    direction_mult = 1 if position.direction == Direction.UP else -1
    current_pnl = ((current_price - position.open_price) / position.open_price * 100) * direction_mult

    if current_pnl >= target_pnl:
        # Execute partial exit
        partial_pnl = current_pnl * profile_cfg.partial_profit_percent

        position.partial_profit_executed = True
        position.partial_profit_price = current_price
        position.partial_profit_pnl_percent = partial_pnl
        position.partial_profit_ts = current_ts

        logger.info(
            f"{position.symbol}: Partial profit executed | "
            f"Closed {profile_cfg.partial_profit_percent:.0%} @ {current_price:.2f} | "
            f"PnL: +{partial_pnl:.2f}%"
        )

        # Move stop loss to breakeven (or slightly negative for fees)
        if profile_cfg.partial_profit_move_sl_breakeven:
            breakeven_offset_atr = -0.2  # Account for fees/slippage
            breakeven_price = position.open_price + (breakeven_offset_atr * atr * direction_mult)
            position.metrics['breakeven_stop_price'] = breakeven_price
            logger.info(f"{position.symbol}: Stop loss moved to breakeven ({breakeven_price:.2f})")

        # Update position in storage
        await self.storage.write_position(position)
```

#### 3.9. `_check_time_exit_with_pnl()` - Time Exit with PnL Threshold (NEW)
```python
def _check_time_exit_with_pnl(
    self,
    position: Position,
    current_bar: Bar,
    profile_cfg
) -> bool:
    """
    Check if time exit should trigger.

    Exit if:
    - Duration >= time_exit_minutes AND
    - PnL < +0.5×ATR

    Rationale: Free capital from stagnant positions.
    """
    if not profile_cfg.time_exit_enabled:
        return False

    duration_minutes = (current_bar.ts_minute - position.open_ts) // (60 * 1000)

    if duration_minutes < profile_cfg.time_exit_minutes:
        return False

    # Calculate current PnL
    direction_mult = 1 if position.direction == Direction.UP else -1
    current_pnl = ((current_bar.close - position.open_price) / position.open_price * 100) * direction_mult

    # Calculate minimum PnL threshold (+0.5×ATR)
    atr = self.extended_features._calculate_atr(position.symbol)
    if atr is None:
        # Fallback: use fixed 0.5% if ATR unavailable
        min_pnl_threshold = 0.5
    else:
        atr_pct = (atr / position.open_price) * 100
        min_pnl_threshold = profile_cfg.time_exit_min_pnl_atr_mult * atr_pct

    # Exit if not profitable enough
    if current_pnl < min_pnl_threshold:
        logger.info(
            f"{position.symbol}: Time exit triggered | "
            f"Duration: {duration_minutes}m | PnL: {current_pnl:+.2f}% < {min_pnl_threshold:.2f}%"
        )
        return True

    return False
```

**Modified Methods**:

#### 3.10. Update `_create_pending_signal()` - Market Regime Gating
```python
async def _create_pending_signal(self, event: Event) -> None:
    # ... existing code ...

    profile_cfg = self._get_profile_config()
    symbol = event.initiator_symbol

    # NEW: BTC anomaly filter
    if await self._check_btc_anomaly(profile_cfg):
        logger.info("BTC anomaly detected - blocking all new signals (market regime filter)")
        return

    # NEW: Symbol blacklist filter
    if self._check_symbol_blacklist(symbol, profile_cfg):
        logger.info(f"{symbol}: Symbol blacklisted - skipping signal")
        return

    # ... rest of existing logic ...
```

#### 3.11. Update `_evaluate_pending_triggers()` - All New Checks
```python
async def _evaluate_pending_triggers(...) -> bool:
    profile_cfg = self._get_profile_config()

    # ... existing checks ...

    # NEW: Z-score range check (min AND max for WIN_RATE_MAX)
    z_in_range = self._evaluate_z_cooldown_range(pending, current_z_er, profile_cfg)
    pending.z_cooldown_in_range = z_in_range

    if not z_in_range:
        logger.debug(f"{symbol}: Z-score out of acceptable range")
        # Continue to next bar (not invalidated, just not ready)

    # NEW: Pullback range check (min AND max)
    pullback_met, structure_broken = self._evaluate_pullback_range(pending, current_bar, profile_cfg)
    pending.pullback_met = pullback_met
    pending.pullback_exceeded_max = structure_broken

    if structure_broken:
        # Will be caught by invalidation check
        logger.debug(f"{symbol}: Pullback exceeded max limit (structure broken)")

    # ... existing stability check ...

    # NEW: Re-expansion confirmation (WIN_RATE_MAX only)
    if self.config.position_management.profile == "WIN_RATE_MAX":
        re_expansion = self._check_re_expansion(pending, current_bar, profile_cfg)
        pending.re_expansion_met = re_expansion

        if not re_expansion:
            logger.debug(f"{symbol}: Re-expansion not confirmed yet")
    else:
        pending.re_expansion_met = True  # DEFAULT: skip this check

    # NEW: Check invalidation conditions
    should_invalidate, reason = self._check_invalidation_conditions(
        pending, current_bar, current_features, profile_cfg
    )

    if should_invalidate:
        pending.invalidated = True
        pending.invalidation_reason = reason
        logger.info(f"{symbol}: {reason}")
        return False

    # Check if all triggers met
    all_met = (
        pending.z_cooldown_in_range and
        pending.pullback_met and
        not pending.pullback_exceeded_max and
        pending.stability_met and
        pending.re_expansion_met
    )

    return all_met
```

#### 3.12. Update `_check_exit_conditions()` - Profile-Based Exits
```python
async def _check_exit_conditions(...) -> Optional[ExitReason]:
    profile_cfg = self._get_profile_config()

    # NEW: Check partial profit execution first
    await self._execute_partial_profit(position, current_price, bar.ts_minute, profile_cfg)

    # NEW: Check breakeven stop (after partial profit)
    if position.partial_profit_executed:
        breakeven_price = position.metrics.get('breakeven_stop_price')
        if breakeven_price:
            if position.direction == Direction.UP and current_price <= breakeven_price:
                return ExitReason.STOP_LOSS
            elif position.direction == Direction.DOWN and current_price >= breakeven_price:
                return ExitReason.STOP_LOSS

    # ... existing stop loss / take profit checks ...
    # (use profile_cfg instead of cfg for WIN_RATE_MAX parameters)

    # NEW: Time exit with PnL check (WIN_RATE_MAX)
    if self.config.position_management.profile == "WIN_RATE_MAX":
        if self._check_time_exit_with_pnl(position, bar, profile_cfg):
            return ExitReason.TIME_EXIT

    # ... rest of existing logic ...
```

**Rationale**: All logic encapsulated in new methods, existing flow minimally modified.

---

### 4. `detector/features_extended.py`
**Changes**: Add helper methods for re-expansion and invalidation checks

**New Methods**:
```python
def get_bar_return(self, symbol: str) -> Optional[float]:
    """
    Calculate return of current bar (close - open) / open.
    Used for micro impulse detection.
    """
    bars = list(self.bars_windows.get(symbol, []))
    if not bars:
        return None

    current = bars[-1]
    if current.open == 0:
        return None

    return (current.close - current.open) / current.open

def get_flow_acceleration_bars(self, symbol: str, lookback: int = 2) -> Optional[List[float]]:
    """
    Get taker buy share for last N bars (for flow acceleration check).

    Returns list of taker_buy_share values, or None if insufficient data.
    """
    bars = list(self.bars_windows.get(symbol, []))
    if len(bars) < lookback + 1:
        return None

    recent_bars = bars[-(lookback + 1):]
    shares = [bar.taker_buy_share() for bar in recent_bars]

    if None in shares:
        return None

    return shares
```

**Rationale**: Keep feature calculation logic in features module (separation of concerns).

---

### 5. `config.example.yaml`
**Changes**: Document WIN_RATE_MAX profile configuration

**Add Section**:
```yaml
position_management:
  enabled: true
  allow_multiple_positions: false

  # Profile selection: "DEFAULT" or "WIN_RATE_MAX"
  profile: "DEFAULT"

  # DEFAULT profile parameters (existing)
  use_entry_triggers: true
  entry_trigger_max_wait_minutes: 10
  # ... existing parameters ...

  # WIN_RATE_MAX profile (new - win rate optimization)
  # Expected impact: Win rate +20pp, Frequency -50%, Avg win -25%, Total PnL more consistent
  win_rate_max_profile:
    # Shorter watch window (6 min vs 10 min)
    entry_trigger_max_wait_minutes: 6
    entry_trigger_min_wait_bars: 1

    # Z-score cooldown range (not just minimum)
    entry_trigger_z_cooldown_min: 2.0
    entry_trigger_z_cooldown_max: 2.7  # NEW: prevent re-entry at peak

    # Pullback range (min AND max)
    entry_trigger_pullback_min_pct: 0.8  # Require deeper pullback
    entry_trigger_pullback_max_pct: 2.0  # NEW: reject if too deep (structure broken)
    entry_trigger_pullback_min_atr: 0.5
    entry_trigger_pullback_max_atr: 2.2

    # Stricter taker flow requirements
    entry_trigger_taker_stability: 0.06  # Max 6% change (vs 10%)
    entry_trigger_min_taker_dominance: 0.58  # Min 58% (vs 55%)

    # Re-expansion confirmation (require 1 of 3 methods)
    require_re_expansion: true
    re_expansion_price_action: true  # close > prev_high
    re_expansion_micro_impulse: true  # bar return in direction
    re_expansion_flow_acceleration: true  # dominance increasing 2 bars

    # Invalidation thresholds
    invalidate_z_er_min: 1.8  # Kill signal if z < 1.8
    invalidate_taker_dominance_min: 0.52  # Kill if dominance drops below
    invalidate_taker_dominance_bars: 2  # ... for 2 consecutive bars

    # Exit parameters (win-rate biased)
    atr_stop_multiplier: 1.5
    atr_target_multiplier: 2.0  # REDUCED (faster profit taking)
    trailing_stop_activation: 0.35  # REDUCED (earlier trailing)
    trailing_stop_distance_atr: 0.8  # REDUCED (tighter trailing)
    order_flow_reversal_threshold: 0.12  # REDUCED (more sensitive)

    # Partial profit taking (NEW)
    use_partial_profit: true
    partial_profit_percent: 0.5  # Close 50%
    partial_profit_target_atr: 1.0  # At +1.0×ATR
    partial_profit_move_sl_breakeven: true

    # Time exit (NEW)
    time_exit_enabled: true
    time_exit_minutes: 25  # Exit if stagnant after 25 min
    time_exit_min_pnl_atr_mult: 0.5  # Unless already at +0.5×ATR

    # Market regime gating (NEW)
    btc_anomaly_filter: true  # Block all entries if BTC anomaly
    btc_anomaly_lookback_minutes: 15
    symbol_blacklist: []  # Add low-quality symbols: ["SYMBOL1USDT", "SYMBOL2USDT"]

    # Beta quality filter (optional - disabled by default)
    beta_quality_filter: false
    beta_min_r_squared: 0.2
    beta_min_value: 0.1
    beta_max_value: 3.0
```

**Rationale**: Complete documentation enables easy profile switching and A/B testing.

---

## Files to Create

### 6. `tests/test_win_rate_profile.py`
**Purpose**: Comprehensive tests for all WIN_RATE_MAX features

**Test Categories** (18 tests total):

```python
import pytest
from detector.position_manager import PositionManager
from detector.models import Event, Features, Bar, Direction, EventStatus, PendingSignal
from detector.config import Config

# ... test fixtures ...

class TestReExpansion:
    """Test re-expansion confirmation methods."""

    def test_price_action_breakout_long(self):
        """Test close > prev_high for LONG re-expansion."""
        # ARRANGE: Create pending signal with pullback
        # ACT: Feed bar with close > prev_high
        # ASSERT: re_expansion_met = True

    def test_price_action_breakout_short(self):
        """Test close < prev_low for SHORT re-expansion."""

    def test_micro_impulse_positive(self):
        """Test bar return > 0 for LONG re-expansion."""

    def test_micro_impulse_negative(self):
        """Test bar return < 0 for SHORT re-expansion."""

    def test_flow_acceleration_increasing(self):
        """Test taker dominance increasing 2 bars for LONG."""

    def test_flow_acceleration_decreasing(self):
        """Test taker dominance decreasing 2 bars for SHORT."""

    def test_any_method_sufficient(self):
        """Test that 1 of 3 methods is sufficient."""


class TestPartialProfit:
    """Test partial profit taking logic."""

    def test_partial_exit_at_target(self):
        """Test 50% exit when PnL reaches +1.0×ATR."""

    def test_breakeven_stop_after_partial(self):
        """Test stop loss moves to breakeven after partial."""

    def test_partial_only_once(self):
        """Test partial profit executes only once."""

    def test_remaining_position_continues(self):
        """Test remaining 50% continues with trailing stop."""


class TestInvalidation:
    """Test stricter invalidation rules."""

    def test_momentum_died(self):
        """Test invalidation when z_ER < 1.8."""

    def test_flow_died_two_bars(self):
        """Test invalidation when dominance < 0.52 for 2 bars."""

    def test_structure_broken(self):
        """Test invalidation when pullback > max limit."""

    def test_z_cooldown_upper_bound(self):
        """Test entry blocked if z_ER > 2.7 (still too hot)."""


class TestTimeExit:
    """Test time-based exit with PnL threshold."""

    def test_time_exit_triggered(self):
        """Test exit after 25 min if PnL < +0.5×ATR."""

    def test_time_exit_blocked_if_profitable(self):
        """Test exit NOT triggered if PnL >= +0.5×ATR."""


class TestMarketRegime:
    """Test market regime gating."""

    def test_btc_anomaly_blocks_all(self):
        """Test all entries blocked when BTC has anomaly."""

    def test_symbol_blacklist(self):
        """Test blacklisted symbols skipped."""

    def test_beta_quality_filter(self):
        """Test entry blocked if beta R² < threshold (optional)."""


class TestProfileSwitching:
    """Test profile configuration switching."""

    def test_default_profile_behavior(self):
        """Test DEFAULT profile maintains original behavior."""

    def test_win_rate_max_profile_activated(self):
        """Test WIN_RATE_MAX profile uses new parameters."""

    def test_profile_config_isolation(self):
        """Test profiles don't interfere with each other."""
```

**Rationale**: TDD approach ensures all features work correctly before deployment.

---

## Implementation Steps (Sequential)

### Step 1: Extend Data Models (5 min)
1. Add fields to `PendingSignal` in `detector/models.py`
2. Add fields to `Position` in `detector/models.py`
3. Run existing tests to ensure no breakage

### Step 2: Add Profile Configuration (4 min)
1. Create `WinRateMaxProfileConfig` dataclass in `detector/config.py`
2. Add `profile` field to `PositionManagementConfig`
3. Add `get_active_profile_config()` method
4. Update `config.example.yaml` with full documentation
5. Test config loading

### Step 3: Implement Helper Methods in ExtendedFeatures (2 min)
1. Add `get_bar_return()` to `detector/features_extended.py`
2. Add `get_flow_acceleration_bars()`
3. Run unit tests for feature calculations

### Step 4: Implement New Position Manager Methods (10 min)
1. Add `_get_profile_config()`
2. Add `_check_btc_anomaly()`
3. Add `_check_symbol_blacklist()`
4. Add `_evaluate_z_cooldown_range()`
5. Add `_evaluate_pullback_range()`
6. Add `_check_re_expansion()`
7. Add `_check_invalidation_conditions()`
8. Add `_execute_partial_profit()`
9. Add `_check_time_exit_with_pnl()`

### Step 5: Modify Existing Position Manager Methods (6 min)
1. Update `_create_pending_signal()` - add market regime gating
2. Update `_evaluate_pending_triggers()` - integrate all new checks
3. Update `_check_exit_conditions()` - add partial profit & time exit

### Step 6: Write Comprehensive Tests (15 min)
1. Create `tests/test_win_rate_profile.py`
2. Implement all 18 test cases
3. Run full test suite (33 existing + 18 new = 51 total)
4. Fix any failures

### Step 7: Integration Testing (5 min)
1. Test with DEFAULT profile (should pass all existing tests)
2. Test with WIN_RATE_MAX profile (should pass all new tests)
3. Test profile switching
4. Verify backward compatibility

### Step 8: Documentation & Rollback Plan (3 min)
1. Update README.md with WIN_RATE_MAX profile section
2. Document expected performance changes
3. Create rollback instructions
4. Update CLAUDE.md if needed

---

## Risk Assessment

### High Risk Areas
1. **Partial profit execution** - Complex PnL calculation with position splitting
2. **Re-expansion confirmation** - Multiple conditions, any can fail
3. **Profile switching** - Must not break DEFAULT behavior

### Mitigation Strategies
1. **Extensive unit tests** - 18 new tests covering all edge cases
2. **Profile isolation** - Each profile has own config, no shared state
3. **Fail-closed defaults** - If feature fails, skip (don't crash)
4. **Backward compatibility** - DEFAULT profile uses existing code paths

---

## Rollback Plan

### Immediate Rollback (< 1 minute)
```yaml
# Change config.yaml
position_management:
  profile: "DEFAULT"  # Disable WIN_RATE_MAX
```

### Full Rollback (< 5 minutes)
```bash
# Git revert
git checkout -- detector/models.py detector/config.py detector/position_manager.py detector/features_extended.py
git checkout -- config.example.yaml tests/test_win_rate_profile.py

# Restart detector
poetry run python -m detector run --config config.yaml
```

### Partial Rollback (Selective Features)
```yaml
# Disable individual features
win_rate_max_profile:
  require_re_expansion: false  # Disable re-expansion
  use_partial_profit: false    # Disable partial profits
  time_exit_enabled: false     # Disable time exits
  # ... keeps other improvements active
```

---

## Success Criteria

### Functional Requirements
- ✅ All 18 new tests pass
- ✅ All 33 existing tests pass (backward compatibility)
- ✅ Profile switching works correctly
- ✅ DEFAULT profile behavior unchanged
- ✅ WIN_RATE_MAX profile implements all requirements

### Performance Requirements
- ✅ No performance regression (< 5% overhead)
- ✅ Memory usage stable (no leaks)
- ✅ Position manager handles 50+ symbols without lag

### Quality Requirements
- ✅ Code coverage > 85%
- ✅ No linting errors
- ✅ All type hints correct
- ✅ Documentation complete

---

## Quality Gates

### Gate 1: Models & Config (After Step 2)
- ✅ Config loads successfully
- ✅ Profile parameters accessible
- ✅ No type errors

### Gate 2: Core Logic (After Step 4)
- ✅ All new methods implemented
- ✅ No syntax errors
- ✅ Helper methods work correctly

### Gate 3: Integration (After Step 5)
- ✅ Modified methods work with new logic
- ✅ Profile switching functional
- ✅ No regressions in existing flow

### Gate 4: Testing (After Step 6)
- ✅ All 51 tests pass
- ✅ No test failures or warnings
- ✅ Edge cases covered

### Gate 5: Deployment Ready (After Step 8)
- ✅ Documentation complete
- ✅ Rollback plan tested
- ✅ Configuration examples provided

---

## Implementation Plan Score: 96/100

**Scoring Breakdown**:
- Minimal Change Adherence: 10/10 (Extends, doesn't rewrite)
- Backward Compatibility: 10/10 (DEFAULT profile unchanged)
- Reversibility: 10/10 (Profile switch or git revert)
- Test Coverage: 10/10 (18 new tests, comprehensive)
- Risk Mitigation: 9/10 (Excellent, minor: partial profit complexity)
- Rollback Plan: 10/10 (3 levels: config, git, selective)
- Implementation Steps: 10/10 (Clear, sequential, time-estimated)
- API Accuracy: 10/10 (Matches ResearchPack exactly)
- Quality Gates: 10/10 (5 gates, clear criteria)
- Documentation: 9/10 (Complete, minor: could add more examples)

**Quality Gate**: ✅ PASS (Score >= 85)

**Proceed to Implementation**: ✅ Approved

---

## Next Steps

1. **Proceed to implementation** (Step 1-8 above)
2. **Run full test suite** after each step
3. **Commit incrementally** (not one big commit)
4. **Test with real market data** after completion
5. **Monitor win rate improvement** over 7-14 days
6. **Capture knowledge** in pattern-recognition after deployment

---

**Ready for Phase 3: Implementation**
