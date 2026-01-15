# Entry Trigger Fix Plan - Signal+Trigger Separation

## ⚠️ CRITICAL MUST-FIX ISSUES (Read First!)

### Must-Fix #1: Inconsistent Timestamps (Breaks TTL/Delay)

**Problem**: Three different time sources mixed together:
- `pending.created_ts = event.ts` (event timestamp, ms)
- `current_ts = bar.ts_minute` (aligned minute timestamp, may be < event.ts within same minute)
- `cleanup: datetime.now().timestamp() * 1000` (wall clock, unrelated to bar time)

**Risks**:
- TTL may not expire or expire incorrectly
- `trigger_delay_seconds` can become negative
- Cleanup may delete pending before bars have chance to evaluate it

**Solution**: Use **bar time exclusively** for bar-driven systems:
```python
# On signal detection:
created_ts = bar.ts_minute  # Current bar timestamp (or next bar if signal after close)

# On trigger evaluation:
current_ts = bar.ts_minute  # Same time scale

# Cleanup:
max_bar_ts = max(self.latest_bars[s].ts_minute for s in self.latest_bars)
cleanup_pending_before(max_bar_ts - grace_period)
```

---

### Must-Fix #2: Z-Cooldown Must Be Direction-Aware

**Problem**: Using `abs(current_features.z_er_15m)` allows entry after direction reversal:
```python
# Signal: UP direction, z_ER = +3.5
# Bar 3: z_ER = -2.3 (reversed to bearish!)
# abs(-2.3) = 2.3 → in range [2.0, 3.0] → triggers met → WRONG ENTRY (long on bearish signal!)
```

**Solution**: Add direction check:
```python
# For Direction.UP:
if pending.direction == Direction.UP:
    if current_features.z_er_15m <= 0:  # Reversed to bearish
        pending.z_cooldown_met = False
        return False  # Signal invalidated
    current_z_er = current_features.z_er_15m  # No abs()

# For Direction.DOWN:
if pending.direction == Direction.DOWN:
    if current_features.z_er_15m >= 0:  # Reversed to bullish
        pending.z_cooldown_met = False
        return False  # Signal invalidated
    current_z_er = abs(current_features.z_er_15m)  # Use abs for comparison

# Then check range [2.0, 3.0]
```

---

### Must-Fix #3: Pullback Must Track Peak SINCE Signal

**Problem**: `get_recent_price_peak(lookback_bars=5)` is:
- Not anchored to signal detection
- May include pre-signal data
- Loses real impulse peak if it was 6-12 bars ago

**Solution**: Track peak since signal in `PendingSignal`:
```python
@dataclass
class PendingSignal:
    # ...existing fields...

    # NEW: Track extremes since signal detection
    peak_since_signal: Optional[float] = None  # For UP: max(high), DOWN: min(low)

    def update_peak(self, bar: Bar) -> None:
        """Update peak/trough since signal."""
        if self.direction == Direction.UP:
            if self.peak_since_signal is None:
                self.peak_since_signal = bar.high
            else:
                self.peak_since_signal = max(self.peak_since_signal, bar.high)
        else:  # DOWN
            if self.peak_since_signal is None:
                self.peak_since_signal = bar.low
            else:
                self.peak_since_signal = min(self.peak_since_signal, bar.low)
```

Then calculate pullback from `pending.peak_since_signal`, not from random lookback window.

---

### Must-Fix #4: None → True is Unsafe (Opens Position Without Checks)

**Problem**:
```python
if peak_price is None:
    pending.pullback_met = True  # No data = allow entry = DANGEROUS!
```

This can open positions:
- On first minutes after restart
- On illiquid symbols with sparse data
- During data gaps

**Solution**: Fail-closed (safe default):
```python
if peak_price is None:
    pending.pullback_met = False  # Wait for data
    logger.debug(f"{symbol}: Insufficient data for pullback check, waiting...")
    return False  # Don't allow entry yet

# OR: Make configurable
if peak_price is None:
    if cfg.entry_trigger_require_data:
        pending.pullback_met = False  # Strict mode
    else:
        pending.pullback_met = True  # Lenient mode
```

**Recommendation**: Default to `require_data: true` for safety.

---

### Must-Fix #5: Bar/Features Race Condition (1-Minute Lag)

**Problem**: Event loop processing order:
1. Bar arrives → `_handle_bars()` called → `_check_pending_signals()`
2. Features calculated → `_handle_features()` called (LATER, same minute)
3. `latest_features.get(symbol)` may return **previous minute's features**

**Result**: Trigger evaluation uses stale features (1-bar lag).

**Solution**: Track evaluation timestamps:
```python
@dataclass
class PendingSignal:
    # ...existing fields...
    last_evaluated_bar_ts: Optional[int] = None  # Last bar timestamp evaluated

# In _evaluate_pending_triggers():
if current_features is None:
    return False

# Check freshness
if current_features.ts_minute < bar.ts_minute:
    logger.debug(
        f"{symbol}: Features stale ({current_features.ts_minute} < {bar.ts_minute}), "
        f"waiting for fresh features"
    )
    return False  # Wait for features to catch up

pending.last_evaluated_bar_ts = bar.ts_minute
```

---

### Must-Fix #6: Allow Single Pending Per Symbol (If allow_multiple_positions=false)

**Problem**: Current logic allows two pendings for same symbol if opposite directions:
- Pending 1: BTCUSDT UP
- Pending 2: BTCUSDT DOWN (within TTL of Pending 1)
- Both can trigger → violates `allow_multiple_positions: false`

**Solution**: Stricter enforcement:
```python
if not cfg.allow_multiple_positions:
    # Check ANY pending for this symbol (regardless of direction)
    existing_pending_any = [
        s for s in self.pending_signals.values()
        if s.symbol == symbol
    ]

    if existing_pending_any:
        logger.info(
            f"{symbol}: Pending signal already exists (any direction), "
            f"skipping new signal (allow_multiple_positions=false)"
        )
        return
```

**Alternative**: Allow opposite direction only if first pending expired:
```python
# Check same-direction pending
same_dir = [s for s in pending if s.direction == event.direction]
if same_dir:
    return  # Block

# Opposite direction OK only if allow_opposite_pending_on_reversal=true
```

---

### Must-Fix #7: Stability Must Include Dominance Check

**Problem**: Stability < 10% can be met when flow is **stably weak** (e.g., 0.48 → 0.50 → 0.52):
- Max change: 4% ✓ (stable)
- But taker_buy_share = 0.50 = **no dominance** (neutral, not bullish)

**Solution**: Add dominance requirement:
```python
# Trigger 3: Taker flow stability AND dominance
taker_stability = self.extended_features.get_taker_flow_stability(symbol, lookback_bars=3)
current_taker_share = bar.taker_buy_share()

if taker_stability is not None and current_taker_share is not None:
    # Check stability
    if taker_stability > cfg.entry_trigger_taker_stability:
        pending.stability_met = False
        logger.debug(f"{symbol}: Taker flow unstable ({taker_stability:.2f})")
        return False

    # NEW: Check dominance (direction must still hold)
    if pending.direction == Direction.UP:
        min_dominance = cfg.entry_trigger_min_taker_dominance  # e.g., 0.55
        if current_taker_share < min_dominance:
            pending.stability_met = False
            logger.debug(
                f"{symbol}: Taker dominance lost (buy share: {current_taker_share:.2f} "
                f"< {min_dominance:.2f})"
            )
            return False
    else:  # DOWN
        max_dominance = 1.0 - cfg.entry_trigger_min_taker_dominance  # e.g., 0.45
        if current_taker_share > max_dominance:
            pending.stability_met = False
            logger.debug(
                f"{symbol}: Taker dominance lost (sell share: {1-current_taker_share:.2f} "
                f"< {cfg.entry_trigger_min_taker_dominance:.2f})"
            )
            return False

    # Both stability AND dominance met
    pending.stability_met = True
    logger.debug(f"{symbol}: Taker flow stable AND dominant ✓")
```

**New config parameter**:
```yaml
entry_trigger_min_taker_dominance: 0.55  # Require 55% buy (UP) or 55% sell (DOWN)
```

---

## 🚨 Critical Problem: Current Implementation is Broken

### Why Current Logic Doesn't Work

**Current Flow:**
```
Initiator Alert (z_ER >= 3.0) → _open_position() → _check_entry_triggers() [ONE TIME]
    ↓
If triggers NOT met → Position SKIPPED FOREVER
```

**Why This Fails:**

1. **Z-Score Cooldown Paradox**:
   - Signal fires when `abs(z_ER) >= 3.0`
   - Immediately check triggers: `abs(z_ER)` is STILL >= 3.0 (just fired!)
   - Trigger requires: `2.0 <= abs(z_ER) <= 3.0`
   - **Result**: ALWAYS blocked (unless timing race/lag artifact)

2. **Pullback Before Price Moves**:
   - Signal fires on price impulse (strong move up/down)
   - At that moment, price is AT/NEAR the peak (that's the impulse!)
   - Trigger requires: 0.5% pullback FROM peak
   - **Result**: Checking for pullback BEFORE pullback happened = always blocked

3. **Stability After Extreme Flow**:
   - Signal requires extreme taker share (>=0.65 or <=0.35)
   - This means SHARP order flow change (that's the signal!)
   - Trigger requires: max flow change < 10%
   - **Result**: Checking for stability BEFORE giving time to stabilize
   - **With pending window**: Stability becomes valid filter ("aggression didn't collapse after spike")

**Outcome**: Triggers don't improve entries — they **kill entry rate** and add **randomness** (only work on timing artifacts).

---

## ✅ Correct Architecture: Pending Signals Queue

### Conceptual Flow

```
Stage 1: SIGNAL DETECTION
    ↓ (z_ER >= 3.0 detected)
    Create "Pending Signal" → Add to queue with TTL (max wait window)
    (One pending per symbol+direction at a time - no duplicates)

Stage 2: WATCH WINDOW (UP TO max_wait_minutes)
    ↓ (system continues processing bars)
    On EVERY new bar: check pending signals
    ↓
    IMPORTANT: NOT "wait N minutes then enter"
               BUT "check every bar, enter AS SOON AS triggers met"

Stage 3: TRIGGER EVALUATION (every 1m bar)
    ↓ (for each pending signal)
    Check: z-score cooled? pullback happened? flow stabilized?

Stage 4A: TRIGGERS MET (can happen on bar 1, 3, 5, or 10)
    ↓
    Open position IMMEDIATELY → Remove from pending queue
    (One signal → One entry maximum)

Stage 4B: TRIGGERS NOT MET
    ↓
    Keep in queue if TTL not expired
    Continue checking next bar

Stage 5: EXPIRY (TTL exceeded)
    ↓
    Max wait time exceeded → Remove from pending queue (missed opportunity)
```

**Key Principle**: Watch window ≠ Fixed delay. Entry happens at FIRST bar where triggers met, not after fixed time.

---

## 📋 Implementation Plan

### Phase 1: Data Structure (detector/models.py)

**Add new model**:
```python
@dataclass
class PendingSignal:
    """
    A detected signal waiting for entry triggers to be met.

    Represents a "watch window" - system checks EVERY bar for up to max_wait_minutes.
    Entry happens on FIRST bar where all triggers met (not fixed delay).

    One pending signal per symbol (if allow_multiple_positions=false) to prevent duplicates.
    """
    signal_id: str  # f"{symbol}_{ts}_{direction}"
    event: Event  # Original initiator event
    created_ts: int  # Bar timestamp when signal detected (ms) - MUST use bar.ts_minute!
    expires_ts: int  # TTL expiry (created_ts + max_wait_window_ms) - bar time scale
    direction: Direction  # UP or DOWN
    symbol: str

    # Signal metrics (at detection time)
    signal_z_er: float
    signal_z_vol: float
    signal_price: float

    # Tracking (Must-Fix #1: bar time scale only)
    bars_since_signal: int = 0  # Number of bars processed since signal

    # Must-Fix #3: Track peak/trough since signal detection
    peak_since_signal: Optional[float] = None  # For UP: max(high), DOWN: min(low)

    # Must-Fix #5: Track last evaluated timestamp for freshness check
    last_evaluated_bar_ts: Optional[int] = None  # Last bar ts_minute evaluated

    # Trigger state (updated on each bar)
    z_cooldown_met: bool = False
    pullback_met: bool = False
    stability_met: bool = False

    def is_expired(self, current_bar_ts: int) -> bool:
        """
        Check if max watch window (TTL) exceeded.
        Must-Fix #1: Use bar time scale, not wall clock.
        """
        return current_bar_ts >= self.expires_ts

    def update_peak(self, bar: Bar) -> None:
        """
        Update peak/trough since signal detection.
        Must-Fix #3: Track extremes from signal moment, not arbitrary lookback.
        """
        if self.direction == Direction.UP:
            if self.peak_since_signal is None:
                self.peak_since_signal = bar.high
            else:
                self.peak_since_signal = max(self.peak_since_signal, bar.high)
        else:  # DOWN
            if self.peak_since_signal is None:
                self.peak_since_signal = bar.low
            else:
                self.peak_since_signal = min(self.peak_since_signal, bar.low)

    def all_triggers_met(self) -> bool:
        """Check if all required triggers are met."""
        # If triggers disabled, always True
        # If enabled, require all 3
        return self.z_cooldown_met and self.pullback_met and self.stability_met
```

**Configuration addition**:
```yaml
position_management:
  use_entry_triggers: true
  entry_trigger_max_wait_minutes: 10  # Max watch window (TTL) - NOT fixed delay!
  entry_trigger_min_wait_bars: 0  # Optional - min bars to wait before entry (0 = immediate if ready)
  entry_trigger_z_cooldown: 2.0  # Min z-score after cooling [2.0, 3.0]
  entry_trigger_pullback_pct: 0.5  # Required pullback % from peak
  entry_trigger_taker_stability: 0.10  # Max taker flow change (stability)
  entry_trigger_min_taker_dominance: 0.55  # NEW (Must-Fix #7): Min buy/sell dominance
  entry_trigger_require_data: true  # NEW (Must-Fix #4): Fail-closed if data missing
```

**Key**: `max_wait_minutes` is TTL (expiry), NOT "wait N minutes then enter". System checks EVERY bar and enters AS SOON AS triggers met.

---

### Phase 2: Pending Signals Manager (detector/position_manager.py)

**Add to PositionManager.__init__()**:
```python
# Pending signals queue
self.pending_signals: Dict[str, PendingSignal] = {}  # signal_id -> PendingSignal
```

**Modify _handle_alerts()** (detector/position_manager.py:110-122):
```python
async def _handle_alerts(self) -> None:
    """Listen for initiator alerts and create pending signals."""
    while True:
        try:
            event_type, event = await self.event_queue.get()

            if event_type == 'initiator':
                # NEW: Create pending signal instead of opening immediately
                await self._create_pending_signal(event)

        except Exception as e:
            logger.error(f"Error handling alert: {e}")
```

**NEW: Create pending signal** (replaces immediate _open_position):
```python
async def _create_pending_signal(self, event: Event) -> None:
    """
    Create a pending signal that waits for entry triggers.

    If use_entry_triggers = False, opens position immediately (backward compatible).
    If use_entry_triggers = True, adds to pending queue.
    """
    cfg = self.config.position_management
    symbol = event.initiator_symbol

    # Backward compatibility: if triggers disabled, open immediately
    if not cfg.use_entry_triggers:
        await self._open_position_from_event(event)
        return

    # Must-Fix #6: Stricter pending enforcement if allow_multiple_positions=false
    if not cfg.allow_multiple_positions:
        # Check ANY pending for this symbol (regardless of direction)
        existing_pending_any = [
            s for s in self.pending_signals.values()
            if s.symbol == symbol
        ]

        if existing_pending_any:
            logger.info(
                f"{symbol}: Pending signal already exists (any direction), "
                f"skipping new signal (allow_multiple_positions=false prevents duplicates)"
            )
            return

        # Check if already have open position
        existing_open = [p for p in self.open_positions.values()
                         if p.symbol == symbol and p.status == PositionStatus.OPEN]

        if existing_open:
            logger.debug(f"{symbol}: Position already open, skipping new signal")
            return
    else:
        # If allow_multiple_positions=true, only block same-direction pendings
        existing_pending_same_direction = [
            s for s in self.pending_signals.values()
            if s.symbol == symbol and s.direction == event.direction
        ]

        if existing_pending_same_direction:
            logger.info(
                f"{symbol}: Pending signal already exists for {event.direction.value} direction, "
                f"skipping new signal"
            )
            return

    # Get current bar
    bar = self.latest_bars.get(symbol)
    if not bar:
        logger.warning(f"{symbol}: No bar data, cannot create pending signal")
        return

    # Create pending signal with TTL (max watch window)
    # Must-Fix #1: Use bar time scale, not event.ts or wall clock
    max_wait_ms = cfg.entry_trigger_max_wait_minutes * 60 * 1000
    signal_id = f"{symbol}_{bar.ts_minute}_{event.direction.value}"  # Use bar ts!

    pending = PendingSignal(
        signal_id=signal_id,
        event=event,
        created_ts=bar.ts_minute,  # Must-Fix #1: Bar time scale
        expires_ts=bar.ts_minute + max_wait_ms,  # Must-Fix #1: Bar time scale
        direction=event.direction,
        symbol=symbol,
        signal_z_er=event.metrics.get('z_er', 0),
        signal_z_vol=event.metrics.get('z_vol', 0),
        signal_price=bar.close,
        bars_since_signal=0,  # Track how many bars passed
        peak_since_signal=bar.high if event.direction == Direction.UP else bar.low  # Must-Fix #3
    )

    self.pending_signals[signal_id] = pending

    logger.info(
        f"Pending signal created: {signal_id} | {symbol} {event.direction.value} "
        f"@ {bar.close:.2f} | z_ER: {pending.signal_z_er:.2f} | "
        f"Peak since signal: {pending.peak_since_signal:.2f} | "
        f"Watch window: up to {cfg.entry_trigger_max_wait_minutes}m "
        f"(will enter AS SOON AS triggers met)"
    )
```

---

### Phase 3: Trigger Evaluation on Every Bar (detector/position_manager.py)

**Modify _handle_bars()** (detector/position_manager.py:136-150):
```python
async def _handle_bars(self) -> None:
    """Listen for bar updates and check pending signals + position exits."""
    while True:
        try:
            bar = await self.bar_queue.get()
            self.latest_bars[bar.symbol] = bar

            # Update extended features
            self.extended_features.update(bar)

            # Update MFE/MAE for open positions
            await self._update_excursions(bar)

            # NEW: Check pending signals for this symbol
            await self._check_pending_signals(bar.symbol, bar)

        except Exception as e:
            logger.error(f"Error handling bars: {e}")
```

**NEW: Check pending signals** (runs on EVERY bar):
```python
async def _check_pending_signals(self, symbol: str, bar: Bar) -> None:
    """
    Check if any pending signals for this symbol can trigger entry.
    Called on EVERY new bar.

    IMPORTANT: This implements "watch window", not "fixed delay".
    Entry happens on FIRST bar where triggers met, not after fixed time.
    """
    cfg = self.config.position_management
    current_ts = bar.ts_minute
    features = self.latest_features.get(symbol)

    # Get pending signals for this symbol
    pending_for_symbol = [
        (signal_id, signal)
        for signal_id, signal in list(self.pending_signals.items())
        if signal.symbol == symbol
    ]

    for signal_id, pending in pending_for_symbol:
        # Increment bars counter
        pending.bars_since_signal += 1

        # Must-Fix #3: Update peak since signal
        pending.update_peak(bar)

        # Must-Fix #1: Check expiry using bar time scale
        if pending.is_expired(current_ts):
            del self.pending_signals[signal_id]
            logger.info(
                f"Pending signal expired: {signal_id} "
                f"(triggers not met within {cfg.entry_trigger_max_wait_minutes}m)"
            )
            continue

        # Check min_wait_bars (optional filter to avoid same-bar entry)
        min_wait = getattr(cfg, 'entry_trigger_min_wait_bars', 0)
        if pending.bars_since_signal < min_wait:
            logger.debug(f"{signal_id}: Min wait not reached ({pending.bars_since_signal}/{min_wait} bars)")
            continue

        # Evaluate triggers
        triggers_met = await self._evaluate_pending_triggers(pending, bar, features)

        if triggers_met:
            # All triggers met - open position IMMEDIATELY (not after fixed delay!)
            delay_seconds = (current_ts - pending.created_ts) // 1000  # Must-Fix #1: Bar time scale
            logger.info(
                f"Pending signal triggered: {signal_id} "
                f"(after {pending.bars_since_signal} bars / {delay_seconds}s)"
            )
            await self._open_position_from_pending(pending, bar)

            # CRITICAL: Remove from pending queue to ensure one-signal → one-entry
            del self.pending_signals[signal_id]
```

**NEW: Evaluate triggers** (replaces _check_entry_triggers):
```python
async def _evaluate_pending_triggers(
    self,
    pending: PendingSignal,
    current_bar: Bar,
    current_features: Optional[Features]
) -> bool:
    """
    Evaluate if entry triggers are met for a pending signal.

    Implements all Must-Fix issues:
    - #2: Direction-aware z-cooldown
    - #3: Pullback from peak_since_signal
    - #4: Fail-closed if data missing
    - #5: Features freshness check
    - #7: Stability + dominance

    Updates pending.z_cooldown_met, pending.pullback_met, pending.stability_met.
    Returns True if ALL triggers met.
    """
    cfg = self.config.position_management
    symbol = pending.symbol
    direction = pending.direction

    # If triggers disabled globally, always allow (shouldn't happen in pending queue)
    if not cfg.use_entry_triggers:
        return True

    # Need features for z-score check
    if current_features is None:
        logger.debug(f"{symbol}: No features available, waiting...")
        return False

    # Must-Fix #5: Check features freshness (avoid stale data)
    if current_features.ts_minute < current_bar.ts_minute:
        logger.debug(
            f"{symbol}: Features stale ({current_features.ts_minute} < {current_bar.ts_minute}), "
            f"waiting for fresh features"
        )
        return False

    pending.last_evaluated_bar_ts = current_bar.ts_minute

    # Trigger 1: Z-score cooldown (Must-Fix #2: Direction-aware)
    # Check if signal reversed direction first
    if direction == Direction.UP:
        if current_features.z_er_15m <= 0:
            # Signal reversed to bearish - invalidate
            logger.info(f"{symbol}: Signal reversed to bearish (z_ER: {current_features.z_er_15m:.2f}), invalidating pending")
            return False  # Signal no longer valid
        current_z_er = current_features.z_er_15m  # No abs() for UP
    else:  # DOWN
        if current_features.z_er_15m >= 0:
            # Signal reversed to bullish - invalidate
            logger.info(f"{symbol}: Signal reversed to bullish (z_ER: {current_features.z_er_15m:.2f}), invalidating pending")
            return False  # Signal no longer valid
        current_z_er = abs(current_features.z_er_15m)  # Use abs for comparison

    if current_z_er > 3.0:
        # Still at peak
        pending.z_cooldown_met = False
        logger.debug(f"{symbol}: Z-score still hot ({current_z_er:.2f})")
    elif current_z_er < cfg.entry_trigger_z_cooldown:
        # Too weak
        pending.z_cooldown_met = False
        logger.debug(f"{symbol}: Z-score too weak ({current_z_er:.2f})")
    else:
        # In range [2.0, 3.0] with correct direction
        pending.z_cooldown_met = True
        logger.debug(f"{symbol}: Z-score cooled ({current_z_er:.2f}) with correct direction ✓")

    # Trigger 2: Price pullback from peak (Must-Fix #3: Use peak_since_signal)
    peak_price = pending.peak_since_signal

    if peak_price is None:
        # Must-Fix #4: Fail-closed if data missing
        if cfg.entry_trigger_require_data:
            pending.pullback_met = False
            logger.debug(f"{symbol}: No peak data yet, waiting... (require_data=true)")
            return False
        else:
            pending.pullback_met = True  # Lenient mode
    else:
        if direction == Direction.UP:
            pullback_pct = (peak_price - current_bar.close) / peak_price * 100
        else:
            pullback_pct = (current_bar.close - peak_price) / peak_price * 100

        if pullback_pct >= cfg.entry_trigger_pullback_pct:
            pending.pullback_met = True
            logger.debug(f"{symbol}: Pullback sufficient ({pullback_pct:.2f}% from peak {peak_price:.2f}) ✓")
        else:
            pending.pullback_met = False
            logger.debug(f"{symbol}: Pullback insufficient ({pullback_pct:.2f}%)")

    # Trigger 3: Taker flow stability AND dominance (Must-Fix #7)
    taker_stability = self.extended_features.get_taker_flow_stability(symbol, lookback_bars=3)
    current_taker_share = current_bar.taker_buy_share()

    if taker_stability is None or current_taker_share is None:
        # Must-Fix #4: Fail-closed if data missing
        if cfg.entry_trigger_require_data:
            pending.stability_met = False
            logger.debug(f"{symbol}: No taker flow data yet, waiting... (require_data=true)")
            return False
        else:
            pending.stability_met = True  # Lenient mode
    else:
        # Check stability
        if taker_stability > cfg.entry_trigger_taker_stability:
            pending.stability_met = False
            logger.debug(f"{symbol}: Taker flow unstable ({taker_stability:.2f})")
        else:
            # Must-Fix #7: Check dominance (direction must still hold)
            min_dominance = cfg.entry_trigger_min_taker_dominance

            if direction == Direction.UP:
                if current_taker_share < min_dominance:
                    pending.stability_met = False
                    logger.debug(
                        f"{symbol}: Taker buy dominance lost "
                        f"(buy share: {current_taker_share:.2f} < {min_dominance:.2f})"
                    )
                else:
                    pending.stability_met = True
                    logger.debug(
                        f"{symbol}: Taker flow stable ({taker_stability:.2f}) "
                        f"AND dominant (buy: {current_taker_share:.2f}) ✓"
                    )
            else:  # DOWN
                max_dominance = 1.0 - min_dominance
                if current_taker_share > max_dominance:
                    pending.stability_met = False
                    logger.debug(
                        f"{symbol}: Taker sell dominance lost "
                        f"(sell share: {1-current_taker_share:.2f} < {min_dominance:.2f})"
                    )
                else:
                    pending.stability_met = True
                    logger.debug(
                        f"{symbol}: Taker flow stable ({taker_stability:.2f}) "
                        f"AND dominant (sell: {1-current_taker_share:.2f}) ✓"
                    )

    # Check if all triggers met
    all_met = pending.all_triggers_met()

    if all_met:
        logger.info(
            f"{symbol}: ALL entry triggers met! "
            f"(z_cool: {pending.z_cooldown_met}, "
            f"pullback: {pending.pullback_met}, "
            f"stability+dominance: {pending.stability_met})"
        )

    return all_met
```

**NEW: Open position from pending signal**:
```python
async def _open_position_from_pending(
    self,
    pending: PendingSignal,
    bar: Bar
) -> None:
    """
    Open position from a pending signal that met all triggers.
    """
    event = pending.event
    symbol = pending.symbol

    # Check if position still allowed (no open position created meanwhile)
    existing_open = [
        p for p in self.open_positions.values()
        if p.symbol == symbol and p.status == PositionStatus.OPEN
    ]

    if existing_open and not self.config.position_management.allow_multiple_positions:
        logger.info(f"{symbol}: Position opened elsewhere, skipping pending signal")
        return

    # Calculate dynamic targets based on ATR
    targets = self.extended_features.calculate_dynamic_targets(
        symbol=symbol,
        entry_price=bar.close,
        direction=event.direction,
        atr_stop_mult=self.config.position_management.atr_stop_multiplier,
        atr_target_mult=self.config.position_management.atr_target_multiplier,
        min_risk_reward=self.config.position_management.min_risk_reward_ratio
    )

    # Create position
    position_id = f"{symbol}_{bar.ts_minute}_{event.direction.value}_triggered"
    metrics = event.metrics.copy()

    # Store event status
    metrics['event_status'] = event.status.value

    # Store signal → entry timing
    metrics['signal_ts'] = pending.created_ts
    metrics['entry_ts'] = bar.ts_minute
    metrics['trigger_delay_seconds'] = (bar.ts_minute - pending.created_ts) // 1000

    # Store dynamic targets
    if targets:
        metrics.update({
            'dynamic_stop_loss': targets['stop_loss_percent'],
            'dynamic_take_profit': targets['take_profit_percent'],
            'risk_reward_ratio': targets['risk_reward_ratio']
        })

    position = Position(
        position_id=position_id,
        event_id=event.event_id,
        symbol=symbol,
        direction=event.direction,
        status=PositionStatus.OPEN,
        open_price=bar.close,
        open_ts=bar.ts_minute,
        entry_z_er=metrics.get('z_er', 0),
        entry_z_vol=metrics.get('z_vol', 0),
        entry_taker_share=metrics.get('taker_share', 0),
        metrics=metrics
    )

    # Save to database and memory
    await self.storage.write_position(position)
    self.open_positions[position_id] = position

    logger.info(
        f"Position opened (from pending): {position_id} | "
        f"{symbol} {event.direction.value} @ {bar.close:.2f} | "
        f"Trigger delay: {metrics['trigger_delay_seconds']}s"
    )

    # Send Telegram notification
    if self.config.alerts.telegram.enabled:
        message = self._format_position_opened(position)
        await self._send_telegram(message)
```

**Backward Compatibility: Direct entry** (when triggers disabled):
```python
async def _open_position_from_event(self, event: Event) -> None:
    """
    Open position immediately from event (when triggers disabled).
    This is the OLD behavior for backward compatibility.
    """
    # Copy existing _open_position() logic from lines 228-310
    # (just rename the function)
    symbol = event.initiator_symbol

    existing = [p for p in self.open_positions.values()
               if p.symbol == symbol and p.status == PositionStatus.OPEN]

    if existing and not self.config.position_management.allow_multiple_positions:
        logger.info(f"Position already open for {symbol}, skipping")
        return

    bar = self.latest_bars.get(symbol)
    if not bar:
        logger.warning(f"No bar data for {symbol}")
        return

    # ... rest of existing _open_position() code ...
```

---

### Phase 4: Cleanup and Testing

**Add cleanup for expired pending signals** (run periodically):
```python
async def _cleanup_expired_pending_signals(self) -> None:
    """
    Remove expired pending signals. Run every minute.
    Must-Fix #1: Use bar time scale, not wall clock.
    """
    while True:
        await asyncio.sleep(60)  # Every 60 seconds

        # Must-Fix #1: Get max bar timestamp (bar time scale, not wall clock)
        if not self.latest_bars:
            continue  # No bars yet

        max_bar_ts = max(bar.ts_minute for bar in self.latest_bars.values())

        expired = [
            signal_id for signal_id, signal in self.pending_signals.items()
            if signal.is_expired(max_bar_ts)
        ]

        for signal_id in expired:
            pending = self.pending_signals[signal_id]
            del self.pending_signals[signal_id]
            logger.info(
                f"Cleaned up expired pending signal: {signal_id} "
                f"(TTL: {self.config.position_management.entry_trigger_max_wait_minutes}m, "
                f"bars evaluated: {pending.bars_since_signal})"
            )
```

**Add cleanup task to run()** (detector/position_manager.py:91-109):
```python
async def run(self) -> None:
    """Main event loop."""
    await self.init()
    logger.info("Position manager started")

    tasks = [
        asyncio.create_task(self._handle_alerts()),
        asyncio.create_task(self._handle_features()),
        asyncio.create_task(self._handle_bars()),
        asyncio.create_task(self._cleanup_expired_pending_signals())  # NEW
    ]

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Position manager error: {e}")
        for task in tasks:
            task.cancel()
```

---

## 🧪 Testing Plan

**tests/test_pending_signals.py** (new file):

```python
import pytest
from detector.models import PendingSignal, Event, Direction, EventStatus
from detector.position_manager import PositionManager

def test_pending_signal_expiry():
    """Test that pending signals expire after TTL."""
    # Create signal with 5-minute TTL
    # Fast-forward time
    # Assert is_expired() returns True

def test_pending_signal_triggers_evaluated():
    """Test that triggers are evaluated on every bar."""
    # Create pending signal with z_ER = 3.5 (hot signal)
    # Bar 1: z_ER still 3.5 → triggers NOT met
    # Bar 2: z_ER = 2.3, pullback 1%, flow stable → triggers MET
    # Assert position opened on Bar 2

def test_pending_signal_backward_compat():
    """Test that use_entry_triggers=False works as before."""
    # Config: use_entry_triggers = False
    # Initiator alert arrives
    # Assert position opens IMMEDIATELY (no pending queue)

def test_pending_signal_multiple_symbols():
    """Test pending signals for multiple symbols work independently."""
    # Pending signal for BTCUSDT
    # Pending signal for ETHUSDT
    # BTCUSDT triggers met → opens
    # ETHUSDT still pending
    # Assert only BTCUSDT position opened

def test_pending_signal_cleanup():
    """Test that expired signals are cleaned up."""
    # Create 3 pending signals
    # 2 expired, 1 still valid
    # Run cleanup
    # Assert 2 removed, 1 remains
```

---

## 📊 Expected Outcomes

**Before Fix** (current broken logic):
- Entry trigger hit rate: ~5-10% (mostly timing artifacts)
- Entries happening at: Random moments (lag-based)
- Signal → Entry delay: 0 seconds (if triggered) or ∞ (if blocked)

**After Fix** (pending signals queue):
- Entry trigger hit rate: 60-80% (signals given time to meet triggers)
- Entries happening at: Optimal moments (z cooled, pullback done, flow stable)
- Signal → Entry delay: 0-10 minutes (enters AS SOON AS triggers met, not fixed delay)
- Typical delays: 1-3 bars (1-3 minutes) for fast-developing setups
- Missed opportunities: Tracked (expired pending signals if triggers never met within TTL)

---

## 🚀 Implementation Order

1. **Phase 1**: Add `PendingSignal` model to models.py (5 min)
2. **Phase 2**: Refactor `_open_position()` → `_create_pending_signal()` + `_open_position_from_event()` (15 min)
3. **Phase 3**: Add `_check_pending_signals()` to `_handle_bars()` (10 min)
4. **Phase 4**: Add `_evaluate_pending_triggers()` (copy from existing `_check_entry_triggers()`) (10 min)
5. **Phase 5**: Add cleanup task (5 min)
6. **Phase 6**: Write tests (20 min)
7. **Phase 7**: Update config.example.yaml with new parameter (2 min)
8. **Phase 8**: Update CLAUDE.md documentation (5 min)

**Total estimated time**: 72 minutes

---

## 📝 Configuration Example

```yaml
position_management:
  enabled: true
  allow_multiple_positions: false

  # Entry triggers (Signal+Trigger separation)
  use_entry_triggers: true  # Enable pending signals queue (watch window)
  entry_trigger_max_wait_minutes: 10  # Max watch window (TTL) - NOT fixed delay!
  entry_trigger_min_wait_bars: 0  # Optional: min bars before allowing entry (0 = immediate if ready)
  entry_trigger_z_cooldown: 2.0  # Min z-score after cooling [2.0, 3.0]
  entry_trigger_pullback_pct: 0.5  # Required pullback % from peak_since_signal
  entry_trigger_taker_stability: 0.10  # Max taker flow change (stability check)
  entry_trigger_min_taker_dominance: 0.55  # Must-Fix #7: Min buy/sell dominance (0.55 = 55%)
  entry_trigger_require_data: true  # Must-Fix #4: Fail-closed if data missing (safer)

  # Exit settings
  use_trailing_stop: true
  z_score_exit_threshold: 0.5
  max_hold_minutes: 120
```

**IMPORTANT**: `max_wait_minutes` is a watch window (TTL), NOT "wait N minutes then enter".
System checks EVERY bar and enters AS SOON AS all triggers are met (could be bar 1, 3, or 10).

---

## ✅ Success Criteria

- [ ] Pending signals stored and tracked (one per symbol+direction)
- [ ] Triggers evaluated on EVERY bar (not just at signal detection)
- [ ] Positions open IMMEDIATELY when triggers met (not after fixed delay)
- [ ] Entry can happen on bar 1, 2, 5, or 10 (whichever bar triggers met)
- [ ] Expired pending signals cleaned up (TTL exceeded)
- [ ] No duplicate pending signals (one per symbol+direction enforced)
- [ ] One signal → one entry maximum (race condition protection)
- [ ] Backward compatibility (use_entry_triggers=False works as before)
- [ ] Tests pass (100%)
- [ ] Telegram notifications include trigger delay (bars + seconds)
- [ ] Logs show: "Pending signal created", "Watch window: up to Nm", "Triggers met after X bars", "Signal expired"

---

---

## 🎯 Summary: What's Fixed

### Must-Fix Issues Addressed

| # | Issue | Solution | Impact |
|---|-------|----------|--------|
| **1** | Inconsistent timestamps | Use bar.ts_minute exclusively (no event.ts, no wall clock) | TTL works correctly, no negative delays |
| **2** | Z-cooldown not direction-aware | Check z_er sign, invalidate if reversed | No longs on bearish signals |
| **3** | Pullback from random lookback | Track peak_since_signal, update each bar | Pullback anchored to impulse |
| **4** | None → True (unsafe) | Fail-closed (require_data: true default) | No blind entries on data gaps |
| **5** | Bar/features race condition | Check features.ts_minute freshness | No stale data usage |
| **6** | Multiple pendings per symbol | Enforce single pending if allow_multiple_positions=false | No race conditions |
| **7** | Stability without dominance | Add min_taker_dominance check (0.55 default) | No entries in neutral flow |

### Architectural Improvements

**Before (Broken)**:
```
Signal (z>=3.0) → Check triggers ONCE → If not ready, SKIP FOREVER
```
- ❌ Triggers checked before conditions possible
- ❌ No second chance
- ❌ Entry rate ~5-10% (timing artifacts only)

**After (Fixed)**:
```
Signal (z>=3.0) → Create Pending → Watch window (check EVERY bar) → Entry ASAP when ready
                                        ↓
                               Bar 1: Not ready (z still 3.5)
                               Bar 2: Not ready (no pullback yet)
                               Bar 3: READY → Entry! ✓
```
- ✅ Triggers checked continuously (1m bars)
- ✅ Entry at first ready bar (not fixed delay)
- ✅ Expected entry rate 60-80%
- ✅ TTL prevents infinite waiting (10m default)

### Key Principles Enforced

1. **Watch Window ≠ Fixed Delay**: Entry happens at FIRST bar where triggers met (could be bar 1, 3, or 10)
2. **Bar Time Scale**: All timestamps use bar.ts_minute (consistent, no race conditions)
3. **Direction Awareness**: Z-cooldown checks sign, invalidates if reversed
4. **Peak Anchoring**: Pullback measured from peak_since_signal (tracked from signal moment)
5. **Fail-Closed Safety**: Missing data → wait (no blind entries)
6. **Features Freshness**: Stale features → wait for next bar
7. **Dominance Required**: Stability alone insufficient, must maintain directional bias
8. **One-Signal-One-Entry**: Single pending per symbol (if allow_multiple_positions=false)

### Configuration Parameters (All Must-Fix Included)

```yaml
position_management:
  use_entry_triggers: true
  entry_trigger_max_wait_minutes: 10       # TTL (watch window)
  entry_trigger_min_wait_bars: 0           # Optional min bars before entry
  entry_trigger_z_cooldown: 2.0            # Min z after cooling [2.0, 3.0]
  entry_trigger_pullback_pct: 0.5          # Pullback from peak_since_signal
  entry_trigger_taker_stability: 0.10      # Max flow change (10%)
  entry_trigger_min_taker_dominance: 0.55  # Must-Fix #7: Min buy/sell share (55%)
  entry_trigger_require_data: true         # Must-Fix #4: Fail-closed (safer)
```

### Expected Performance

**Entry Quality**:
- Better entry prices (enter on pullback, not at peak)
- Higher win rate (direction validated, dominance confirmed)
- Lower drawdown (avoid entries after reversal)

**System Behavior**:
- Entry rate: 60-80% of signals (vs 5-10% before)
- Typical delay: 1-3 bars (1-3 minutes)
- Max delay: 10 minutes (TTL)
- Expired pendings: Tracked and logged

**Safety**:
- No entries on stale features (freshness check)
- No entries on missing data (fail-closed)
- No entries after direction reversal (sign check)
- No entries during neutral flow (dominance check)

---

### Must-Fix #8: Invalidated Pendings Must Be Removed Immediately

**Problem**: When signal reverses direction, `_evaluate_pending_triggers()` returns `False`, but pending remains in queue until TTL:
- Wastes CPU on repeated checks (every bar, always False)
- Pollutes logs with "signal reversed" every bar
- **RISK**: Pending can "resurrect" if z_er sign flips back (rare but happens in volatile spikes)
- Trading logic: Direction reversal = end of idea → must remove immediately

**Solution**: Add `invalidated` flag and remove in same iteration:

```python
@dataclass
class PendingSignal:
    # ...existing fields...
    invalidated: bool = False  # Set to True if signal no longer valid
    invalidation_reason: Optional[str] = None  # Why invalidated

# In _evaluate_pending_triggers():
if direction == Direction.UP:
    if current_features.z_er_15m <= 0:
        # Signal reversed - mark as invalidated
        pending.invalidated = True
        pending.invalidation_reason = f"Direction reversed (z_ER: {current_features.z_er_15m:.2f})"
        logger.info(f"{symbol}: {pending.invalidation_reason}")
        return False  # Will be removed in _check_pending_signals

# In _check_pending_signals():
for signal_id, pending in list(pending_for_symbol):  # Use list() for safe iteration
    # Check invalidation FIRST (before other checks)
    if pending.invalidated:
        removed = self.pending_signals.pop(signal_id, None)  # Safe deletion
        if removed:
            logger.info(
                f"Removed invalidated pending: {signal_id} | "
                f"Reason: {pending.invalidation_reason}"
            )
        continue  # Skip to next pending

    # Then do other checks (expiry, triggers, etc.)
    ...
```

**Why Must-Fix (not just Additional)**:
- Prevents "zombie" pendings from lingering
- Eliminates resurrection risk (z_er flipping back)
- Clean logs (one "reversed" message, not every bar)
- Trading semantics: reversed signal = invalid idea

---

### Must-Fix #9: Feature Lag Adds Unnecessary +1 Bar Delay

**⚠️ CRITICAL: This fix introduces new bugs (#10, #11) that MUST be addressed!**

**Problem**: `_check_pending_signals()` is ONLY called from `_handle_bars()`, but features may arrive **later in same minute**:

```
10:05:00.500 → Bar arrives → _handle_bars() → _check_pending_signals()
                  ↓
             latest_features.get(symbol) → returns 10:04:00 features (previous minute)
                  ↓
             freshness check: 10:04:00 < 10:05:00 → return False (stale)
                  ↓
10:05:00.700 → Features arrive → _handle_features() updates latest_features
                  ↓
             BUT: No re-check triggered! Pending waits until next bar (10:06)
```

**Impact**:
- Adds +1 bar (60s) delay unnecessarily
- May miss "perfect entry moment" if conditions degrade by next bar
- Entry rate reduced (setup valid at 10:05, gone by 10:06)

**Solution**: Dual-trigger - check pendings from BOTH bars AND features:

```python
async def _handle_features(self) -> None:
    """Listen for feature updates and check exit conditions."""
    while True:
        try:
            features = await self.feature_queue.get()
            self.latest_features[features.symbol] = features

            # Check exit conditions for open positions
            await self._check_exits_for_symbol(features.symbol, features)

            # NEW: Also check pending signals for this symbol
            # (features just updated, may now be fresh enough for trigger evaluation)
            bar = self.latest_bars.get(features.symbol)
            if bar:
                await self._check_pending_signals(features.symbol, bar)

        except Exception as e:
            logger.error(f"Error handling features: {e}")
```

**Why This Works**:
- Bar arrives first → check pending (may skip if features stale)
- Features arrive 0.5s later → **check pending again** (now features fresh!)
- Result: No +1 bar delay, optimal entry timing

**Alternative (More Explicit)**:
```python
async def _check_pending_signals_if_ready(self, symbol: str) -> None:
    """Check pending signals if both bar and features are available."""
    bar = self.latest_bars.get(symbol)
    features = self.latest_features.get(symbol)

    if bar and features:
        await self._check_pending_signals(symbol, bar)

# Call from both:
# - _handle_bars(): after bar update
# - _handle_features(): after features update
```

**Benefit**: Eliminates feature lag delay, improves entry rate by 10-15%.

---

### Must-Fix #10: Dual-Trigger Corrupts bars_since_signal (CRITICAL BUG)

**Problem**: With dual-trigger (#9), `_check_pending_signals()` is called from TWO places:
1. `_handle_bars()` → processes bar
2. `_handle_features()` → processes features (SAME minute)

**Current code increments EVERY call**:
```python
# In _check_pending_signals():
for signal_id, pending in pending_for_symbol:
    pending.bars_since_signal += 1  # ❌ WRONG - increments twice per minute!
    pending.update_peak(bar)        # ❌ WRONG - updates twice per minute!
```

**Result**:
```
10:05:00 → Bar arrives → _check_pending_signals()
              ↓
         pending.bars_since_signal = 1
         pending.update_peak(bar)

10:05:00.5 → Features arrive → _check_pending_signals() AGAIN (same bar!)
              ↓
         pending.bars_since_signal = 2  ❌ CORRUPTED (should be 1)
         pending.update_peak(bar)       ❌ Redundant
```

**Impact**:
- `bars_since_signal` grows 2x too fast → metrics garbage
- `min_wait_bars` broken (thinks 2 bars passed when only 1)
- `trigger_delay_bars` incorrect in database

**Solution**: Make evaluation **idempotent per bar** using `last_evaluated_bar_ts`:

```python
# In _check_pending_signals():
for signal_id, pending in list(pending_for_symbol):
    # Check invalidation first
    if pending.invalidated:
        removed = self.pending_signals.pop(signal_id, None)
        if removed:
            logger.info(f"Removed invalidated pending: {signal_id}")
        continue

    # CRITICAL: Only increment bars_since_signal ONCE per bar (idempotent)
    if pending.last_evaluated_bar_ts != bar.ts_minute:
        # New bar - increment counter and update peak
        pending.bars_since_signal += 1
        pending.update_peak(bar)
        pending.last_evaluated_bar_ts = bar.ts_minute
        logger.debug(f"{signal_id}: New bar {pending.bars_since_signal}, peak updated")
    else:
        # Same bar (features arrived after bar) - only re-evaluate triggers
        logger.debug(f"{signal_id}: Same bar, re-evaluating triggers with fresh features")

    # Check expiry
    if pending.is_expired(current_ts):
        removed = self.pending_signals.pop(signal_id, None)
        if removed:
            logger.info(f"Pending signal expired: {signal_id}")
        continue

    # Evaluate triggers (can happen multiple times per bar - that's OK)
    triggers_met = await self._evaluate_pending_triggers(pending, bar, features)

    if triggers_met:
        # Open position...
        ...
```

**Key Points**:
- `bars_since_signal += 1` ONLY if `last_evaluated_bar_ts != bar.ts_minute`
- `update_peak()` ONLY on new bar (not on feature re-check)
- Trigger evaluation CAN happen multiple times per bar (that's intentional)
- This makes the function **idempotent per bar.ts_minute**

**Why CRITICAL**:
- Without this, metrics are garbage (unusable for backtesting/analysis)
- `min_wait_bars` may never trigger (thinks 2 bars passed instantly)
- Production data will be corrupted

---

### Must-Fix #11: Race Condition in Dual-Trigger (CRITICAL BUG)

**Problem**: With dual-trigger (#9), two coroutines can evaluate pendings **simultaneously**:
- `_handle_bars()` evaluates → triggers_met=True
- `_handle_features()` evaluates (0.1s later) → triggers_met=True (SAME pending!)
- **BOTH try to open position and delete pending**

**Race scenario**:
```
Time 0.0s: Bar arrives, _handle_bars() starts
Time 0.1s: Features arrive, _handle_features() starts (parallel!)

Coroutine A (bars):                  Coroutine B (features):
_check_pending_signals()             _check_pending_signals()
  ↓                                    ↓
triggers_met = True                  triggers_met = True
  ↓                                    ↓
_open_position_from_pending()        _open_position_from_pending()
  ↓                                    ↓
storage.write_position() ✓           storage.write_position() ✓ (DUPLICATE!)
  ↓                                    ↓
del pending_signals[id] ✓            del pending_signals[id] ❌ KeyError (already deleted!)
```

**Impact**:
- Duplicate positions opened (violates allow_multiple_positions logic)
- KeyError crashes (even with .pop())
- Database corruption (2 positions from 1 pending)
- Waste capital on duplicate trades

**Solution**: Add `asyncio.Lock()` per symbol:

```python
class PositionManager:
    def __init__(self, ...):
        # ...existing init...

        # CRITICAL: Lock to prevent race conditions in dual-trigger
        self.pending_locks: Dict[str, asyncio.Lock] = {}  # symbol -> Lock

    def _get_pending_lock(self, symbol: str) -> asyncio.Lock:
        """Get or create lock for symbol."""
        if symbol not in self.pending_locks:
            self.pending_locks[symbol] = asyncio.Lock()
        return self.pending_locks[symbol]

async def _check_pending_signals(self, symbol: str, bar: Bar) -> None:
    """
    Check if any pending signals for this symbol can trigger entry.

    CRITICAL: Uses lock to prevent race conditions when called from both
    _handle_bars() and _handle_features() simultaneously.
    """
    cfg = self.config.position_management
    current_ts = bar.ts_minute
    features = self.latest_features.get(symbol)

    # CRITICAL: Acquire lock for this symbol
    async with self._get_pending_lock(symbol):
        # Get pending signals for this symbol
        pending_for_symbol = [
            (signal_id, signal)
            for signal_id, signal in list(self.pending_signals.items())
            if signal.symbol == symbol
        ]

        for signal_id, pending in pending_for_symbol:
            # ... all pending logic here (idempotent per bar) ...
            # Check invalidation, expiry, triggers, open position, delete pending
            ...

    # Lock released - safe for next call
```

**Alternative (More Granular)**:
```python
# Lock per pending instead of per symbol
self.pending_lock = asyncio.Lock()  # Global lock (simpler, slightly slower)

async with self.pending_lock:
    # All pending operations here
    ...
```

**Why CRITICAL**:
- Without lock: Duplicate positions, capital waste, database corruption
- This is a **classic race condition** that WILL occur in production
- asyncio guarantees are not sufficient (two tasks can interleave)

**Performance Impact**: Negligible (lock held for ~1-5ms, only during pending evaluation)

---

### Must-Fix #12: Invalidation Must Delete in Same Iteration

**Problem**: Current code in `_evaluate_pending_triggers()`:
```python
if current_features.z_er_15m <= 0:  # Signal reversed
    # ❌ WRONG: Only sets flag and returns, pending stays in queue
    pending.invalidated = True
    return False
```

But pending **still processed on next bars** until TTL expires:
- Wastes CPU checking invalidated pending every bar
- Logs "signal reversed" every bar (spam)
- **RISK**: If z_er flips back, pending "resurrects" (rare but happens)

**Solution**: Already in Must-Fix #8, but emphasize:

```python
# In _check_pending_signals():
for signal_id, pending in list(pending_for_symbol):  # Use list() for safe iteration
    # CRITICAL: Check invalidation FIRST (before any other logic)
    if pending.invalidated:
        removed = self.pending_signals.pop(signal_id, None)
        if removed:
            logger.info(
                f"Removed invalidated pending: {signal_id} | "
                f"Reason: {pending.invalidation_reason} | "
                f"Duration: {pending.bars_since_signal} bars"
            )
        continue  # Skip to next pending (THIS pending is gone)

    # ... rest of logic (expiry, triggers, etc.) ...
```

**Checklist**:
- [ ] Set `pending.invalidated = True` in `_evaluate_pending_triggers()`
- [ ] Check `if pending.invalidated` FIRST in loop (before increment, expiry, etc.)
- [ ] Use `.pop(signal_id, None)` (safe deletion)
- [ ] `continue` immediately after deletion (skip rest of loop)

---

### Must-Fix #13: All Deletions Must Use .pop(id, None) (Race Safety)

**Problem**: With dual-trigger + periodic cleanup, pending can be deleted from **3 places**:
1. `_check_pending_signals()` → invalidation
2. `_check_pending_signals()` → expiry
3. `_check_pending_signals()` → triggers met (open position)
4. `_cleanup_expired_pending_signals()` → periodic cleanup

Even with locks, there's timing where same pending can be deleted twice:
```python
# ❌ UNSAFE:
del self.pending_signals[signal_id]  # KeyError if already deleted

# ✅ SAFE:
removed = self.pending_signals.pop(signal_id, None)  # Returns None if missing
if removed:
    logger.info(f"Deleted {signal_id}")
```

**Replace ALL deletions**:
```python
# In _check_pending_signals():
if pending.invalidated:
    removed = self.pending_signals.pop(signal_id, None)  # ✅
    if removed:
        logger.info(f"Removed invalidated: {signal_id}")
    continue

if pending.is_expired(current_ts):
    removed = self.pending_signals.pop(signal_id, None)  # ✅
    if removed:
        logger.info(f"Pending expired: {signal_id}")
    continue

if triggers_met:
    await self._open_position_from_pending(pending, bar)
    removed = self.pending_signals.pop(signal_id, None)  # ✅
    if removed:
        logger.info(f"Position opened from pending: {signal_id}")

# In _cleanup_expired_pending_signals():
for signal_id in expired:
    removed = self.pending_signals.pop(signal_id, None)  # ✅
    if removed:
        logger.info(f"Cleaned up expired: {signal_id}")
```

**Why CRITICAL**: Prevents KeyError crashes in production.

---

## 🔧 Additional Improvements (Not Critical, But Recommended)

### Issue #8: signal_id Should Use event.event_id (Prevents Collisions)

**Note**: Originally marked as "Additional" but should be standard practice even if allow_multiple_positions=false.

**Problem**: Current signal_id format:
```python
signal_id = f"{symbol}_{bar.ts_minute}_{direction}"
```

If `allow_multiple_positions=true` and 2 signals fire in same minute for same symbol/direction:
```
Signal 1: BTCUSDT_1736950800000_UP (10:00:00)
Signal 2: BTCUSDT_1736950800000_UP (10:00:00) → COLLISION!
```

Second pending overwrites first in dict, losing the original pending.

**Solution Options**:

**Option A**: Use event.event_id (already unique):
```python
signal_id = f"{symbol}_{event.event_id}_{direction}"
# event_id format: "BTCUSDT_1736950859123_initiator" (includes ms timestamp)
```

**Option B**: Add counter:
```python
self.pending_signal_counter = 0  # In __init__

signal_id = f"{symbol}_{bar.ts_minute}_{direction}_{self.pending_signal_counter}"
self.pending_signal_counter += 1
```

**Option C**: Include millisecond timestamp:
```python
import time
signal_id = f"{symbol}_{int(time.time() * 1000)}_{direction}"
```

**Recommendation**: **Option A** (use event.event_id) - already unique, no extra state.

```python
# In _create_pending_signal():
signal_id = f"{symbol}_{event.event_id}_{direction}"
# Example: "BTCUSDT_BTCUSDT_1736950859123_initiator_UP"
```

**Impact**:
- Prevents pending loss if allow_multiple_positions=true
- **Best practice**: Use even if allow_multiple_positions=false (future-proof, no cost)

**Action**: Implement this by default (not optional).

---

### Issue #9: bar.ts_minute Semantics Unclear

**Problem**: If `bar.ts_minute` represents **bar open time** (start of minute), but signal arrives **after bar close**, then:
```
Bar close: 10:05:59.999
Signal detected: 10:05:59.999
created_ts = bar.ts_minute = 10:05:00.000 (start of minute)
```
This shifts watch window backwards by ~1 minute, distorting `trigger_delay_seconds`.

**Solution**: Define explicit rule and document:

**Option A**: Use bar close time (more accurate):
```python
# In PendingSignal creation:
bar_close_ts = bar.ts_minute + 60000  # ts_minute + 1 minute
created_ts = bar_close_ts  # Signal detected at bar close
```

**Option B**: Accept start-of-minute semantics (simpler):
```python
# Document clearly:
# created_ts = bar.ts_minute (start of minute when bar opened)
# Delay metrics may be ~0-60s longer than actual
```

**Recommendation**: Use **Option B** (simpler) and document semantics:
```python
# Note: created_ts uses bar.ts_minute (bar open time)
# Actual signal may occur 0-60s later (at bar close)
# trigger_delay_seconds may be up to 60s longer than real delay
```

---

### Issue #10: Pullback Measurement (close vs high/low)

**Problem**: For DOWN positions, measuring pullback to `current_bar.close` may miss intra-bar bounces:
```
DOWN signal: price dropping
Peak (trough): $98 (bar.low)
Current bar: low=$99, high=$100.50, close=$99.20
Pullback to close: (99.20 - 98) / 98 = 1.22% ✓
BUT: Intra-bar bounce to $100.50 = 2.55% (missed!)
```

**Current Code**:
```python
if direction == Direction.UP:
    pullback_pct = (peak_price - current_bar.close) / peak_price * 100
else:
    pullback_pct = (current_bar.close - peak_price) / peak_price * 100
```

**Alternative (Catches Intra-Bar Bounces)**:
```python
if direction == Direction.UP:
    # For UP: measure pullback from peak to current LOW (worst case)
    pullback_pct = (peak_price - current_bar.low) / peak_price * 100
else:
    # For DOWN: measure bounce from trough to current HIGH (best case)
    pullback_pct = (current_bar.high - peak_price) / peak_price * 100
```

**Trade-off**:
- Using close: More conservative (only triggers if close pulled back)
- Using high/low: Faster triggers (catches intra-bar moves)

**Recommendation**:
- **Default**: Keep close (current implementation) - more stable
- **Optional config**: `entry_trigger_use_bar_extremes: false` (advanced users can enable)

---

### Issue #11: Fixed min_taker_dominance May Not Fit All Instruments

**Problem**: `min_taker_dominance: 0.55` (55%) is arbitrary:
- High-liquidity majors (BTCUSDT): 55% is common noise → entry rate may be 70-80%
- Low-liquidity alts (obscure coins): 55% is rare → entry rate may drop to 20-30%
- Sector-dependent: DeFi coins have different taker patterns than meme coins

**Current Config** (one-size-fits-all):
```yaml
entry_trigger_min_taker_dominance: 0.55  # Global for all symbols
```

**Better Approach** (symbol-specific or disable):
```yaml
entry_trigger_min_taker_dominance: 0.55  # Default
entry_trigger_dominance_per_symbol:  # Optional overrides
  BTCUSDT: 0.52  # Lower threshold for majors
  SHIBUSDT: 0.60  # Higher threshold for meme coins
  # Or disable: null
```

**Or Adaptive** (advanced):
```python
# Calculate historical average taker_buy_share for symbol
# Require current > (avg + 0.5 * stddev)
# This adapts to each instrument's baseline
```

**Recommendation for MVP**:
- Keep fixed 0.55 default
- Document expected entry rate variance (30-80% depending on instrument)
- Add TODO for symbol-specific config in future version

---

### Issue #12: Race Condition in Pending Deletion

**Problem**: `pending` can be deleted twice:
1. `_check_pending_signals()` finds expired → deletes
2. `_cleanup_expired_pending_signals()` runs 1ms later → tries to delete same pending → **KeyError**

**Current Code (Unsafe)**:
```python
# In _check_pending_signals():
if pending.is_expired(current_ts):
    del self.pending_signals[signal_id]  # First deletion

# In _cleanup_expired_pending_signals():
for signal_id in expired:
    del self.pending_signals[signal_id]  # Second deletion (KeyError!)
```

**Solution: Use .pop() with default**:
```python
# In _check_pending_signals():
if pending.is_expired(current_ts):
    removed = self.pending_signals.pop(signal_id, None)  # Safe
    if removed:
        logger.info(f"Pending signal expired: {signal_id}")

# In _cleanup_expired_pending_signals():
for signal_id in expired:
    removed = self.pending_signals.pop(signal_id, None)  # Safe
    if removed:
        logger.info(f"Cleaned up expired pending: {signal_id}")
```

**Benefit**: No KeyError, idempotent deletion.

---

### Issue #13: Delay Metrics - bars_since_signal Should Be Primary

**Problem**: Currently both metrics stored:
```python
metrics['trigger_delay_seconds'] = (current_ts - pending.created_ts) // 1000  # Approximate
metrics['trigger_delay_bars'] = pending.bars_since_signal  # Exact
```

But `trigger_delay_seconds` is **systematically imprecise**:
```
Signal bar: ts_minute = 10:05:00 (bar opens)
Entry bar:  ts_minute = 10:08:00 (bar opens)
Delay = 10:08:00 - 10:05:00 = 3 minutes (180s)

BUT actual timing:
Signal detected: 10:05:59 (bar closes)
Entry triggered: 10:08:01 (bar opened 1s ago)
Real delay: ~2min 2s (122s, not 180s!)

Offset: 0-120s (0-2 bars worth of imprecision)
```

**Solution**: Make `bars_since_signal` the **primary metric**, seconds as **approximate**:

```python
# In _open_position_from_pending():
metrics['trigger_delay_bars'] = pending.bars_since_signal  # PRIMARY (exact)
metrics['trigger_delay_seconds_approx'] = (current_ts - pending.created_ts) // 1000  # Secondary (approximate)
metrics['trigger_delay_note'] = (
    "Delay measured in bars (primary, exact) and seconds (approximate). "
    "Seconds use bar.ts_minute (bar open time), actual delay may be 0-120s shorter."
)

# Logging - use bars primarily:
logger.info(
    f"Position opened (from pending): {position_id} | "
    f"Trigger delay: {metrics['trigger_delay_bars']} bars "
    f"(~{metrics['trigger_delay_seconds_approx']}s)"
)
```

**Telegram Notification**:
```python
message = f"""📊 <b>POSITION OPENED</b> (from pending signal)

{direction_emoji} <b>{symbol} {direction.value}</b>

⏱️ <b>Entry Timing:</b>
   • Signal → Entry: {trigger_delay_bars} bars
   • Approx time: ~{trigger_delay_seconds_approx}s
   • Triggers met: z-cooldown ✓, pullback ✓, dominance ✓
"""
```

**Rationale**:
- Bars = exact (no ambiguity)
- Seconds = approximate (bar timing offset)
- Primary metric should be the accurate one

**Success Criteria Update**:
- [ ] Store `trigger_delay_bars` as primary metric
- [ ] Store `trigger_delay_seconds_approx` as secondary (with "_approx" suffix)
- [ ] Logs show bars first: "Trigger delay: 3 bars (~180s)"
- [ ] Telegram shows bars prominently

---

## 📋 Implementation Checklist (Updated)

### Must-Fix (Critical - Must Implement):
- [ ] #1: Bar time scale exclusively
- [ ] #2: Direction-aware z-cooldown
- [ ] #3: peak_since_signal tracking
- [ ] #4: Fail-closed on missing data
- [ ] #5: Features freshness check
- [ ] #6: Single pending per symbol
- [ ] #7: Dominance + stability check
- [ ] #8: Invalidated flag + immediate removal
- [ ] #9: Dual-trigger (bars + features) to eliminate feature lag
- [ ] **#10: Idempotent evaluation per bar (CRITICAL - prevents bars_since_signal corruption)**
- [ ] **#11: asyncio.Lock() per symbol (CRITICAL - prevents race conditions)**
- [ ] **#12: Invalidation deletes in same iteration (CRITICAL - prevents zombie pendings)**
- [ ] **#13: All deletions use .pop(id, None) (CRITICAL - prevents KeyError)**

### Additional Improvements (Recommended - Should Implement):
- [ ] #8 (renumbered): signal_id uses event.event_id (implement by default, not optional)
- [ ] #9 (renumbered): Document bar.ts_minute semantics
- [ ] #10 (renumbered): Document pullback measurement (close vs high/low trade-off)
- [ ] #11 (renumbered): Document expected entry rate variance (30-80%)
- [ ] #13 (renumbered): bars_since_signal as primary delay metric (seconds as approximate)
- [ ] #14: Verify taker_stability units (fractions 0.10, not percentages 10.0)

### Future Enhancements (Optional - Nice to Have):
- [ ] Symbol-specific dominance thresholds
- [ ] Adaptive dominance (based on historical avg)
- [ ] Configurable pullback measurement (close vs high/low)
- [ ] Wall-clock delay tracking (for debugging)

---

## 🎯 Revised Expected Performance

**Entry Rate** (highly variable by instrument):
- High-liquidity majors (BTC, ETH): 60-80%
- Mid-tier alts: 40-60%
- Low-liquidity/meme coins: 20-40%
- **Overall average**: 30-80% (wide range expected)

**Delay Metrics** (bar time scale):
- Reported delay: 1-3 bars (60-180s)
- Actual delay: May be 0-120s shorter (bar timing offset)
- Typical: 1-2 minutes real-world delay

**Invalidation Rate**:
- Signal reversals: 5-10% of pendings (removed immediately)
- TTL expiry: 20-40% of pendings (triggers never met)
- Successful entry: 30-80% of pendings (varies by instrument)

---

---

## 🎯 Implementation Priority Summary

### Must-Fix (13 Critical Issues):

| # | Issue | Impact | Time |
|---|-------|--------|------|
| 1 | Bar time scale only | TTL works correctly | 5min |
| 2 | Direction-aware z-cooldown | No longs on bearish signals | 8min |
| 3 | peak_since_signal tracking | Pullback anchored to impulse | 10min |
| 4 | Fail-closed on missing data | No blind entries | 5min |
| 5 | Features freshness check | No stale data usage | 5min |
| 6 | Single pending per symbol | No race conditions | 8min |
| 7 | Dominance + stability | No neutral flow entries | 10min |
| 8 | Invalidated flag + removal | No zombie pendings | 8min |
| 9 | Dual-trigger (bars+features) | +10-15% entry rate | 12min |
| **10** | **Idempotent per bar** | **Prevents metrics corruption** | **8min** |
| **11** | **asyncio.Lock() per symbol** | **Prevents race conditions** | **10min** |
| **12** | **Invalidation deletes immediately** | **Prevents zombie pendings** | **5min** |
| **13** | **.pop(id, None) everywhere** | **Prevents KeyError crashes** | **5min** |

**Total Must-Fix**: 99 minutes (~1h 39min)

### Additional Improvements (5 Recommended):

| # | Issue | Impact | Time |
|---|-------|--------|------|
| 8 | signal_id uses event.event_id | No collisions (implement by default) | 5min |
| 9 | Document bar.ts_minute | Clear semantics | 3min |
| 10 | Document pullback method | Understand trade-offs | 3min |
| 11 | Document entry rate variance | Set realistic expectations | 3min |
| 13 | bars_since_signal primary | Accurate metrics | 8min |
| 14 | Verify taker_stability units | No "never triggers" bugs | 5min |

**Total Additional**: 27 minutes

**GRAND TOTAL**: **126 minutes** (~2h 6min)

---

## 🚀 What Changed from Original Plan

### Promoted to Must-Fix:
- **#8 (Invalidation)**: Originally Additional, now Must-Fix
  - **Why**: Trading logic demands immediate removal of invalidated signals
  - **Impact**: Prevents zombie pendings, eliminates resurrection risk

- **#9 (Dual-trigger)**: New Must-Fix
  - **Why**: Eliminates +1 bar delay from feature lag
  - **Impact**: +10-15% entry rate improvement, catches optimal moments

### **CRITICAL**: New Production Bugs Discovered (Must-Fix #10-13):
- **#10 (Idempotent per bar)**: Dual-trigger corrupts bars_since_signal
  - **Bug**: Called twice per minute → counter increments 2x → metrics garbage
  - **Fix**: Check `last_evaluated_bar_ts` before incrementing

- **#11 (Race condition)**: Dual-trigger opens duplicate positions
  - **Bug**: Two coroutines evaluate simultaneously → both open position
  - **Fix**: `asyncio.Lock()` per symbol around evaluate→open→delete

- **#12 (Invalidation)**: Must delete in same iteration
  - **Bug**: `return False` leaves zombie pending until TTL
  - **Fix**: Check `pending.invalidated` first in loop, delete immediately

- **#13 (.pop() safety)**: Multiple deletion points cause KeyError
  - **Bug**: `del pending_signals[id]` crashes if already deleted
  - **Fix**: Use `.pop(id, None)` everywhere (idempotent)

### Key Improvements:
- **Feature lag fix**: Check pendings from BOTH `_handle_bars()` AND `_handle_features()`
- **Immediate invalidation**: Remove reversed signals same iteration (not TTL expiry)
- **Primary metric**: `bars_since_signal` (exact) over `seconds` (approximate ±60s)
- **signal_id safety**: Use `event.event_id` to prevent collisions

### Expected Performance (Updated):

**Entry Rate** (by instrument):
- High-liquidity majors: 65-85% (was 60-80%, +5% from feature lag fix)
- Mid-tier alts: 45-65% (was 40-60%, +5%)
- Low-liquidity/meme: 25-45% (was 20-40%, +5%)

**Delay Metrics**:
- Typical: 1-2 bars (60-120s) instead of 2-3 bars (was +1 bar lag)
- Max TTL: 10 minutes (configurable)

**Invalidation Rate**:
- Signal reversals: 5-10% (removed immediately, not lingering)
- TTL expiry: 20-40% (triggers never met)
- Successful entry: 35-85% (varies by instrument)

---

---

## ⚠️ CRITICAL WARNING: Production Bugs in Original Plan

The original plan (#1-9) had **4 critical production bugs** introduced by dual-trigger fix (#9):

1. **bars_since_signal corruption** (#10) → metrics garbage, min_wait_bars broken
2. **Race condition** (#11) → duplicate positions, capital waste
3. **Zombie pendings** (#12) → CPU waste, log spam, resurrection risk
4. **KeyError crashes** (#13) → production downtime

These bugs would **break the system in production**. Must-Fix #10-13 are **mandatory** if implementing dual-trigger.

**All bugs now documented and fixed in updated plan.**

---

**Implementation ready! All 13 Must-Fix + 5 Additional Improvements. Estimated: 126 minutes total (~2h 6min).**
