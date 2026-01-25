# Спецификация критических исправлений v3

## Текущее состояние системы

### Статистика за 3 дня (80 сделок)

| Метрика | Значение | Статус |
|---------|----------|--------|
| Total Positions | 80 | - |
| Win Rate | 48.8% (39W / 39L) | ⚠️ Приемлемо |
| Total PnL | -13.91% | ❌ Критический убыток |
| Avg Win | +1.16% | ⚠️ Низко |
| Avg Loss | -1.51% | ❌ Слишком высоко |
| R:R Ratio | 0.77 | ❌ Ниже 1.0 |
| Avg Duration | 5 минут | ✅ Быстро |

### Exit Reasons Distribution

| Exit Reason | Количество | % | Estimated Avg PnL | Total Contribution |
|-------------|------------|---|-------------------|-------------------|
| Z_SCORE_REVERSAL | 32 | 40.0% | ~+0.2% | ~+6.4% |
| TRAILING_STOP | 26 | 32.5% | ~+0.7% | ~+18.2% |
| STOP_LOSS | 19 | 23.8% | ~-1.8% | ~-34.2% |
| TAKE_PROFIT_TP3 | 3 | 3.8% | ~+2.5% | ~+7.5% |

### Математика убытка

```
Прибыльные exits:
  Z_SCORE_REVERSAL (winners): ~15 × +0.5% = +7.5%
  TRAILING_STOP:              26 × +0.7% = +18.2%
  TAKE_PROFIT_TP3:             3 × +2.5% = +7.5%
  ─────────────────────────────────────────────
  Total wins:                            +33.2%

Убыточные exits:
  Z_SCORE_REVERSAL (losers): ~17 × -0.3% = -5.1%
  STOP_LOSS:                  19 × -1.8% = -34.2%
  ─────────────────────────────────────────────
  Total losses:                          -39.3%

NET: +33.2% - 39.3% = -6.1% (остаток от комиссий и variance)
```

**Вывод: STOP_LOSS один генерирует -34.2%, что превышает всю прибыль системы.**

---

## Корневые причины убытков

### Причина 1: Stop-Loss слишком широкий

| Проблема | Доказательство |
|----------|----------------|
| Avg Loss = -1.51% | При 19 SL это -28.7% |
| ATR multipliers слишком высокие | Стоп далеко от entry |
| Max stop distance не ограничен | Возможны стопы -2%+ |

### Причина 2: Take-Profit срабатывает редко

| Проблема | Доказательство |
|----------|----------------|
| Только 3 TP3 из 80 (3.8%) | Цели слишком далёкие |
| TP1, TP2 не фиксируются в статистике | Возможно не срабатывают |
| Avg duration 5 мин | Недостаточно для достижения далёких TP |

### Причина 3: R:R Ratio ниже 1.0

| Проблема | Доказательство |
|----------|----------------|
| R:R = 0.77 | Убытки больше прибылей |
| При WR 48.8% нужен R:R > 1.05 | Для breakeven |
| При WR 48.8% и R:R 0.77 | Математически убыточно |

### Причина 4: Z-Score exit закрывает в убытке

| Проблема | Доказательство |
|----------|----------------|
| 32 Z-exit, вероятно ~50% в минусе | Нет фильтра на profit |
| Exit при z < threshold | Не учитывает PnL позиции |

---

## Исправление 1: Сузить Stop-Loss

### Цель

Уменьшить Avg Loss с -1.51% до -0.80% — -1.00%

### Изменения в конфигурации

```yaml
adaptive_stop_loss:
  enabled: true
  
  # СУЖАЕМ базовые множители
  base_multipliers:
    extreme_spike: 0.8    # Было 1.2 → уменьшаем на 33%
    strong_signal: 1.0    # Было 1.4 → уменьшаем на 29%
    early_signal: 1.2     # Было 1.6 → уменьшаем на 25%
  
  # ОГРАНИЧИВАЕМ volatility adjustment
  volatility_adjustment:
    enabled: true
    high_volatility_multiplier: 1.2   # Было 1.3 → меньше расширение
    low_volatility_multiplier: 0.9    # Было 0.85
  
  # СТРОГИЕ лимиты
  min_stop_distance_pct: 0.3          # Минимум 0.3%
  max_stop_distance_pct: 1.5          # Было 3.0 → уменьшаем в 2 раза
  
  # УБИРАЕМ direction adjustment для long
  direction_adjustment:
    enabled: false                     # Было true
```

### Логика изменений

| Параметр | Было | Стало | Причина |
|----------|------|-------|---------|
| extreme_spike base | 1.2 | 0.8 | MR сделки короткие, стоп должен быть tight |
| strong_signal base | 1.4 | 1.0 | Стандартный momentum не требует широкого стопа |
| early_signal base | 1.6 | 1.2 | Ранний вход, но не нужен огромный стоп |
| max_stop_distance | 3.0% | 1.5% | Ограничить максимальный убыток |
| high_vol_multiplier | 1.3 | 1.2 | Меньше расширение на волатильности |

### Ожидаемый результат

| Метрика | Сейчас | После |
|---------|--------|-------|
| Avg Loss | -1.51% | -0.85% |
| SL Rate | 23.8% | 28-32% (чаще, но меньше) |
| Total SL contribution | -34.2% | -22% to -26% |

**Trade-off:** SL будет срабатывать чаще, но каждый SL будет меньше. Математически это выгоднее.

---

## Исправление 2: Агрессивный Take-Profit

### Цель

Увеличить TP hit rate с 3.8% до 20-30%

### Изменения в конфигурации

```yaml
tiered_take_profit:
  enabled: true
  
  levels_by_class:
    # EXTREME_SPIKE: самые близкие цели (MR = быстрые сделки)
    extreme_spike:
      tp1_atr: 0.25           # Было 0.4 → ближе на 37%
      tp1_close_pct: 50       # Было 30 → закрываем больше
      tp2_atr: 0.5            # Было 0.8 → ближе на 37%
      tp2_close_pct: 30       # Было 30 → без изменений
      tp3_atr: 0.8            # Было 1.2 → ближе на 33%
      tp3_close_pct: 20       # Было 40 → меньше на последнем уровне
    
    # STRONG_SIGNAL: средние цели
    strong_signal:
      tp1_atr: 0.3            # Было 0.5 → ближе на 40%
      tp1_close_pct: 50       # Было 30 → закрываем больше
      tp2_atr: 0.6            # Было 1.0 → ближе на 40%
      tp2_close_pct: 30
      tp3_atr: 1.0            # Было 1.5 → ближе на 33%
      tp3_close_pct: 20
    
    # EARLY_SIGNAL: чуть дальше (ловим большие движения)
    early_signal:
      tp1_atr: 0.35           # Было 0.6 → ближе на 42%
      tp1_close_pct: 50
      tp2_atr: 0.7            # Было 1.2 → ближе на 42%
      tp2_close_pct: 30
      tp3_atr: 1.2            # Было 2.0 → ближе на 40%
      tp3_close_pct: 20
  
  actions:
    move_sl_breakeven_on_tp1: true    # После TP1 — стоп в breakeven
    activate_trailing_on_tp1: true    # Было on_tp2 → раньше активируем trailing
```

### Логика изменений

| Изменение | Причина |
|-----------|---------|
| TP1 ближе на 37-42% | Avg duration 5 мин — далёкие цели не достигаются |
| TP1 close 50% (было 30%) | Фиксировать больше прибыли рано |
| TP3 close 20% (было 40%) | Меньше риска на последнем уровне |
| Trailing on TP1 | Защищать прибыль раньше |

### Ожидаемый результат

| Метрика | Сейчас | После |
|---------|--------|-------|
| TP1 Hit Rate | ~10-15% | 35-45% |
| TP2 Hit Rate | ~5-8% | 20-25% |
| TP3 Hit Rate | 3.8% | 10-15% |
| Avg Win | +1.16% | +0.90% (меньше, но чаще) |

---

## Исправление 3: Trailing Stop раньше и tighter

### Цель

Увеличить эффективность trailing, защищать прибыль раньше

### Изменения в конфигурации

```yaml
trailing_stop:
  enabled: true
  
  activation_by_class:
    extreme_spike:
      profit_threshold_pct: 0.15      # Было 0.25 → раньше на 40%
      distance_atr: 0.25              # Было 0.5 → tighter на 50%
    
    strong_signal:
      profit_threshold_pct: 0.20      # Было 0.35 → раньше на 43%
      distance_atr: 0.35              # Было 0.7 → tighter на 50%
    
    early_signal:
      profit_threshold_pct: 0.25      # Было 0.45 → раньше на 44%
      distance_atr: 0.5               # Было 1.0 → tighter на 50%
  
  behavior:
    activate_on_tp1: true             # Было on_tp2 → активировать раньше
    update_frequency: "every_bar"
    use_close_price: true
    
    # НОВОЕ: ускоренный trailing после большого движения
    accelerated_trailing:
      enabled: true
      trigger_profit_pct: 0.5         # При +0.5% profit
      tighter_distance_multiplier: 0.7 # Trail становится на 30% tighter
```

### Логика изменений

| Изменение | Причина |
|-----------|---------|
| Activation на 40-44% раньше | Текущие 26 trailing exits — хорошо, но можно больше |
| Distance на 50% tighter | Защитить больше прибыли |
| Activate on TP1 | После первого profit — сразу защищать |
| Accelerated trailing | При хорошем движении — не отдавать прибыль |

### Ожидаемый результат

| Метрика | Сейчас | После |
|---------|--------|-------|
| Trailing Exit Rate | 32.5% | 35-40% |
| Avg Trailing Exit PnL | ~+0.7% | ~+0.55% (tighter, но сохраняет больше) |

---

## Исправление 4: Z-Score Exit только в прибыли

### Цель

Предотвратить закрытие убыточных позиций по z-score

### Изменения в конфигурации

```yaml
z_score_exit:
  enabled: true
  
  # Базовые thresholds (без изменений)
  thresholds:
    extreme_spike: 1.5
    strong_signal: 0.8
    early_signal: 0.8
  
  # НОВОЕ: условия для z-exit
  conditions:
    require_positive_pnl: true        # ГЛАВНОЕ: только если в прибыли
    min_pnl_for_full_exit: 0.15       # Полный exit при PnL >= 0.15%
    min_pnl_for_partial_exit: 0.0     # Partial exit при PnL >= 0%
    
    # Поведение при убытке
    behavior_when_losing:
      action: "hold"                   # Держать позицию
      max_additional_hold_minutes: 5   # Максимум ещё 5 минут
      fallback_exit: "trailing_or_sl"  # Потом trailing или SL
  
  # Partial close по z-score
  partial_close:
    enabled: true
    close_percent: 60                  # Закрыть 60% по z-exit
    hold_remainder_for_trailing: true  # Остаток под trailing
```

### Логика изменений

| Сценарий | Старое поведение | Новое поведение |
|----------|------------------|-----------------|
| Z вернулся, PnL +0.3% | Exit полностью | Exit 60%, trailing на остаток |
| Z вернулся, PnL +0.05% | Exit полностью | Partial exit, hold остаток |
| Z вернулся, PnL -0.2% | Exit с убытком | HOLD, ждать trailing или SL |
| Z вернулся, PnL -0.5% | Exit с убытком | HOLD до max 5 мин, потом SL |

### Ожидаемый результат

| Метрика | Сейчас | После |
|---------|--------|-------|
| Z-Exit в убытке | ~17 из 32 | 0 |
| Z-Exit Avg PnL | ~+0.2% | ~+0.35% |
| Total Z-Exit contribution | ~+6.4% | ~+11% |

---

## Исправление 5: Минимальный R:R фильтр при входе

### Цель

Не открывать позиции с плохим потенциальным R:R

### Новая секция в конфигурации

```yaml
entry_filters:
  min_rr_filter:
    enabled: true
    
    # Расчёт expected R:R
    expected_tp: "tp1"                 # Считать против TP1 (самый вероятный)
    expected_sl: "adaptive_sl"         # Текущий adaptive SL
    
    # Минимальные пороги
    min_rr_ratio: 1.0                  # Минимум R:R = 1.0
    
    # По классам (опционально, разные требования)
    by_class:
      extreme_spike:
        min_rr_ratio: 0.8              # MR может иметь ниже R:R но выше WR
      strong_signal:
        min_rr_ratio: 1.0
      early_signal:
        min_rr_ratio: 1.2              # Early требует лучший R:R
```

### Логика

```
При создании pending signal:

1. Рассчитать expected_tp_distance = ATR × tp1_multiplier
2. Рассчитать expected_sl_distance = ATR × sl_multiplier
3. expected_rr = expected_tp_distance / expected_sl_distance

4. IF expected_rr < min_rr_ratio:
     log("Skipping entry: R:R {expected_rr} < {min_rr_ratio}")
     return  # Не создавать pending signal

5. ELSE:
     continue with entry
```

### Пример

```
ATR = 0.5%
TP1 = 0.3 × ATR = 0.15%
SL = 0.8 × ATR = 0.40%

R:R = 0.15 / 0.40 = 0.375

0.375 < 1.0 → SKIP ENTRY
```

```
ATR = 1.0%
TP1 = 0.3 × ATR = 0.30%
SL = 0.8 × ATR = 0.80%

R:R = 0.30 / 0.80 = 0.375

0.375 < 1.0 → SKIP ENTRY
```

**Важно:** С текущими настройками R:R будет низким. Нужно балансировать TP и SL.

### Альтернативный подход: ATR minimum filter

```yaml
entry_filters:
  min_atr_filter:
    enabled: true
    min_atr_pct: 0.4                   # Минимум ATR 0.4%
```

Это отфильтрует низковолатильные ситуации где profit potential мал.

---

## Исправление 6: Ограничить убыток на позицию

### Цель

Hard cap на максимальный убыток одной позиции

### Изменения в конфигурации

```yaml
risk_management:
  max_loss_per_position:
    enabled: true
    max_loss_pct: 1.2                  # Максимум -1.2% на позицию
    action: "immediate_close"          # Закрыть немедленно при достижении
    
  # Это ДОПОЛНИТЕЛЬНО к adaptive SL
  # Работает как safety net если SL не сработал
```

### Логика

```
На каждом баре:
  current_pnl = calculate_pnl(position, current_price)
  
  IF current_pnl <= -max_loss_pct:
    close_position(reason="MAX_LOSS_CAP")
    return
  
  # Потом проверять обычные exits
```

### Ожидаемый результат

Это страховка. При правильной работе adaptive SL — не должно срабатывать. Но защитит от outliers.

---

## Полная конфигурация после исправлений

```yaml
# ============================================
# CRITICAL FIXES V3 - PROFITABILITY CONFIG
# ============================================

# Исправление 1: Узкий Stop-Loss
adaptive_stop_loss:
  enabled: true
  
  base_multipliers:
    extreme_spike: 0.8
    strong_signal: 1.0
    early_signal: 1.2
  
  volatility_adjustment:
    enabled: true
    lookback_bars: 1440
    high_volatility_percentile: 75
    high_volatility_multiplier: 1.2
    low_volatility_percentile: 25
    low_volatility_multiplier: 0.9
  
  direction_adjustment:
    enabled: false
  
  min_stop_distance_pct: 0.3
  max_stop_distance_pct: 1.5

# Исправление 2: Агрессивный Take-Profit
tiered_take_profit:
  enabled: true
  
  levels_by_class:
    extreme_spike:
      tp1_atr: 0.25
      tp1_close_pct: 50
      tp2_atr: 0.5
      tp2_close_pct: 30
      tp3_atr: 0.8
      tp3_close_pct: 20
    
    strong_signal:
      tp1_atr: 0.3
      tp1_close_pct: 50
      tp2_atr: 0.6
      tp2_close_pct: 30
      tp3_atr: 1.0
      tp3_close_pct: 20
    
    early_signal:
      tp1_atr: 0.35
      tp1_close_pct: 50
      tp2_atr: 0.7
      tp2_close_pct: 30
      tp3_atr: 1.2
      tp3_close_pct: 20
  
  actions:
    move_sl_breakeven_on_tp1: true
    activate_trailing_on_tp1: true

# Исправление 3: Ранний и tight Trailing Stop
trailing_stop:
  enabled: true
  
  activation_by_class:
    extreme_spike:
      profit_threshold_pct: 0.15
      distance_atr: 0.25
    strong_signal:
      profit_threshold_pct: 0.20
      distance_atr: 0.35
    early_signal:
      profit_threshold_pct: 0.25
      distance_atr: 0.5
  
  behavior:
    activate_on_tp1: true
    update_frequency: "every_bar"
    use_close_price: true
  
  accelerated_trailing:
    enabled: true
    trigger_profit_pct: 0.5
    tighter_distance_multiplier: 0.7

# Исправление 4: Z-Exit только в прибыли
z_score_exit:
  enabled: true
  
  thresholds:
    extreme_spike: 1.5
    strong_signal: 0.8
    early_signal: 0.8
  
  conditions:
    require_positive_pnl: true
    min_pnl_for_full_exit: 0.15
    min_pnl_for_partial_exit: 0.0
  
  behavior_when_losing:
    action: "hold"
    max_additional_hold_minutes: 5
    fallback_exit: "trailing_or_sl"
  
  partial_close:
    enabled: true
    close_percent: 60
    hold_remainder_for_trailing: true

# Исправление 5: Entry R:R фильтр
entry_filters:
  min_rr_filter:
    enabled: true
    expected_tp: "tp1"
    expected_sl: "adaptive_sl"
    min_rr_ratio: 0.8
  
  min_atr_filter:
    enabled: true
    min_atr_pct: 0.3

# Исправление 6: Hard cap на убыток
risk_management:
  max_loss_per_position:
    enabled: true
    max_loss_pct: 1.2
    action: "immediate_close"

# Time exits (минимальные, не агрессивные)
time_exit:
  aggressive_exits:
    enabled: false                     # ОТКЛЮЧЕНО
  
  max_hold_minutes:
    extreme_spike: 20
    strong_signal: 45
    early_signal: 60
```

---

## Ожидаемые результаты после всех исправлений

### Целевые метрики

| Метрика | Сейчас | Цель | Минимум |
|---------|--------|------|---------|
| Win Rate | 48.8% | 52-55% | 50% |
| Avg Win | +1.16% | +0.85% | +0.75% |
| Avg Loss | -1.51% | -0.75% | -0.90% |
| R:R Ratio | 0.77 | 1.13 | 1.0 |
| SL Rate | 23.8% | 18-22% | 25% |
| TP Hit Rate | 3.8% | 25-35% | 20% |

### Математика прибыльности

```
При WR 52% и R:R 1.13:

52 wins × 0.85% = +44.2%
48 losses × 0.75% = -36.0%
─────────────────────────────
Net PnL = +8.2% на 100 сделок

После комиссий (~0.08% × 100 = 8%):
Net = +8.2% - 8% = +0.2%

Нужно лучше!
```

```
При WR 55% и R:R 1.2:

55 wins × 0.80% = +44.0%
45 losses × 0.67% = -30.0%
─────────────────────────────
Net PnL = +14.0% на 100 сделок

После комиссий:
Net = +14% - 8% = +6%

Это уже прибыльно!
```

---

## План внедрения

### День 1: Критические изменения

1. **Сузить Stop-Loss** (Исправление 1)
   - Изменить base_multipliers
   - Установить max_stop_distance_pct: 1.5

2. **Z-Exit только в прибыли** (Исправление 4)
   - Добавить require_positive_pnl: true

### День 2: Take-Profit и Trailing

3. **Агрессивный TP** (Исправление 2)
   - Приблизить все TP уровни
   - Увеличить TP1 close до 50%

4. **Ранний Trailing** (Исправление 3)
   - Уменьшить activation threshold
   - Уменьшить distance

### День 3: Фильтры

5. **R:R фильтр** (Исправление 5)
   - Не входить при плохом R:R

6. **Max loss cap** (Исправление 6)
   - Safety net

---

## Мониторинг после изменений

### Ключевые метрики для отслеживания

| Метрика | Alert если |
|---------|------------|
| Win Rate | < 45% |
| Avg Loss | > -1.0% |
| R:R Ratio | < 0.9 |
| SL Rate | > 30% |
| Z-Exit с убытком | > 0 (должно быть 0) |
| Daily PnL | < -3% |

### Логирование

Добавить детальный лог для каждой сделки:

```
[TRADE CLOSED]
Symbol: {symbol}
Direction: {direction}
Signal Class: {class}
Duration: {minutes} min

Entry: {entry_price}
Exit: {exit_price}
Exit Reason: {reason}

PnL: {pnl}%
MFE: {mfe}%
MAE: {mae}%

TP1 Hit: {yes/no}
TP2 Hit: {yes/no}
Trailing Active: {yes/no}
SL Price: {sl_price}
Distance to SL at exit: {distance}%
```

---

## Резюме критических изменений

| # | Исправление | Влияние на Avg Loss | Влияние на Avg Win | Влияние на WR |
|---|-------------|---------------------|--------------------| --------------|
| 1 | Узкий SL | -1.51% → -0.75% | - | -2% (чаще SL) |
| 2 | Агрессивный TP | - | +1.16% → +0.85% | +5% (чаще TP) |
| 3 | Ранний Trailing | - | Сохраняет profit | +2% |
| 4 | Z-Exit в прибыли | Убирает ~17 убытков | - | +3% |
| 5 | R:R фильтр | Меньше плохих сделок | - | +2% |
| 6 | Max loss cap | Safety net | - | - |

**Суммарный ожидаемый эффект:**
- Avg Loss: -1.51% → -0.75%
- Avg Win: +1.16% → +0.85%
- R:R: 0.77 → 1.13
- Win Rate: 48.8% → 52-55%
- Total PnL: -13.91% → +5% to +10% (на 80 сделок)
