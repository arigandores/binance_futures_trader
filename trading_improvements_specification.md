# Спецификация улучшений гибридной стратегии

## Контекст и проблема

### Текущие результаты (62 сделки)

| Метрика | Значение | Оценка |
|---------|----------|--------|
| Win Rate | 51.6% | ⚠️ На грани |
| Gross PnL | +1.24% | ❌ Недостаточно |
| Avg Win | +0.69% | ⚠️ Низко |
| Avg Loss | -0.78% | ❌ Высоко |
| R:R Ratio | 0.89 | ❌ Ниже 1.0 |
| Avg Duration | 5.7 мин | ✅ Быстро |
| MFE Capture | 49% | ❌ Упускаем прибыль |

### Расчёт реальной прибыльности

```
Gross PnL:                    +1.24%
Funding fee (~0.01% × 124):   -1.24%
Trading fee (~0.04% × 124):   -4.96%
───────────────────────────────────
Net PnL:                      ≈ -4.96%
```

Система убыточна после учёта комиссий. Требуются улучшения.

### Ключевые проблемы

| Проблема | Доказательство | Impact |
|----------|----------------|--------|
| Stop-loss слишком tight | 13 SL = -16.43% (съедает всю прибыль) | Критический |
| Take-profit слишком далёкий | Только 3 TP из 62 сделок (5%) | Высокий |
| Z-exit слишком ранний | MFE +0.45% vs Exit +0.22% | Высокий |
| Лонги убыточные | 45.5% WR, -1.23% PnL | Средний |
| Trailing не работает | 1 срабатывание из 62 | Средний |

---

## Улучшение 1: Адаптивный Stop-Loss

### Проблема

Фиксированный ATR multiplier (1.2-1.5x) не учитывает:
- Силу сигнала (EXTREME vs EARLY)
- Текущую рыночную волатильность
- Направление позиции (long vs short)

Результат: 21% сделок закрываются по стоп-лоссу с общим убытком -16.43%.

### Решение

Stop-loss multiplier зависит от трёх факторов:
1. Signal class (база)
2. Volatility regime (корректировка)
3. Direction (корректировка для long)

### Базовые множители по классам

| Signal Class | Base ATR Multiplier | Логика |
|--------------|---------------------|--------|
| EXTREME_SPIKE | 1.5x | MR сделки короткие, можно tight |
| STRONG_SIGNAL | 1.8x | Стандартный momentum |
| EARLY_SIGNAL | 2.0x | Ранний вход, нужно больше пространства |

### Корректировка по волатильности

Определение volatility regime:
- Рассчитать текущий ATR
- Сравнить с ATR percentile за последние 24 часа (1440 баров)
- Определить regime

| ATR Percentile | Regime | Multiplier Adjustment |
|----------------|--------|----------------------|
| > 75% | High Volatility | × 1.5 (расширить стоп) |
| 25-75% | Normal | × 1.0 (без изменений) |
| < 25% | Low Volatility | × 0.75 (сузить стоп) |

### Корректировка по направлению

Лонги более рискованные (45.5% WR vs 55% для шортов), добавить buffer:

| Direction | Additional Adjustment |
|-----------|----------------------|
| SHORT | × 1.0 (без изменений) |
| LONG | × 1.15 (на 15% шире) |

### Итоговая формула

```
Final_SL_Multiplier = Base_Multiplier × Volatility_Adjustment × Direction_Adjustment

Примеры:
- EXTREME_SPIKE + High Vol + LONG = 1.5 × 1.5 × 1.15 = 2.59x ATR
- EXTREME_SPIKE + Normal + SHORT = 1.5 × 1.0 × 1.0 = 1.5x ATR
- EARLY_SIGNAL + Low Vol + SHORT = 2.0 × 0.75 × 1.0 = 1.5x ATR
```

### Реализация

Файл: `position_manager.py`

Новая функция: `_calculate_adaptive_stop_loss`

Параметры:
- symbol: str
- signal_class: SignalClass
- direction: Direction
- entry_price: float

Логика:
1. Получить base_multiplier по signal_class из конфига
2. Получить текущий ATR для symbol
3. Рассчитать ATR percentile за 24h
4. Определить volatility_adjustment
5. Определить direction_adjustment
6. Вычислить final_multiplier
7. Вернуть stop_loss_price = entry_price ± (ATR × final_multiplier)

Вызов: В `_open_position_from_pending` перед созданием Position

### Конфигурация

```yaml
adaptive_stop_loss:
  enabled: true
  
  base_multipliers:
    extreme_spike: 1.5
    strong_signal: 1.8
    early_signal: 2.0
  
  volatility_adjustment:
    enabled: true
    lookback_bars: 1440              # 24 часа на 1m bars
    high_volatility_percentile: 75
    high_volatility_multiplier: 1.5
    low_volatility_percentile: 25
    low_volatility_multiplier: 0.75
  
  direction_adjustment:
    enabled: true
    long_additional_multiplier: 1.15
    short_additional_multiplier: 1.0
  
  # Safety limits
  min_stop_distance_pct: 0.3         # Минимум 0.3% от entry
  max_stop_distance_pct: 5.0         # Максимум 5% от entry
```

### Ожидаемый результат

| Метрика | Сейчас | После |
|---------|--------|-------|
| Stop-Loss Rate | 21% | 10-12% |
| SL Total PnL | -16.43% | -8 to -10% |
| Avg Loss | -0.78% | -0.90% (шире, но реже) |

---

## Улучшение 2: Tiered Take-Profit

### Проблема

Take-profit на 2-3x ATR достигается только в 5% сделок (3 из 62). Средняя длительность 5.7 минут — недостаточно для достижения далёких целей.

MFE показывает что средняя позиция видит +0.45% прибыли, но закрывается только с +0.22%.

### Решение

Вместо одного далёкого TP — три уровня с частичным закрытием.

### Структура уровней

| Уровень | Target | Закрыть | Действие после |
|---------|--------|---------|----------------|
| TP1 | 0.5x ATR | 30% позиции | Перенести SL в breakeven |
| TP2 | 1.0x ATR | 30% позиции | Активировать trailing stop |
| TP3 | 1.5x ATR | 40% позиции (остаток) | Полное закрытие |

### Логика работы

```
При открытии позиции:
  - Рассчитать TP1, TP2, TP3 уровни
  - remaining_position = 100%
  - tp1_hit = false, tp2_hit = false

На каждом баре:
  IF price >= TP1 AND NOT tp1_hit:
    - Закрыть 30% позиции
    - Перенести SL на entry_price (breakeven)
    - tp1_hit = true
    - Логировать partial close
  
  IF price >= TP2 AND NOT tp2_hit:
    - Закрыть 30% позиции
    - Активировать trailing stop для остатка
    - tp2_hit = true
    - Логировать partial close
  
  IF price >= TP3:
    - Закрыть остаток (40%)
    - Позиция полностью закрыта
    - Exit reason = TAKE_PROFIT_FULL
```

### Расчёт уровней по классам

Разные классы имеют разные характеристики движения:

| Signal Class | TP1 | TP2 | TP3 |
|--------------|-----|-----|-----|
| EXTREME_SPIKE | 0.4x ATR | 0.8x ATR | 1.2x ATR |
| STRONG_SIGNAL | 0.5x ATR | 1.0x ATR | 1.5x ATR |
| EARLY_SIGNAL | 0.6x ATR | 1.2x ATR | 2.0x ATR |

EXTREME_SPIKE имеет самые близкие цели (быстрые MR сделки).
EARLY_SIGNAL имеет самые далёкие цели (ловим большое движение).

### Реализация

Файл: `position_manager.py`

Новые поля в Position:
- tp1_price: float
- tp2_price: float
- tp3_price: float
- tp1_hit: bool = False
- tp2_hit: bool = False
- remaining_quantity_pct: float = 100.0
- sl_moved_to_breakeven: bool = False

Новая функция: `_check_tiered_take_profit`

Параметры:
- position: Position
- current_price: float
- current_bar: Bar

Возвращает:
- None (ничего не делать)
- PartialCloseAction (закрыть часть)
- FullCloseAction (закрыть всё)

Вызов: В `_check_exits_for_symbol` перед другими exit проверками

### Конфигурация

```yaml
tiered_take_profit:
  enabled: true
  
  levels_by_class:
    extreme_spike:
      tp1_atr: 0.4
      tp1_close_pct: 30
      tp2_atr: 0.8
      tp2_close_pct: 30
      tp3_atr: 1.2
      tp3_close_pct: 40
    
    strong_signal:
      tp1_atr: 0.5
      tp1_close_pct: 30
      tp2_atr: 1.0
      tp2_close_pct: 30
      tp3_atr: 1.5
      tp3_close_pct: 40
    
    early_signal:
      tp1_atr: 0.6
      tp1_close_pct: 30
      tp2_atr: 1.2
      tp2_close_pct: 30
      tp3_atr: 2.0
      tp3_close_pct: 40
  
  actions:
    move_sl_breakeven_on_tp1: true
    activate_trailing_on_tp2: true
```

### Ожидаемый результат

| Метрика | Сейчас | После |
|---------|--------|-------|
| TP Hit Rate | 5% | 35-45% (TP1), 20-25% (TP2), 10-15% (TP3) |
| Avg Win | +0.69% | +0.85% |
| MFE Capture | 49% | 65-70% |

---

## Улучшение 3: Отложенный Z-Score Exit

### Проблема

Z_SCORE_REVERSAL — основной exit (44 из 62, 71%). Средний profit +0.22%, но MFE +0.45%.

Z-exit срабатывает когда z возвращается к threshold (1.5σ), но:
- Цена может продолжить движение после z-normalization
- Ранний exit упускает +0.23% движения на каждой сделке

### Решение

Добавить условия задержки для Z-exit:
1. Минимальный profit
2. Минимальное время в позиции
3. Частичное закрытие вместо полного

### Новые условия Z-Exit

```
Z-Score Exit разрешён ТОЛЬКО если:
  (z вернулся к threshold)
  AND (
    (pnl >= min_profit_pct)
    OR (time_in_position >= min_hold_minutes)
    OR (position already partial closed via TP1)
  )
```

### Частичное закрытие по Z-Exit

Вместо полного закрытия:
- Если НЕТ partial closes ранее → закрыть 50%, держать остаток
- Если ЕСТЬ partial close (TP1 hit) → закрыть всё

### Параметры по классам

| Signal Class | Min Profit для Z-Exit | Min Hold | Partial Close % |
|--------------|----------------------|----------|-----------------|
| EXTREME_SPIKE | 0.15% | 2 мин | 60% |
| STRONG_SIGNAL | 0.20% | 3 мин | 50% |
| EARLY_SIGNAL | 0.25% | 4 мин | 50% |

EXTREME_SPIKE имеет самые мягкие условия (MR сделки должны быть быстрыми).

### Реализация

Файл: `position_manager.py`

Модификация: `_check_exit_conditions`

Текущая логика Z-exit:
```
IF abs(z_er_15m) <= z_threshold:
    return ExitReason.Z_SCORE_REVERSAL
```

Новая логика:
```
IF abs(z_er_15m) <= z_threshold:
    # Проверить условия задержки
    profit_ok = position.current_pnl_pct >= min_profit_for_class
    time_ok = position.duration_minutes >= min_hold_for_class
    already_partial = position.tp1_hit
    
    IF profit_ok OR time_ok OR already_partial:
        IF NOT already_partial AND delayed_z_exit.partial_enabled:
            # Частичное закрытие
            return PartialExitAction(percent=50, reason=Z_SCORE_REVERSAL_PARTIAL)
        ELSE:
            return ExitReason.Z_SCORE_REVERSAL
    ELSE:
        # Не закрывать, продолжить мониторинг
        return None
```

### Конфигурация

```yaml
delayed_z_exit:
  enabled: true
  
  conditions_by_class:
    extreme_spike:
      min_profit_pct: 0.15
      min_hold_minutes: 2
    strong_signal:
      min_profit_pct: 0.20
      min_hold_minutes: 3
    early_signal:
      min_profit_pct: 0.25
      min_hold_minutes: 4
  
  partial_close:
    enabled: true
    close_percent: 50
    skip_if_tp1_hit: true    # Если TP1 уже был — закрыть полностью
```

### Ожидаемый результат

| Метрика | Сейчас | После |
|---------|--------|-------|
| Avg Z-Exit PnL | +0.22% | +0.32% |
| MFE Capture | 49% | 60-65% |

---

## Улучшение 4: Асимметричная фильтрация Long/Short

### Проблема

| Direction | Win Rate | PnL |
|-----------|----------|-----|
| SHORT | 55.0% | +2.47% |
| LONG | 45.5% | -1.23% |

Лонги значительно хуже. Причины:
- Mean-reversion на дампах менее предсказуем (дамп может продолжиться)
- Крипторынок имеет bias к резким дампам без отскока
- BTC dump влияет на все альты одновременно

### Решение

Разные требования для входа в long и short позиции.

### Требования для LONG (строже)

| Параметр | SHORT | LONG | Комментарий |
|----------|-------|------|-------------|
| Min Z для EXTREME_SPIKE | 5.0 | 6.0 | Лонг только на сильных дампах |
| Pullback requirement | 20% | 25% | Больший отскок для подтверждения |
| BTC filter | Нет | Да | Не лонговать если BTC падает |
| Volume requirement | 1.0x | 1.2x | Нужен повышенный объём |

### BTC Filter для Long

Логика: Не открывать LONG если BTC в сильном нисходящем тренде.

```
BTC Filter для LONG:
  btc_z_er = get_btc_z_score()
  
  IF btc_z_er < -2.0:
    # BTC сильно падает, не лонговать альты
    return BLOCKED
  
  IF btc_z_er < -1.0 AND signal_class != EXTREME_SPIKE:
    # BTC падает, лонговать только extreme dips
    return BLOCKED
```

### Опционально: Отключение LONG для EARLY_SIGNAL

EARLY_SIGNAL лонги имеют наименьшую надёжность. Опция полного отключения:

```yaml
direction_filters:
  long:
    disable_for_early_signal: true  # Не открывать LONG для z < 3.0
```

### Реализация

Файл: `position_manager.py`

Модификация: `_create_pending_signal` или `_apply_class_aware_filters`

Добавить проверку после классификации:

```
IF trade_direction == Direction.UP:  # LONG
    # Проверить усиленные требования
    IF NOT _check_long_requirements(signal_class, features):
        logger.info(f"[FILTER] {symbol}: LONG blocked, stricter requirements not met")
        return  # Не создавать pending signal
```

Новая функция: `_check_long_requirements`

### Конфигурация

```yaml
direction_filters:
  enabled: true
  
  long:
    # Усиленные пороги
    min_extreme_spike_z: 6.0        # Выше чем для short (5.0)
    min_strong_signal_z: 3.5        # Выше чем для short (3.0)
    pullback_multiplier: 1.25       # На 25% больше pullback
    volume_multiplier: 1.2          # На 20% больше volume
    
    # BTC filter
    btc_filter_enabled: true
    btc_block_threshold: -2.0       # Блокировать если BTC z < -2.0
    btc_restrict_threshold: -1.0    # Ограничить если BTC z < -1.0
    
    # Опционально
    disable_for_early_signal: false # Полностью отключить LONG для EARLY
  
  short:
    # Стандартные пороги (без изменений)
    min_extreme_spike_z: 5.0
    min_strong_signal_z: 3.0
    pullback_multiplier: 1.0
    volume_multiplier: 1.0
    btc_filter_enabled: false
```

### Ожидаемый результат

| Метрика | Сейчас | После |
|---------|--------|-------|
| Long Win Rate | 45.5% | 52-55% |
| Long PnL | -1.23% | +0.5 to +1.0% |
| Long Count | 22 | 12-15 (меньше, но качественнее) |

---

## Улучшение 5: Активация Trailing Stop

### Проблема

Trailing stop сработал 1 раз из 62 (1.6%). Причина: activation threshold слишком высокий — требует достижения 50% от TP, который сам редко достигается.

### Решение

Снизить порог активации и привязать к реальным profit уровням.

### Новые параметры активации

| Signal Class | Activation Profit | Trail Distance |
|--------------|-------------------|----------------|
| EXTREME_SPIKE | +0.25% | 0.5x ATR |
| STRONG_SIGNAL | +0.35% | 0.7x ATR |
| EARLY_SIGNAL | +0.45% | 1.0x ATR |

EXTREME_SPIKE имеет самую раннюю активацию и самый tight trail (быстрые MR сделки).

### Интеграция с Tiered TP

Trailing stop автоматически активируется при достижении TP2:

```
IF tp2_hit:
    trailing_stop_active = true
    trailing_distance = config.distance_atr_by_class[signal_class]
```

Но также может активироваться раньше по profit threshold.

### Логика trailing

```
На каждом баре (если trailing активен):
  IF direction == UP:
    new_trail_price = current_price - (ATR × trail_distance)
    IF new_trail_price > current_trail_price:
      current_trail_price = new_trail_price  # Поднять trail
    IF current_price <= current_trail_price:
      return ExitReason.TRAILING_STOP
  
  ELSE:  # direction == DOWN
    new_trail_price = current_price + (ATR × trail_distance)
    IF new_trail_price < current_trail_price:
      current_trail_price = new_trail_price  # Опустить trail
    IF current_price >= current_trail_price:
      return ExitReason.TRAILING_STOP
```

### Реализация

Файл: `position_manager.py`

Новые поля в Position:
- trailing_active: bool = False
- trailing_price: Optional[float] = None
- trailing_activation_profit: float
- trailing_distance_atr: float

Модификация: `_update_trailing_stop`

Вызов: В `_check_exits_for_symbol` после tiered TP проверки

### Конфигурация

```yaml
trailing_stop:
  enabled: true
  
  activation_by_class:
    extreme_spike:
      profit_threshold_pct: 0.25
      distance_atr: 0.5
    strong_signal:
      profit_threshold_pct: 0.35
      distance_atr: 0.7
    early_signal:
      profit_threshold_pct: 0.45
      distance_atr: 1.0
  
  behavior:
    activate_on_tp2: true           # Авто-активация при TP2
    update_frequency: "every_bar"   # Обновлять каждый бар
    use_close_price: true           # Сравнивать с close, не high/low
```

### Ожидаемый результат

| Метрика | Сейчас | После |
|---------|--------|-------|
| Trailing Activation Rate | 1.6% | 25-35% |
| Trailing Exit Rate | 1.6% | 15-20% |
| Avg Trailing Exit PnL | +0.18% | +0.50% |

---

## Улучшение 6: Интеллектуальный Time Exit

### Проблема

Time exit сработал 1 раз из 62. Max hold (20/45/90 мин по классам) слишком длинный — другие exits срабатывают раньше.

Нет логики для "мёртвых" позиций — которые не двигаются ни в profit, ни в loss.

### Решение

Добавить aggressive time exit для:
1. Убыточных позиций, застрявших в убытке
2. Flat позиций, которые не двигаются

### Новые time exit правила

| Условие | Max Time | Действие |
|---------|----------|----------|
| PnL < -0.3% | 5 мин | Закрыть (cut losses early) |
| PnL между -0.1% и +0.1% | 8 мин | Закрыть (opportunity cost) |
| PnL > +0.1% | Max hold по классу | Стандартный exit |

### Логика

```
IF position.duration_minutes >= 5:
    IF position.pnl_pct < -0.3:
        return ExitReason.TIME_EXIT_LOSING

IF position.duration_minutes >= 8:
    IF abs(position.pnl_pct) < 0.1:
        return ExitReason.TIME_EXIT_FLAT

IF position.duration_minutes >= max_hold_for_class:
    return ExitReason.TIME_EXIT_MAX_HOLD
```

### Параметры по классам

| Signal Class | Losing Threshold | Losing Max Time | Flat Max Time | Max Hold |
|--------------|------------------|-----------------|---------------|----------|
| EXTREME_SPIKE | -0.25% | 4 мин | 6 мин | 20 мин |
| STRONG_SIGNAL | -0.30% | 5 мин | 8 мин | 45 мин |
| EARLY_SIGNAL | -0.35% | 6 мин | 10 мин | 90 мин |

EXTREME_SPIKE — самые строгие time limits (MR должен работать быстро).

### Реализация

Файл: `position_manager.py`

Модификация: `_check_exit_conditions`

Добавить проверки перед текущими time exit:

```
# Aggressive time exits
losing_exit = _check_losing_time_exit(position)
if losing_exit:
    return losing_exit

flat_exit = _check_flat_time_exit(position)
if flat_exit:
    return flat_exit

# Standard max hold
if position.duration >= max_hold:
    return ExitReason.TIME_EXIT_MAX_HOLD
```

### Конфигурация

```yaml
time_exit:
  enabled: true
  
  aggressive_exits:
    enabled: true
    
    losing_position:
      extreme_spike:
        threshold_pct: -0.25
        max_minutes: 4
      strong_signal:
        threshold_pct: -0.30
        max_minutes: 5
      early_signal:
        threshold_pct: -0.35
        max_minutes: 6
    
    flat_position:
      extreme_spike:
        threshold_pct: 0.1      # |pnl| < 0.1% = flat
        max_minutes: 6
      strong_signal:
        threshold_pct: 0.1
        max_minutes: 8
      early_signal:
        threshold_pct: 0.1
        max_minutes: 10
  
  max_hold_minutes:
    extreme_spike: 20
    strong_signal: 45
    early_signal: 90
```

### Ожидаемый результат

| Метрика | Сейчас | После |
|---------|--------|-------|
| Time Exit Rate | 1.6% | 10-15% |
| Avg Losing Hold Time | ~5-6 мин | ~4 мин |
| Freed Capital | - | Быстрее переиспользуется |

---

## Улучшение 7: Минимальный Profit Filter

### Проблема

При ATR = 0.2% и target 1.5x ATR, ожидаемый profit = 0.3%.
После комиссий (0.08% round trip) остаётся 0.22%.
R:R с таким profit слишком низкий.

### Решение

Не открывать позицию если expected profit слишком мал для покрытия комиссий с приемлемым R:R.

### Формула фильтрации

```
expected_tp1_profit = ATR × tp1_multiplier × direction_sign
expected_net_profit = expected_tp1_profit - estimated_fees

IF expected_net_profit < min_expected_profit_pct:
    SKIP position
```

### Параметры

| Параметр | Значение | Комментарий |
|----------|----------|-------------|
| Estimated fees | 0.10% | 0.04% × 2 (taker) + buffer |
| Min expected profit | 0.35% | После комиссий должно остаться 0.25%+ |
| Check against | TP1 | Первый уровень (самый вероятный) |

### Пример

```
ATR = 0.4%
TP1 = 0.5 × ATR = 0.2%
Expected net = 0.2% - 0.1% = 0.1%
Min required = 0.35%
Decision: SKIP (недостаточный profit potential)
```

```
ATR = 1.0%
TP1 = 0.5 × ATR = 0.5%
Expected net = 0.5% - 0.1% = 0.4%
Min required = 0.35%
Decision: ENTER (достаточный profit potential)
```

### Реализация

Файл: `position_manager.py`

Новая функция: `_check_min_profit_filter`

Вызов: В `_create_pending_signal` или `_evaluate_*_triggers` перед открытием

### Конфигурация

```yaml
min_profit_filter:
  enabled: true
  
  estimated_fees_pct: 0.10         # Round-trip fees
  min_expected_profit_pct: 0.35    # Минимум profit до комиссий
  
  check_against: "tp1"             # Проверять против TP1 (самый вероятный)
  
  # Или разные пороги по классам
  by_class:
    extreme_spike: 0.30            # MR может иметь меньший target
    strong_signal: 0.35
    early_signal: 0.40             # Early требует больше потенциала
```

### Ожидаемый результат

| Метрика | Сейчас | После |
|---------|--------|-------|
| Trades Count | 62 | 45-50 (меньше) |
| Avg Expected Profit | Variable | >= 0.35% |
| Win Rate | 51.6% | 53-55% (лучше качество) |

---

## Порядок внедрения

### Фаза 1: Критические улучшения (Неделя 1)

| № | Улучшение | Приоритет | Причина |
|---|-----------|-----------|---------|
| 1 | Tiered Take-Profit | Высший | +8-10% total PnL impact |
| 2 | Адаптивный Stop-Loss | Высший | +5-8% total PnL impact |

Эти два улучшения решают главные проблемы: SL съедает прибыль, TP не достигается.

### Фаза 2: Оптимизация exits (Неделя 2)

| № | Улучшение | Приоритет | Причина |
|---|-----------|-----------|---------|
| 3 | Trailing Stop activation | Высокий | +3-5% total PnL impact |
| 4 | Отложенный Z-Exit | Высокий | +2-4% total PnL impact |

Улучшают capture существующих profitable движений.

### Фаза 3: Фильтрация качества (Неделя 3)

| № | Улучшение | Приоритет | Причина |
|---|-----------|-----------|---------|
| 5 | Long/Short Asymmetry | Средний | +2-3% total PnL impact |
| 6 | Time Exit | Средний | +1-2% total PnL impact |
| 7 | Min Profit Filter | Средний | Улучшает качество |

Тонкая настройка после основных улучшений.

---

## Тестирование

### Unit Tests

Для каждого улучшения создать тесты:

1. **Adaptive SL:**
   - Test volatility calculation
   - Test multiplier combinations
   - Test safety limits (min/max)

2. **Tiered TP:**
   - Test level calculations
   - Test partial close logic
   - Test SL move to breakeven

3. **Delayed Z-Exit:**
   - Test delay conditions
   - Test partial close
   - Test interaction with TP1

4. **Direction Filters:**
   - Test BTC filter
   - Test elevated thresholds
   - Test EARLY_SIGNAL disable option

5. **Trailing Stop:**
   - Test activation conditions
   - Test trail price updates
   - Test exit trigger

6. **Time Exit:**
   - Test losing position exit
   - Test flat position exit
   - Test max hold

7. **Min Profit Filter:**
   - Test profit calculation
   - Test filter decision

### Integration Testing

Запустить систему в parallel mode:
- Старая логика делает решения
- Новая логика логирует что бы она сделала
- Сравнить результаты за 48-72 часа

### Backtest

Если есть исторические данные:
- Прогнать backtest с новыми параметрами
- Сравнить метрики: Win Rate, Avg PnL, R:R, Max Drawdown

---

## Мониторинг после внедрения

### Ключевые метрики

| Метрика | Цель | Alert Threshold |
|---------|------|-----------------|
| Win Rate | >= 55% | < 50% |
| R:R Ratio | >= 1.5 | < 1.0 |
| Stop-Loss Rate | <= 12% | > 18% |
| TP1 Hit Rate | >= 35% | < 25% |
| Trailing Activation | >= 25% | < 15% |
| Net PnL (daily) | > 0 | < -2% |

### Dashboards

Создать dashboard с:
- PnL по часам/дням
- Exit reasons distribution
- Win rate по классам и направлениям
- MFE/MAE tracking
- Trailing stop performance

### Alerts

Настроить alerts:
- Win rate падает ниже 48% за последние 20 сделок
- 3 consecutive stop-losses
- Daily PnL < -3%
- Unusual long losing streak

---

## Целевые метрики после всех улучшений

| Метрика | Сейчас | Цель | Минимум для production |
|---------|--------|------|----------------------|
| Win Rate | 51.6% | 57-60% | 55% |
| Avg Win | +0.69% | +0.90% | +0.80% |
| Avg Loss | -0.78% | -0.55% | -0.65% |
| R:R Ratio | 0.89 | 1.6 | 1.3 |
| Stop-Loss Rate | 21% | 10% | 14% |
| TP Hit Rate (any) | 5% | 40% | 30% |
| MFE Capture | 49% | 70% | 60% |
| Net PnL (after fees) | -4.96% | +8-12% | +3% |

---

## Конфигурация: Полный пример

```yaml
# ============================================
# HYBRID STRATEGY IMPROVEMENTS CONFIGURATION
# ============================================

# Улучшение 1: Адаптивный Stop-Loss
adaptive_stop_loss:
  enabled: true
  
  base_multipliers:
    extreme_spike: 1.5
    strong_signal: 1.8
    early_signal: 2.0
  
  volatility_adjustment:
    enabled: true
    lookback_bars: 1440
    high_volatility_percentile: 75
    high_volatility_multiplier: 1.5
    low_volatility_percentile: 25
    low_volatility_multiplier: 0.75
  
  direction_adjustment:
    enabled: true
    long_additional_multiplier: 1.15
    short_additional_multiplier: 1.0
  
  min_stop_distance_pct: 0.3
  max_stop_distance_pct: 5.0

# Улучшение 2: Tiered Take-Profit
tiered_take_profit:
  enabled: true
  
  levels_by_class:
    extreme_spike:
      tp1_atr: 0.4
      tp1_close_pct: 30
      tp2_atr: 0.8
      tp2_close_pct: 30
      tp3_atr: 1.2
      tp3_close_pct: 40
    strong_signal:
      tp1_atr: 0.5
      tp1_close_pct: 30
      tp2_atr: 1.0
      tp2_close_pct: 30
      tp3_atr: 1.5
      tp3_close_pct: 40
    early_signal:
      tp1_atr: 0.6
      tp1_close_pct: 30
      tp2_atr: 1.2
      tp2_close_pct: 30
      tp3_atr: 2.0
      tp3_close_pct: 40
  
  actions:
    move_sl_breakeven_on_tp1: true
    activate_trailing_on_tp2: true

# Улучшение 3: Отложенный Z-Exit
delayed_z_exit:
  enabled: true
  
  conditions_by_class:
    extreme_spike:
      min_profit_pct: 0.15
      min_hold_minutes: 2
    strong_signal:
      min_profit_pct: 0.20
      min_hold_minutes: 3
    early_signal:
      min_profit_pct: 0.25
      min_hold_minutes: 4
  
  partial_close:
    enabled: true
    close_percent: 50
    skip_if_tp1_hit: true

# Улучшение 4: Асимметрия Long/Short
direction_filters:
  enabled: true
  
  long:
    min_extreme_spike_z: 6.0
    min_strong_signal_z: 3.5
    pullback_multiplier: 1.25
    volume_multiplier: 1.2
    btc_filter_enabled: true
    btc_block_threshold: -2.0
    btc_restrict_threshold: -1.0
    disable_for_early_signal: false
  
  short:
    min_extreme_spike_z: 5.0
    min_strong_signal_z: 3.0
    pullback_multiplier: 1.0
    volume_multiplier: 1.0
    btc_filter_enabled: false

# Улучшение 5: Trailing Stop
trailing_stop:
  enabled: true
  
  activation_by_class:
    extreme_spike:
      profit_threshold_pct: 0.25
      distance_atr: 0.5
    strong_signal:
      profit_threshold_pct: 0.35
      distance_atr: 0.7
    early_signal:
      profit_threshold_pct: 0.45
      distance_atr: 1.0
  
  behavior:
    activate_on_tp2: true
    update_frequency: "every_bar"
    use_close_price: true

# Улучшение 6: Time Exit
time_exit:
  enabled: true
  
  aggressive_exits:
    enabled: true
    
    losing_position:
      extreme_spike:
        threshold_pct: -0.25
        max_minutes: 4
      strong_signal:
        threshold_pct: -0.30
        max_minutes: 5
      early_signal:
        threshold_pct: -0.35
        max_minutes: 6
    
    flat_position:
      extreme_spike:
        threshold_pct: 0.1
        max_minutes: 6
      strong_signal:
        threshold_pct: 0.1
        max_minutes: 8
      early_signal:
        threshold_pct: 0.1
        max_minutes: 10
  
  max_hold_minutes:
    extreme_spike: 20
    strong_signal: 45
    early_signal: 90

# Улучшение 7: Min Profit Filter
min_profit_filter:
  enabled: true
  estimated_fees_pct: 0.10
  min_expected_profit_pct: 0.35
  check_against: "tp1"
```

---

## Резюме

| Улучшение | Решает проблему | Expected Impact |
|-----------|-----------------|-----------------|
| Adaptive SL | SL съедает прибыль | +5-8% PnL |
| Tiered TP | TP не достигается | +8-10% PnL |
| Delayed Z-Exit | Ранний exit | +2-4% PnL |
| Long/Short Asymmetry | Лонги убыточны | +2-3% PnL |
| Trailing Stop | Не активируется | +3-5% PnL |
| Time Exit | Мёртвые позиции | +1-2% PnL |
| Min Profit Filter | Низкий R:R сделки | Quality improvement |

**Суммарный ожидаемый impact: +21-32% improvement в total PnL**

При текущем gross PnL +1.24% и net PnL -4.96%, улучшения должны вывести систему в стабильный плюс +5-10% net PnL.
