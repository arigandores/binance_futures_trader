# Спецификация критических исправлений v2

## Диагноз проблемы

### Сравнение результатов

| Метрика | До улучшений | После улучшений | Проблема |
|---------|--------------|-----------------|----------|
| Win Rate | 51.6% | 34.6% | ❌ Катастрофа |
| Total PnL | +1.24% | -5.21% | ❌ Убыток |
| Avg Win | +0.69% | +3.18% | ✅ Улучшилось |
| Avg Loss | -0.78% | -1.99% | ❌ Ухудшилось |
| SL Rate | 21% | 7.7% | ✅ Улучшилось |
| TIME_EXIT_LOSING | 0% | 38.5% | ❌ Новая проблема |

### Корневая причина

**TIME_EXIT_LOSING убивает систему.** 

10 из 26 позиций (38.5%) закрыты этим правилом с убытком ~-2% каждая. Это новое правило из Улучшения #6, которое оказалось слишком агрессивным.

### Расчёт влияния

```
TIME_EXIT_LOSING contribution: 10 × (-2.0%) = -20.0%
All other losing exits:        ~6.0%
Total losses:                  -26.0%

All winning exits:             +17.5%
────────────────────────────────────────
Net PnL:                       ≈ -8.5%
```

TIME_EXIT_LOSING генерирует больше убытка чем все прибыльные сделки вместе.

---

## Исправление 1: Отключить или радикально смягчить TIME_EXIT_LOSING

### Вариант A: Полное отключение (рекомендуется для начала)

```yaml
time_exit:
  aggressive_exits:
    enabled: false  # ОТКЛЮЧИТЬ полностью
```

Это вернёт систему к состоянию близкому к "до улучшений", но с работающими Tiered TP и Trailing Stop.

### Вариант B: Радикальное смягчение

Если хочется сохранить логику, но сделать её безопасной:

| Параметр | Было | Стало | Логика |
|----------|------|-------|--------|
| Losing threshold | -0.25% to -0.35% | -0.8% to -1.0% | Только глубокие убытки |
| Max time | 4-6 мин | 12-15 мин | Дать больше времени |
| Activation delay | 0 мин | 5 мин | Не проверять первые 5 мин |

```yaml
time_exit:
  aggressive_exits:
    enabled: true
    
    # СМЯГЧЁННЫЕ настройки
    activation_delay_minutes: 5     # НЕ проверять первые 5 минут
    
    losing_position:
      extreme_spike:
        threshold_pct: -0.8         # Было -0.25
        max_minutes: 12             # Было 4
      strong_signal:
        threshold_pct: -0.9         # Было -0.30
        max_minutes: 14             # Было 5
      early_signal:
        threshold_pct: -1.0         # Было -0.35
        max_minutes: 15             # Было 6
```

### Вариант C: Условное TIME_EXIT (продвинутый)

Закрывать по времени ТОЛЬКО если:
- Убыток глубокий (> -0.5%)
- И z-score развернулся против позиции
- И taker flow против позиции

```yaml
time_exit:
  aggressive_exits:
    enabled: true
    mode: "conditional"  # Не просто по времени, а по комбинации факторов
    
    conditions:
      min_loss_pct: -0.5
      require_z_reversal: true      # z должен быть против нас
      require_flow_reversal: true   # taker flow должен быть против
      min_time_minutes: 8
```

---

## Исправление 2: Убрать FLAT position exit

Текущая настройка закрывает позиции с |pnl| < 0.1% после 6-10 минут.

### Проблема

Flat позиция может ждать breakout. Закрытие flat позиции = упущенная возможность.

### Решение

```yaml
time_exit:
  aggressive_exits:
    flat_position:
      enabled: false  # ОТКЛЮЧИТЬ полностью
```

---

## Исправление 3: Смягчить Delayed Z-Exit

### Проблема

Delayed Z-Exit требует min profit (0.15-0.25%) перед закрытием по z-score. Если позиция не достигает этого profit, она держится и попадает в TIME_EXIT_LOSING.

### Решение

Убрать требование min profit, вернуть обычный Z-exit, но с частичным закрытием:

```yaml
delayed_z_exit:
  enabled: true
  
  # УБРАТЬ требование min profit
  require_min_profit: false
  
  # Оставить только partial close
  partial_close:
    enabled: true
    close_percent: 50             # Закрыть 50% по z-exit
    hold_remainder: true          # Держать остаток для trailing
```

Альтернативно — полностью отключить delayed z-exit:

```yaml
delayed_z_exit:
  enabled: false  # Вернуться к обычному z-exit
```

---

## Исправление 4: Уменьшить Adaptive Stop-Loss multipliers

### Проблема

Avg Loss вырос с -0.78% до -1.99%. Адаптивный SL сделал стопы слишком широкими.

### Решение

Уменьшить базовые множители:

| Signal Class | Было | Стало |
|--------------|------|-------|
| EXTREME_SPIKE | 1.5x | 1.2x |
| STRONG_SIGNAL | 1.8x | 1.4x |
| EARLY_SIGNAL | 2.0x | 1.6x |

```yaml
adaptive_stop_loss:
  enabled: true
  
  base_multipliers:
    extreme_spike: 1.2    # Было 1.5
    strong_signal: 1.4    # Было 1.8
    early_signal: 1.6     # Было 2.0
  
  # Уменьшить volatility adjustment
  volatility_adjustment:
    high_volatility_multiplier: 1.3   # Было 1.5
    low_volatility_multiplier: 0.85   # Было 0.75
  
  # Строже ограничить максимум
  max_stop_distance_pct: 3.0          # Было 5.0
```

---

## Исправление 5: Изменить приоритет Exit Conditions

### Проблема

Если TIME_EXIT_LOSING проверяется раньше чем Trailing Stop или TP, позиции закрываются в убытке вместо того чтобы дать им достичь profit.

### Правильный порядок проверки

```
1. TAKE_PROFIT (TP1, TP2, TP3)     - Первый приоритет (прибыль)
2. TRAILING_STOP                   - Второй (защита прибыли)
3. STOP_LOSS                       - Третий (защита от больших убытков)
4. Z_SCORE_REVERSAL                - Четвёртый (сигнал закончился)
5. TIME_EXIT_MAX_HOLD              - Пятый (максимальное время)
6. TIME_EXIT_LOSING                - ПОСЛЕДНИЙ (только если ничего не сработало)
```

### Реализация

В `_check_exit_conditions` порядок проверок должен быть:

```
async def _check_exit_conditions(self, position, features, bar):
    # 1. Сначала проверить profitable exits
    tp_result = self._check_tiered_take_profit(position, bar)
    if tp_result:
        return tp_result
    
    trailing_result = self._check_trailing_stop(position, bar)
    if trailing_result:
        return trailing_result
    
    # 2. Потом protective exits
    sl_result = self._check_stop_loss(position, bar)
    if sl_result:
        return sl_result
    
    # 3. Потом signal-based exits
    z_result = self._check_z_score_exit(position, features)
    if z_result:
        return z_result
    
    # 4. Time exits ПОСЛЕДНИМИ
    max_hold_result = self._check_max_hold(position)
    if max_hold_result:
        return max_hold_result
    
    # 5. Aggressive time exit только если ничего другого
    losing_result = self._check_time_exit_losing(position)
    if losing_result:
        return losing_result
    
    return None
```

---

## Исправление 6: Добавить Grace Period для новых позиций

### Проблема

Позиции могут быть в минусе сразу после открытия из-за spread или slippage. TIME_EXIT_LOSING может закрыть их слишком рано.

### Решение

Не применять никакие aggressive exits первые N минут:

```yaml
position_management:
  grace_period_minutes: 3   # Первые 3 минуты не применять aggressive exits
```

### Логика

```
IF position.duration_minutes < grace_period_minutes:
    # Применять ТОЛЬКО критические exits
    - STOP_LOSS (если цена пробила стоп)
    - TAKE_PROFIT (если цена достигла TP)
    
    # НЕ применять
    - TIME_EXIT_LOSING
    - TIME_EXIT_FLAT
    - Z_SCORE_REVERSAL (можно оставить, но лучше подождать)
```

---

## Рекомендуемая конфигурация (Quick Fix)

Для быстрого возврата к прибыльности применить минимальные изменения:

```yaml
# ==========================================
# QUICK FIX CONFIGURATION
# ==========================================

# Исправление 1: ОТКЛЮЧИТЬ агрессивные time exits
time_exit:
  aggressive_exits:
    enabled: false              # ГЛАВНОЕ ИЗМЕНЕНИЕ
  
  # Оставить только max hold
  max_hold_minutes:
    extreme_spike: 20
    strong_signal: 45
    early_signal: 90

# Исправление 3: Вернуть обычный Z-exit
delayed_z_exit:
  enabled: false                # Отключить задержку
  # ИЛИ убрать требование min profit:
  # require_min_profit: false

# Исправление 4: Уменьшить стопы
adaptive_stop_loss:
  enabled: true
  base_multipliers:
    extreme_spike: 1.2          # Было 1.5
    strong_signal: 1.4          # Было 1.8
    early_signal: 1.6           # Было 2.0
  max_stop_distance_pct: 3.0    # Было 5.0

# Исправление 6: Grace period
position_management:
  grace_period_minutes: 3

# Оставить работающие улучшения без изменений:
# - tiered_take_profit: enabled (работает - 3 TP3 + 1 TP)
# - trailing_stop: enabled (работает - 3 срабатывания)
```

---

## План действий

### Шаг 1: Немедленно (сегодня)

1. Отключить `time_exit.aggressive_exits.enabled: false`
2. Перезапустить систему
3. Мониторить 24 часа

### Шаг 2: После стабилизации (через 24-48 часов)

Если Win Rate вернулся к 50%+:
1. Уменьшить adaptive_stop_loss multipliers
2. Отключить или смягчить delayed_z_exit
3. Мониторить ещё 24 часа

### Шаг 3: Тонкая настройка (через 3-5 дней)

Если система прибыльна:
1. Попробовать включить TIME_EXIT_LOSING с мягкими настройками (Вариант B)
2. Добавить grace_period
3. A/B тестировать разные параметры

---

## Ожидаемые результаты после Quick Fix

| Метрика | Сейчас | После Quick Fix |
|---------|--------|-----------------|
| Win Rate | 34.6% | 48-52% |
| TIME_EXIT_LOSING | 38.5% | 0% (отключено) |
| Avg Loss | -1.99% | -1.2% to -1.5% |
| Total PnL | -5.21% | +1% to +3% |

---

## Анализ: Что работает, что не работает

### ✅ Работает хорошо (оставить)

| Улучшение | Доказательство | Действие |
|-----------|----------------|----------|
| Tiered Take-Profit | 3 TP3 + 1 TP = 4 profitable exits | Оставить |
| Trailing Stop | 3 срабатывания с прибылью | Оставить |
| Reduced SL Rate | 7.7% vs 21% раньше | Оставить концепцию, уменьшить width |

### ❌ Не работает (исправить/отключить)

| Улучшение | Проблема | Действие |
|-----------|----------|----------|
| TIME_EXIT_LOSING | 38.5% сделок, все в убытке | Отключить |
| TIME_EXIT_FLAT | Закрывает потенциальные winners | Отключить |
| Delayed Z-Exit | Держит losers слишком долго | Отключить или смягчить |
| Wide Adaptive SL | Avg Loss -1.99% | Уменьшить multipliers |

### ⚠️ Требует мониторинга

| Улучшение | Статус | Действие |
|-----------|--------|----------|
| Long/Short Asymmetry | Недостаточно данных | Мониторить |
| Min Profit Filter | Недостаточно данных | Мониторить |

---

## Диагностические метрики для мониторинга

После применения Quick Fix отслеживать:

| Метрика | Цель | Alert если |
|---------|------|------------|
| Win Rate | > 48% | < 42% |
| TIME_EXIT_LOSING count | 0 | > 0 (должно быть отключено) |
| Avg Loss | < -1.5% | > -2.0% |
| TP Hit Rate (any level) | > 25% | < 15% |
| Trailing Stop Rate | > 10% | < 5% |
| Total PnL (daily) | > 0% | < -3% |

---

## Логирование для отладки

Добавить детальное логирование каждого exit:

```
[EXIT] {symbol} {direction}
  - Reason: {exit_reason}
  - Duration: {minutes} min
  - Entry Price: {entry}
  - Exit Price: {exit}
  - PnL: {pnl}%
  - MFE: {mfe}% (max profit seen)
  - MAE: {mae}% (max loss seen)
  - Signal Class: {class}
  - Trading Mode: {mode}
  - TP1 Hit: {yes/no}
  - Trailing Active: {yes/no}
```

Это поможет понять почему каждая позиция закрылась и была ли возможность лучшего exit.

---

## Резюме

**Главная ошибка:** TIME_EXIT_LOSING слишком агрессивный. Он закрывает 38.5% позиций в убытке, не давая им шанса восстановиться.

**Quick Fix:** Отключить `time_exit.aggressive_exits.enabled: false`

**Ожидаемый результат:** Win Rate вернётся к ~50%, Total PnL станет положительным.

**Долгосрочно:** После стабилизации можно попробовать включить TIME_EXIT_LOSING с гораздо более мягкими настройками (threshold -0.8%, time 12+ минут, с grace period).
