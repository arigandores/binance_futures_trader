# Спецификация: Рефакторинг фильтров по классам сигналов

## Проблема

Текущая архитектура применяет WIN_RATE_MAX фильтры ДО классификации сигнала. Это приводит к тому, что все сигналы фильтруются одинаково, независимо от их силы и типа торговли.

```
ТЕКУЩИЙ ПОРЯДОК:
Event → WIN_RATE_MAX filters → Hybrid classification → Pending signal
         (одинаковые для всех)

ПРОБЛЕМА:
- EXTREME_SPIKE (z≥5) блокируется теми же фильтрами что и EARLY_SIGNAL (z=1.5)
- Mean-reversion на экстремумах работает даже на низколиквидных инструментах
- Early signals на низколиквидных инструментах = шум
- Единые пороги либо слишком строгие, либо слишком мягкие
```

---

## Целевая архитектура

```
НОВЫЙ ПОРЯДОК:
Event → Hybrid classification → Class-aware filters → Pending signal
                                 (разные по классам)

ПРИНЦИП:
- Чем сильнее сигнал, тем мягче фильтры (сигнал сам по себе надёжен)
- Чем слабее сигнал, тем строже фильтры (нужна дополнительная валидация)
```

---

## Часть 1: Изменения в порядке обработки

### 1.1 Текущий flow (detector.py + position_manager.py)

```
detector.py:
  _process_features()
    → _classify_signal() или _check_initiator_trigger()
    → создать Event
    → broadcast в queue

position_manager.py:
  _handle_alerts()
    → получить Event
    → _create_pending_signal()
        → WIN_RATE_MAX filters (строки 224-240)  ← БЛОКИРОВКА ЗДЕСЬ
        → создать PendingSignal
```

### 1.2 Новый flow

```
detector.py:
  _process_features()
    → _classify_signal()  ← Классификация ПЕРВОЙ
    → создать Event с signal_class
    → broadcast в queue

position_manager.py:
  _handle_alerts()
    → получить Event
    → _create_pending_signal()
        → _apply_class_aware_filters(event.signal_class)  ← НОВАЯ ФУНКЦИЯ
        → создать PendingSignal
```

---

## Часть 2: Новая структура фильтров

### 2.1 Конфигурация

Добавить в `config.yaml` новую секцию внутри `hybrid_strategy`:

```yaml
hybrid_strategy:
  enabled: true
  
  # ... existing classification thresholds ...
  
  # НОВОЕ: Фильтры по классам сигналов
  class_aware_filters:
    enabled: true  # Включить class-aware фильтрацию
    
    # ========== EXTREME_SPIKE (z ≥ 5.0) — Mean-Reversion ==========
    # Экстремальные аномалии надёжны сами по себе
    # Ослабленные фильтры: MR работает даже на низколиквидных инструментах
    extreme_spike:
      # Liquidity
      min_volume_usd: 25000           # Ослаблено (было 100000)
      min_trades_per_bar: 15          # Ослаблено (было 50)
      
      # Quality
      apply_btc_anomaly_filter: true  # Оставить — BTC хаос влияет на всё
      apply_beta_quality_filter: false # Отключить — для MR бета менее важна
      
      # Symbol restrictions
      use_global_blacklist: true
      additional_blacklist: []        # Можно добавить специфичные для MR
      
    # ========== STRONG_SIGNAL (3.0 ≤ z < 5.0) — Conditional Momentum ==========
    # Стандартные фильтры WIN_RATE_MAX
    strong_signal:
      # Liquidity
      min_volume_usd: 100000          # Стандартно
      min_trades_per_bar: 50          # Стандартно
      
      # Quality
      apply_btc_anomaly_filter: true
      apply_beta_quality_filter: true
      beta_min_abs: 0.1
      beta_max_abs: 3.0
      beta_min_r_squared: 0.2
      
      # Symbol restrictions
      use_global_blacklist: true
      additional_blacklist: []
      
    # ========== EARLY_SIGNAL (1.5 ≤ z < 3.0) — Early Momentum ==========
    # Усиленные фильтры: слабые сигналы требуют высокой ликвидности
    early_signal:
      # Liquidity — СТРОЖЕ
      min_volume_usd: 150000          # Усилено (было 100000)
      min_trades_per_bar: 75          # Усилено (было 50)
      
      # Quality — ВСЕ ВКЛЮЧЕНЫ
      apply_btc_anomaly_filter: true
      apply_beta_quality_filter: true
      beta_min_abs: 0.15              # Строже (было 0.1)
      beta_max_abs: 2.5               # Строже (было 3.0)
      beta_min_r_squared: 0.3         # Строже (было 0.2)
      
      # Symbol restrictions
      use_global_blacklist: true
      additional_blacklist: []        # Можно добавить high-risk символы
      
      # Дополнительный фильтр для EARLY
      require_recent_volume_spike: true  # Объём должен быть выше среднего
      recent_volume_spike_threshold: 1.5 # В 1.5 раза выше среднего
```

### 2.2 Global blacklist

Остаётся в `win_rate_max_profile` и применяется ко всем классам:

```yaml
win_rate_max_profile:
  symbol_blacklist:
    - "USDCUSDT"     # Стейблкоин
    - "BTCDOMUSDT"   # Индекс
    # ... другие
```

---

## Часть 3: Реализация функции фильтрации

### 3.1 Новая функция в position_manager.py

Файл: `position_manager.py`
Расположение: после текущих helper функций фильтрации (около строки 240)

Название: `_apply_class_aware_filters`

Сигнатура:
```
async def _apply_class_aware_filters(
    self,
    symbol: str,
    signal_class: SignalClass,
    features: Features,
    bar: Bar
) -> Tuple[bool, Optional[str]]
```

Возвращает:
- `(True, None)` — фильтры пройдены
- `(False, reason)` — заблокировано, reason содержит причину

### 3.2 Логика функции

```
ВХОД: symbol, signal_class, features, bar

1. Получить конфигурацию фильтров для данного класса:
   IF signal_class == EXTREME_SPIKE:
       filter_cfg = config.hybrid_strategy.class_aware_filters.extreme_spike
   ELIF signal_class == STRONG_SIGNAL:
       filter_cfg = config.hybrid_strategy.class_aware_filters.strong_signal
   ELIF signal_class == EARLY_SIGNAL:
       filter_cfg = config.hybrid_strategy.class_aware_filters.early_signal

2. Проверка global blacklist:
   IF filter_cfg.use_global_blacklist:
       IF symbol in win_rate_max_profile.symbol_blacklist:
           RETURN (False, "global_blacklist")

3. Проверка additional blacklist для класса:
   IF symbol in filter_cfg.additional_blacklist:
       RETURN (False, "class_blacklist")

4. Проверка ликвидности:
   volume_usd = self._get_recent_volume_usd(symbol, bar)
   trades_count = self._get_recent_trades_count(symbol, bar)
   
   IF volume_usd < filter_cfg.min_volume_usd:
       RETURN (False, "low_volume")
   
   IF trades_count < filter_cfg.min_trades_per_bar:
       RETURN (False, "low_trades")

5. Проверка BTC anomaly (если включена):
   IF filter_cfg.apply_btc_anomaly_filter:
       IF NOT self._check_btc_anomaly_filter(symbol):
           RETURN (False, "btc_anomaly")

6. Проверка beta quality (если включена):
   IF filter_cfg.apply_beta_quality_filter:
       beta_info = self._get_beta_info(symbol)
       
       IF abs(beta_info.beta) < filter_cfg.beta_min_abs:
           RETURN (False, "beta_too_low")
       
       IF abs(beta_info.beta) > filter_cfg.beta_max_abs:
           RETURN (False, "beta_too_high")
       
       IF beta_info.r_squared < filter_cfg.beta_min_r_squared:
           RETURN (False, "beta_unreliable")

7. Дополнительная проверка для EARLY_SIGNAL:
   IF signal_class == EARLY_SIGNAL AND filter_cfg.require_recent_volume_spike:
       avg_volume = self._get_average_volume(symbol, lookback=60)
       current_volume = self._get_recent_volume_usd(symbol, bar)
       
       IF current_volume < avg_volume * filter_cfg.recent_volume_spike_threshold:
           RETURN (False, "no_volume_spike")

8. Все фильтры пройдены:
   RETURN (True, None)
```

### 3.3 Интеграция в _create_pending_signal

Файл: `position_manager.py`
Функция: `_create_pending_signal`
Строки: 224-240 (текущие WIN_RATE_MAX фильтры)

Изменения:

```
БЫЛО (строки 224-240):
  if profile == "WIN_RATE_MAX":
      if not self._check_btc_anomaly_filter(symbol):
          return
      if not self._check_symbol_quality_filter(symbol):
          return
      if not self._check_beta_quality_filter(symbol):
          return

СТАЛО:
  # Class-aware filtering (заменяет старые WIN_RATE_MAX фильтры)
  if self.config.hybrid_strategy.class_aware_filters.enabled:
      signal_class = event.signal_class
      passed, block_reason = await self._apply_class_aware_filters(
          symbol, signal_class, features, bar
      )
      
      if not passed:
          logger.info(
              f"[FILTER] {symbol}: blocked for {signal_class.name}, "
              f"reason={block_reason}"
          )
          return
  
  # Legacy WIN_RATE_MAX filters (если class-aware отключены)
  elif profile == "WIN_RATE_MAX":
      # ... старый код без изменений ...
```

---

## Часть 4: Вспомогательные функции

### 4.1 Получение volume в USD

Если функция `_get_recent_volume_usd` не существует, добавить:

```
Название: _get_recent_volume_usd
Параметры: symbol, bar
Логика:
  - Получить quote_volume из bar (или из extended_features)
  - Для USDT-пар это уже USD
  - Вернуть значение
```

### 4.2 Получение количества trades

Если функция `_get_recent_trades_count` не существует, добавить:

```
Название: _get_recent_trades_count
Параметры: symbol, bar
Логика:
  - Получить num_trades из bar
  - Вернуть значение
```

### 4.3 Получение средней volume

Для проверки volume spike в EARLY_SIGNAL:

```
Название: _get_average_volume
Параметры: symbol, lookback (в минутах)
Логика:
  - Получить исторические данные volume за lookback баров
  - Вычислить среднее
  - Вернуть значение
```

---

## Часть 5: Логирование

### 5.1 При блокировке сигнала

Формат лога:
```
[FILTER] {symbol}: blocked for {signal_class}, reason={reason}, metrics={{volume_usd}, {trades}, {beta}}
```

Пример:
```
[FILTER] MITOUSDT: blocked for EARLY_SIGNAL, reason=low_volume, metrics={volume=$45000, trades=32, beta=0.85}
```

### 5.2 При прохождении фильтров

Формат лога:
```
[FILTER] {symbol}: passed for {signal_class}, filters_applied={{list}}
```

Пример:
```
[FILTER] ETHUSDT: passed for STRONG_SIGNAL, filters_applied={liquidity, btc_anomaly, beta_quality}
```

### 5.3 Метрики для мониторинга

Добавить счётчики:

| Метрика | Описание |
|---------|----------|
| `signals_received_by_class` | Количество сигналов по классам до фильтрации |
| `signals_blocked_by_class` | Количество заблокированных по классам |
| `signals_passed_by_class` | Количество прошедших по классам |
| `block_reasons_by_class` | Распределение причин блокировки по классам |

---

## Часть 6: Миграция

### 6.1 Этапы

```
Этап 1: Добавить конфигурацию
  - Добавить секцию class_aware_filters в config.yaml
  - enabled: false (пока отключено)
  - Задеплоить без изменения поведения

Этап 2: Реализовать функцию фильтрации
  - Добавить _apply_class_aware_filters
  - Добавить вспомогательные функции если нужны
  - Покрыть unit тестами

Этап 3: Интегрировать в _create_pending_signal
  - Добавить условную логику (if class_aware_filters.enabled)
  - Сохранить старую логику как fallback

Этап 4: Parallel mode тестирование
  - enabled: true
  - Логировать решения обеих систем
  - Сравнивать результаты 24-48 часов

Этап 5: Полное включение
  - Убрать параллельное логирование
  - Удалить legacy фильтры (опционально, можно оставить как fallback)
```

### 6.2 Rollback план

Если class-aware фильтры показывают худшие результаты:

```yaml
hybrid_strategy:
  class_aware_filters:
    enabled: false  # Откатиться на legacy WIN_RATE_MAX
```

---

## Часть 7: Тестовые сценарии

### 7.1 EXTREME_SPIKE проходит мягкие фильтры

Входные данные:
- Symbol: MITOUSDT
- Z-score: +6.5 (EXTREME_SPIKE)
- Volume: $30,000 (ниже стандартного порога 100k)
- Trades: 20 (ниже стандартного порога 50)

Ожидаемый результат:
- Старая система: BLOCKED (low_volume)
- Новая система: PASSED (порог для EXTREME_SPIKE = $25k, 15 trades)

### 7.2 EARLY_SIGNAL блокируется строгими фильтрами

Входные данные:
- Symbol: BANDUSDT
- Z-score: +2.0 (EARLY_SIGNAL)
- Volume: $120,000 (выше стандартного порога)
- Trades: 60 (выше стандартного порога)

Ожидаемый результат:
- Старая система: PASSED
- Новая система: BLOCKED (порог для EARLY_SIGNAL = $150k, 75 trades)

### 7.3 STRONG_SIGNAL с плохой beta

Входные данные:
- Symbol: XYZUSDT
- Z-score: +4.0 (STRONG_SIGNAL)
- Volume: $200,000
- Trades: 100
- Beta: 0.05 (слишком низкая)
- R-squared: 0.15 (ненадёжная)

Ожидаемый результат:
- BLOCKED (beta_too_low + beta_unreliable)

### 7.4 EARLY_SIGNAL без volume spike

Входные данные:
- Symbol: ETHUSDT
- Z-score: +1.8 (EARLY_SIGNAL)
- Current volume: $160,000
- Average volume (60m): $150,000
- Ratio: 1.07x (ниже порога 1.5x)

Ожидаемый результат:
- BLOCKED (no_volume_spike)

### 7.5 Global blacklist применяется ко всем классам

Входные данные:
- Symbol: USDCUSDT (в global blacklist)
- Z-score: +7.0 (EXTREME_SPIKE)

Ожидаемый результат:
- BLOCKED (global_blacklist) — даже для EXTREME_SPIKE

---

## Часть 8: Ожидаемые результаты

### 8.1 Изменение распределения сигналов

| Класс | До рефакторинга | После рефакторинга |
|-------|-----------------|-------------------|
| EXTREME_SPIKE | Много блокировок из-за ликвидности | Больше проходят (мягкие фильтры) |
| STRONG_SIGNAL | Стандартно | Без изменений |
| EARLY_SIGNAL | Много проходят | Меньше проходят (строгие фильтры) |

### 8.2 Ожидаемое влияние на качество

| Метрика | Ожидаемое изменение | Причина |
|---------|---------------------|---------|
| Entry rate EXTREME | ↑ Увеличится | Мягкие фильтры |
| Entry rate EARLY | ↓ Уменьшится | Строгие фильтры |
| Win rate EXTREME | → Без изменений | MR работает на любой ликвидности |
| Win rate EARLY | ↑ Улучшится | Только качественные сигналы |
| Avg PnL EARLY | ↑ Улучшится | Меньше шума |

---

## Резюме изменений

| Файл | Изменение |
|------|-----------|
| `config.yaml` | Добавить секцию `class_aware_filters` |
| `position_manager.py` | Добавить функцию `_apply_class_aware_filters` |
| `position_manager.py` | Изменить `_create_pending_signal` для использования новых фильтров |
| `position_manager.py` | Добавить вспомогательные функции если отсутствуют |
| `tests/` | Добавить тесты для class-aware фильтрации |
