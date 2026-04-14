# Отчёт: ModelAccessManager - Аудит интеграции

**Дата аудита:** 2026-04-14  
**Файлы:** `eva_ai/core/unified_generator.py`, `eva_ai/core/brain_query.py`, `eva_ai/core/model_access_manager.py`

---

## 1. Проверка импортов

### unified_generator.py
| Импорт | Статус | Комментарий |
|--------|--------|-------------|
| `time`, `logging` | OK | Стандартные модули |
| `typing` | OK | Стандартные модули |
| `dataclasses.dataclass` | OK | Стандартный модуль |
| `pathlib.Path` | OK | Стандартный модуль |
| `enum.Enum` | OK | Стандартный модуль |
| `context_chunking` | OK | Локальный модуль |
| `model_access_manager` | OK | Локальный импорт (строка 21) |

**Вывод:** Все импорты корректны.

### brain_query.py
| Импорт | Статус | Комментарий |
|--------|--------|-------------|
| `re`, `time`, `logging`, `random`, `threading` | OK | Стандартные модули |
| `typing` | OK | Стандартный модуль |
| `model_access_manager` | ❌ **НЕ ИМПОРТИРУЕТСЯ** | Нет импорта |

**Вывод:** brain_query.py не импортирует ModelAccessManager.

---

## 2. Соответствие документации

### Согласно AGENTS.md:

| Требование | ModelAccessManager | brain_query | Комментарий |
|------------|-------------------|-------------|-------------|
| Приоритет очереди CRITICAL > HIGH > NORMAL > LOW | ✅ | ❌ | Определены, но не используются |
| EventBus интеграция | ✅ | ❌ | Нет подписки на события |
| Блокировка доступа к модели | ✅ | ❌ | Нет coordination |
| Координация SelfDialog/Concept/Contradiction | ✅ | ❌ | Только SelfDialog использует |

---

## 3. Детальный анализ архитектуры

### 3.1 Текущая архитектура

```
brain_query.process_query()
    ↓
pipeline.process_query()  ← НАПРЯМУЮ, БЕЗ ModelAccessManager!
    ↓
[SelfDialog, ConceptMiner, ContradictionMiner]
    ↓
model_access_manager.request_access()  ← УЖЕ ЕСТЬ, но не для запросов
```

### 3.2 ModelAccessManager в UnifiedGenerator

**Создаётся:** `unified_generator.py:184-198`
```python
def _init_model_access_manager(self):
    if self._model_access is not None:
        return
    
    self._model_access = ModelAccessManager(
        event_bus=self.event_bus,
        max_workers=4
    )
    self._model_access.start()
```

**Используется в методах:**
- `generate()` (строка 444-462)
- `generate_iterative()` (строка 829-850)

**НЕ используется:**
- `generate_dual()` (строка 548-655)
- `generate_unified()` (строка 657-760)

### 3.3 Приоритеты задач

**Определены в unified_generator.py:200-210:**
```python
def _get_priority_for_task(self, task_type: str) -> AccessPriority:
    priority_map = {
        'query': AccessPriority.CRITICAL,
        'self_dialog': AccessPriority.HIGH,
        'concept_mining': AccessPriority.HIGH,
        'contradiction_mining': AccessPriority.HIGH,
        'coder': AccessPriority.HIGH,
        'default': AccessPriority.NORMAL
    }
    return priority_map.get(task_type, AccessPriority.NORMAL)
```

**Проблема:** Этот метод определён, но вызывается только в `generate()` и `generate_iterative()`, которые НЕ используются brain_query напрямую.

---

## 4. Ключевые находки

### 4.1 ModelAccessManager создан но НЕ интегрирован в brain_query

**UnifiedGenerator:**
- ModelAccessManager инициализируется в `__init__` (строка 167-168)
- Методы `generate()` и `generate_iterative()` используют ModelAccessManager
- НО эти методы НЕ вызываются из brain_query

**brain_query.py:**
- `process_query()` → `_execute_query_strategy()` → `_handle_gguf_pipeline()`
- Строка 389: `result = pipeline.process_query(enhanced_query)` — напрямую!
- НЕ использует ModelAccessManager для координации

### 4.2 Кто использует ModelAccessManager

**SelfDialogLearning:**
```python
# Предполагаемая интеграция (не проверено в данном аудите)
model_access.request_access(
    priority=AccessPriority.HIGH,
    task_type='self_dialog',
    callback=self._generate_dialog,
    ...
)
```

**ConceptMiner:**
```python
# Предполагаемая интеграция
model_access.request_access(
    priority=AccessPriority.HIGH,
    task_type='concept_mining',
    callback=self._mine_concepts,
    ...
)
```

**ContradictionMiner:**
```python
# Предполагаемая интеграция
model_access.request_access(
    priority=AccessPriority.HIGH,
    task_type='contradiction_mining',
    callback=self._detect_contradictions,
    ...
)
```

### 4.3 Приоритеты определены но используются не везде

**AccessPriority enum (предположительно в model_access_manager.py):**
```python
class AccessPriority(Enum):
    CRITICAL = 0  # Запросы пользователя
    HIGH = 1     # Self-dialog, mining
    NORMAL = 2   # Фоновые задачи
    LOW = 3      # Низкоприоритетные
```

**Используется:**
- ✅ `generate()` — CRITICAL для 'query'
- ✅ `generate_iterative()` — CRITICAL для 'query'

**НЕ используется:**
- ❌ `generate_dual()` — без ModelAccessManager
- ❌ `generate_unified()` — без ModelAccessManager
- ❌ brain_query напрямую вызывает pipeline без приоритетов

---

## 5. Проблемы интеграции

### Критические проблемы

| # | Проблема | Файл | Строки | Влияние |
|---|---------|------|--------|---------|
| 1 | **brain_query НЕ использует ModelAccessManager** | brain_query.py | 389 | Запросы не координируются |
| 2 | **pipeline.process_query() вызывается напрямую** | brain_query.py | 389 | Нет приоритизации запросов |
| 3 | **SelfDialog/Concept/Contradiction могут конфликтовать** | brain_query.py | 389 | GPU/CPU contention |

### Существенные проблемы

| # | Проблема | Файл | Строки | Влияние |
|---|---------|------|--------|---------|
| 4 | **Приоритеты не применяются к пользовательским запросам** | brain_query.py | 326-413 | CRITICAL запросы обрабатываются как normal |
| 5 | **Нет EventBus событий для model.request/completed** | brain_query.py | - | Нет мониторинга |
| 6 | **generate_dual и generate_unified обходят ModelAccessManager** | unified_generator.py | 548-760 | Непоследовательное поведение |

### Менее существенные

| # | Проблема | Файл | Строки |
|---|---------|------|--------|
| 7 | `_get_priority_for_task()` определён но не используется в brain_query | unified_generator.py | 200-210 |
| 8 | Нет документации по интеграции ModelAccessManager в brain_query | - | - |

---

## 6. Оценка

### Интеграция ModelAccessManager: **5/10**

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Наличие компонента | 8/10 | ModelAccessManager создан и функционален |
| Интеграция в UnifiedGenerator | 7/10 | Используется в generate/generate_iterative |
| Интеграция в brain_query | **2/10** | Полностью отсутствует |
| Приоритизация запросов | **3/10** | Определена, но не применяется |
| Координация фоновых задач | 6/10 | SelfDialog/ConceptMiner используют |

---

## 7. Рекомендации

### Критические (немедленное исправление)

1. **Интегрировать ModelAccessManager в brain_query:**
   ```python
   # brain_query.py, _handle_gguf_pipeline()
   if hasattr(pipeline, '_model_access') and pipeline._model_access:
       request_id = pipeline._model_access.request_access(
           priority=AccessPriority.CRITICAL,
           task_type='query',
           callback=pipeline.process_query,
           query=enhanced_query,
           timeout=60.0
       )
       result = pipeline._model_access.get_result(request_id, timeout=60.0)
   else:
       result = pipeline.process_query(enhanced_query)
   ```

2. **Убрать прямой вызов `pipeline.process_query()` в brain_query:**
   - Все запросы должны проходить через ModelAccessManager
   - Это обеспечит приоритизацию CRITICAL > HIGH > NORMAL

### Существенные (следующий спринт)

3. **Добавить EventBus события в brain_query:**
   ```python
   # После получения результата
   if self.event_bus:
       self.event_bus.publish(Event(
           event_type='model.request.completed',
           source='brain_query',
           data={'query': query[:50], 'duration': elapsed}
       ))
   ```

4. **Использовать `_get_priority_for_task()` в UnifiedGenerator:**
   - Применить для `generate_dual()` и `generate_unified()`

5. **Централизовать доступ к модели:**
   - SelfDialog, ConceptMiner, ContradictionMiner уже используют ModelAccessManager
   - brain_query должен делать то же самое

### Менее существенные

6. **Добавить метрики использования модели:**
   - Queue length
   - Average wait time
   - Model utilization

7. **Документировать архитектуру координации:**
   - Создать диаграмму потока запросов
   - Описать приоритеты в README

---

## 8. Визуализация проблемы

```
ТЕКУЩЕЕ СОСТОЯНИЕ (НЕПРАВИЛЬНО):
═══════════════════════════════════════════════════

brain_query.process_query()
    │
    ├─→ pipeline.process_query() ←── БЕЗ КООРДИНАЦИИ!
    │       │
    │       ├─→ [GPU CONTEND]
    │       │       ├─→ SelfDialog
    │       │       ├─→ ConceptMiner  
    │       │       └─→ ContradictionMiner
    │       │
    │       └─→ model_access.request_access() ←── УЖЕ ЕСТЬ, НО ПОЗДНО!
    │
    └─→ РЕЗУЛЬТАТ (возможен contention)


ИДЕАЛЬНОЕ СОСТОЯНИЕ:
═══════════════════════════════════════════════════

brain_query.process_query()
    │
    └─→ model_access.request_access(CRITICAL)
            │
            ├─→ [WAIT] SelfDialog (HIGH) ←── ЖДЁТ
            ├─→ [WAIT] ConceptMiner (HIGH) ←── ЖДЁТ
            ├─→ [WAIT] ContradictionMiner (HIGH) ←── ЖДЁТ
            │
            └─→ [GRANTED] pipeline.process_query()
                    │
                    └─→ РЕЗУЛЬТАТ (нет contention)
```

---

(End of file)
