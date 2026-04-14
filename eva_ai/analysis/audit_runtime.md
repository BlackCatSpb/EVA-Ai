# Runtime Subsystem Audit

**Дата:** 2026-04-14  
**Аудитор:** EVA AI Analysis System  
**Директория:** C:\Users\black\OneDrive\Desktop\CogniFlex\eva_ai\runtime\  
**Вердикт:** 3/10 — КРИТИЧЕСКИ НЕИСПОЛЬЗУЕТСЯ

---

## 1. Обзор Runtime Подсистемы

### 1.1 Файлы в eva_ai/runtime/

| Файл | Строк | Описание | Статус |
|------|-------|----------|--------|
| worker_pool.py | 195 | InferenceWorkerPool — multiprocessing пул воркеров | **НЕ ИСПОЛЬЗУЕТСЯ** |
| simple_model.py | 43 | xample_model_fn — пример-заглушка модели | **НЕ ИСПОЛЬЗУЕТСЯ** |

### 1.2 Связанные файлы (поиск "runtime")

| Файл | Описание |
|------|----------|
| va_ai/mlearning/hot_deployment/onnx_runtime.py | **Отдельная подсистема** — ONNX Runtime инференс (395 строк) |
| va_ai/mlearning/hot_deployment/openvino_convert.py | Конвертация моделей в OpenVINO (154 строки) |
| va_ai/analysis/audit_system_runtime.md | Предыдущий аудит системы |

**ВНИМАНИЕ:** onnx_runtime.py — это НЕ часть va_ai/runtime/. Это отдельная подсистема в mlearning/hot_deployment/.

---

## 2. Детальный анализ файлов

### 2.1 worker_pool.py (195 строк)

#### Назначение
Multiprocessing-based пул воркеров для параллельного инференса моделей.

#### Архитектура
\\\python
class InferenceWorkerPool:
    model_fn_path: str        # Путь к model_fn (torch_adapter)
    num_workers: int          # mp.cpu_count() // 2
    torch_threads: int = 2    # OMP_NUM_THREADS
    interop_threads: int = 1  # MKL_NUM_THREADS
    queue_maxsize: int = 64   # Размер очереди
    _events: Any = None       # Optional EventBus для телеметрии
\\\

#### Ключевые методы
| Метод | Описание |
|-------|----------|
| start() | Создаёт daemon mp.Process воркеры |
| stop() | Шлёт None в in_q, join(5s) |
| submit(batch) | Добавляет batch в очередь |
| ecv() | Читает результаты из out_q |
| infer_batches() | Pipeline submit + collect |

#### Worker Entry (_worker_entry)
1. Настраивает torch threading (NUM_THREADS, no_grad)
2. Резолвит device и precision локально
3. Ожидает batch_id, batch из очереди
4. Вызывает model_fn(batch) в autocast_context
5. Кладёт результат в out_q или ошибку

---

### 2.2 simple_model.py (43 строк)

#### Назначение
Пример-заглушка, демонстрирующая ожидаемую сигнатуру model_fn.

\\\python
def example_model_fn(batch: Batch) -> Dict[str, Any]:
    # Если input_ids — sum по dim=1
    # Иначе — fallback zeros
    return {"logits": logits, "meta_idx": meta_idx}
\\\

**Вывод:** Это ТОЛЬКО пример, не реальная модель.

---

## 3. EventBus Integration Analysis

### 3.1 Текущее состояние

| Компонент | EventBus | Использование |
|-----------|----------|---------------|
| worker_pool._events | ❌ НЕ ИСПОЛЬЗУЕТСЯ | Параметр есть, но не передаётся в submit() |
| emit_wrapper_event() | ✅ Вызывается | Но НЕ через EventBus напрямую |

### 3.2 Анализ emit_wrapper_event()

\\\python
# batch_wrapper.py:116-143
def emit_wrapper_event(event_type, meta, *, brain=None, events=None, extra=None):
    ev = events or getattr(brain, "events", None)
    if ev is None:
        return
    payload = {"kind": "batch_wrapper", "event": str(event_type), "meta": asdict(meta)}
    if extra:
        payload["extra"] = extra
    ev.trigger("metrics", payload)  # Это EventSystem, НЕ EventBus!
\\\

**Проблема:** mit_wrapper_event вызывает v.trigger("metrics", payload) — это метод EventSystem, а не EventBus. Это разные системы!

### 3.3 EventSystem vs EventBus

| Характеристика | EventSystem | EventBus |
|----------------|-------------|----------|
| Расположение | core/event_system.py | core/event_bus.py |
| Интерфейс | .trigger() | .publish() |
| Используется в | batch_wrapper, worker_pool | ConceptMiner, ContradictionMiner, SelfDialogLearning |
| Приоритеты | Нет | Да (CRITICAL > HIGH > NORMAL > LOW) |

**Вывод:** worker_pool НЕ интегрирован с EventBus. Он использует EventSystem для метрик, но это изолированная телеметрия.

---

## 4. Duplicates Analysis

### 4.1 Дублирование функциональности

| Функция | eva_ai/runtime/ | mlearning/hot_deployment/ | Система |
|---------|------------------|---------------------------|---------|
| Multiprocessing пул | ✅ worker_pool.py | ❌ Нет | **ДУБЛЬ 1** |
| Model execution | ✅ simple_model.py | ❌ Нет | **ДУБЛЬ 2** |
| Runtime оптимизация | ❌ Нет | ✅ onnx_runtime.py | Нет |

### 4.2 Дублирование Batch

\\\
eva_ai/adapters/torch_adapter.py:30 — class Batch
eva_ai/core/batch_wrapper.py:32 — class BatchEnvelope
eva_ai/nlp_fallbacks.py:259 — class BatchProcessor
\\\

**Это НЕ дубликаты** — разные абстракции для разных целей.

### 4.3 SimpleModel дублирование

В openvino_convert.py:123 есть локальный класс:
\\\python
class SimpleModel(torch.nn.Module):
    def __init__(self, base_model):
        self.model = base_model
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
\\\

**Это НЕ связан с va_ai/runtime/simple_model.py** — разные классы с разным назначением.

---

## 5. Usage Analysis

### 5.1 Поиск импортов

\\\ash
# Результаты grep "from eva_ai.runtime" — НЕТ СОВПАДЕНИЙ
# Результаты grep "eva_ai.runtime" — НЕТ СОВПАДЕНИЙ
\\\

**Вывод:** Ни один файл в проекте НЕ импортирует va_ai/runtime/.

### 5.2 Поиск использования InferenceWorkerPool

\\\ash
grep "InferenceWorkerPool" — только self-reference в worker_pool.py:102
grep "worker_pool" — только self-reference и аналитические файлы
\\\

**Вывод:** InferenceWorkerPool НЕ используется в системе.

### 5.3 Поиск использования example_model_fn

\\\ash
grep "example_model_fn" — только в simple_model.py и аналитических файлах
\\\

**Вывод:** xample_model_fn НЕ используется в системе.

### 5.4 Кто использует Batch из torch_adapter

| Файл | Использование |
|------|---------------|
| untime/worker_pool.py | Импортирует, но сам не используется |
| untime/simple_model.py | Импортирует, но сам не используется |
| core/batch_wrapper.py | Работает с BatchEnvelope |
| dapters/torch_adapter.py | Реализация |

---

## 6. System Integration Map

### 6.1 Текущая архитектура использования

\\\
                    ┌─────────────────────────────────────┐
                    │         CoreBrain                   │
                    │  (init_factories.py)                 │
                    └──────────┬──────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
    ┌────▼─────┐    ┌──────────▼───────┐    ┌──────▼─────┐
    │SelfDialog│    │  ConceptMiner   │    │Contradiction│
    │Learning  │    │  (knowledge/)    │    │Miner        │
    └────┬─────┘    └──────────┬───────┘    └──────┬─────┘
         │                     │                  │
         └─────────────────────┼──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │     EventBus         │
                    │  (core/event_bus.py) │
                    └──────────────────────┘
                               ▲
                               │
                    ┌──────────┴──────────┐
                    │  DeferredCommand   │
                    │     System         │
                    └───────────────────┘

    ┌──────────────────────────────────────────────────────┐
    │              eva_ai/runtime/                         │
    │   (НЕ ИНТЕГРИРОВАН В СИСТЕМУ)                        │
    │   ┌─────────────────┐  ┌────────────────┐          │
    │   │  worker_pool.py │  │ simple_model.py │          │
    │   │ InferenceWorker │  │ example_model_fn│          │
    │   │      Pool       │  │                 │          │
    │   └─────────────────┘  └─────────────────┘          │
    └──────────────────────────────────────────────────────┘
\\\

### 6.2 Связи worker_pool

\\\
worker_pool.py imports:
├── eva_ai.adapters.torch_adapter (Batch) ✓
├── eva_ai.core.device_resolver (DeviceConfig, resolve_device, etc.) ✓
├── eva_ai.core.batch_wrapper (assert_clean_batch, emit_wrapper_event) ✓
└── multiprocessing, torch — стандартные библиотеки

worker_pool.py USES:
├── mp.Queue, mp.Process — multiprocessing
├── torch.set_num_threads — PyTorch threading
└── emit_wrapper_event — телеметрия (НЕ EventBus!)
\\\

---

## 7. Проблемы и Риски

### 7.1 Критические проблемы

| Проблема | Серьёзность | Описание |
|----------|-------------|----------|
| **Не используется** | 🔴 КРИТИЧЕСКАЯ | worker_pool не импортируется нигде в системе |
| **Мёртвый код** | 🔴 КРИТИЧЕСКАЯ | simple_model.py — только пример без использования |
| **Нет EventBus** | 🟠 ВЫСОКАЯ | worker_pool._events не используется |
| **Изоляция** | 🟠 ВЫСОКАЯ | Подсистема полностью изолирована |
| **Дублирование концепции** | 🟡 СРЕДНЯЯ | Multiprocessing пул изолирован от DeferredCommandSystem |

### 7.2 Потенциальные области применения

Если бы worker_pool использовался, он мог бы:
1. **Параллельный инференс** — несколько моделей одновременно
2. **Бэкенд-agnostic execution** — унифицированный интерфейс
3. **Нагрузочное тестирование** — stress testing моделей

### 7.3 Почему не используется?

Вероятные причины:
1. **Альтернативная архитектура** — SelfDialogLearning и ConceptMiner используют ThreadPoolExecutor/DeferredCommandSystem вместо multiprocessing
2. **Сложность** — multiprocessing с torch имеет проблемы (KMP_DUPLICATE_LIB_OK и т.д.)
3. **Не завершена** — подсистема была написана, но не интегрирована

---

## 8. Сравнение с Альтернативами

### 8.1 WorkerPool vs DeferredCommandSystem

| Характеристика | InferenceWorkerPool | DeferredCommandSystem |
|----------------|--------------------|-----------------------|
| Параллелизм | Multiprocessing | ThreadPoolExecutor |
| Модель доступа | model_fn_path | ModelAccessManager |
| Приоритеты | Нет | CRITICAL > HIGH > NORMAL > LOW |
| EventBus | Нет (EventSystem) | Да |
| Load Shedding | Нет | Да |
| Recovery | Нет | Автоматический перезапуск |
| Статус | **НЕ ИСПОЛЬЗУЕТСЯ** | **АКТИВНО ИСПОЛЬЗУЕТСЯ** |

### 8.2 Вывод

DeferredCommandSystem делает то же что InferenceWorkerPool, но лучше:
- Интегрирован с EventBus
- Поддерживает приоритеты
- Имеет load shedding
- Автоматически восстанавливается после сбоев

---

## 9. onnx_runtime.py — Отдельная подсистема

### 9.1 Статус

va_ai/mlearning/hot_deployment/onnx_runtime.py — это **НЕ часть** va_ai/runtime/.

### 9.2 Анализ

| Характеристика | onnx_runtime.py |
|----------------|-----------------|
| Строк | 395 |
| Класс | OnnxRuntimeGenerator, OnnxConverter |
| Функции | test_onnx_conversion(), test_existing_onnx() |
| EventBus | Нет |
| Использование | Только если установлен onnxruntime |
| Статус | **ГОТОВ К ИСПОЛЬЗОВАНИЮ** |

---

## 10. Рекомендации

### 10.1 Immediate Actions

| Действие | Приоритет | Описание |
|----------|-----------|----------|
| **Удалить или интегрировать** | 🔴 ВЫСОКИЙ | Либо удалить мёртвый код, либо интегрировать |
| **Document intended use** | 🟠 СРЕДНИЙ | Если должен использоваться — написать документацию |
| **Убрать false dependencies** | 🟠 СРЕДНИЙ | Если не используется — убрать импорты Batch |

### 10.2 Опция 1: Удаление

\\\ash
# Удалить eva_ai/runtime/
rm -rf eva_ai/runtime/
\\\

**Плюсы:** Убирает мёртвый код, уменьшает сложность  
**Минусы:** Если понадобится — нужно будет переписывать

### 10.3 Опция 2: Интеграция

1. Интегрировать worker_pool с ModelAccessManager
2. Добавить EventBus публикацию событий
3. Использовать DeferredCommandSystem для координации
4. Добавить в brain_components.py

**Плюсы:** Потенциально полезная функциональность  
**Минусы:** Требует значительной переработки

---

## 11. Итоговая Оценка

### 11.1 Критерии оценки

| Критерий | Оценка (1-10) | Комментарий |
|----------|---------------|-------------|
| **Функциональность** | 7/10 | Код качественный, но не используется |
| **Интеграция** | 1/10 | Нет EventBus, нет импортов |
| **Покрытие тестами** | 0/10 | Нет тестов |
| **Документация** | 2/10 | Минимальные комментарии |
| **Актуальность** | 2/10 | Создан 07.04.2026, не обновлялся |
| **Использование** | 0/10 | НЕ ИСПОЛЬЗУЕТСЯ НИГДЕ |

### 11.2 Общая оценка

**3/10 — КРИТИЧЕСКИ НЕИСПОЛЬЗУЕТСЯ**

### 11.3 Breakdown

| Категория | Оценка |
|-----------|--------|
| Архитектура | 6/10 |
| Код | 7/10 |
| Интеграция | 1/10 |
| Полезность | 2/10 |
| **ИТОГО** | **3/10** |

---

## 12. Резюме

### Что хорошо
1. Качественный код с type hints
2. Хорошая обработка ошибок
3. Clean separation of concerns
4. Демонстрация multiprocessing patterns

### Что плохо
1. **НЕ ИСПОЛЬЗУЕТСЯ** — главная проблема
2. Нет EventBus интеграции
3. Нет тестов
4. Изолирован от остальной системы
5. Конкурирует с DeferredCommandSystem

### Вердикт
**eva_ai/runtime/ — мёртвый код**, который мог бы быть полезен, но не интегрирован в систему. Рекомендуется либо удалить, либо серьёзно переработать для интеграции.

---

*Аудит проведён: 2026-04-14*
