# System Health & Runtime Audit

**Дата:** 2026-04-14  
**Файлы:** va_ai/system/health_monitor.py, va_ai/system/fault_tolerance.py, va_ai/system/system_types.py, va_ai/runtime/simple_model.py, va_ai/runtime/worker_pool.py

---

## 1. Health Monitor

### Архитектура
**HealthMonitor** (system/health_monitor.py, 409 строк) — централизованный агрегатор здоровья системы EVA.

### Метрики
Собирает данные о 7 компонентах с весовой системой:

| Компонент | Вес | Источник |
|-----------|-----|----------|
| ml_core | 0.30 | rain.ml_core.get_system_health() |
| knowledge_graph | 0.20 | rain.knowledge_graph.get_system_health() |
| memory_manager | 0.15 | rain.memory_manager.get_system_health() |
| adaptation_manager | 0.10 | rain.adaptation_manager.get_system_health() |
| ethical_framework | 0.10 | rain.ethical_framework.get_system_health() |
| neuromorphic_simulator | 0.10 | rain.neuromorphic_simulator.get_system_health() |
| web_search_engine | 0.05 | rain.web_search_engine.get_system_health() |

### Статусы
- health_score > 0.7 → **healthy**
- health_score > 0.4 → **warning**
- health_score <= 0.4 → **critical**

### Функциональность
1. **Синхронный анализ** — вызывается по требованию, непрерывного цикла нет
2. **Анализ эволюции** — хранит историю в SQLite, вычисляет тренды (increasing/decreasing/stable)
3. **Learning Opportunities** — при критических проблемах добавляет opportunities через nalyzer_core
4. **Рекомендации** — статические текстовые рекомендации на основе статуса компонентов

### Проблемы
- **Нет EventBus интеграции** — не публикует события, не подписывается на изменения
- **Нет непрерывного мониторинга** — только ручные вызовы nalyze_system_health()
- **Слабая аналитика** — анализ трендов примитивный (линейная разница start/end)
- **Жесткая связь** — зависит от конкретных атрибутов brain (hasattr проверки)

---

## 2. Fault Tolerance

### Архитектура
**FaultTolerance** (system/fault_tolerance.py, 92 строки) — минималистичная заглушка.

### Реализация
`python
fault_handlers: Dict[str, Callable]      # Регистрация обработчиков
recovery_strategies: Dict[str, Callable] # Стратегии восстановления (пустой словарь!)
fault_history: List[Dict[str, Any]]      # История ошибок
`

### Методы
- egister_fault_handler() — регистрация обработчика (только логирование)
- handle_fault() — добавляет в историю, вызывает обработчик
- get_system_health() — считает health_score как 100 - штрафы за recent_faults

### Health Score
`python
if len(recent_faults) > 10: health_score -= 30
elif len(recent_faults) > 5: health_score -= 15
# recent_faults = faults за последний час
`

### Критические проблемы
1. **Нет восстановления** — ecovery_strategies словарь всегда пуст, методы для восстановления отсутствуют
2. **Тривиальная логика** — просто считает количество ошибок, не анализирует типы
3. **Нет EventBus** — не публикует события об ошибках
4. **Нет автоматического реагирования** — только ручная регистрация обработчиков
5. **Не интегрирован с brain** — есть ссылка self.brain, но не используется

---

## 3. Runtime (Worker Pool)

### InferenceWorkerPool (untime/worker_pool.py, 195 строк)

#### Архитектура
Multiprocessing-based пул воркеров для параллельного инференса.

#### Конфигурация
`python
model_fn_path: str           # Путь к model_fn (torch_adapter)
num_workers: int             # mp.cpu_count() // 2
torch_threads: int = 2       # OMP_NUM_THREADS
interop_threads: int = 1     # MKL_NUM_THREADS
queue_maxsize: int = 64      # Размер очереди
start_method: str = "spawn"   # Контекст multiprocessing
_events: Any = None          # Optional EventBus для телеметрии
`

#### Worker Entry (_worker_entry)
1. Настраивает torch threading (NUM_THREADS, no_grad)
2. Резолвит device и precision локально
3. Ожидает atch_id, batch из очереди
4. Вызывает model_fn(batch) в autocast_context
5. Кладёт результат в out_q или ошибку

#### Ключевые методы
- start() — создаёт mp.Process воркеры (daemon=True)
- stop() — шлёт None в in_q, join(5s)
- submit(batch) — добавляет batch в очередь, возвращает batch_id
- ecv() — читает из out_q
- infer_batches() — submit pipeline + collect

#### EventBus интеграция (minimal)
`python
self._events = events  # Сохраняется но НЕ используется в коде!
`
Есть mit_wrapper_event() вызов (line 164), но _events не передаётся в submit().

### simple_model (untime/simple_model.py, 43 строки)

**Пример-заглушка**, демонстрирующая сигнатуру:
`python
def example_model_fn(batch: Batch) -> Dict[str, Any]:
    # Если input_ids — sum по dim=1
    # Иначе — fallback zeros
    return {"logits": logits, "meta_idx": meta_idx}
`

**Проблема:** Это только пример, не реальная модель. Не используется в системе.

---

## 4. Integration Issues

### 4.1 Дублирование функциональности

| Функция | system/health_monitor.py | monitoring/system_monitor.py |
|---------|---------------------------|-------------------------------|
| Сбор метрик | Частично | Да (psutil) |
| Health checks | Компоненты EVA | system_resources, python_process |
| Alerts | Нет | Да (AlertManager) |
| Хранение истории | SQLite | В памяти (list) |
| Фоновый поток | Нет | Да (threading) |

**Вывод:** monitoring/system_monitor.py более развит, но не используется компонентами EVA.

### 4.2 EventBus Интеграция

| Модуль | EventBus |
|--------|----------|
| HealthMonitor | **Отсутствует** — не публикует, не подписывается |
| FaultTolerance | **Отсутствует** |
| SystemMonitor | **Отсутствует** (495 строк, но нет event_bus) |
| InferenceWorkerPool | **Зачаточная** — _events сохраняется, но не используется |

**Проблема:** Все системы мониторинга изолированы от EventBus, хотя другие компоненты (dialog_core, model_access_manager, deferred_command_system) активно его используют.

### 4.3 Инициализация

**monitoring/system_monitor.py** (строка 495):
`python
_auto_start_monitoring()  # Запускается при импорте!
`
Автоматический запуск мониторинга при импорте — потенциальная проблема:
- Создаёт фоновый поток
- Использует psutil
- Может конфликтовать с HealthMonitor

### 4.4 Связь с SelfAnalyzer

В learning/self_analyzer.py:
`python
if HealthMonitor is not None:
    self.health_monitor = HealthMonitor(brain, self.analyzer_core)
`
HealthMonitor — опциональный компонент, подключается черезbrain.

---

## 5. Overall Assessment

### Плюсы
1. **WorkerPool** — хорошо спроектирован для multiprocess inference
2. **SystemMonitor** — полноценный мониторинг ресурсов с алертами
3. **HealthMonitor** — взвешенная агрегация компонентов

### Критические проблемы
1. **Фрагментация мониторинга** — 3 разных системы (HealthMonitor, FaultTolerance, SystemMonitor), нет координации
2. **Слабая отказоустойчивость** — FaultTolerance по сути заглушка
3. **Изоляция от EventBus** —错过了 единую точку координации
4. **Дублирование** — SystemMonitor и HealthMonitor пересекаются
5. **Ручной запуск HealthMonitor** — нет автоматического фонового мониторинга

### Рекомендации
1. **Интегрировать HealthMonitor с EventBus** — публиковать health.status_changed, подписываться на system.state_changed
2. **Расширить FaultTolerance** — добавить реальные recovery_strategies, автоматическое восстановление
3. **Убрать дублирование** — SystemMonitor или HealthMonitor должен быть основным
4. **Автоматизировать HealthMonitor** — добавить фоновый поток как в SystemMonitor
5. **Использовать _events в WorkerPool** — передавать EventBus для телеметрии

### Архитектурный риск: MEDIUM
Система мониторинга и runtime изолированы от основной координации через EventBus. При росте нагрузки возможны проблемы с обнаружением и реакцией на сбои.

---

*Аудит проведён: 2026-04-14*
