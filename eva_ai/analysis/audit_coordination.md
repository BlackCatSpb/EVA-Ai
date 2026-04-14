# Отчёт: Координационная инфраструктура

**Дата аудита:** 2026-04-14  
**Файлы:** `eva_ai/core/event_bus.py`, `eva_ai/core/deferred_command_system.py`

---

## 1. Проверка импортов

### EventBus (event_bus.py)
| Импорт | Статус | Комментарий |
|--------|--------|-------------|
| `logging` | OK | Стандартный модуль |
| `time`, `threading`, `weakref` | OK | Стандартные модули |
| `typing` | OK | Стандартный модуль |
| `enum.Enum` | OK | Стандартный модуль |
| `dataclasses.dataclass` | OK | Стандартный модуль |
| `collections.defaultdict, deque` | OK | Стандартные модули |
| `queue` | OK | Стандартный модуль |

**Вывод:** Все импорты корректны.

### DeferredCommandSystem (deferred_command_system.py)
| Импорт | Статус | Комментарий |
|--------|--------|-------------|
| `time`, `logging`, `threading`, `queue` | OK | Стандартные модули |
| `typing` | OK | Стандартный модуль |
| `enum.Enum` | OK | Стандартный модуль |
| `dataclasses.dataclass` | OK | Стандартный модуль |
| `concurrent.futures.ThreadPoolExecutor` | OK | Стандартный модуль |
| `eva_ai.core.event_bus` | OK | Локальный импорт |

**Вывод:** Все импорты корректны.

---

## 2. Соответствие документации

### Согласно AGENTS.md:

| Требование | EventBus | DeferredCommandSystem |
|------------|----------|----------------------|
| `publish()` | ✅ Есть (строка 246) | N/A |
| `subscribe()` | ✅ Есть (строка 151) | N/A |
| `add_command()` | N/A | ✅ Есть (строка 128) |
| `execute()` | N/A | ✅ `_execute_command()` (строка 239) |
| Приоритеты LOW/NORMAL/HIGH/CRITICAL | ⚠️ Частично (см.проблемы) | ✅ Полностью (строки 32-39) |
| Pub/Sub паттерн | ✅ Реализован | N/A |
| Load Shedding | ❌ Нет | ✅ Есть (строки 117-124, 571-643) |
| Восстановление после сбоев | ❌ Нет | ⚠️ Частично (см.проблемы) |

---

## 3. Детальный анализ EventBus

### 3.1 Архитектура
```
EventBus
├── _subscribers: Dict[str, List[(subscription_id, weak_handler)]]
├── _event_history: deque(maxlen=max_history)
├── _event_queue: queue.Queue()
├── _worker_thread: threading.Thread
└── _lock: threading.RLock()
```

### 3.2 Приоритизация событий

**Документация утверждает:** 4 уровня приоритета (LOW=1, NORMAL=2, HIGH=3, CRITICAL=4)

**Реальность:**
```python
# EventPriority enum - OK
class EventPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

# Subscribe имеет параметр priority, но ОН НИГДЕ НЕ ИСПОЛЬЗУЕТСЯ!
def subscribe(self, event_type: str, handler: Callable, priority: int = 5) -> str:
    # priority принимается, записывается в лог, но...
    # ...подписчик добавляется в список БЕЗ учёта приоритета
    self._subscribers[event_type].append((subscription_id, weak_handler))
```

**Критическая проблема:** Параметр `priority` в `subscribe()` полностью игнорируется. Все подписчики одного события вызываются в порядке добавления, без учёта приоритета.

### 3.3 Event.priority тоже не используется
```python
# Event имеет поле priority
@dataclass
class Event:
    event_type: str
    source: str
    data: Dict[str, Any]
    timestamp: float = 0
    priority: EventPriority = EventPriority.NORMAL  # Устанавливается...

# НО в _process_event() приоритет НЕ проверяется!
def _process_event(self, event: Event) -> int:
    # ...нет кода вида:
    # if event.priority == EventPriority.CRITICAL:
    #     process_first()
```

### 3.4 Обработка событий
- **Асинхронный режим:** `publish()` → `_event_queue` → worker thread → `_process_event()`
- **Синхронный режим:** `publish_sync()` → немедленный вызов `_process_event()`
- **Очистка мёртвых ссылок:** выполняется внутри `_process_event()` при каждом вызове
- **Weakref для handler'ов:** корректно используется для автоматической очистки

### 3.5 Нет Load Shedding
EventBus не имеет механизма сброса нагрузки при перегрузке. Если события поступают быстрее, чем обрабатываются, очередь будет расти неограниченно.

### 3.6 Нет автоматического перезапуска
Если обработчик падает с ошибкой, EventBus ловит исключение, логирует и продолжает работу. Нет механизма:
- Retry для упавших обработчиков
- Circuit breaker
- Fallback обработчиков

---

## 4. Детальный анализ DeferredCommandSystem

### 4.1 Архитектура
```
DeferredCommandSystem
├── command_queues: Dict[Priority, queue.PriorityQueue]
├── commands: Dict[command_id, DeferredCommand]
├── executor: ThreadPoolExecutor(max_workers)
├── monitor_thread: threading.Thread (_monitor_modules)
├── load_monitor_thread: threading.Thread (_monitor_load_shedding)
└── _ls_callbacks: List[Dict] (load shedding)
```

### 4.2 Приоритизация команд - РЕАЛЬНО РАБОТАЕТ
```python
# Приоритеты определены корректно
class CommandPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    MEDIUM = NORMAL  # Алиас
    LOW = 3

# При добавлении используется отрицательное значение для правильной сортировки
self.command_queues[priority].put((-priority.value, time.time(), command_id))

# При обработке - строго по порядку приоритетов
for priority in [CommandPriority.CRITICAL, CommandPriority.HIGH, 
               CommandPriority.NORMAL, CommandPriority.LOW]:
```

**Вывод:** Приоритизация команд реализована корректно.

### 4.3 Load Shedding - РЕАЛЬНО РАБОТАЕТ

**Встроенные коллбеки:**

1. **CPU High Load Shedding** (строки 430-460)
   - Условие: `cpu_usage > 0.8` (80%)
   - Действие: сброс всех LOW приоритет команд
   - Cooldown: 30 секунд

2. **Queue Overflow Shedding** (строки 463-502)
   - Условие: `total_commands > 100`
   - Действие: сброс половины LOW приоритет команд
   - Cooldown: 15 секунд

**API для регистрации:**
```python
def register_load_shed_callback(
    self,
    condition: Callable[[], bool],
    action: Callable[[], None],
    name: Optional[str] = None,
    cooldown_sec: float = 10.0,
    priority: CommandPriority = CommandPriority.HIGH,
) -> str:
```

**Работает в фоновом потоке `_monitor_load_shedding()`** с интервалом 2 секунды.

### 4.4 Восстановление после сбоев

**Retry механизм:**
```python
# _execute_command() при неудаче
if cmd.attempts < cmd.max_retries and not self._shutting_down:
    cmd.status = CommandStatus.RETRYING
    # Планируем повтор через _schedule_retry()
    retry_thread.start()
```

**Мониторинг модулей:**
```python
# _monitor_modules() проверяет каждые 30 секунд
for module_name, health_check in self.module_health_checks.items():
    if not health_check():
        self._recover_module(module_name)
```

**Проблемы:**

1. **Health check нужно регистрировать вручную** - нет автообнаружения
2. **Recovery добавляет команду с приоритетом HIGH**, но нет защиты от спама
3. **Нет max recovery attempts** - если модуль не восстанавливается, будет вечный цикл

### 4.5 Интеграция EventBus ↔ DeferredCommandSystem

**Связь через глобальные функции:**
```python
# deferred_command_system.py
_global_event_bus = None
_event_bus_lock = __import__('threading').Lock()

def set_event_bus(event_bus):
    global _global_event_bus
    with _event_bus_lock:
        _global_event_bus = event_bus

# В core_brain.py при инициализации:
if hasattr(self, '_new_event_bus') and self._new_event_bus:
    set_event_bus(self._new_event_bus)
```

**Публикация событий команд:**
```python
# _publish_command_event() публикует:
# - command.completed
# - command.failed
event = Event(
    event_type=event_type,
    source="deferred_command_system",
    data={...}
)
eb.publish(event)
```

**Проблема:** Глобальный синглтон `_global_event_bus` может рассинхронизироваться с реальным EventBus в core_brain.

---

## 5. Проблемы

### Критические проблемы

| # | Проблема | Файл | Строки | Влияние |
|---|---------|------|--------|---------|
| 1 | **Priority в subscribe() не используется** | event_bus.py | 151-188 | Подписчики вызываются не по приоритету |
| 2 | **Event.priority игнорируется** | event_bus.py | 314-405 | События обрабатываются FIFO, а не по приоритету |
| 3 | **Нет Load Shedding в EventBus** | event_bus.py | - | Очередь событий может расти бесконечно |

### Существенные проблемы

| # | Проблема | Файл | Строки | Влияние |
|---|---------|------|--------|---------|
| 4 | **Глобальный синглтон EventBus ненадёжен** | deferred_command_system.py | 18-30 | Возможна рассинхронизация |
| 5 | **Нет защиты от спама восстановления** | deferred_command_system.py | 364-380 | Бесконечный цикл при сбое модуля |
| 6 | **Health check нужно регистрировать вручную** | deferred_command_system.py | 345-362 | Мониторинг не работает без явной регистрации |
| 7 | **Retry блокирует поток** | deferred_command_system.py | 313-318 | `time.sleep()` в отдельном потоке, но создаёт новый |

### Менее существенные

| # | Проблема | Файл | Строки |
|---|---------|------|--------|
| 8 | MEDIUM = NORMAL избыточен | deferred_command_system.py | 38 |
| 9 | Очистка мёртвых подписчиков только при обработке | event_bus.py | 337-351 |
| 10 | Нет метода получить список pending retries | deferred_command_system.py | - |

---

## 6. Оценка

### EventBus: 6/10

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Функциональность | 7/10 | Базовый pub/sub работает |
| Приоритизация | 3/10 | Приоритеты объявлены, но не работают |
| Надёжность | 6/10 | Логирование ошибок, weakref |
| Load Shedding | 0/10 | Отсутствует |
| Масштабируемость | 5/10 | Очередь без ограничений |

### DeferredCommandSystem: 7/10

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Функциональность | 8/10 | Команды с приоритетами работают |
| Приоритизация | 9/10 | Реализовано корректно |
| Надёжность | 6/10 | Retry работает, но нет защиты от спама |
| Load Shedding | 8/10 | Два встроенных механизма + API |
| Масштабируемость | 7/10 | ThreadPoolExecutor с max_workers |

### Общая интеграция: 6/10

EventBus и DeferredCommandSystem связаны через глобальный синглтон, что создаёт риски. Нет общей политики приоритизации - каждая система работает по-своему.

---

## Рекомендации

### Для EventBus:
1. Реализовать реальное использование `priority` в `subscribe()` и `_process_event()`
2. Добавить механизм Load Shedding (макс. размер очереди, таймауты)
3. Рассмотреть возможность автоматического retry/callback при сбоях обработчиков

### Для DeferredCommandSystem:
1. Заменить глобальный синглтон на внедрение зависимостей (dependency injection)
2. Добавить `max_recovery_attempts` и `recovery_cooldown` для защиты от спама
3. Добавить автообнаружение модулей для мониторинга (scan brain.components)

### Для интеграции:
1. Создать общий `PriorityPolicy` для обеих систем
2. Рассмотреть единую точку входа для load shedding
