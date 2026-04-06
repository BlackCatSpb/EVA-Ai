# EVA-Ai — Архитектура и Принципы Работы

**Версия:** 2.2  
**Дата:** 2026-04-06  
**Репозиторий:** https://github.com/BlackCatSpb/EVA-Ai

---

## 1. Обзор Системы

EVA-Ai — когнитивная нейросетевая система с рекурсивным рассуждением, фрактальной памятью и самообучением. Система построена на архитектуре Three-GGUF Pipeline, где три специализированные модели работают последовательно, обеспечивая качество ответов через перекрёстную проверку и адаптивную настройку параметров.

### 1.1 Философия Системы

EVA не просто генерирует текст — она **рассуждает**. Каждый ответ проходит через:
1. **Логическую проверку** (Model A извлекает факты)
2. **Концептуальное развитие** (Model B расширяет мысль)
3. **Контроль качества** (проверка на зацикливание, язык, шаблонность)
4. **Семантический анализ** (сравнение с предыдущими попытками через эмбеддинги)
5. **Адаптацию** (корректировка параметров при провалах)

### 1.2 Ключевые особенности

- **Three-GGUF Pipeline** — три модели (3B + 3B + 1.5B Coder) работают последовательно через llama.cpp
- **Адаптивная генерация** — AdaptiveParameterController анализирует причины провалов
- **Семантический анализ застрялости** — через эмбеддер (multilingual-e5-base)
- **Фрактальная память** — иерархический граф знаний
- **Самообучение** — система создаёт самодиалоги
- **GPU-ускорение** — embedding-модель на CUDA
- **Веб-интерфейс** — Flask сервер с 30+ endpoints, SSE streaming, SPA на vanilla JS
- **EventBus** — централизованная событийная шина нового поколения (50+ типов событий)
- **DeferredCommandSystem** — система отложенных команд с приоритетами, retry, load shedding
- **GenerationTracker** — отслеживание жизненного цикла генераций с SSE-трансляцией
- **CoreBrain** — координация через 8 миксинов (State, Query, Memory, Monitoring, EventSubscription, CommandIssuer, ProcessTracker, Config)

### 1.3 Аппаратные требования

| Компонент | Требование | Назначение |
|-----------|------------|------------|
| CPU | 6+ ядер (AVX2) | GGUF инференс (llama.cpp) |
| RAM | 16 ГБ минимум | Модели 3B Q4_K_M × 2 = ~4GB, система ~6GB |
| GPU | CUDA 2+ ГБ VRAM | Embedding-модель |
| SSD | 10+ ГБ свободно | GGUF модели, кэши, граф памяти |

---

## 2. Архитектура Компонентов

### 2.1 CoreBrain — Центральный Координатор

**Файл:** `eva/core/core_brain.py`

CoreBrain — тонкий координатор, использующий **8 миксинов** через множественное наследование:

```
CoreBrain(
    ConfigMixin,           # Загрузка brain_config.json, masking secrets
    ComponentMixin,        # Жизненный цикл компонентов, управление модулями
    QueryMixin,            # Multi-level fallback chain (7+ уровней)
    MonitoringMixin,       # Health checks, метрики
    MemoryMixin,           # Операции с памятью
    StateMixin,            # State machine (INITIALIZING → READY → RUNNING → SHUTTING_DOWN)
    EventSubscriptionMixin,# 13 обработчиков событий EventBus
    CommandIssuerMixin,    # 20+ типов команд
    ProcessTrackerMixin    # Трекинг запросов и команд
)
```

**Жизненный цикл:**
1. `__init__()` — создание EventBus, EventSystem, EventBusBridge, DeferredCommandSystem
2. `initialize()` — инициализация всех компонентов, переход в READY
3. `start()` — запуск фоновых служб, переход в RUNNING
4. `stop()` — остановка, переход в SHUTTING_DOWN → OFFLINE
5. `reboot()` — полный цикл stop → reinitialize → start

**Подписки на события (13 событий):**
```python
eb.subscribe("pipeline.start", self._on_pipeline_start)
eb.subscribe("pipeline.model_a.complete", self._on_model_a_complete)
eb.subscribe("pipeline.model_b.complete", self._on_model_b_complete)
eb.subscribe("pipeline.complete", self._on_pipeline_complete)
eb.subscribe("pipeline.failed", self._on_pipeline_failed)
eb.subscribe("component.error", self._on_component_error)
eb.subscribe("component.initialized", self._on_component_ready)
eb.subscribe("system.error", self._on_system_error)
eb.subscribe("contradiction.detected", self._on_contradiction)
eb.subscribe("learning.progress", self._on_learning_progress)
eb.subscribe("learning.completed", self._on_learning_completed)
eb.subscribe("memory.warning", self._on_memory_warning)
eb.subscribe("memory.optimized", self._on_memory_optimized)
```

**Типы команд (20+):**
```
reload_pipeline, adjust_pipeline_params, flush_cache, compact_memory,
trigger_learning, resolve_contradiction, recover_component, abort_generation,
scale_resources, rebuild_knowledge, initiate_search, publish_alert,
set_timeout_limit, force_model_fallback, reset_generation_attempts,
set_max_retries, get_system_status, restart_component,
update_event_subscription, set_log_level
```

**BackgroundCoordinator** координирует фоновые задачи:
- Training jobs
- Web indexing
- Module recovery
- Self-dialog learning

### 2.2 EventBus — Событийная Шина

**Файл:** `eva/core/event_bus.py`

**Архитектура:**
- **Singleton** через `get_event_bus()` — автозапуск при первом обращении
- **Queue-based async processing** — события ставятся в очередь, обрабатываются фоновым потоком
- **Weak references** — подписчики через `weakref.WeakMethod` / `weakref.ref` для предотвращения утечек памяти
- **Dead subscriber cleanup** — автоматическое удаление собранных GC обработчиков
- **Event history** — кольцевой буфер на 10,000 событий

**50+ типов событий:**
```
system.start, system.stop, system.error, system.ready
component.initialized, component.error, component.stopped, component.started
learning.started, learning.completed, learning.failed, learning.progress
contradiction.detected, contradiction.resolved, contradiction.failed
ethics.violation, ethics.warning, ethics.assessment
analytics.insight, analytics.report, analytics.alert
knowledge.updated, knowledge.added, knowledge.deleted
feedback.processed, feedback.received
web.search.started, web.search.completed, web.search.failed
memory.cleared, memory.optimized, memory.warning
adaptation.started, adaptation.completed, adaptation.failed
pipeline.start, pipeline.model_a.start, pipeline.model_a.complete
pipeline.model_b.start, pipeline.model_b.complete, pipeline.complete, pipeline.failed
command.completed, command.failed
generation.started, generation.progress, generation.completed, generation.failed, generation.timeout
```

**EventBusBridge** (`event_bus_bridge.py`):
- Двунаправленный адаптер между старой EventSystem и новой EventBus
- Патчит `old.trigger()` → `new.publish()`
- Патчит `new.publish_sync()` → `old.trigger()`
- Маппинг 23 старых имён событий на новые (напр. `query_received` → `pipeline.start`)

### 2.3 DeferredCommandSystem — Система Отложенных Команд

**Файл:** `eva/core/deferred_command_system.py` (641 строка)

**Архитектура:**
- **4 очереди приоритетов**: CRITICAL, HIGH, NORMAL, LOW (каждая — `PriorityQueue`)
- **ThreadPoolExecutor** с настраиваемым `max_workers` (по умолчанию 6)
- **Command retry** с экспоненциальным backoff
- **Module health monitoring** (интервал 30 секунд)
- **Load shedding** (CPU > 80%, переполнение очереди > 100 команд)

**Жизненный цикл команды:**
```
PENDING → RUNNING → COMPLETED
                → FAILED → (retry) → PENDING
                → RETRYING → PENDING
```

**Load Shedding:**
- **CPU threshold**: >80% сбрасывает все LOW приоритет команды
- **Queue overflow**: >100 команд сбрасывает 50% LOW приоритет
- **Cooldown**: предотвращает повторное срабатывание (15-30 секунд)

**Мост к EventBus:**
- Публикует `command.completed` и `command.failed` события
- Двунаправленная связь с `ResourceManager` для load shedding

### 2.4 GenerationTracker — Трекинг Генераций

**Файл:** `eva/core/generation_tracker.py`

**Жизненный цикл генерации:**
```
start_generation() → update_progress() → complete() / fail() / timeout()
```

- Уникальный `command_id` для каждой генерации (формат: `gen_<12 hex chars>`)
- Публикует события: `generation.started`, `generation.progress`, `generation.completed`, `generation.failed`, `generation.timeout`
- Автоочистка завершённых записей старше 300 секунд
- Интеграция с `DeferredCommandSystem` для отслеживания таймаутов

### 2.5 RecursiveModelPipeline — Three-GGUF Pipeline

**Файл:** `eva/core/pipeline_core.py`

**3-Model GGUF Pipeline** через `llama_cpp.Llama`:

| Модель | Файл | Назначение | Параметры |
|--------|------|------------|-----------|
| **Model A** | `qwen2.5-3b-instruct-q4_k_m.gguf` | Логическое ядро — извлечение фактов, макс 3 предложения | temp=0.3, top_p=0.9, top_k=40, repeat_penalty=1.5, max_tokens=1024 |
| **Model B** | `qwen2.5-3b-instruct-q4_k_m_model_b.gguf` | Развитие концепций — расширение фактов примерами и деталями | temp=0.3, top_p=0.9, top_k=40, repeat_penalty=2.0, max_tokens=512 |
| **Model C** | `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | Генерация кода — ленивая загрузка только при запросе кода | temp=0.1, top_p=0.9, top_k=50, repeat_penalty=1.3, max_tokens=512 |

**AdaptiveParameterController** (`pipeline_adaptive.py`):
- **Semantic stuck detection** — эмбеддинги `sentence-transformers` + cosine similarity (threshold 0.85)
- **Failure-based adaptation** — корректировка параметров по причинам провалов (зацикливание, китайские символы, filler words, пустой вывод)
- **Resource-based adaptation** — снижение max_tokens при CPU > 70-85% или RAM > 90%

**Quality Checking** (`pipeline_quality.py`):
- Детекция гласных (кириллица)
- Детекция повторений (уровень слов, уровень строк)
- Детекция filler фраз
- Детекция китайских символов
- Детекция соотношения английских слов
- Санитизация ответов (смешанные кириллица/латиница, сохранение блоков кода)
- Удаление зацикленных блоков

### 2.6 Query Processing — Fallback Chain

**Файл:** `eva/core/brain_query.py` (815+ строк)

Метод `_execute_query_strategy()` диспетчеризует по режиму:

**Режим 1: `qwen_only_mode`**
1. Preprocessing pipeline (детекция уточнений, извлечение сущностей)
2. Обогащение контекстом из knowledge graph
3. Two-Model Pipeline (если активен)
4. LlamaCpp GGUF генерация
5. QwenModelManager PyTorch генерация

**Режим 2: `disable_pytorch` (GGUF mode — текущий конфиг)**
1. Two-Model Pipeline (Model A → B → C)
2. Fallback на финальную LlamaCpp генерацию

**Режим 3: Полный fallback chain (legacy)**
1. SelfReasoningEngine
2. EnhancedReasoningEngine
3. QwenModelManager (ленивая загрузка)
4. LlamaCpp финальный
5. Generation coordinator
6. Fractal model manager
7. Query processor
8. MLUnit прямой
9. Memory manager (похожие взаимодействия)
10. Basic keyword-based fallback
11. Финальное fallback сообщение

### 2.7 Память (Memory)

| Компонент | Файл | Описание |
|-----------|------|----------|
| **UnifiedFractalMemory** | `memory/unified_fractal_memory.py` | Единая фрактальная память |
| **GraphLearning** | `memory/graph_learning.py` | Обучение на графе |
| **HybridTokenCache** | `memory/hybrid_token_cache.py` | Гибридный кэш токенов |
| **EmbeddingCache** | `memory/embedding_cache.py` | SQLite LRU кэш эмбеддингов |

### 2.8 Машинное обучение (MLearning)

| Компонент | Файл | Описание |
|-----------|------|----------|
| **FractalModelManager** | `mlearning/fractal_model_manager.py` | Управление GGUF моделями |
| **QwenModelManager** | `mlearning/qwen_model_manager.py` | Управление PyTorch Qwen |
| **UnifiedTextProcessor** | `mlearning/unified_text_processor.py` | Текст + эмбеддинги |
| **MLUnit** | `mlearning/ml_unit.py` | ML ядро системы |
| **SentenceTransformersCache** | `mlearning/sentence_transformers_cache.py` | Кэш embedding-моделей |

### 2.9 Экосистема Модулей EVA

| Модуль | Назначение |
|--------|------------|
| `core/` | Центральный мозг, pipeline, event bus, отложенные команды |
| `knowledge/` | Knowledge graph, Wikipedia интеграция, онлайн знания |
| `learning/` | Самодиалоги, анализ производительности, самоанализ |
| `memory/` | Фрактальная память, кэш токенов, гибридный кэш |
| `mlearning/` | ML модели (Qwen, фрактальные, BitNet), токенизаторы |
| `websearch/` | Веб-поиск, кэширование, управление БД |
| `ethics/` | Этический фреймворк, детекция нарушений |
| `contradiction/` | Детекция противоречий, разрешение, стратегии |
| `reasoning/` | Self-reasoning engine, enhanced reasoning |
| `preprocess/` | Предобработка запросов, уточнения, сущности |
| `neuromorphic/` | Симулированные синапсы |
| `distributed/` | Кластерное управление, планирование задач |
| `security/` | Фреймворк безопасности |
| `monitoring/` | Мониторинг системы |
| `recovery/` | Система восстановления |
| `training/` | GGUF training система |
| `adaptation/` | Адаптивные системы |
| `nlp/` | Обработка текста |
| `generation/` | Координация генерации |
| `storage/` | Абстракции хранилища |

---

## 3. Веб-интерфейс

### 3.1 Архитектура

```
Browser (Vanilla JS SPA)
    ↓ HTTP/JSON + SSE
Flask Server (server_routes.py — 30+ endpoints)
    ├── Аутентификация
    ├── Сессии
    ├── Чат
    ├── Файлы
    ├── Аналитика
    ├── Обучение
    ├── Знания
    ├── Wikipedia
    ├── Настройки
    ├── Debug
    └── SSE Streaming
```

### 3.2 Все API Endpoints

#### Основные endpoints:

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/` | Статический `index.html` |
| POST | `/api/login` | Аутентификация, создание/переиспользование сессии |
| GET/POST/DELETE | `/api/sessions` | Список, создание, удаление сессий |
| GET | `/api/session/<session_id>` | Детали сессии с контекстом/сущностями |
| DELETE/PUT | `/api/session/<session_id>` | Удаление/обновление сессии |
| POST | `/api/upload` | Загрузка файлов (txt, pdf, docx, изображения с OCR, код) |
| POST | `/api/chat` | Обработка сообщения через brain |
| POST | `/api/v1/chat` | Версионированный chat endpoint |
| GET | `/api/entities/<session_id>` | Извлечённые сущности для сессии |
| DELETE | `/api/entities/<session_id>` | Удаление сущностей |
| POST | `/api/feedback` | Отзыв пользователя (рейтинг, точность, когерентность) |
| GET | `/api/status` | Статус системы (brain connected, running, кол-во компонентов) |
| GET | `/api/metrics` | CPU, memory, cache hit rate |
| GET | `/api/memory-graph` | Данные для визуализации графа памяти (nodes/edges) |
| GET/POST | `/api/self-dialog` | Получение/создание самодиалогов |

#### Аналитика и обучение:

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/api/analytics` | Полная аналитика (запросы, среднее время, success rate, диалоги, пробелы, кэш) |
| GET | `/api/learning` | Возможности обучения, статистика, недавние диалоги |
| GET/POST | `/api/settings` | Получение/обновление настроек (auto_learning, model_name, etc.) |

#### Документы и знания:

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/api/documents` | Документы для сессии |
| DELETE | `/api/documents/<file_id>` | Удаление документа |
| GET/POST | `/api/knowledge-graph` | Получение/поиск/добавление узлов knowledge graph |
| GET/POST/DELETE | `/api/knowledge` | Knowledge CRUD |
| GET | `/api/cache-stats` | Hybrid cache, search cache, memory stats |
| GET | `/api/system` | Системная информация (версия, модель, модули, фичи) |

#### Модели и генерация:

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/api/model-status` | Статус загрузки моделей |
| GET | `/api/generation-status` | Все статусы генераций |
| GET | `/api/generation-status/<command_id>` | Статус конкретной генерации |
| GET | `/api/events/stream` | **SSE streaming** для real-time событий |

#### Графы и экспорт:

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/api/fractal-graph` | Данные фрактального графа |
| POST | `/api/export` | Экспорт данных системы |
| POST | `/api/import` | Импорт данных системы |
| GET/POST | `/api/snapshots` | Системные снапшоты |

#### Wikipedia:

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET/POST | `/api/wikipedia` | Поиск/обучение Wikipedia |

#### Debug endpoints:

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/api/debug/test` | Debug: статус, время, пользователи |
| GET | `/api/debug/deferred` | Debug: состояние deferred command system |
| GET | `/api/debug/events` | Debug: история событий (последние 50) |
| GET | `/api/debug/auth` | Debug: детальный процесс auth |

### 3.3 Файлы Web GUI

| Файл | Описание | Строк |
|------|----------|-------|
| `gui/web_gui/server_main.py` | WebGUI класс, Flask app, create_app() | — |
| `gui/web_gui/server_routes.py` | 30+ API endpoints, SSE streaming | — |
| `gui/web_gui/server_auth.py` | AuthManager, SessionManager, EntityExtractor, EthicsChecker | — |
| `gui/web_gui/server_models.py` | Model status, fractal graph endpoints | — |
| `gui/web_gui/server_api_knowledge.py` | Knowledge CRUD endpoints | — |
| `gui/web_gui/server_api_wikipedia.py` | Wikipedia endpoints | — |
| `gui/web_gui/server_api_export.py` | Import/export endpoints | — |
| `gui/web_gui/server_types.py` | Dataclasses: WebSession, UserCredentials, SessionStatus | — |
| `gui/web_gui/bridge.py` | GUIBridge — интеграция с CoreBrain | — |
| `gui/web_gui/templates/index.html` | HTML шаблон (SPA) | 393 |
| `gui/web_gui/static/js/app.js` | JavaScript логика (vanilla JS, IIFE) | 1870 |
| `gui/web_gui/static/css/style.css` | Стили (dark theme, CSS custom properties) | 2287 |
| `gui/web_gui/static/js/emoji.js` | Emoji data (отключено) | 11 |

### 3.4 Frontend Архитектура

**Подход:** Single-page application (SPA) на **vanilla JavaScript**, без фреймворков.

**State Management:**
```javascript
let token = null;           // Auth token (boolean flag, не JWT)
let userId = null;          // ID текущего пользователя
let sessions = [];          // Массив объектов сессий
let activeSessionId = null; // Текущая выбранная сессия
let sidebarOpen = false;
let settingsState = { auto_learn: true, memory_enabled: true, sre_enabled: true };
```

**7 представлений (views):**
1. **Chat View** — основной интерфейс чата с markdown рендерингом
2. **Memory View** — граф знаний, статистика узлов/рёбер, документы
3. **Analytics View** — производительность, системные метрики (CPU/RAM/VRAM)
4. **Learning View** — возможности обучения, статистика, диалоги
5. **Knowledge View** — поиск по базе знаний, сущности/отношения
6. **Wikipedia View** — поиск, загрузка тем/категорий, автообучение
7. **Settings View** — аккаунт, тогглы, экспорт/импорт, статус моделей

**Markdown рендеринг:** Кастомный regex-based парсер (без библиотек):
- Заголовки (h1-h3), bold, italic, strikethrough
- Inline code и блоки кода с языковой меткой + кнопка копирования
- Списки, blockquotes, горизонтальные линии
- HTML escaping перед обработкой (XSS prevention)

**Reasoning отображение:** Collapsible panel со структурированными шагами:
- Фаза, мысль, уверенность (цветовая кодировка: зелёный/жёлтый/красный)

### 3.5 SSE (Server-Sent Events) Streaming

**Клиент (frontend):**
```javascript
const eventSource = new EventSource('/api/events/stream')
```

**Подписки на события:**
| Событие | Действие |
|---------|----------|
| `pipeline.start` | Показ UI прогресса генерации |
| `pipeline.model_a.start` | Активация шага A |
| `pipeline.model_a.complete` | Отметка шага A завершённым |
| `pipeline.model_b.start` | Активация шага B |
| `pipeline.model_b.complete` | Отметка шага B завершённым |
| `pipeline.complete` | Завершение, скрытие прогресса через 800ms |
| `pipeline.failed` | Показ ошибки, скрытие через 1500ms |
| `generation.progress` | Обновление прогресс-бара |
| `generation.started` | Запуск таймера elapsed |
| `generation.completed` | Остановка таймера |
| `generation.failed` | Остановка таймера |

**UI генерации:**
- Динамически создаваемый `div.generation-progress` в чате
- 3-шаговый pipeline: Model A → Model B → Completion
- Анимированный прогресс-бар с gradient shimmer эффектом
- Счётчик elapsed time (обновление каждые 200ms)
- Через 30 секунд — уведомление "Генерация занимает больше времени"
- Автоочистка при завершении или ошибке

**Серверная часть SSE:**
- Flask `Response(stream_with_context(...))` с `mimetype='text/event-stream'`
- Подписка на EventBus brain для pipeline/generation событий
- `queue.Queue` для thread-safe форвардинга событий
- Heartbeat каждую 1 секунду (`: heartbeat\n\n`)
- Очистка подписок при `GeneratorExit`

**Reconnection:** При `onerror`, если не CLOSED, переподключение через 5 секунд.

### 3.6 Управление Сессиями (Frontend)

**Жизненный цикл сессии:**
1. **Login** возвращает `{user, session_id, sessions[]}` — первая сессия автовыбрана
2. **New Chat**: `POST /api/sessions` создаёт новую сессию, очищает чат
3. **Selection сессии**: Клик в sidebar → `loadSessionMessages(id)` загружает контекст
4. **Переименование**: Двойной клик → inline edit → `PUT /api/session/<id>`
5. **Удаление**: Клик X → `DELETE /api/session/<id>` → перерисовка списка

**Хранение на бэкенде:** Сессии в `eva/gui/eva_gui_cache/sessions.json` с thread-safe locking.
- Context nodes: макс 20
- Chat history: макс 50
- Entities: макс 30 на сессию

### 3.7 Аутентификация

- **Логин**: `admin`
- **Пароль**: `cogniflex`
- **Конфиг**: `eva/gui/eva_config.json`
- **Хэширование паролей**: PBKDF2-SHA256, 100,000 итераций, случайная 16-байтная соль
- **Auth token**: Boolean flag (не JWT). Идентификация через заголовок `X-User-ID`

### 3.8 Загрузка файлов

**Поддерживаемые форматы:** `.txt`, `.pdf`, `.docx`, `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.h`, `.html`, `.css`, `.json`, `.xml`, `.yaml`, `.yml`, `.md`, `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`

**Извлечение текста:** PyMuPDF, pdfplumber, PyPDF2, python-docx, Tesseract OCR (с fallback chain)

**Хранение:** UUID-именованные файлы в `eva/gui/web_gui/uploads/`

### 3.9 Логирование

Логи пишутся в:
- Консоль (stdout)
- Файл `logs/cogniflex.log`

Настраивается в `start_webgui.py`:
```python
from logging.handlers import RotatingFileHandler

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=20*1024*1024, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
```

---

## 4. Детальный Поток Данных

### 4.1 Полный цикл обработки запроса

```
Пользователь → Web GUI (Flask) → process_message()
    ├── Ethics check (blocked patterns)
    ├── Entity extraction
    └── Session context loading
    ↓
CoreBrain.process_query()
    ↓
_execute_query_strategy()
    ├── Mode check (qwen_only / gguf / fallback)
    └── Two-Model Pipeline (текущий режим)
        ├── Ethics check
        ├── Fractal memory context enrichment
        ├── Model A generation (logic)
        │   ├── Adaptive params
        │   ├── Quality check
        │   └── Retry on failure
        ├── Model B generation (concept expansion)
        │   ├── Adaptive params
        │   ├── Quality check
        │   └── Logit bias on retry
        ├── Model C generation (code, if needed)
        │   ├── Lazy load
        │   └── Code-specific prompt
        ├── Contradiction check
        ├── Knowledge rollback
        └── Save to fractal memory
    ↓
Response formatting
    ├── Reasoning steps extraction
    ├── Self-dialog triggering
    └── Session context update
    ↓
JSON Response → Web GUI → Пользователь
```

### 4.2 Event Flow

```
Pipeline publishes events → EventBus queues them → Worker thread processes →
Subscribers react (CoreBrain handlers) → Commands issued via DeferredCommandSystem →
Command executor runs → Results published back as events
```

**Детальный flow:**
```
CoreBrain.start()
    ↓
EventBus.publish(EventTypes.SYSTEM_START)
    ↓
BackgroundCoordinator._setup_event_integration()
    ↓
Subscribes to: SYSTEM_READY, SYSTEM_STOP, COMPONENT_*, LEARNING_*, PIPELINE_*, MEMORY_*
    ↓
Query received → EventBus.publish(EventTypes.PIPELINE_START)
    ↓
Pipeline processing → EventBus.publish(EventTypes.PIPELINE_MODEL_A_COMPLETE)
    ↓
Response generated → EventBus.publish(EventTypes.PIPELINE_COMPLETE)
    ↓
GUIBridge receives events → Emits to frontend via SSE
```

### 4.3 Storage / Хранение данных

**Без ORM — file-based storage повсеместно:**
- **Сессии**: JSON файл (`eva/gui/eva_gui_cache/sessions.json`)
- **Память**: JSON файлы в `memory/fractal_torch_storage/unified_memory/` (concepts, experiences, nodes)
- **Search cache**: JSON файл (`cogniflex_web_search_cache/search_cache.json`)
- **Token cache**: Disk cache с index JSON
- **Knowledge graph**: In-memory с JSON persistence
- **SQLite**: Используется в `contradiction/core_detection.py` и `learning/dialog_core.py` для специфичных подсистем

---

## 5. Структура Директорий

```
CogniFlex/
├── run.py                          # Root launcher (делегирование в eva/run.py)
├── wsgi.py                         # WSGI entry point (Flask app из eva.gui.web_gui.server)
├── gunicorn_config.py              # Gunicorn production config (bind 0.0.0.0:5555)
├── start_webgui.py                 # Web GUI launcher (init CoreBrain + Flask)
├── brain_config.json               # Central configuration file
├── requirements.txt                # Python зависимости
├── pyproject.toml                  # Project metadata
│
├── eva/
│   ├── __init__.py
│   ├── __main__.py                 # Точка входа (python -m eva)
│   ├── run.py                      # Primary entry point (singleton, signal handling)
│   ├── server.py                   # Re-exports из server_main
│   ├── server_main.py              # Flask app, WebGUI class, SessionManager, AuthManager
│   ├── server_routes.py            # Legacy API routes (sessions, chat, upload, feedback, status)
│   ├── server_handlers.py          # Extended handlers (analytics, learning, settings, knowledge-graph)
│   ├── nlp_fallbacks.py
│   │
│   ├── core/                       # CORE BRAIN MODULE
│   │   ├── core_brain.py           # CoreBrain class (тонкий координатор, 8 миксинов)
│   │   ├── core_brain_types.py     # SystemState enum, ComponentStatus
│   │   ├── brain_config.py         # Загрузка конфига, secret masking
│   │   ├── brain_components.py     # Инициализация компонентов
│   │   ├── brain_init.py           # Init orchestration
│   │   ├── brain_query.py          # QueryMixin (multi-level fallback chain, 7+ уровней)
│   │   ├── brain_coordination.py   # EventSubscriptionMixin, CommandIssuerMixin, ProcessTrackerMixin
│   │   ├── brain_state.py          # StateMixin, SystemState enum
│   │   ├── brain_memory.py         # MemoryMixin
│   │   ├── brain_monitoring.py     # MonitoringMixin
│   │   ├── event_bus.py            # EventBus (pub/sub, weak refs, queue-based worker)
│   │   ├── event_bus_bridge.py     # Мост старой EventSystem ↔ новой EventBus
│   │   ├── event_system.py         # Legacy EventSystem wrapper
│   │   ├── deferred_command_system.py  # DeferredCommandSystem (priority queues, retry, load shedding)
│   │   ├── deferred_commands.py    # Legacy DeferredCommandSystem (простая версия)
│   │   ├── generation_tracker.py   # GenerationTracker (lifecycle tracking)
│   │   ├── pipeline_core.py        # RecursiveModelPipeline (3-model GGUF orchestration)
│   │   ├── pipeline_models.py      # Model A/B/C generation methods
│   │   ├── pipeline_adaptive.py    # AdaptiveParameterController (semantic stuck detection)
│   │   ├── pipeline_quality.py     # Quality checking, sanitization, looping detection
│   │   ├── resource_manager.py     # System resource monitoring
│   │   ├── contradiction_resolver.py
│   │   ├── knowledge_rollback.py
│   │   ├── query_processor.py
│   │   ├── processor_pipeline.py
│   │   ├── processor_core.py
│   │   ├── processor_handlers.py
│   │   ├── reasoning_engine.py
│   │   ├── self_learning_system.py
│   │   ├── response_generator.py
│   │   ├── token_processor.py
│   │   ├── system_metrics.py
│   │   ├── system_optimizer.py
│   │   ├── system_state.py
│   │   ├── background_coordinator.py
│   │   ├── component_initializer.py
│   │   ├── feedback_processor.py
│   │   ├── memory_graph_ml.py
│   │   ├── config_manager.py
│   │   ├── fractal_attention_system.py
│   │   ├── utils.py
│   │   ├── api_compat.py
│   │   ├── base_component.py
│   │   └── recursive_model_pipeline.py
│   │
│   ├── gui/web_gui/                # Веб-интерфейс
│   │   ├── server_routes.py        # 30+ Flask endpoints
│   │   ├── server_main.py          # WebGUI class
│   │   ├── server_auth.py          # SessionManager, AuthManager, EntityExtractor, EthicsChecker
│   │   ├── server_models.py        # Model status, fractal graph endpoints
│   │   ├── server_api_export.py    # Import/export endpoints
│   │   ├── server_api_knowledge.py # Knowledge CRUD
│   │   ├── server_api_wikipedia.py # Wikipedia endpoints
│   │   ├── server_types.py         # Dataclasses
│   │   ├── bridge.py               # GUI-to-Core bridge
│   │   ├── templates/index.html    # SPA template (393 lines)
│   │   ├── static/css/style.css    # Стили (2287 lines, dark theme)
│   │   ├── static/js/app.js        # JavaScript (1870 lines, vanilla JS)
│   │   ├── static/js/emoji.js      # Emoji data (disabled)
│   │   ├── uploads/                # File upload storage
│   │   └── eva_gui_cache/          # Session cache
│   │
│   ├── memory/                     # Память
│   ├── mlearning/                  # ML компоненты
│   ├── learning/                   # Обучение
│   ├── knowledge/                  # Знания
│   ├── contradiction/              # Противоречия
│   ├── ethics/                     # Этика
│   ├── reasoning/                  # Рассуждение
│   ├── preprocess/                 # Предобработка
│   ├── neuromorphic/               # Нейроморфные системы
│   ├── distributed/                # Распределённые системы
│   ├── security/                   # Безопасность
│   ├── monitoring/                 # Мониторинг
│   ├── recovery/                   # Восстановление
│   ├── training/                   # Обучение моделей
│   ├── adaptation/                 # Адаптация
│   ├── nlp/                        # NLP
│   ├── generation/                 # Генерация
│   ├── storage/                    # Хранилище
│   ├── websearch/                  # Веб-поиск
│   ├── models/                     # Модели
│   ├── config/                     # Конфигурация
│   ├── cache/                      # Кэши
│   ├── scripts/                    # Скрипты
│   ├── tools/                      # Инструменты
│   ├── utils/                      # Утилиты
│   ├── adapters/                   # Адаптеры
│   └── analysis/                   # Анализ
│
├── tests/                          # 50+ тестовых файлов
├── backup/                         # Backup snapshot
├── backup_pre_audit/               # Pre-audit backup
├── logs/                           # Логи
│   └── cogniflex.log
├── cache/                          # Кэш
├── docs/                           # Документация
├── DESIGN.md                       # Этот документ
├── README.md
├── DEPLOYMENT.md
├── IMPROVEMENT_PLAN.md
├── LOG_ANALYSIS.md
├── COMPONENT_ANALYSIS_REPORT.md
├── IMPORT_ANALYSIS_REPORT.md
├── SYSTEM_FLOW.md
└── eva.bat / run.bat / start.sh / eva.bat  # Launch scripts
```

---

## 6. Конфигурация

### 6.1 brain_config.json

```json
{
  "model": {
    "name": "qwen3.5-0.8b",
    "type": "qwen",
    "disable_pytorch": true,
    "use_two_model_pipeline": true,
    "model_a_gguf_path": "eva/memory/fractal_torch_storage/gguf_models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf",
    "model_b_gguf_path": "...qwen2.5-3b-instruct-q4_k_m_model_b.gguf",
    "model_c_gguf_path": "...qwen2.5-coder-1.5b-instruct/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
  },
  "system": {
    "disable_learning_threads": true,
    "disable_background_training": true,
    "max_concurrent_operations": 4,
    "query_timeout": 30
  },
  "reasoning": {
    "weights": { "ethics": 0.3, "contradiction": 0.3, "knowledge": 0.4 }
  },
  "web_gui": { "host": "127.0.0.1", "port": 5555 }
}
```

### 6.2 Точки входа

| Файл | Назначение |
|------|------------|
| `run.py` | Делегирует в `eva/run.py:main()` |
| `eva/run.py` | **Primary entry**: создаёт CoreBrain, вызывает `initialize()` + `start()`, запускает GUI, обрабатывает сигналы (SIGINT/SIGTERM), singleton через PID file |
| `wsgi.py` | Production WSGI: импортирует `eva.gui.web_gui.server.app` |
| `gunicorn_config.py` | Gunicorn: bind `0.0.0.0:5555`, workers = `cpu_count*2+1`, sync workers, preload_app |
| `start_webgui.py` | Standalone GUI launcher: инициализирует CoreBrain, создаёт Flask app, бесконечный цикл |

---

## 7. Запуск

### 7.1 Запуск Web GUI

```bash
python start_webgui.py
```

Последовательность:
1. `start_webgui.py` — настройка логирования
2. `init_brain()` — инициализация CoreBrain (опционально)
3. `server.create_app(brain=brain)` — создание Flask app
4. `WebGUI.start()` — запуск Flask сервера

### 7.2 Запуск Core (полный)

```bash
python -m eva
```

или

```bash
python eva/__main__.py
```

### 7.3 Production (Gunicorn)

```bash
gunicorn -c gunicorn_config.py wsgi:app
```

### 7.4 Debug Endpoints в браузере

После запуска открыть:
- `http://127.0.0.1:5555/api/debug/test` — тест
- `http://127.0.0.1:5555/api/debug/auth` — состояние auth
- `http://127.0.0.1:5555/api/debug/deferred` — Deferred + EventBus
- `http://127.0.0.1:5555/api/debug/events` — история событий

---

## 8. Взаимосвязи Ключевых Классов

```
CoreBrain
  ├── ConfigMixin (brain_config.json loading)
  ├── ComponentMixin (component lifecycle, module management)
  ├── QueryMixin (multi-level fallback query processing)
  ├── MonitoringMixin (health, metrics)
  ├── MemoryMixin (memory operations)
  ├── StateMixin (system state machine)
  ├── EventSubscriptionMixin (13 event handlers)
  ├── CommandIssuerMixin (20+ command types)
  ├── ProcessTrackerMixin (query/command tracking)
  │
  ├── EventBus (новая событийная шина)
  ├── EventSystem (legacy событийная шина)
  ├── EventBusBridge (двунаправленный адаптер)
  ├── DeferredCommandSystem (priority queue + thread pool)
  ├── GenerationTracker (generation lifecycle)
  ├── RecursiveModelPipeline (3-model GGUF)
  │   ├── AdaptiveParameterController (semantic stuck detection)
  │   ├── Model A (Llama - Logic)
  │   ├── Model B (Llama - Concepts)
  │   └── Model C (Llama - Code, lazy)
  ├── ResourceManager (CPU/RAM monitoring)
  ├── UnifiedFractalMemory (experience storage)
  ├── BackgroundCoordinator (training, web indexing, recovery)
  └── SelfDialogLearningSystem (autonomous learning)

WebGUI (Flask server)
  ├── SessionManager
  ├── AuthManager
  ├── EntityExtractor
  └── EthicsChecker
```

---

## 9. История Версий

| Версия | Дата | Описание |
|--------|------|----------|
| 1.0 | 2026-03-23 | Начальная архитектура |
| 1.5 | 2026-03-26 | Qwen-only модель |
| 2.0 | 2026-04-03 | Three-GGUF Pipeline, Web GUI |
| 2.1 | 2026-04-05 | **EventBus**, **DeferredCommandSystem**, debug endpoints, login fix |
| 2.2 | 2026-04-06 | **GenerationTracker**, **SSE streaming**, **CoreBrain 8 mixins**, **полная документация API (30+ endpoints)**, **frontend архитектура**, **fallback chain**, **pipeline детали**, **модульная экосистема** |

---

## 10. Контакты и Ссылки

- **Репозиторий**: https://github.com/BlackCatSpb/EVA-Ai
- **Web GUI**: http://127.0.0.1:5555
- **Логин**: admin / cogniflex
- **Debug**: /api/debug/deferred, /api/debug/events, /api/debug/auth, /api/debug/test
