# EVA-Ai — Архитектура и Принципы Работы

**Версия:** 2.1  
**Дата:** 2026-04-05  
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
- **Веб-интерфейс** — Flask сервер с чатом, отладкой, системными endpoints
- **EventBus** — централизованная событийная шина нового поколения
- **DeferredCommandSystem** — система отложенных команд с восстановлением

### 1.3 Аппаратные требования

| Компонент | Требование | Назначение |
|-----------|------------|------------|
| CPU | 6+ ядер (AVX2) | GGUF инференс (llama.cpp) |
| RAM | 16 ГБ минимум | Модели 3B Q4_K_M × 2 = ~4GB, система ~6GB |
| GPU | CUDA 2+ ГБ VRAM | Embedding-модель |
| SSD | 10+ ГБ свободно | GGUF модели, кэши, граф памяти |

---

## 2. Архитектура Компонентов

### 2.1 Ядро (Core) — основные компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **CoreBrain** | `core/core_brain.py` | Центральный координатор. Загружает конфиг, инициализирует компоненты, управляет жизненным циклом. |
| **ComponentInitializer** | `core/component_initializer.py` | Фабрика компонентов. Создаёт компоненты через registered factories. |
| **EventBus** | `core/event_bus.py` | Pub/sub событийная шина нового поколения. Поддержка очереди, синхронная и асинхронная публикация. |
| **EventSystem** | `core/event_system.py` | Обёртка для обратной совместимости. Обертка вокруг EventBus с timeline. |
| **DeferredCommandSystem** | `core/deferred_command_system.py` | Очередь отложенных команд. Health checks, recovery strategies, приоритеты. |
| **BackgroundCoordinator** | `core/background_coordinator.py` | Координатор фоновых задач, автопилот. |
| **BrainCoordination** | `core/brain_coordination.py` | EventSubscriptionMixin - подписки на события, CommandIssuerMixin - issuing commands. |
| **RecursiveModelPipeline** | `core/recursive_model_pipeline.py` | Three-GGUF pipeline + AdaptiveController |
| **ResourceManager** | `core/resource_manager.py` | Мониторинг CPU/RAM/GPU |

### 2.2 Системы Событий и Команд

#### EventBus (core/event_bus.py)

Основные классы:
- `Event` — dataclass события (event_type, source, data, priority, timestamp)
- `EventTypes` — предопределённые типы событий (SYSTEM_*, COMPONENT_*, LEARNING_*, PIPELINE_*, MEMORY_*, KNOWLEDGE_*)
- `EventBus` — основной класс с методами:
  - `subscribe(event_type, handler)` — подписка на событие
  - `unsubscribe(event_type, handler_or_id)` — отписка
  - `publish(event)` — асинхронная публикация
  - `publish_sync(event)` — синхронная публикация
  - `get_stats()` — статистика

Поддержка логирования:
- Логирование публикации событий
- Логирование подписки/отписки
- Логирование обработки событий с дампом данных

#### DeferredCommandSystem (core/deferred_command_system.py)

Основные классы:
- `CommandPriority` — CRITICAL(0), HIGH(1), NORMAL(2), LOW(3)
- `CommandStatus` — PENDING, RUNNING, COMPLETED, FAILED, RETRYING
- `DeferredCommand` — dataclass команды
- `DeferredCommandSystem` — основной класс с методами:
  - `add_command(command, args, kwargs, priority, max_retries, ...)` — добавление команды
  - `add_module_health_check(module_name, check_func)` — health check
  - `add_module_recovery_strategy(module_name, recovery_func)` — recovery strategy
  - `get_stats()` — статистика системы

### 2.3 Память (Memory)

| Компонент | Файл | Описание |
|-----------|------|----------|
| **UnifiedFractalMemory** | `memory/unified_fractal_memory.py` | Единая фрактальная память |
| **GraphLearning** | `memory/graph_learning.py` | Обучение на графе |
| **HybridTokenCache** | `memory/hybrid_token_cache.py` | Гибридный кэш токенов |
| **EmbeddingCache** | `memory/embedding_cache.py` | SQLite LRU кэш эмбеддингов |

### 2.4 Машинное обучение (MLearning)

| Компонент | Файл | Описание |
|-----------|------|----------|
| **FractalModelManager** | `mlearning/fractal_model_manager.py` | Управление GGUF моделями |
| **QwenModelManager** | `mlearning/qwen_model_manager.py` | Управление PyTorch Qwen |
| **UnifiedTextProcessor** | `mlearning/unified_text_processor.py` | Текст + эмбеддинги |
| **MLUnit** | `mlearning/ml_unit.py` | ML ядро системы |
| **SentenceTransformersCache** | `mlearning/sentence_transformers_cache.py` | Кэш embedding-моделей |

---

## 3. Веб-интерфейс

### 3.1 Архитектура

```
Browser (jQuery + JavaScript)
    ↓ HTTP/JSON
Flask Server (server_routes.py)
    ├── /api/login — аутентификация
    ├── /api/chat — обработка сообщений
    ├── /api/status — статус системы
    ├── /api/system — информация о системе
    ├── /api/deferred/stats — DeferredCommandSystem
    ├── /api/deferred/commands — добавить команду
    ├── /api/debug/auth — debug: состояние auth
    ├── /api/debug/deferred — debug: Deferred + EventBus
    ├── /api/debug/events — debug: история событий
    ├── /api/debug/test — debug: тестовый endpoint
    └── /api/debug/login — debug: подробный логин
```

### 3.2 Файлы Web GUI

| Файл | Описание |
|------|----------|
| `gui/web_gui/server_main.py` | WebGUI класс, Flask app, create_app() |
| `gui/web_gui/server_routes.py` | API endpoints (login, chat, sessions, debug) |
| `gui/web_gui/server_auth.py` | AuthManager, SessionManager |
| `gui/web_gui/bridge.py` | GUIBridge — интеграция с CoreBrain |
| `gui/web_gui/templates/index.html` | HTML шаблон |
| `gui/web_gui/static/js/app.js` | JavaScript логика |
| `gui/web_gui/static/css/style.css` | Стили |

### 3.3 Debug Endpoints

Для отладки и мониторинга системы:

```bash
# Тестовый endpoint
GET /api/debug/test

# Состояние аутентификации
GET /api/debug/auth
# Ответ: {users: {admin: {username, salt, hash_prefix, has_user_id}}, users_count, lock_exists}

# Deferred Command System + Event Bus
GET /api/debug/deferred
# Ответ: {
#   available: true/false,
#   deferred_system: {type, commands, commands_count, recovery_strategies, health_checks, stats, running, shutting_down},
#   event_bus: {type, running, stats, subscribers: {event_type: count}},
#   event_system: {type, available},
#   brain_components: {component_name: {available, type}}
# }

# История событий из EventBus
GET /api/debug/events
# Ответ: {events: [{event_type, source, timestamp, data}], event_bus_stats}

# Debug login с подробным дампом
POST /api/debug/login
Body: {"username": "admin", "password": "cogniflex"}
# Ответ: {success, step, details, error, traceback}
```

### 3.4 Аутентификация

- **Логин**: `admin`
- **Пароль**: `cogniflex`
- **Конфиг**: `eva/gui/eva_config.json`
```json
{
  "web_gui": {
    "admin_password": "cogniflex",
    "admin_salt": "default_salt_2024"
  }
}
```

### 3.5 Логирование

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
    ↓
CoreBrain.process_query()
    ↓
SelfReasoningEngine / Two-Model Pipeline
    ↓
Model A (логика) → Quality Check → Model B (развитие) → Quality Check
    ↓
GraphLearning (сохранение опыта)
    ↓
SelfDialogLearning (самодиалог)
    ↓
Ответ → Web GUI → Пользователь
```

### 4.2 Event Flow

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

---

## 5. Структура Директорий

```
CogniFlex/
├── eva/
│   ├── core/                          # Ядро системы
│   │   ├── core_brain.py              # CoreBrain
│   │   ├── event_bus.py                # EventBus (новый)
│   │   ├── event_system.py             # EventSystem (обёртка)
│   │   ├── event_bus_bridge.py         # Мост между системами
│   │   ├── deferred_command_system.py  # DeferredCommandSystem
│   │   ├── brain_components.py          # Компоненты brain
│   │   ├── brain_coordination.py        # EventSubscriptionMixin
│   │   ├── background_coordinator.py   # Фоновые задачи
│   │   ├── component_initializer.py    # Фабрика компонентов
│   │   ├── recursive_model_pipeline.py # Three-GGUF pipeline
│   │   ├── memory_graph_ml.py          # ML на графе
│   │   ├── graph_ml_core.py            # Graph ML Core
│   │   ├── graph_ml_training.py        # Graph ML Training
│   │   └── ...
│   │
│   ├── memory/                        # Память
│   ├── mlearning/                      # ML компоненты
│   ├── gui/web_gui/                    # Веб-интерфейс
│   │   ├── server_main.py               # WebGUI класс
│   │   ├── server_routes.py             # API endpoints
│   │   ├── server_auth.py               # Auth + Session
│   │   ├── bridge.py                    # GUIBridge
│   │   ├── templates/index.html
│   │   └── static/
│   │       ├── js/app.js
│   │       └── css/style.css
│   ├── learning/                      # Обучение
│   ├── knowledge/                      # Знания
│   ├── contradiction/                  # Противоречия
│   ├── ethics/                         # Этика
│   └── __main__.py                     # Точка входа
│
├── DESIGN.md                          # Этот документ
├── start_webgui.py                    # Альтернативный запуск GUI
├── logs/                              # Логи
│   └── cogniflex.log
└── eva/gui/eva_config.json            # Конфиг Web GUI
```

---

## 6. Запуск

### 6.1 Запуск Web GUI

```bash
python start_webgui.py
```

Последовательность:
1. `start_webgui.py` — настройка логирования
2. `init_brain()` — инициализация CoreBrain (опционально)
3. `server.create_app(brain=brain)` — создание Flask app
4. `WebGUI.start()` — запуск Flask сервера

### 6.2 Запуск Core (полный)

```bash
python -m eva
```

или

```bash
python eva/__main__.py
```

### 6.3 Debug Endpoints в браузере

После запуска открыть:
- `http://127.0.0.1:5555/api/debug/test` — тест
- `http://127.0.0.1:5555/api/debug/auth` — состояние auth
- `http://127.0.0.1:5555/api/debug/deferred` — Deferred + EventBus
- `http://127.0.0.1:5555/api/debug/events` — история событий

---

## 7. История Версий

| Версия | Дата | Описание |
|--------|------|----------|
| 1.0 | 2026-03-23 | Начальная архитектура |
| 1.5 | 2026-03-26 | Qwen-only модель |
| 2.0 | 2026-04-03 | Three-GGUF Pipeline, Web GUI |
| 2.1 | 2026-04-05 | **EventBus**, **DeferredCommandSystem**, debug endpoints, login fix |

---

## 8. Контакты и Ссылки

- **Репозиторий**: https://github.com/BlackCatSpb/EVA-Ai
- **Web GUI**: http://127.0.0.1:5555
- **Логин**: admin / cogniflex
- **Debug**: /api/debug/deferred, /api/debug/events, /api/debug/auth
