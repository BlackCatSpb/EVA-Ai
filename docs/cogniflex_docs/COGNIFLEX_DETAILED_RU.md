# CogniFlex — Подробное описание проекта (RU)

Обновлено: 2025-08-19 11:33 (+03)

## Резюме
CogniFlex — модульная AI/ML-платформа, объединяющая обработку текста, локальный офлайн-инференс моделей, гибридное кэширование и рассуждение на основе графа знаний в едином рантайме. Основные акценты:
- Локальная работа без сети (офлайн-инференс)
- «Безопасный тестовый режим» для быстрых и детерминированных проверок
- Расширяемый менеджмент моделей с фоновым (щадящим) прогревом
- Богатая телеметрия, проверки здоровья и плавная деградация

Ключевые директории:
- `cogniflex/` — ядро рантайма
- `cogniflex/mlearning/` — ML-стек (модели, токенизация, генерация)
- `cogniflex/core/` — оркестрация, события, планирование
- `cogniflex/websearch/` — модуль веб-поиска
- `tools/` — утилиты, отладка и бенчмарки
- `docs/` — документация
- `cogniflex_models/` — локальные снапшоты моделей (игнор в Git)

---

## Архитектура

### Вершинные компоненты
- **CoreBrain (`cogniflex/core/`)**
  - Хранит глобальную конфигурацию, жизненный цикл и шину событий (`brain.events`).
  - Интегрирует подсистемы: ML, кэши, планировщики, граф знаний, веб-поиск.

- **MLUnit (`cogniflex/mlearning/ml_unit.py`)**
  - Главный фасад ML для ядра.
  - Инициализирует гибридные кэши, текстовый процессор, `ModelManager` и получает `ResponseGenerator` из `brain`.
  - Предоставляет API задач (например, генерация текста) и координирует выбор подходящих моделей.

- **ModelManager (`cogniflex/mlearning/model_manager.py`)**
  - Регистрирует модели, хранит метаданные в SQLite (`core/cogniflex_cache/models/models.db`).
  - Фоновая щадящая загрузка, load-shedding, безопасные фоллбэки.
  - Офлайн-провизия дефолтной Qwen и автообнаружение локальных моделей в `model_dir`.
  - Исправлено: мягкая остановка потоков без `executor.shutdown()`; полное закрытие в `close()`.

- **ResponseGenerator (`cogniflex/core/response_generator.py`)**
  - Токенизация, генерация, постобработка и кэширование.
  - Метрики, логирование и безопасные фоллбэки при недоступности моделей.

- **Unified Text Processor (`cogniflex/mlearning/unified_text_processor.py`)**
  - Унифицирует токенизацию между разными моделями.
  - Интегрируется в `ModelManager` через событие готовности.

- **WebSearchEngine (`cogniflex/websearch/web_search_engine.py`)**
  - Синхронный и квазиявно-асинхронный веб-поиск с кэшированием и базой истории.
  - Вспомогательные движки: `cogniflex/websearch/search_engines.py` (Google/Yandex синтетические ответы для оффлайна, Bing — реальный HTTP), модели данных — `search_models.py`.

- **Deferred Command System (`cogniflex/core/deferred_command_system.py`)**
  - Планирование фоновых задач и кооперативный load-shedding при тяжёлых операциях (например, загрузка модели).

- **Knowledge Graph (`cogniflex/knowledge/knowledge_graph.py`)**
  - Хранилище концептов и связей для более высокого уровня рассуждений и памяти.

---

## Данные и кэширование
- **Гибридный кэш** (`core/cogniflex_cache/`, `ml_cache/`, `benchmark_cache/`, `test_cache/`):
  - Комбинирует in-memory структуры с дисковым хранилищем.
  - Токенный и блочный кэши ускоряют токенизацию и потоковую генерацию.
- **SQLite-реестр моделей (`models.db`)** под управлением `ModelManager`.
- **Кэш веб-поиска** (`cogniflex/websearch/web_search_engine.py`):
  - JSON-кэш `search_cache.json` и база `search_history.db` в каталоге кэша модуля (по умолчанию `cogniflex/websearch/cogniflex_web_search_cache/`).

---

## Установка и запуск

### Предварительные требования
- Python 3.11+ (папка `.venv311/` указывает на использование 3.11)
- Windows (разработано и тестируется на Windows; Linux/macOS также возможны)

### Установка зависимостей (минимум)
Если у вас нет общего `requirements.txt`, установите ключевые пакеты вручную:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch transformers huggingface_hub requests beautifulsoup4 pytest
```

### Переменные окружения (базовые)
- `COGNIFLEX_TEST_MODE` / `COGNIFLEX_SAFE_TEST_MODE` — включает безопасный тестовый режим
- `TRANSFORMERS_OFFLINE` / `HF_HUB_OFFLINE` — офлайн-режим HuggingFace
- `COGNIFLEX_MODELS_DIR` — корневой каталог локальных снапшотов моделей
- `COGNIFLEX_QWEN_REPO`, `COGNIFLEX_QWEN_LOCAL_NAME` — источник/имя локальной Qwen

### Быстрый офлайн E2E-debug (полный режим)
```powershell
$env:COGNIFLEX_TEST_MODE="0"; $env:TRANSFORMERS_OFFLINE="1"; $env:HF_HUB_OFFLINE="1";
$env:COGNIFLEX_MODELS_DIR="c:\\Users\\black\\OneDrive\\Desktop\\CogniFlex\\cogniflex_models";
python tools/simple_debug.py
```
Если в локальной папке модели нет весов — система вернётся к безопасным ответам и залогирует предупреждение.

### Развёртывание локального снапшота Qwen (разово, онлайн)
```powershell
$env:TRANSFORMERS_OFFLINE="0"; $env:HF_HUB_OFFLINE="0"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-1.5B-Instruct',
    local_dir=r'c:\\Users\\black\\OneDrive\\Desktop\\CogniFlex\\cogniflex_models\\qwen25_1p5b_instruct',
    local_dir_use_symlinks=False,
)
PY
$env:TRANSFORMERS_OFFLINE="1"; $env:HF_HUB_OFFLINE="1"
```

---

## Модуль веб-поиска

Файл: `cogniflex/websearch/web_search_engine.py`

- **Назначение**: высокоуровневый движок веб-поиска с кэшем и историей (SQLite).
- **Ключевые зависимости**: `requests`, `beautifulsoup4`, `sqlite3`, `threading`.
- **Поддерживаемые движки**: Google (синтетическая выдача), Yandex (синтетическая выдача), Bing (реальный HTTP).
- **Важное по офлайн-режиму**:
  - По умолчанию Bing выключен (`active_search_engines['bing'] = False`), чтобы не идти в сеть.
  - Google/Yandex возвращают реалистично-демонстрационные результаты (см. `cogniflex/websearch/search_engines.py`).

### Основные методы
- `search(query: str, max_results: int = None, use_cache: bool = None) -> Dict`
- `search_async(query: str, max_results: int = None) -> str` + `get_task_status(task_id: str) -> Dict`
- `set_search_engines(use_google: bool, use_yandex: bool, use_bing: bool)`
- `configure_settings(**kwargs)` / `get_settings()`
- `get_stats()` / `get_recent_queries(limit: int)`
- `clear_cache()` / `get_cache_info()`
- `web_search_and_learn(concept: str, num_results: int) -> List[Dict[str, Any]]`

### Пример использования (без `CoreBrain`)
```python
from cogniflex.websearch.web_search_engine import WebSearchEngine

engine = WebSearchEngine()
engine.set_search_engines(use_google=True, use_yandex=True, use_bing=False)  # офлайн-дружественно
engine.configure_settings(max_results=5, cache_ttl=3600, use_cache=True)

resp = engine.search("Python programming basics", max_results=3)
print(resp["status"], len(resp.get("results", [])))

# Асинхронный запуск
engine.start()
task_id = engine.search_async("artificial intelligence", max_results=3)
# ... подождать или опрашивать статус
d = engine.get_task_status(task_id)
engine.stop()

# Преобразование результатов в знания
knowledge = engine.web_search_and_learn("machine learning", num_results=3)
```

### Интеграция в ядро
В e2e-тестах (`tests/test_web_search_e2e.py`) модуль доступен как `brain.web_search_engine` после `CoreBrain.initialize()`. Инициализация выполняется `ComponentInitializer` внутри ядра.

---

## Примеры API (ML)

### Быстрая генерация текста через `CoreBrain`
```python
from cogniflex.core.core_brain import CoreBrain

brain = CoreBrain()
assert brain.initialize()
# Убедитесь, что модель доступна (не safe mode и есть веса)
resp = brain.response_generator.generate_response(
    prompt="Кратко опиши, что такое искусственный интеллект.",
    max_length=120,
    temperature=0.7,
    top_p=0.9,
    task="text-generation",
)
print(resp.get("text", ""))
```

### Прямой доступ к `MLUnit`
```python
ml = brain.ml_unit
answer = ml.generate_response("Привет! Объясни, что такое токенизация в NLP.", task="text-generation")
print(answer.get("text"))
```

### Работа с `ModelManager`
```python
mm = brain.ml_unit.model_manager
# Зарегистрированные модели
models = mm.get_available_models()
# Подобрать модель для задачи
model = mm.get_model_for_task("text-generation")
```

---

## Управление моделями и провизия
- **Регистрация**: `register_model()` пишет модель в БД и синхронизирует её с `MLUnit`.
- **Автопровизия Qwen**: `_provision_qwen_default()` создаёт/обновляет алиас `default_text_gen` на локальный снапшот `Qwen/Qwen2.5-1.5B-Instruct` (по умолчанию).
- **Автообнаружение**: `scan_models_directory()` находит валидные локальные модели в `model_dir`.
- **Выбор модели**: `_select_best_model()` сортирует кандидатов по `priority` и соответствию задаче (например, `"text-generation"`).
- **Фоновая загрузка**: `load_model_low_impact()` троттлит CPU/IO и взаимодействует с `DeferredCommandSystem`.
- **Здоровье и метрики**: `_background_monitoring()` и `_background_loading()` с событиями `_emit_model_load_event()` и `_emit_metrics()`.

### Офлайн-поведение
- `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1` — запрет сетевых обращений.
- `COGNIFLEX_MODELS_DIR` — корень локальных снапшотов.
- Нужны веса: `model.safetensors` или `pytorch_model*.bin` в папке модели.

### Безопасный тестовый режим
- `COGNIFLEX_TEST_MODE=1` или `COGNIFLEX_SAFE_TEST_MODE=1`.
- Лёгкие стабы моделей/токенизаторов, CPU-only, отключение фоновых служб.

---

## Уникальные решения
- **Двухрежимный рантайм (Safe vs Full)** — стабы для CI и реальный инференс для продакшн-проверок.
- **Load-shedding при загрузке** через `DeferredCommandSystem` — приоритеты, пауза тяжёлых задач.
- **Устойчивые фоновые сервисы** — мягкий рестарт потоков без остановки `ThreadPoolExecutor` (полный shutdown — только в `close()`).
- **SQLite-реестр моделей** — идемпотентные миграции, единый источник правды.
- **Событийная интеграция** — прогресс/метрики через `brain.events`.
- **Унифицированная токенизация** — `UnifiedTextProcessor` скрывает специфику моделей.
- **Офлайн-провизия снапшотов** — поддержка air-gapped окружений.

---

## Потоки выполнения

### Генерация текста (полный режим)
1. `MLUnit.generate_response()` запрашивает модель `"text-generation"` через `ModelManager.get_model_for_task()`.
2. `ModelManager` выбирает кандидата и инициирует загрузку.
3. `ResponseGenerator` выполняет токенизацию/сэмплинг/постобработку; эмитит метрики.
4. При недоступности — безопасный фоллбэк.

### Генерация текста (safe test mode)
- Лёгкие синтетические ответы без реальных моделей.

---

## Эксплуатационные моменты
- **Память**: крупные модели могут не влезать в RAM/VRAM; во время загрузки активируется load-shedding.
- **Жизненный цикл executor**: `stop()` — мягко, `close()` — полное выключение (включая `executor.shutdown()`).
- **Квантизация**: динамическая CPU-квантизация (Linear/LSTM/GRU) по конфигу.
- **Логи/метрики**: подробные логи (`tools/logs/`, корневые `*.log`), метрики доступны как события.

---

## Дорожная карта
- Плагинные адаптеры моделей (vision, ASR) рядом с текстовыми.
- Проверка целостности локальных снапшотов и авто-ремонт.
- Продвинутый роутинг промптов и ансамбли моделей.
- Более тесная интеграция с графом знаний для retrieval и few-shot памяти.

---

## Чек-лист устранения неполадок
- **Нет весов** в каталоге модели (`model.safetensors`/`pytorch_model*.bin`).
- **Конфликт путей**: проверьте `COGNIFLEX_MODELS_DIR` и реальный `model_dir` (`ModelManager.__init__`).
- **Executor shutdown**: используйте `close()` только при завершении процесса.
- **Случайно включён Safe-режим**: `COGNIFLEX_TEST_MODE` = "0" для реального инференса.
- **Веб-поиск уходит в сеть**: отключите Bing (`set_search_engines(..., use_bing=False)`), оставьте Google/Yandex (синтетические результаты).
