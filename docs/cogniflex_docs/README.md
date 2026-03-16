# CogniFlex — Обзор функциональности проекта

CogniFlex — модульная когнитивная система с упором на устойчивость, производительность и удобство интеграции. Проект включает ядро, ML‑подсистему, кэширование токенов, аналитику метрик и GUI.

## Ключевые возможности
- **Модульная архитектура**: разделение на менеджеры конфигураций, состояния системы, ресурсов и метрик.
- **ML‑менеджер моделей**: реестр моделей, фоновая/ручная загрузка, поддержка офлайн‑режима, асинхронная инициализация.
- **Гибридный токен‑кэш + асинхронная токенизация**: значительный прирост скорости на простых/средних текстах.
- **Система метрик и здоровья**: мониторинг производительности, панели для GUI.
- **Надёжная обработка ошибок**: фолбэки, частичная инициализация компонентов, устранение циклических зависимостей.
- **Интерактивный GUI**: обновления дашборда, уведомления, симуляторы взаимодействий.

## Документация
- **Архитектура**: `docs/architecture.md`
- **Деплой/DevOps**: `docs/deployment.md`
- **API справочник**: `docs/api_reference.md`

## Структура проекта (основное)
- `cogniflex/`
  - `core/`
    - `token_processor.py` — унифицированная токенизация, потоковая обработка.
    - менеджеры: ConfigManager, SystemStateManager, ResourceManager, SystemMetricsManager (после рефакторинга из `core_brain.py`).
  - `mlearning/`
    - `model_manager.py` — реестр и загрузка моделей (см. ниже).
    - `ml_core.py` — `ModelHealth` и базовые сущности ML.
  - `tools/`
    - `dependency_scan.py` — анализ зависимостей/циклов.
- `core/cogniflex_cache/` — кэши и БД моделей: `models/models.db`.
- `tools/` — утилиты, `train_from_path.py` и др.
- Тесты/скрипты запуска: `minimal_test.py`, `final_test.py`, `system_smoke_test.py`, `test_*`, `run_gui.py`.

## Менеджер моделей (`cogniflex/mlearning/model_manager.py`)
- **Реестр и БД**: хранение метаданных моделей в SQLite (`core/cogniflex_cache/models/models.db`).
- **Дефолтные модели**: добавляются при первом запуске, если БД пуста (Qwen 2.5, RuGPT‑3, DialoGPT, GPT‑2 и др.).
- **Фоновые службы**: мониторинг и автозагрузка по приоритетам (можно отключить).
- **Офлайн‑режим**: учитывает `TRANSFORMERS_OFFLINE=1` и/или `HF_HUB_OFFLINE=1`, передаёт `local_files_only=True` в `from_pretrained()`.
- **Параметры конструктора**:
  - `use_gpu: bool` — выбор устройства (CUDA при наличии).
  - `max_workers: int` — параллельные загрузки.
  - `autoload: bool` — запуск фоновых служб (мониторинг/автозагрузка).
- **API (основное)**:
  - `get_available_models()` — список зарегистрированных моделей.
  - `load_model(model_id)` — асинхронная загрузка.
  - Автоматическая регистрация `UnifiedTextProcessor` при событии `text_processor_ready`.

### Безопасная инициализация без скачиваний
- PowerShell (только текущая сессия):
  ```powershell
  $env:TRANSFORMERS_OFFLINE="1"
  $env:HF_HUB_OFFLINE="1"
  ```
- Инициализация (без фоновой автозагрузки):
  ```bash
  python -c "from cogniflex.mlearning.model_manager import ModelManager; mm=ModelManager(brain=None,use_gpu=False,max_workers=1,autoload=False); print('OK')"
  ```

### Провизия локальной ruGPT3 и поведение скачивания
- При инициализации `ModelManager` в обычном режиме выполняется провизия ruGPT3 (по умолчанию `sberbank-ai/rugpt3large_based_on_gpt2`) в локальную папку `cogniflex/mlearning/cogniflex_models/<имя>`.
- Если в целевой папке уже есть валидная локальная модель (минимум: `config.json`, веса `pytorch_model.bin` или `model.safetensors`, токенайзер `tokenizer.json` или `merges.txt+vocab.json`), скачивание из HF пропускается.
- В офлайн‑режиме скачивание всегда пропускается.
- Можно принудительно перекачать снапшот при онлайне, установив `COGNIFLEX_FORCE_MODEL_REFRESH=1`.
- Имя локальной подпапки можно задать через `COGNIFLEX_RUGPT3_LOCAL_NAME` (дефолт: `rugpt3_large`).

### Проверка БД моделей напрямую
```bash
python -c "import sqlite3, json; db=r'core/cogniflex_cache/models/models.db'; conn=sqlite3.connect(db); cur=conn.cursor(); cur.execute('select id,name,model_path,model_type,priority,tags from models order by priority desc, timestamp asc'); rows=cur.fetchall(); conn.close(); print(json.dumps([{'id':r[0],'name':r[1],'path':r[2],'type':r[3],'priority':r[4],'tags': (json.loads(r[5]) if r[5] else [])} for r in rows], ensure_ascii=False, indent=2))"
```

## Токенизация и кэш
- **Асинхронная токенизация** в `cogniflex/core/token_processor.py`.
- **Гибридный кэш токенов**: повышение пропускной способности, снижение задержек; потокобезопасность и метрики.

## Метрики и здоровье системы
- **SystemMetricsManager**: сбор и публикация метрик производительности.
- **SystemStateManager/ResourceManager**: контроль статуса компонентов и ресурсов (CPU/GPU/память), предупреждения при 90%+ памяти.
- **API ядра**: `get_system_health()`, `get_system_metrics()`, `get_system_dashboard_data()` — для GUI/панелей.

## GUI и инструменты
- `run_gui.py` — запуск графического интерфейса.
- Уведомления GUI: `show_notification()` реализован.
- Симуляторы взаимодействия: `gui_interaction_simulator.py`, результаты и логи в корне.
- Утилиты: `dependency_scan.py`, `train_from_path.py`, примеры документов в `tools/sample_docs/`.

## Производительность
- Итоги бенчмарков (асинхронная токенизация + гибридный кэш):
  - Средний прирост: **9.54×**
  - Простые тексты: **34.59×**
  - Средняя сложность: **1.61×**
  - Сложные/большие контексты: ~1.0× (предмет для дальнейшей оптимизации)
- Качество понимания контекста сохранено; требуется улучшить hit‑rate кэша в некоторых сценариях.

## Надёжность и рефакторинг
- Устранены циклические зависимости между модулями ядра и ML.
- Разделение обязанностей: `ConfigManager`, `SystemStateManager`, `ResourceManager`, `SystemMetricsManager`.
- Добавлены фолбэки во всех критичных путях, частичная инициализация компонентов.
- Исправлены ключевые ошибки времени выполнения:
  - Отсутствующие методы: `process_query()` в Core, `get_system_metrics()` и др.
  - Статистика графа знаний: `get_statistics()`, `get_domain_statistics()`.
  - GUI‑методы: `show_notification()`.
  - Совместимость имён: алиас `system_metrics_manager` и обновлённые обращения в дашборде.

## Быстрый старт
1. Установите зависимости (см. `requirements.txt`, при необходимости — создайте venv).
2. При первом запуске запустите минимальные тесты: `python minimal_test.py`.
3. Для работы офлайн:
   - Установите переменные окружения (см. выше) и инициализируйте `ModelManager(autoload=False)`.
4. Для GUI: `python run_gui.py`.

## Конфигурация и переменные окружения
- `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1` — строгий офлайн‑режим (без загрузок из сети).
- Настройки путей кэша/моделей задаются параметрами `ModelManager` (`cache_dir`, `model_dir`).

### Выбор дефолтной модели генерации текста
- `COGNIFLEX_DEFAULT_TEXT_GEN` — явное переопределение алиаса `default_text_gen`.
  - Допустимые значения:
    - HF repo id (например: `ai-forever/rugpt3small_based_on_gpt2`).
    - Абсолютный путь к локальной папке модели.
    - Имя подпапки внутри директории моделей (`model_dir`), если модель хранится локально в `cogniflex_models/`.
  - При наличии этой переменной `ModelManager` регистрирует/обновляет alias `default_text_gen` с высоким приоритетом, без изменения тестов и остального кода.

Примеры: оставить только ruGPT3 Small (локально скачанную) как дефолт

```powershell
$env:COGNIFLEX_DEFAULT_TEXT_GEN = "ai-forever/rugpt3small_based_on_gpt2"
# При желании принудительно обновить локальную копию (онлайн)
# $env:COGNIFLEX_FORCE_MODEL_REFRESH = "1"
```

Примеры установки:

- PowerShell (на время текущей сессии):
  ```powershell
  $env:COGNIFLEX_DEFAULT_TEXT_GEN = "ai-forever/rugpt3small_based_on_gpt2"
  # Опционально офлайн-режим
  $env:TRANSFORMERS_OFFLINE = "1"
  $env:HF_HUB_OFFLINE = "1"
  ```

- Bash:
  ```bash
  export COGNIFLEX_DEFAULT_TEXT_GEN="ai-forever/rugpt3small_based_on_gpt2"
  # Опционально офлайн-режим
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  ```

Быстрая проверка без загрузки весов:

```powershell
python -c "import json; from cogniflex.mlearning.model_manager import ModelManager; mm=ModelManager(use_gpu=False, autoload=False, safe_test_mode=True); mm.scan_models_directory(); md=getattr(mm,'model_metadata',{}).get('default_text_gen'); print(json.dumps({'model_path':getattr(md,'model_path',None),'model_type':getattr(md,'model_type',None),'priority':int(getattr(md,'priority',0) or 0)}, ensure_ascii=False))"
```

## Тестирование
- Сценарии: `system_smoke_test.py`, `test_*` в корне проекта.
- Полезные проверки: `final_test.py`, `test_e2e_chat_query.py`, бенчмарки `test_*performance*` и `comprehensive_performance_benchmark.py`.

## Известные ограничения
- GUI: отдельные вопросы жизненного цикла виджетов (TclError) возможны в редких сценариях.
- Большие модели (Qwen 7B/14B) требуют значительных ресурсов и времени для загрузки; рекомендуется управлять приоритетами или работать офлайн.

## Лицензирование и данные моделей
- Используемые модели из Hugging Face подчиняются их лицензиям. Перед загрузкой убедитесь в совместимости и наличии ресурсов.

---
Дополнительные материалы смотрите в разделе "Документация": `docs/architecture.md`, `docs/deployment.md`, `docs/api_reference.md`.
