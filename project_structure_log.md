# CogniFlex Project Structure Analysis Log
# Создан: 2025-08-29T12:13:12+03:00
# Анализатор: Cascade AI Assistant

## 1. ОБЩАЯ ИНФОРМАЦИЯ О ПРОЕКТЕ

### Основные характеристики:
- **Название**: CogniFlex - локальная когнитивная нейросеть
- **Точка входа**: python -m cogniflex.run
- **Основной язык**: Python
- **Архитектура**: Модульная система с разделением на компоненты

### Структура запуска:
- `cogniflex/run.py` - точка входа с настройкой CUDA
- `cogniflex/core/core_brain.py` - основное ядро системы (main функция)
- `cogniflex/core/core_brain.py::main()` - функция запуска

## 2. АРХИТЕКТУРА ПРОЕКТА

### Основные компоненты:
1. **Core** - ядро системы (`cogniflex/core/`)
2. **Adaptation** - система адаптации (`cogniflex/adaptation/`)
3. **Contradiction** - анализ противоречий (`cogniflex/contradiction/`)
4. **Memory** - система памяти (`cogniflex/memory/`)
5. **Knowledge** - управление знаниями (`cogniflex/knowledge/`)
6. **Learning** - система обучения (`cogniflex/learning/`)
7. **Generation** - генерация контента (`cogniflex/generation/`)
8. **Adapters** - адаптеры для внешних систем (`cogniflex/adapters/`)

### Вспомогательные компоненты:
- **GUI** - графический интерфейс (`cogniflex/gui/`)
- **Runtime** - среда выполнения (`cogniflex/runtime/`)
- **Tools** - инструменты (`cogniflex/tools/`)
- **System** - системные утилиты (`cogniflex/system/`)

## 3. КЛАССИФИКАЦИЯ ФАЙЛОВ

### ФУНКЦИОНАЛЬНЫЕ ФАЙЛЫ (основной код):
```
cogniflex/
├── run.py                           # Точка входа
├── __init__.py                      # Инициализатор пакета
├── core/
│   ├── core_brain.py               # Основное ядро (145KB)
│   ├── background_coordinator.py   # Координатор фоновых задач
│   ├── component_initializer.py    # Инициализатор компонентов
│   ├── query_processor.py          # Обработчик запросов
│   ├── response_generator.py       # Генератор ответов
│   ├── system_state.py             # Состояние системы
│   ├── resource_manager.py         # Менеджер ресурсов
│   └── utils.py                    # Утилиты
├── adaptation/
│   ├── adaptation_manager.py       # Менеджер адаптации
│   ├── adaptation_core.py          # Ядро адаптации
│   └── adaptation_analytics.py     # Аналитика адаптации
├── contradiction/
│   ├── contradiction_core.py       # Ядро анализа противоречий
│   ├── contradiction_detection.py  # Обнаружение противоречий
│   ├── contradiction_resolution.py # Разрешение противоречий
│   └── contradiction_manager.py    # Менеджер противоречий
├── memory/
│   ├── memory_manager.py           # Менеджер памяти
│   ├── hybrid_token_cache.py       # Гибридный кэш токенов
│   ├── memory_core.py              # Ядро памяти
│   └── working_memory.py           # Рабочая память
├── knowledge/
│   ├── knowledge_graph.py          # Граф знаний (311KB)
│   ├── knowledge_core.py           # Ядро знаний
│   ├── knowledge_integrator.py     # Интегратор знаний
│   └── knowledge_manager.py        # Менеджер знаний
└── learning/
    ├── learning_scheduler.py       # Планировщик обучения
    ├── self_analyzer.py            # Самоанализатор
    └── memory_graph_trainer.py     # Тренер графов памяти
```

### ОТЛАДОЧНЫЕ И ТЕСТОВЫЕ ФАЙЛЫ:
```
cogniflex/
├── system_selftest.py              # Самотестирование системы
├── system_selftest.log             # Лог самотестирования
├── system_selftest copy.log        # Копия лога тестирования
├── test_fractal_store.py           # Тест хранилища фракталов
├── nlp_fallbacks.py                # Резервные NLP функции
└── dependency_report.log           # Отчет о зависимостях

scripts/                           # Скрипты для отладки
├── auto_archive_chat_logs.py
├── batch_load_to_graph.py
├── build_state_from_fractal.py
├── cleanup_models_db.py
└── ...
```

### КЭШ-ФАЙЛЫ И ДАННЫЕ:
```
cogniflex/
├── cache/                         # Основной кэш
├── cogniflex_cache/               # Кэш CogniFlex
├── search_cache.json              # Кэш поиска
├── base_knowledge_v2.json         # База знаний v2
├── initial_knowledge.json         # Начальные знания
├── cogniflex_knowledge.db         # База знаний SQLite
└── cogniflex_dialog_history.json  # История диалогов

Корневые кэши:
├── cogniflex_cache/
├── ethics_cache/
├── hybrid_cache/
├── ml_cache/
├── tokenizer_cache/
└── ...
```

### КОНФИГУРАЦИОННЫЕ ФАЙЛЫ:
```
├── pyproject.toml                 # Конфигурация проекта
├── requirements.txt               # Зависимости
├── pytest.ini                     # Конфигурация тестирования
├── universal_analyzer_config.ini  # Конфиг анализатора
└── setup.py                       # Скрипт установки
```

### ДОКУМЕНТАЦИЯ:
```
documentation/
├── docs/
│   ├── api_reference.md
│   ├── architecture.md
│   └── ...
└── *.md файлы в корне
```

## 4. ЗАВИСИМОСТИ ПРОЕКТА

### Основные зависимости (requirements.txt):
```
# Core Dependencies
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
tqdm>=4.65.0
safetensors>=0.3.0

# GUI Dependencies
tkinter>=8.6.12
matplotlib>=3.7.1
Pillow>=9.5.0

# Data Processing
pandas>=2.0.0
scikit-learn>=1.2.0
scipy>=1.10.0

# Utilities
pyyaml>=6.0
python-dotenv>=1.0.0
loguru>=0.7.0

# Testing
pytest>=7.3.1
pytest-cov>=4.0.0

# Documentation
sphinx>=6.1.3
sphinx-rtd-theme>=1.2.0
```

## 5. ПРОБЛЕМЫ ПРОЕКТА

### Критические проблемы:
1. **Разбросанность кода**: Функциональный код смешан с отладочными файлами
2. **Отсутствие чистой структуры**: Много временных и тестовых файлов в основной директории
3. **Кэши в корне проекта**: Нарушают чистоту репозитория
4. **Отладочные логи**: Логи тестирования в основной директории

### Рекомендации по реорганизации:
1. **Очистить основную директорию** от кэшей и отладочных файлов
2. **Создать отдельную папку для отладки** (`debug/`)
3. **Организовать кэши** в специальную папку (`cache/`)
4. **Переместить тестовые скрипты** в `tests/` или `scripts/`
5. **Очистить старые логи** и временные файлы

## 6. ПЛАН РЕОРГАНИЗАЦИИ

### Этап 1: Анализ и классификация
- [x] Проанализировать структуру проекта
- [x] Классифицировать файлы по категориям
- [x] Выявить основные зависимости

### Этап 2: Создание новой структуры
- [ ] Создать папку `cogniflex_clean/` для чистого кода
- [ ] Создать папку `debug/` для отладочных файлов
- [ ] Создать папку `cache/` для всех кэшей
- [ ] Создать папку `logs/` для логов

### Этап 3: Перемещение файлов
- [ ] Переместить функциональный код в `cogniflex_clean/`
- [ ] Переместить отладочные файлы в `debug/`
- [ ] Переместить кэши в `cache/`
- [ ] Переместить логи в `logs/`

### Этап 4: Тестирование
- [ ] Проверить работоспособность после реорганизации
- [ ] Обновить пути импортов
- [ ] Протестировать запуск системы

## 7. ТЕКУЩИЙ СТАТУС
- Анализ структуры: ЗАВЕРШЕН
- Классификация файлов: ЗАВЕРШЕНА
- Создание лог-файла: ЗАВЕРШЕНО
- Ожидание дальнейших инструкций по реорганизации

---
*Лог-файл создан Cascade AI Assistant для проекта CogniFlex*
*Время создания: 2025-08-29T12:13:12+03:00*
