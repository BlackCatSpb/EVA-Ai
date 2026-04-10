# Deprecated Modules

Эта папка содержит модули, которые были заменены новой архитектурой или не используются в текущей системе.

## Целевая архитектура (активная):
```
eva_ai/
├── core/                    # Ядро системы (CoreBrain)
│   ├── brain_query.py      # Обработка запросов
│   ├── hybrid_pipeline_adapter.py  # Two-Model Pipeline (DualGenerator)
│   └── ...
├── memory/
│   └── fractal_graph_v2/   # FractalGraph V2 - основная память
│       ├── dual_generator.py   # DualGenerator (2 модели)
│       ├── storage.py
│       └── ...
├── websearch/              # Tavily веб-поиск
├── contradiction/           # Управление противоречиями
├── learning/               # Обучение (SelfDialogLearning)
├── gui/                    # Веб-интерфейс
└── generation/             # Генерация ответов
```

## Что уже перенесено сюда:

### 2026-04-10

#### knowledge_old/ (50 файлов)
Заменены на FractalGraph v2:
- concept_miner.py - старый концепт-майнер
- knowledge_graph*.py - старый Knowledge Graph
- graph_*.py - старая графовая система
- query_*.py - старая система запросов
- и другие...

#### Тестовые файлы
- memory_fractal_graph_v2_test_generation.py - тестовый генератор
- scripts_test_*.py - тестовые скрипты (4 файла)
- hot_deployment_test_*.py - тесты hot deployment (2 файла)

**Всего:** 56+ файлов перенесено в deprecated

## Критерии для переноса в deprecated:
1. Модуль не импортируется из active кодом
2. Функциональность заменена новым модулем
3. Модуль вызывает ошибки импорта из-за отсутствующих зависимостей
4. Код не обновлялся более 6 месяцев и не используется

## Как перенести модуль:
```bash
mv eva_ai/old_module deprecated_modules/
```

## Важно:
- НЕ удалять сразу - сначала перенести
- Дождаться стабильной работы системы
- Проверить что ничего не сломалось
