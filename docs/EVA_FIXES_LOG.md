# CogniFlex - Блокнот исправлений и проблем

## Дата создания: 2026-04-10

---

## АРХИТЕКТУРА СИСТЕМЫ

### Основные компоненты
- **CoreBrain** - ядро системы, координирует все компоненты
- **FractalGraphV2** - основное хранилище памяти (519 узлов, 362 связи)
- **DualGenerator** - генерация ответов (краткий/развёрнутый)
- **SelfDialogLearning** - самодиалог для обучения
- **ConceptMiner** - поиск концептов
- **ContradictionManager** - управление противоречиями
- **ResourceManager** - мониторинг ресурсов

### Текущие модули brain
```
query_processor, background_coordinator, mode_controller, event_bus,
resource_manager, config_manager, memory_manager, hybrid_cache,
fractal_graph_v2, text_processor, ml_unit, model_manager,
response_generator, reasoning_engine, system_monitor, metrics_collector,
analytics_manager, contradiction_manager, ethics_framework,
adaptation_manager, web_search_engine, fractal_storage,
self_reasoning_engine, enhanced_reasoning_engine,
generation_coordinator, reasoning_integration
```

---

## ИЗВЕСТНЫЕ ПРОБЛЕМЫ

### 1. CPU/RAM метрики = 0
**Статус**: 🔴 ТРЕБУЕТ ПЕРЕЗАПУСКА  
**Причина**: Сервер запущен со старым кодом, изменения не применены  
**Исправление**: Добавлен прямой доступ к resource_manager + psutil fallback в server_routes.py

### 2. Concepts = {} (пустой объект)
**Статус**: 🔴 ТРЕБУЕТ ПЕРЕЗАПУСКА  
**Причина**: ConceptMiner в knowledge_old/, не найден импорт  
**Исправление**: Создан адаптер eva_ai/knowledge/concept_miner.py

### 3. SelfDialogLearning не инициализирован
**Статус**: 🔴 ТРЕБУЕТ ПЕРЕЗАПУСКА  
**Причина**: brain.self_dialog_learning = None  
**Проверка**: start_webgui.py строки 129-131 - создаётся если SelfDialogLearningSystem доступен

### 4. ContradictionManager operations_count = 0
**Статус**: 🟡 ТРЕБУЕТ ПРОВЕРКИ  
**Причина**: Система работает, но не обрабатывает данные активно  
**Решение**: Проверить после запуска

### 5. ConceptMiner dry_run: True
**Статус**: 🟡 ТРЕБУЕТ ПРОВЕРКИ  
**Причина**: Конфигурация по умолчанию в knowledge_old/concept_miner.py  
**Решение**: Изменить конфиг или код

### 6. Tavily счётчики = 0
**Статус**: 🟡 ТРЕБУЕТ ПРОВЕРКИ  
**Причина**: Нужно протестировать веб-поиск  
**Решение**: После исправления метрик протестировать

---

## ВНЕСЁННЫЕ ИСПРАВЛЕНИЯ

### 2026-04-10

#### 1. Адаптер ConceptMiner
**Файл**: `eva_ai/knowledge/concept_miner.py`  
**Коммит**: Новый файл  
**Описание**: Адаптер для импорта из knowledge_old/ для обратной совместимости

#### 2. ResourceManager direct access
**Файл**: `eva_ai/gui/web_gui/server_routes.py`  
**Коммит**: 6e31b53e  
**Описание**: Добавлен прямой доступ к resource_manager.get_cpu_usage(), get_memory_usage(), get_current_metrics()
**Fallback**: psutil.cpu_percent(), psutil.virtual_memory() если resource_manager не работает

#### 3. Concepts из FGv2
**Файл**: `eva_ai/gui/web_gui/server_routes.py`  
**Коммит**: 6e31b53e  
**Описание**: Если concept_miner не найден, берём данные из fractal_graph_v2.nodes_by_type

#### 4. GraphCurator health check fix
**Коммит**: c6c01bbe  
**Описание**: gc.get_state() возвращает строку, а не dict

#### 5. websearch_stats endpoint
**Коммит**: c6c01bbe  
**Описание**: Endpoint определён вне функции register_routes

#### 6. FractalGraphV2 storage path
**Коммит**: 44071375  
**Описание**: Исправлен путь на `fractal_graph_v2_data`

#### 7. Stop tokens и очистка ответов
**Коммит**: 6145f68c  
**Описание**: Добавлены stop tokens, улучшена очистка от "Модель B:"

#### 8. Динамический контекст
**Коммит**: 01874a36  
**Описание**: До 50 узлов графа для развёрнутых ответов

---

## TODO ЛИСТ

### После перезапуска сервера:
- [x] Проверить /api/metrics - cpu_usage: 37.0%, memory_usage: 79.7% ✅ РАБОТАЕТ
- [x] Проверить /api/analytics - concepts: 32 (concept_nodes: 30, aci_concepts: 2) ✅ РАБОТАЕТ
- [ ] Протестировать Tavily поиск - сделать запрос и проверить счётчики
- [ ] SelfDialogLearning - initialized в brain, но dialogs=0 (нужны запросы)
- [ ] ContradictionManager - state=running, но operations_count=0 (система idle)

### Результаты тестирования (2026-04-10 10:43):
| Метрика | Значение | Статус |
|---------|----------|--------|
| cpu_usage | 37.0% | ✅ |
| memory_usage | 79.7% | ✅ |
| disk_usage | 51.6% | ✅ |
| concepts | 32 | ✅ |
| contradictions.state | running | ✅ |
| fractal_nodes | 519 | ✅ |
| curator_state | running ✅ (циклы растут: 2→3) |
| tavily_requests | 0 | ⚠️ не триггерится |
| dialogs | 0 | ⚠️ не работает |
| queries | 0 | ⚠️ analytics не получает |
| web_searches | 0 | ⚠️ не триггерится |

### Проблемы для исправления:

#### 1. Tavily API key НЕВАЛИДНЫЙ
**Файл**: `brain_config.json`
**Проблема**: Tavily возвращает 401 Unauthorized
**Статус**: Нужен валидный API key

#### 2. queries=0 - нет трекинга
**Файл**: `eva_ai/gui/web_gui/server_routes.py`
**Проблема**: analytics не получает данные о запросах из brain
**Решение**: Добавить ProcessTrackerMixin данные в analytics

#### 3. dialogs=0
**Файл**: `eva_ai/learning/dialog_core.py`
**Проблема**: SelfDialogLearning.stats обнуляется, нет обновления счётчиков
**Решение**: Интегрировать с brain events для обновления stats

#### 4. contradictions=0
**Проблема**: operations_count=0, система idle
**Решение**: Проверить работу ContradictionManager

#### 5. GPU/VRAM метрики
**Проблема**: vram всегда 0, GPU не отображается
**Решение**: Добавить GPU мониторинг

### Долгосрочные улучшения:
- [ ] Изменить ConceptMiner dry_run: True на False
- [ ] Интегрировать Graph Curator с Concept Miner
- [ ] Удалить knowledge_old/ после полного тестирования
- [ ] Удалить JSON хранилища memory_manager (перейти на FGv2)

---

## КОМАНДЫ ЗАПУСКА

```bash
# Запуск сервера
python -m eva_ai

# Или
python start_webgui.py

# Проверка API
curl http://127.0.0.1:5555/api/metrics
curl http://127.0.0.1:5555/api/analytics
curl http://127.0.0.1:5555/api/system

# Логи
Get-Content -Tail 100 logs/cogniflex.log
```

---

## КОНФИГУРАЦИЯ

### brain_config.json ключевые параметры:
```json
{
  "fractal_graph_v2": {
    "enabled": true,
    "max_context_nodes": 50,
    "max_context_chars": 20000,
    "storage_dir": "fractal_graph_v2_data"
  },
  "concept_miner": {
    "enabled": true,
    "dry_run": true
  },
  "tavily_api_key": "tvly-dev-..."
}
```

---

## ENDPOINTS API

| Endpoint | Метод | Данные |
|----------|-------|--------|
| `/api/metrics` | GET | cpu_usage, memory_usage, graph, contradictions, concepts, health |
| `/api/analytics` | GET | queries, avg_time, cpu, memory, fractal_nodes, curator, tavily |
| `/api/system` | GET | version, modules, features |
| `/api/status` | GET | sessions_count, brain_connected |
| `/api/websearch_stats` | GET | tavily_requests, searches_performed, cache_hits |
| `/api/chat` | POST | query → response |

---

## ЖУРНАЛ ИЗМЕНЕНИЙ

| Дата | Время | Действие | Результат |
|------|-------|----------|-----------|
| 2026-04-10 | 09:00 | Запуск агентов анализа | Собран полный анализ системы |
| 2026-04-10 | 09:30 | Создание md блокнота | Файл создан |
| 2026-04-10 | 10:00 | Запуск сервера | (ожидает) |
| 2026-04-10 | 10:43 | Перезапуск сервера | CPU/RAM/Concepts работают! |
| 2026-04-10 | 10:45 | Тест Tavily | API key невалидный (401) |
| 2026-04-10 | 11:00 | Исправления | ProcessTrackerMixin в analytics, Tavily error tracking |
| 2026-04-10 | 11:05 | Пуш | Коммит 0c86a042 |

## ИСПРАВЛЕНО:

1. ✅ **Tavily** - теперь работает! (tavily_requests: 1, tavily_responses: 1)
2. ✅ **Ctrl+C** - работает корректно (werkzeug make_server)
3. ✅ **WebSearch инициализация** - принудительная инициализация в create function

## ТЕКУЩЕЕ СОСТОЯНИЕ (2026-04-10 12:25):

### ✅ Работает:
| Метрика | Значение | Статус |
|---------|----------|--------|
| CPU | 82.1% | ✅ |
| Tavily requests | 1 | ✅ |
| Tavily responses | 1 | ✅ |
| web_searches | 1 | ✅ |
| results_found | 5 | ✅ |
| fractal_nodes | 519 | ✅ |
| concepts | 32 | ✅ |
| curator_state | running | ✅ |

### ⚠️ Не работает:
| Метрика | Проблема |
|---------|----------|
| queries | 0 - ProcessTrackerMixin не читается |
| dialogs | 0 - SelfDialogLearning не отслеживается |
| avg_time | 0 |
| success_rate | 0 |
| vram | 0 |

## ОСТАЛОСЬ:

1. ✅ **queries, avg_time, success_rate** - добавлен трекинг в process_query
2. ✅ **GPU/VRAM метрики** - добавлен get_gpu_metrics() в resource_manager  
3. ⏳ **Dialogs** - SelfDialogLearning инициализирован, но диалоги не создаются (требует глубокого анализа)

## ГОТОВО К ПРОВЕРКЕ (после перезапуска):

| Метрика | Ожидаемый результат |
|---------|---------------------|
| queries | > 0 при запросах |
| avg_time | > 0 при запросах |
| success_rate | % успешных запросов |
| vram | GPU memory % |

## ОЧИСТКА КОДА (2026-04-10):

### ✅ Перенесено в deprecated_modules:
| Папка/Файл | Количество | Причина |
|------------|------------|---------|
| knowledge_old/ | 50 файлов | Заменено FractalGraph v2 |
| test_generation.py | 1 файл | Не используется |
| scripts/test_*.py | 4 файла | Тестовые скрипты |
| hot_deployment/test_*.py | 2 файла | Тестовые файлы |
| adapters/kg_adapter.py | 1 файл | Дубликат |
| **ИТОГО** | **58+ файлов** | **~15,000 строк кода** |

### Что оставлено (активно используется):
- eva_ai/core/ (102 файла) - ядро системы
- eva_ai/memory/ (72 файла) - память (FractalGraph v2)
- eva_ai/mlearning/ (73 файла) - ML компоненты
- eva_ai/websearch/ (9 файлов) - Tavily
- eva_ai/contradiction/ (24 файла) - противоречия
- eva_ai/learning/ (32 файла) - обучение
- eva_ai/gui/ (50 файлов) - веб-интерфейс

## ИСПРАВЛЕНО:

1. ✅ **Ctrl+C** - добавлен shutdown flag, werkzeug make_server
2. ✅ **Tavily** - только Tavily, без fallback, работает в Two-Model Pipeline
3. ✅ **Очистка кода** - 58+ файлов в deprecated_modules
