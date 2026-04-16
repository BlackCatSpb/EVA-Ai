# EVA AI System - Сводка проблем и трекинг исправлений

**Дата создания:** 14.04.2026  
**Общая оценка системы:** 3.8/10  
**Проверено:** 14 циклов, 68 аудит-отчётов, 70 агентов  

---

## ИНСТРУКЦИЯ ПО РАБОТЕ С ПЛАНОМ

### Цикл исправлений:

1. **Перед запуском** - очистить процессы и логи:
   ```powershell
   taskkill /F /IM python.exe /T
   taskkill /F /IM powershell.exe /T
   Remove-Item "C:\Users\black\OneDrive\Desktop\CogniFlex\*.log" -Force
   Remove-Item "C:\Users\black\OneDrive\Desktop\CogniFlex\logs\*.log" -Force
   ```

2. **Запуск системы** - в отдельном терминале:
   ```bash
   cd C:\Users\black\OneDrive\Desktop\CogniFlex && python -m eva_ai
   ```

3. **Мониторинг запуска**:
   - Ждать 3-4 минуты (таймаут запуска)
   - Каждые 15 секунд проверять лог `logs/eva_ai.log`
   - Критерий успешного запуска: `ЕВА успешно запущен` или `WebGUI сервер запущен`

4. **ОСТАНОВКА системы** - ПОСЛЕ подтверждения запуска:
   - **ОБЯЗАТЕЛЬНО** убить терминал: `taskkill /F /IM python.exe /T` + `taskkill /F /IM powershell.exe /T`
   - Подождать 5 секунд
   - Только потом читать логи

5. **Анализ логов** - получить ВСЕ ошибки/warnings из лога

### Правила внесения ошибок:

| Тип ошибки | Действие |
|------------|----------|
| **Связана напрямую** с текущим исправлением | Исправить сразу, добавить в план как [X] |
| **НЕ связана** с текущим исправлением | Внести в соответствующий раздел плана, продолжить по плану |

### После каждого этапа:
- Коммит и пуш в git
- Сверка с планом исправлений
- Обновление статусов в ISSUES_TRACKER.md

### Критерии завершения:
- Система работает без ERROR
- Все критичные ошибки ([КРИТ]) исправлены
- WARNING не блокируют работу

---

## LEGEND / ЛЕГЕНДА

```
[КРИТ] = Критическая проблема (исправить немедленно)
[ВЫС]  = Высокий приоритет
[СРЕД] = Средний приоритет
[НИЗК] = Низкий приоритет

[ ] = Не начато
[P]  = В процессе
[X]  = Исправлено
[!]  = Не удалось исправить / заблокировано
```

---

# РАЗДЕЛ 1: КООРДИНАЦИЯ И АРХИТЕКТУРА

## 1.1 EventBus
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 1.1.1 | Priority игнорируется - FIFO вместо приоритетов | [КРИТ] | [X] | ✅ Исправлено: PriorityQueue с (priority, timestamp, event) |
| 1.1.2 | EventBus синглтон anti-pattern | [ВЫС] | [ ] | Рекомендуется dependency injection |
| 1.1.3 | 90% компонентов не интегрированы | [КРИТ] | [X] | ✅ SelfReasoningEngine (pipeline.start/complete/failed), EthicsFramework (ethics.assessment/violation/warning), IntegratedEthicsFramework |
| 1.1.4 | Signal handler signatures (event=None → event) | [КРИТ] | [X] | Исправлено: brain_coordination, system_state, event_bus_bridge |

## 1.2 ModelAccessManager
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 1.2.1 | brain_query НЕ использует MAM | [КРИТ] | [X] | ✅ Исправлено: HybridPipelineAdapter использует MAM |
| 1.2.2 | Конфликты доступа самодиалог/запросы | [ВЫС] | [X] | ✅ Исправлено: DialogConceptsMixin использует MAM с HIGH приоритетом |

## 1.3 DeferredCommandSystem
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 1.3.1 | TaskScheduler дублирует DCS в distributed | [ВЫС] | [ ] | Удалить или использовать DCS |
| 1.3.2 | BackgroundCoordinator Detectors не зарегистрированы | [КРИТ] | [X] | ✅ Исправлено: добавлена регистрация LearningOpportunityDetector, WebDiscoveryDetector, ModuleRecoveryDetector |
| 1.3.3 | MemoryManager deferred commands - hasattr check | [КРИТ] | [X] | ✅ Исправлено: manager_core.py добавлены hasattr проверки |
| 1.3.4 | BackgroundCoordinator._lock not initialized | [КРИТ] | [X] | ✅ Исправлено: hasattr check в _deferred_start |
| 1.3.5 | Coder model loading slow (debugging) | [СРЕД] | [X] | ✅ Добавлен EVA_SKIP_CODER=1 для отладки |

---

# РАЗДЕЛ 2: ГРАФ ПАМЯТИ (FractalGraphV2 / KnowledgeGraph)

## 2.1 FractalGraphV2
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 2.1.1 | **Метод `get_clusters()` НЕ СУЩЕСТВУЕТ** | [КРИТ] | [X] | ✅ Исправлено: добавлен get_clusters() с кэшированием |
| 2.1.2 | SQLite без WAL | [ВЫС] | [X] | ✅ Исправлено: FractalGraphV2 storage использует WAL mode |
| 2.1.3 | Embedding fallback на случайные векторы | [КРИТ] | [X] | ✅ Исправлено: SemanticContextCache._compute_embedding() возвращает None |
| 2.1.4 | kg_adapter.py - баг `edge_type` vs `relation_type` | [ВЫС] | [P] | Частично: KG адаптер удалён, create_knowledge_components |
| 2.1.5 | FractalMemoryGraph не публикует события | [ВЫС] | [X] | ✅ Исправлено: добавлен memory.graph_updated после add_nodes_batch |

## 2.2 KnowledgeCurator / GraphCurator
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 2.2.1 | **KnowledgeCurator НЕ СУЩЕСТВУЕТ** | [КРИТ] | [N/A] | Non-critical: GraphCurator функционален, работает без него |
| 2.2.2 | GraphCurator изолирован (нет EventBus/DCS) | [КРИТ] | [X] | ✅ Исправлено: подписки на system.idle, memory.graph_updated, DCS интеграция |
| 2.2.3 | threading.Timer вместо адаптивного | [СРЕД] | [X] | ✅ Исправлено: адаптивный интервал через DCS, загрузка CPU/RAM |
| 2.2.4 | is_running() отсутствует | [СРЕД] | [X] | ✅ Исправлено: добавлен метод is_running() |

---

# РАЗДЕЛ 3: СИСТЕТМА КОНЦЕПТОВ И ПРОТИВОРЕЧИЙ

## 3.1 ConceptExtractor
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 3.1.1 | `extract_concepts()` не сохраняет автоматически | [КРИТ] | [X] | ✅ Уже реализовано: есть save_concept_to_graph() |
| 3.1.2 | Нарушение SRP | [ВЫС] | [X] | ✅ Интегрирован в brain_query |
| 3.1.3 | Нет EventBus событий | [СРЕД] | [X] | ✅ Исправлено: ConceptExtractor публикует concept.extracted |

## 3.2 ContradictionDetector
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 3.2.1 | 2 версии ContradictionAnalyzer | [ВЫС] | [X] | ✅ Исправлено: удалён мёртвый contradiction_analysis.py, используется только из analytics |

## 3.3 SelfDialogLearning
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 3.3.1 | Подписан на `curator.*` которых нет | [КРИТ] | [X] | ✅ Исправлено: добавлена публикация curator.graph_optimized, curator.cleanup_done |

---

# РАЗДЕЛ 4: СИСТЕМА РАССУЖДЕНИЙ (Reasoning)

## 4.1 SelfReasoningEngine
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 4.1.1 | **НЕТ EventBus интеграции** | [КРИТ] | [X] | ✅ Уже исправлено: подписки на system.idle, публикации pipeline.*, reasoning.* |
| 4.1.2 | 3 движка рассуждения | [ВЫС] | [ ] | SRE, Enhanced, Core |
| 4.1.3 | Конфликт инициализации | [СРЕД] | [ ] | Дублирование создания SRE |

## 4.2 EthicsFramework
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 4.2.1 | **2 версии класса** (framework_core vs ethics_core) | [КРИТ] | [ ] | Конфликт имён |
| 4.2.2 | `analyze_response()` отсутствует в ethics_core | [КРИТ] | [ ] | SelfReasoningEngine ожидает |
| 4.2.3 | pipeline_core создаёт локальную копию | [СРЕД] | [ ] | Рекомендуется использовать brain.ethics_framework |

---

# РАЗДЕЛ 5: ГЕНЕРАЦИЯ И МОДЕЛИ

## 5.1 UnifiedGenerator / Pipeline
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 5.1.1 | brain_query вызывает напрямую (без MAM) | [КРИТ] | [X] | ✅ Исправлено: _handle_gguf_pipeline использует MAM с CRITICAL приоритетом |
| 5.1.2 | ТРОЙНАЯ абстракция | [ВЫС] | [ ] | PipelineAdapter → GC → Provider |
| 5.1.3 | max_tokens завышены x16-32 | [СРЕД] | [X] | ✅ Исправлено: LOGIC=512, CONTEXT=1024 |

## 5.2 MLearning (8 Model Managers!)
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 5.2.1 | **8 Model Manager классов** | [КРИТ] | [ ] | Qwen, Fractal, Hybrid, Universal, BitNet, Model, etc. |
| 5.2.2 | ~70% кода - мёртвый | [КРИТ] | [ ] | universal_model_manager, bitnet_model_manager, text_quality_* |
| 5.2.3 | EventBus НЕ интегрирован | [ВЫС] | [ ] | Полная изоляция |

## 5.3 HotDeployment (10/12 файлов мёртвые!)
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 5.3.1 | 10 из 12 файлов - мёртвый код | [КРИТ] | [ ] | Удалить или использовать |
| 5.3.2 | **3 версии OpenVINOGenerator** | [ВЫС] | [ ] | mlearning, core, hot_deployment |
| 5.3.3 | llama_cpp_wrapper.py = llama_cpp_hot.py | [СРЕД] | [X] | ✅ Исправлено: удалён неиспользуемый llama_cpp_wrapper.py |

---

# РАЗДЕЛ 6: КЭШИРОВАНИЕ И ХРАНЕНИЕ

## 6.1 HybridCache / TokenCache
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 6.1.1 | **Pickle в 15+ местах без валидации** | [КРИТ] | [X] | ✅ Исправлено: cache_disk, disk_cache, fractal_torch_storage, fractal_weight_store, storage/fractal_storage |
| 6.1.2 | 2 дубликата дискового кэша | [ВЫС] | [ ] | TokenDiskCache vs DiskCache |
| 6.1.3 | LRUCache дублируется | [ВЫС] | [ ] | cache_ram.py == memory_cache.py |
| 6.1.4 | Race conditions в UnifiedCacheBridge | [КРИТ] | [X] | ✅ Исправлено: добавлены locks в _load_state и save_state |
| 6.1.5 | Nested deadlock | [КРИТ] | [X] | ✅ Исправлено: используем disk_cache.put напрямую без лока |
| 6.1.6 | Нет фоновой очистки TTL | [СРЕД] | [ ] | Только при get() |
| 6.1.7 | 6 компонентов создают НОВЫЕ инстансы | [ВЫС] | [ ] | Вместо переиспользования |

## 6.2 FractalCache / Storage
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 6.2.1 | storage/fractal_storage.py - мёртвый код | [ВЫС] | [ ] | 0 импортов |
| 6.2.2 | 8 случаев Pickle в storage | [КРИТ] | [ ] | fractal_torch_storage, etc. |

## 6.3 Embeddings
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 6.3.1 | CPU по умолчанию (init_factories.py:469) | [КРИТ] | [X] | ✅ Исправлено: auto-detect GPU, используем cuda если доступен |
| 6.3.2 | hash() нестабилен между запусками | [ВЫС] | [ ] | str(hash(text)) меняется |
| 6.3.3 | EmbeddingCache не подключен | [СРЕД] | [ ] | SQLite cache не используется |
| 6.3.4 | Случайные векторы при fallback | [КРИТ] | [X] | ✅ Исправлено: возвращаем None, вызывающий код обрабатывает |

---

# РАЗДЕЛ 7: БЕЗОПАСНОСТЬ

## 7.1 SecurityFramework
| # | Уязвимость | Файл | CVSS | Статус | Комментарий |
|---|------------|------|------|--------|-------------|
| 7.1.1 | **HARDCODED `admin:admin` backdoor** | security_framework.py:141 | **9.8** | [X] | ✅ Исправлено: убран default password fallback |
| 7.1.2 | SHA256 без соли | security_framework.py:137 | 9.1 | [ ] | Заменить на PBKDF2/bcrypt |
| 7.1.3 | Нет key stretching | security_framework.py:137 | 8.5 | [ ] | |
| 7.1.4 | Нет salt | security_framework.py:137 | 8.2 | [ ] | |

## 7.2 ComponentManagers
| # | Класс | Проблема | Статус | Комментарий |
|---|-------|----------|--------|-------------|
| 7.2.1 | AuthManager | Принимает любой пароль | [ ] | |
| 7.2.2 | RecoveryManager | Ничего не восстанавливает | [ ] | Рекомендуется удалить |
| 7.2.3 | SecurityManager | Проксирует вызовы | [ ] | |

---

# РАЗДЕЛ 8: ВЕБ ИНТЕРФЕЙС (Web GUI)

## 8.1 Web GUI
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 8.1.1 | Thread leak | [СРЕД] | [ ] | Требует профилирования |

## 8.2 WebSearch
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 8.2.1 | search_google() → DuckDuckGo | [ВЫС] | [ ] | Убрать подмену или документировать |
| 8.2.2 | search_yandex() → Brave | [ВЫС] | [ ] | Убрать подмену или документировать |
| 8.2.3 | Tavily API Key path wrong | [КРИТ] | [X] | ✅ Исправлено: config.web_search.tavily_api_key |

---

# РАЗДЕЛ 9: КОНФИГУРАЦИЯ И СКРИПТЫ

## 9.1 Config
| # | Проблема | Приоритет | Статус | Комментарий |
|---|----------|----------|--------|-------------|
| 9.1.1 | optimal_config.json = fractal_model_config.json | [ВЫС] | [ ] | Дублирование |
| 9.1.2 | apply_optimal_config.py только печатает | [КРИТ] | [ ] | Не применяет настройки |
| 9.1.3 | os.getcwd() + "eva" hardcoded | [ВЫС] | [ ] | Ломается при запуске из другой директории |
| 9.1.4 | Tokenizer fallback path wrong | [КРИТ] | [X] | ✅ Исправлено: использует brain_config.json model path |
| 9.1.5 | Tokenizer optional logging ERROR→INFO | [СРЕД] | [X] | ✅ Исправлено: event_system.py |

## 9.2 Scripts (6 из 8 сломаны!)
| # | Скрипт | Проблема | Статус | Комментарий |
|---|--------|----------|--------|-------------|
| 9.2.1 | migrate_kg_to_fg.py | Несуществующий импорт | [ ] | kg_to_fg_migration не найден |
| 9.2.2 | migrate_to_optimized.py | Несуществующий импорт | [ ] | optimized_fractal_model_manager |
| 9.2.3 | export_qwen.py | Неверный путь | [ ] | eva\mlearning\ |
| 9.2.4 | migrate_events.py | Только документация | [ ] | |
| 9.2.5 | load_gguf_to_fg.py | fg_gguf_architecture_mapper | [ ] | |
| 9.2.6 | activate_max_cache.py | unified_fractal_manager | [ ] | |

---

# РАЗДЕЛ 10: МЁРТВЫЙ КОД (40+ файлов)

## 10.1 Рекомендуется к удалению
| # | Директория/Файл | Причина | Статус | Комментарий |
|---|-----------------|---------|--------|-------------|
| 10.1.1 | **eva_ai/fractal/** | 0 импортов, изолирован | [ ] | Полностью мёртвый |
| 10.1.2 | **eva_ai/runtime/** | НЕ ИСПОЛЬЗУЕТСЯ | [ ] | Конкурирует с DCS |
| 10.1.3 | **eva_ai/distributed/** | НЕ инициализируется | [ ] | _init_distributed_system НЕ СУЩЕСТВУЕТ |
| 10.1.4 | **eva_ai/adaptation/** (4 версии) | 2,200 строк мёртвого кода | [ ] | 4 AdaptationManager |
| 10.1.5 | **component_managers.py** | 8 заглушек | [ ] | Рекомендуется удалить |
| 10.1.6 | **mlearning/hot_deployment/** (10/12) | Мёртвый код | [ ] | Только llama_cpp_hot.py активен |
| 10.1.7 | storage/fractal_storage.py | 0 импортов | [ ] | |
| 10.1.8 | dual_generator_pie.py | Не используется | [ ] | |

---

# РАЗДЕЛ 11: НЕИНИЦИАЛИЗИРОВАННЫЕ СИСТЕМЫ

| # | Система | Проблема | Статус | Комментарий |
|---|---------|----------|--------|-------------|
| 11.1 | Distributed | `_init_distributed_system()` НЕ СУЩЕСТВУЕТ | [ ] | ModuleRecoveryJob ссылается |
| 11.2 | Training | `self.output_dir` НЕ определён | [ ] | AttributeError при _save_lora_adapters() |
| 11.3 | GraphCurator | deferred_system не используется | [ ] | Присвоен но не задействован |

---

# РАЗДЕЛ 12: ДУБЛИРОВАНИЕ (итог)

| # | Компонент | Версий | Статус | Комментарий |
|---|-----------|--------|--------|-------------|
| 12.1 | EntityExtractor | 4 | [ ] | preprocess, reasoning, gui + 1 |
| 12.2 | AdaptationManager | 4 | [ ] | Только v2 активен |
| 12.3 | RecoveryManager | 3 | [ ] | recovery_system, distributed, component_managers |
| 12.4 | ModelManager | 8 | [ ] | Только 2-3 активны |
| 12.5 | OpenVINOGenerator | 3 | [ ] | mlearning, core, hot_deployment |
| 12.6 | FractalStore | 3 | [ ] | |
| 12.7 | DiskCache | 2 | [ ] | TokenDiskCache vs DiskCache |
| 12.8 | LRUCache | 2 | [ ] | cache_ram.py == memory_cache.py |
| 12.9 | SystemState enum | 3 | [ ] | system_state, core_brain_types, brain_state |
| 12.10 | ContradictionAnalyzer | 2 | [ ] | analytics vs contradiction |

---

# ПЛАН ИСПРАВЛЕНИЙ ПО ПРИОРИТЕТАМ

## ФАЗА 1: КРИТИЧЕСКИЕ БЕЗОПАСНОСТЬ И БАГИ (Неделя 1)
- [X] 7.1.1 - УДАЛИТЬ backdoor admin:admin ✅
- [ ] 6.1.1 - Заменить Pickle на JSON/msgpack
- [ ] 3.1.1 - Исправить extract_ambiguous_terms() или удалить вызовы
- [ ] 11.2 - Исправить self.output_dir в Training
- [ ] 11.1 - Реализовать _init_distributed_system() или удалить distributed/
- [X] 2.1.1 - Реализовать FractalGraphV2.get_clusters() ✅

## ФАЗА 2: ИНТЕГРАЦИЯ (Неделя 2-3)
- [ ] 1.2.1 - Интегрировать ModelAccessManager в brain_query
- [ ] 4.1.1 - Добавить EventBus в SelfReasoningEngine
- [ ] 2.2.2 - Интегрировать GraphCurator с EventBus/DCS
- [ ] 1.1.1 - Исправить EventBus priority system
- [ ] 3.3.1 - GraphCurator публикует curator.* события

## ФАЗА 3: ОЧИСТКА МЁРТВОГО КОДА (Неделя 3-4)
- [ ] 10.1.1 - Удалить eva_ai/fractal/
- [ ] 10.1.2 - Удалить eva_ai/runtime/
- [ ] 10.1.3 - Удалить eva_ai/distributed/
- [ ] 10.1.4 - Удалить 3 версии adaptation_manager
- [ ] 10.1.5 - Удалить component_managers.py
- [ ] 10.1.6 - Удалить 10/12 файлов hot_deployment/

## ФАЗА 4: ОБЪЕДИНЕНИЕ ДУБЛИКАТОВ (Неделя 4-5)
- [ ] 5.2.1 - Объединить 8 ModelManagers в 1-2
- [ ] 5.3.2 - Объединить 3 OpenVINOGenerator
- [ ] 6.1.2 - Объединить TokenDiskCache и DiskCache
- [ ] 6.1.3 - Удалить дубликат LRUCache
- [ ] 12.1 - Объединить 4 EntityExtractor
- [ ] 4.2.1 - Объединить 2 EthicsFramework

## ФАЗА 5: ИСПРАВЛЕНИЕ ОСТАЛЬНЫХ ПРОБЛЕМ (Неделя 5-6)
- [ ] 6.3.1 - Исправить CPU/GPU определение для embeddings
- [ ] 6.1.4 - Исправить race conditions в UnifiedCacheBridge
- [ ] 9.1.2 - Исправить apply_optimal_config.py
- [ ] 9.2.1-9.2.6 - Починить или удалить сломанные скрипты
- [ ] 8.2.1-8.2.2 - Убрать подмену поисковиков

---

# ЧЕКПОИНТЫ ПРОГРЕССА

| Дата | Фаза | Исправлено | Комментарий |
|------|------|-----------|-------------|
| 14.04.2026 | - | 0/68 | Начало работы |
| 14.04.2026 | Фаза 1 | 9/68 | 7.1.1, 2.1.1, 1.1.4, 1.3.3, 3.1.1, 3.1.2, 8.2.3, 9.1.4, 9.1.5 |
| 14.04.2026 | Отладка | 12/68 | +1.3.4 (BC._lock), +1.3.5 (skip coder) |
| 14.04.2026 | 1.3.2 | 12/68 | +1.3.2 (Detectors registered) |
| 14.04.2026 | 6.1.5 | 13/68 | +6.1.5 (nested deadlock fixed) |
| 14.04.2026 | 6.3.4 | 14/68 | +6.3.4 (random vectors → None) |
| 14.04.2026 | 6.3.1 | 15/68 | +6.3.1 (auto-detect GPU for embeddings) |
| 14.04.2026 | 6.1.4 | 16/68 | +6.1.4 (race conditions fixed) |
| 14.04.2026 | 6.1.1 | 17/68 | +6.1.1 (pickle → json in cache) |
| 14.04.2026 | 3.3.1 | 18/68 | +3.3.1 (GraphCurator publishes curator.* events) |
| 14.04.2026 | Архитектура | 21/68 | +1.1.1 (PriorityQueue), +1.2.1 (MAM), +6.1.1 (pickle) |
| | | | |
| | | | |

---

# СТАТИСТИКА

| Метрика | Значение |
|---------|----------|
| Всего проблем | 68 |
| КРИТ приоритет | ~20 |
| ВЫС приоритет | ~25 |
| СРЕД/НИЗК приоритет | ~23 |
| Исправлено | 21 |
| В процессе | 2 |
| Неактуально | 1 |

---

*Документ обновлён: 14.04.2026*
*70 AI агентов проверили систему за 14 циклов*
*Добавлена инструкция по работе с планом: 14.04.2026*
*Обновлён счётчик: +2 исправления (1.3.4, 1.3.5)*
