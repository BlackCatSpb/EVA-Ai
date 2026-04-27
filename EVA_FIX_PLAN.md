# ПЛАН ИСПРАВЛЕНИЙ EVA AI SYSTEM

**Дата создания:** 27 апреля 2026  
**Основано на:** 70 audit-отчётов, ISSUES_TRACKER.md, анализе кодовой базы  
**Всего файлов:** 530 Python файлов, 80 Markdown документов  
**Оценка системы:** 3.8/10

---

## ОБЗОР ПРОБЛЕМ

### Критические проблемы (КРИТ)
1. **Distributed система не инициализируется** - `_init_distributed_system()` не существует
2. **4 версии AdaptationManager** - 2200 строк мёртвого кода
3. **8 Model Manager классов** - только 2-3 используются
4. **EthicsFramework дублирование** - 3 версии класса
5. **EntityExtractor дублирование** - 4 версии в разных модулях
6. **EventBus Priority не работает** - FIFO вместо приоритетов
7. **GraphCurator изолирован** - не использует EventBus/DCS
8. **Pickle без валидации** - 15+ мест с уязвимостью безопасности

### Проблемы высокого приоритета (ВЫС)
1. **10/12 файлов hot_deployment** - мёртвый код
2. **3 версии OpenVINOGenerator** - конфликт имён
3. **TokenDiskCache vs DiskCache** - дублирование
4. **LRUCache дублируется** - cache_ram.py == memory_cache.py
5. **Config скрипты сломаны** - 6 из 8 не работают
6. **WebSearch подмена** - search_google → DuckDuckGo

---

## ФАЗА 1: КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ (Дни 1-3)

### 1.1 Безопасность [КРИТ]
| # | Задача | Файлы | Статус |
|---|--------|-------|--------|
| 1.1.1 | Заменить Pickle на JSON/msgpack | cache_disk.py, disk_cache.py, fractal_torch_storage.py | [ ] |
| 1.1.2 | Удалить hardcoded пароли | security_framework.py | [X] |
| 1.1.3 | Добавить валидацию данных | storage/*.py | [ ] |

### 1.2 Инициализация систем [КРИТ]
| # | Задача | Файлы | Статус |
|---|--------|-------|--------|
| 1.2.1 | Реализовать `_init_distributed_system()` или удалить distributed/ | core/*.py, distributed/ | [ ] |
| 1.2.2 | Исправить `self.output_dir` в Training | training/*.py | [ ] |
| 1.2.3 | Добавить проверку hasattr для всех методов восстановления | module_recovery_job.py | [ ] |

### 1.3 EventBus интеграция [КРИТ]
| # | Задача | Файлы | Статус |
|---|--------|-------|--------|
| 1.3.1 | Исправить PriorityQueue | core/event_system.py | [X] |
| 1.3.2 | Интегрировать SelfReasoningEngine | reasoning/self_reasoning.py | [X] |
| 1.3.3 | Интегрировать GraphCurator | knowledge/graph_curator.py | [X] |
| 1.3.4 | Опубликовать curator.* события | graph_curator.py | [X] |

### 1.4 FractalGraphV2 [КРИТ]
| # | Задача | Файлы | Статус |
|---|--------|-------|--------|
| 1.4.1 | Реализовать `get_clusters()` | memory/fractal_graph_v2.py | [X] |
| 1.4.2 | Исправить SQLite WAL mode | memory/fractal_graph_v2.py | [X] |
| 1.4.3 | Исправить embedding fallback | memory/embeddings_manager.py | [X] |

---

## ФАЗА 2: ОЧИСТКА МЁРТВОГО КОДА (Дни 4-7)

### 2.1 Удаление неиспользуемых модулей
| # | Модуль | Файлы | Строк | Приоритет |
|---|--------|-------|-------|-----------|
| 2.1.1 | eva_ai/fractal/ | 15 файлов | ~2000 | [ВЫС] |
| 2.1.2 | eva_ai/runtime/ | 8 файлов | ~1200 | [ВЫС] |
| 2.1.3 | eva_ai/distributed/ | 10 файлов | ~3200 | [КРИТ] |
| 2.1.4 | adaptation_manager.py (3 версии) | 3 файла | ~2200 | [КРИТ] |
| 2.1.5 | component_managers.py | 1 файл | ~400 | [ВЫС] |
| 2.1.6 | hot_deployment/ (10/12 файлов) | 10 файлов | ~1500 | [ВЫС] |
| 2.1.7 | storage/fractal_storage.py | 1 файл | ~300 | [СРЕД] |
| 2.1.8 | dual_generator_pie.py | 1 файл | ~200 | [НИЗК] |

### 2.2 Проверка перед удалением
```bash
# Для каждого файла:
grep -r "import.*<module>" /workspace/eva_ai --include="*.py"
# Если 0 импортов → безопасно удалять
```

---

## ФАЗА 3: ОБЪЕДИНЕНИЕ ДУБЛИКАТОВ (Дни 8-14)

### 3.1 Model Managers [КРИТ]
**Текущее состояние:** 8 классов
- QwenModelManager
- FractalModelManager
- HybridModelManager
- UniversalModelManager
- BitNetModelManager
- ModelManager
- UnifiedFractalManager
- OptimizedFractalModelManager

**Цель:** 2 класса
- `ModelManager` - базовый менеджер
- `HybridModelManager` - для GGUF + специализированных моделей

**План:**
1. Проанализировать используемые методы в каждом классе
2. Создать единый интерфейс
3. Мигрировать вызовы
4. Удалить дубликаты

### 3.2 EthicsFramework [КРИТ]
**Текущее состояние:** 3 класса
- ethics_core.py: `EthicsFramework`
- framework_core.py: `EthicsFramework` (другой)
- ethics_integrated.py: `IntegratedEthicsFramework`

**Цель:** 1 класс + mixins
- `EthicsFramework` с mixins для расширенной функциональности

### 3.3 EntityExtractor [ВЫС]
**Текущее состояние:** 4 класса
- preprocess/preprocessing_pipeline.py: `GGUFEntityExtractor`
- reasoning/entity_extractor.py: `EntityExtractor`
- gui/web_gui/server_auth.py: `EntityExtractor`
- knowledge/context_entity.py: `EntityExtractor`

**Цель:** 1-2 класса
- `EntityExtractor` - основной
- `GGUFEntityExtractor` - опционально для GGUF моделей

### 3.4 OpenVINOGenerator [ВЫС]
**Текущее состояние:** 3 класса
- mlearning/openvino_generator.py
- core/openvino_generator.py
- hot_deployment/openvino_generator_hot.py

**Цель:** 1 класс

### 3.5 Cache Systems [ВЫС]
| Дубликат | Файлы | Решение |
|----------|-------|---------|
| DiskCache | TokenDiskCache vs DiskCache | Объединить в DiskCache |
| LRUCache | cache_ram.py vs memory_cache.py | Удалить один |
| FractalStore | 3 версии | Оставить одну активную |

---

## ФАЗА 4: ИНТЕГРАЦИЯ КОМПОНЕНТОВ (Дни 15-21)

### 4.1 ModelAccessManager [КРИТ]
**Задача:** Интегрировать MAM во все точки доступа к моделям

| Компонент | Текущее состояние | Требуется |
|-----------|------------------|-----------|
| brain_query | Прямой вызов pipeline | Использовать MAM |
| SelfDialogLearning | Частично | Полная интеграция |
| GenerationCoordinator | Не использует | Интегрировать |

### 4.2 DeferredCommandSystem [ВЫС]
**Задача:** Использовать DCS вместо threading.Timer

| Компонент | Проблема | Решение |
|-----------|----------|---------|
| GraphCurator | threading.Timer | DCS с адаптивным интервалом |
| BackgroundCoordinator | Детекторы не зарегистрированы | Регистрация всех детекторов |
| MemoryManager | Нет hasattr проверок | Добавить проверки |

### 4.3 Monitoring & Health [СРЕД]
**Задача:** Интегрировать monitoring с EventBus

| Проблема | Решение |
|----------|---------|
| Дублирование monitoring/ и system/ | Объединить |
| Нет EventBus | Добавить подписки/публикации |
| Нет auto-start | Добавить в background_jobs |

---

## ФАЗА 5: КОНФИГУРАЦИЯ И СКРИПТЫ (Дни 22-25)

### 5.1 Config файлы [ВЫС]
| Проблема | Файл | Решение |
|----------|------|---------|
| Дублирование конфигов | optimal_config.json, fractal_model_config.json | Объединить |
| apply_optimal_config.py только печатает | config/apply_optimal_config.py | Реализовать применение |
| os.getcwd() hardcoded |多处 | Использовать pathlib |

### 5.2 Скрипты миграции [КРИТ]
| Скрипт | Проблема | Решение |
|--------|----------|---------|
| migrate_kg_to_fg.py | Несуществующий импорт | Починить или удалить |
| migrate_to_optimized.py | Несуществующий импорт | Починить или удалить |
| export_qwen.py | Неверный путь | Исправить путь |
| migrate_events.py | Только документация | Реализовать или удалить |
| load_gguf_to_fg.py | fg_gguf_architecture_mapper | Починить или удалить |
| activate_max_cache.py | unified_fractal_manager | Починить или удалить |

### 5.3 WebSearch [ВЫС]
| Проблема | Файл | Решение |
|----------|------|---------|
| search_google() → DuckDuckGo | websearch/providers.py | Убрать подмену |
| search_yandex() → Brave | websearch/providers.py | Убрать подмену |
| Tavily API Key path | config/*.json | Исправлено ✅ |

---

## ФАЗА 6: ТЕСТИРОВАНИЕ И ВАЛИДАЦИЯ (Дни 26-30)

### 6.1 Unit тесты
- [ ] Протестировать все исправленные компоненты
- [ ] Проверить интеграцию EventBus
- [ ] Валидировать безопасность (Pickle → JSON)

### 6.2 Integration тесты
- [ ] Запуск системы с нуля
- [ ] Проверка всех endpoints API
- [ ] Нагрузочное тестирование

### 6.3 Регрессионное тестирование
- [ ] Сравнение производительности до/после
- [ ] Проверка обратной совместимости
- [ ] Валидация функциональности

---

## МЕТРИКИ УСПЕХА

| Метрика | Текущее | Цель |
|---------|---------|------|
| Оценка системы | 3.8/10 | ≥7.0/10 |
| Критических ошибок | ~20 | 0 |
| Мёртвого кода | 40+ файлов | 0 |
| Дубликатов компонентов | 12 групп | ≤3 |
| Покрытие тестами | <10% | ≥60% |
| Время запуска | 3-4 мин | ≤1 мин |

---

## РИСКИ

| Риск | Вероятность | Влияние | Митигация |
|------|------------|---------|-----------|
| Потеря функциональности при удалении | Средняя | Высокое | Тщательная проверка импортов |
| Конфликты при объединении классов | Высокая | Среднее | Постепенная миграция |
| Регрессия производительности | Низкая | Высокое | Бенчмарки после каждой фазы |
| Поломка интеграций | Средняя | Высокое | Integration тесты |

---

## ИНСТРУКЦИЯ ПО ВЫПОЛНЕНИЮ

### Перед началом каждой фазы:
1. Создать backup текущей версии
2. Запустить существующие тесты
3. Зафиксировать метрики производительности

### После каждой фазы:
1. Запустить полный набор тестов
2. Проверить логи на ошибки
3. Обновить ISSUES_TRACKER.md
4. Сделать git commit с описанием изменений

### Коммиты:
```bash
git add .
git commit -m "Phase X: <описание изменений>
- Исправлено: #issue numbers
- Удалено: <файлы>
- Объединено: <компоненты>"
git push
```

---

*Документ будет обновляться по мере выполнения фаз*
*Ответственный: AI Architect Team*
