# АУДИТ ДУБЛИРУЮЩИХСЯ СИСТЕМ В EVA AI
**Дата:** 14 апреля 2026  
**Аудитор:** File Search Specialist  
**Цель:** Детальный анализ всех дублирующихся компонентов системы

---

## EXECUTIVE SUMMARY

| Метрика | Значение |
|---------|----------|
| **Общая оценка архитектуры** | **3.5/10** |
| **Количество дублей AdaptationManager** | 2 |
| **Количество дублей RecoveryManager** | 3 |
| **Количество дублей FractalStore/Storage** | 3 |
| **Количество дублей EntityExtractor** | 3 |
| **Дублирование генерации (Coordinator vs Unified)** | ДА |
| **Config файлы - дублирование** | ДА |
| **Сломанные миграционные скрипты** | 5 из 8 |

---

## 1. ADAPTATIONMANAGER - 2 ВЕРСИИ

### 1.1 Найденные классы

| # | Путь | Строк | Назначение |
|---|------|-------|------------|
| 1 | `eva_ai/adaptation/adaptation_manager.py` | 729 | Полноценный менеджер адаптации с профилями, фидбэком, концептами |
| 2 | `eva_ai/adaptation/adaptation_core.py` | 604 | Базовый менеджер с SQLite, фоновыми процессами |

### 1.2 Использование в системе

```
eva_ai/core/init_factories.py:409       -> ИМПОРТИРУЕТ adaptation_core.AdaptationManager
eva_ai/adaptation/adaptation_integration.py:14 -> ИМПОРТИРУЕТ adaptation_core.AdaptationManager
eva_ai/adaptation/adaptation_integrated.py:19  -> ИМПОРТИРУЕТ adaptation_core.AdaptationManager
eva_ai/adaptation/adaptation_analytics.py:14  -> ИМПОРТИРУЕТ adaptation_core.AdaptationManager
eva_ai/adaptation/__init__.py:3         -> ЭКСПОРТИРУЕТ adaptation_core.AdaptationManager
```

### 1.3 Сравнение функциональности

| Функция | adaptation_manager.py | adaptation_core.py |
|---------|----------------------|-------------------|
| Профили пользователей | YES | YES |
| UserFeedback | YES | YES |
| Концепты (concept_cache) | YES | NO |
| Извлечение концептов из запроса | YES (300+ lines) | YES (100+ lines, NLP-based) |
| SQLite база данных | NO | YES |
| Фоновые процессы | YES (threading) | YES (threading) |
| Health metrics | YES | YES |
| Анализ паттернов | YES | YES |

### 1.4 Рекомендация: **ОБЪЕДИНИТЬ**

**Действие:** Объединить оба класса в один, взяв лучшее:
- SQLite из `adaptation_core.py`
- Концепты из `adaptation_manager.py`
- NLP-извлечение концептов из `adaptation_core.py`

**Приоритет:** ВЫСОКИЙ (6/10)

---

## 2. RECOVERYMANAGER - 3 ВЕРСИИ

### 2.1 Найденные классы

| # | Путь | Строк | Назначение |
|---|------|-------|------------|
| 1 | `eva_ai/recovery/recovery_system.py` | 777 | Полноценная система восстановления с checkpoint, plan, state manager |
| 2 | `eva_ai/distributed/distributed_recovery_manager.py` | 353 | Распределённый менеджер восстановления |
| 3 | `eva_ai/core/component_managers.py:219` | 141 | Заглушка (stub) - только логирование |

### 2.2 Использование в системе

```
eva_ai/distributed/distributed_system.py:174 -> ИМПОРТИРУЕТ distributed_recovery_manager.RecoveryManager
eva_ai/distributed/__init__.py:6             -> ЭКСПОРТИРУЕТ distributed_recovery_manager.RecoveryManager
```

### 2.3 Детальный анализ

#### recovery_system.py (ОСНОВНОЙ)
```python
class ComponentStateManager:
    # Управление состояниями компонентов
    
class RecoveryManager:
    # Полноценный менеджер восстановления
    - create_checkpoint()
    - restore_from_checkpoint()
    - create_recovery_plan()
    - execute_recovery()
    - handle_failure()
```

#### distributed_recovery_manager.py (ИСПОЛЬЗУЕТСЯ)
```python
class RecoveryManager:
    # Распределённое восстановление
    - _load_checkpoints()
    - create_checkpoint()
    - recover_component()
    - get_recovery_status()
```

#### component_managers.py (ЗАГЛУШКА)
```python
class RecoveryManager:
    def handle_failure(...):  # Только логирует, возвращает False
    def create_backup(...):   # Stub
```

### 2.4 Рекомендация: **УДАЛИТЬ ЗАГЛУШКУ + ОБЪЕДИНИТЬ**

**Действия:**
1. **Удалить** `component_managers.py:RecoveryManager` (заглушка)
2. **Объединить** `recovery_system.py` и `distributed_recovery_manager.py`
   - Взять checkpoint логику из `recovery_system.py`
   - Добавить распределённые функции из `distributed_recovery_manager.py`

**Приоритет:** ВЫСОКИЙ (7/10)

---

## 3. FRACTALSTORE/FRACTALSTORAGE - 4 ВЕРСИИ

### 3.1 Найденные классы

| # | Путь | Класс | Строк | Назначение |
|---|------|-------|-------|------------|
| 1 | `eva_ai/fractal/fractal_store.py` | FractalStore | 708 | Основное фрактальное хранилище |
| 2 | `eva_ai/storage/fractal_storage.py` | FractalStorage | 254 | Базовое хранилище (JSON-файлы) |
| 3 | `eva_ai/reasoning/fractal_ml/fractal_storage.py` | FractalStorage | 734 | Иерархическое хранилище (L0-L3 уровни) |
| 4 | `eva_ai/mlearning/storage/fractal_store_utils.py` | FractalStoreUtils | - | Утилиты для FractalStore |

### 3.2 Использование в системе

```
eva_ai/core/init_factories.py:607           -> ИМПОРТИРУЕТ reasoning.fractal_ml.FractalStorage
eva_ai/reasoning/self_reasoning_engine.py:113 -> ИМПОРТИРУЕТ reasoning.fractal_ml.FractalStorage
eva_ai/core/graph_ml_core.py:18            -> ИМПОРТИРУЕТ fractal.entity_fractal_store.EntityFractalStore
eva_ai/reasoning/__init__.py:12            -> ЭКСПОРТИРУЕТ fractal_ml.FractalStorage
eva_ai/reasoning/fractal_ml/__init__.py:18 -> ЭКСПОРТИРУЕТ fractal_ml.FractalStorage
eva_ai/mlearning/storage/__init__.py:11     -> ЭКСПОРТИРУЕТ fractal_store_utils.FractalStoreUtils
eva_ai/fractal/__init__.py:2               -> ЭКСПОРТИРУЕТ fractal_store.FractalStore
```

### 3.3 Сравнение функциональности

| Функция | fractal_store.py | storage/fractal_storage.py | reasoning/fractal_ml/fractal_storage.py |
|---------|-----------------|---------------------------|----------------------------------------|
| FractalContainer | YES | NO | NO |
| pack_model_weights | YES | NO | NO |
| JSON-файлы | NO | YES | NO |
| Иерархия L0-L3 | NO | NO | YES |
| FractalNode/Edge | NO | NO | YES |
| GPU hot cache | YES | NO | NO |
| Priority queue | YES | NO | NO |

### 3.4 Рекомендация: **УДАЛИТЬ НЕИСПОЛЬЗУЕМОЕ**

**Действия:**
1. **Удалить** `storage/fractal_storage.py` (не используется)
2. **Проверить** `fractal/entity_fractal_store.py` - это отдельный неиспользуемый модуль?
3. **Объединить** `fractal_store.py` и `reasoning/fractal_ml/fractal_storage.py`

**Приоритет:** СРЕДНИЙ (5/10)

---

## 4. ENTITYEXTRACTOR - 3 ВЕРСИИ

### 4.1 Найденные классы

| # | Путь | Строк | Назначение |
|---|------|-------|------------|
| 1 | `eva_ai/knowledge/context_entity.py` | 165 | Обёртка над reasoning.EntityExtractor |
| 2 | `eva_ai/reasoning/entity_extractor.py` | 398 | Полноценный извлекатель (regex, patterns) |
| 3 | `eva_ai/gui/web_gui/server_auth.py:222` | 41 | Простой извлекатель для auth |

### 4.2 Детальный анализ

#### reasoning/entity_extractor.py (ОСНОВНОЙ)
```python
class EntityExtractor:
    - extract_from_query()      # Из запроса
    - extract_from_response()   # Из ответа
    - extract_from_contradiction() # Из противоречий
    - save_to_knowledge_graph() # Сохранение в KG
    - Паттерны: concept, fact, person, event, value
```

#### knowledge/context_entity.py (ОБЁРТКА)
```python
class EntityExtractor:
    # Обёртка над reasoning.EntityExtractor
    # Если reasoningExtractor доступен - использует его
    # Иначе - semantic search через FGv2
```

#### gui/server_auth.py (ПРОСТОЙ)
```python
class EntityExtractor:
    # Простой keyword-based extractor для auth
    # extract_entities() - ищет keyword совпадения
```

### 4.3 Рекомендация: **ОБЪЕДИНИТЬ В ОДИН**

**Действия:**
1. **Удалить** `gui/server_auth.py:EntityExtractor` - примитивный
2. **Сохранить** `reasoning/entity_extractor.py` как основной
3. **Удалить** `knowledge/context_entity.py` - дублирует функциональность

**Приоритет:** ВЫСОКИЙ (7/10)

---

## 5. UNIFIEDGENERATOR vs GENERATIONCOORDINATOR

### 5.1 Найденные классы

| # | Путь | Класс | Строк |
|---|------|-------|-------|
| 1 | `eva_ai/core/unified_generator.py` | UnifiedGenerator | 1733 |
| 2 | `eva_ai/generation/generation_coordinator.py` | UnifiedGenerationCoordinator | 609 |

### 5.2 Дублирование функциональности

| Функция | UnifiedGenerator | GenerationCoordinator |
|---------|------------------|----------------------|
| Единая точка входа | YES | YES |
| Роутинг моделей | YES (SimpleRouter) | YES (провайдеры) |
| Multiple providers | NO | YES |
| Fallback | NO | YES |
| EventBus интеграция | YES (ModelAccessManager) | NO |

### 5.3 Аудит использования

```
Используется напрямую:
- brain.two_model_pipeline -> UnifiedGenerator
- HybridPipelineAdapter -> UnifiedGenerator
- DualGeneratorPie -> Pie Architecture

GenerationCoordinator:
- sre_context.py:125 - используется как fallback
- enhanced_reasoning_engine.py - используется fallback
```

### 5.4 Существующий аудит (из analysis/)

> **Проблема:** GenerationCoordinator избыточен - UnifiedGenerator и HybridPipelineAdapter уже обеспечивают единую точку входа.
> 
> **Рекомендация:** Удалить как избыточный слой.

### 5.5 Рекомендация: **УДАЛИТЬ GENERATIONCOORDINATOR**

**Действия:**
1. **Удалить** `generation/generation_coordinator.py`
2. **Обновить** `sre_context.py` и `enhanced_reasoning_engine.py` использовать напрямую `brain.two_model_pipeline`

**Приоритет:** КРИТИЧЕСКИЙ (9/10)

---

## 6. CONFIG ФАЙЛЫ - ДУБЛИРОВАНИЕ

### 6.1 Найденные файлы

| Путь | Строк | Описание |
|------|-------|----------|
| `eva_ai/config/optimal_config.json` | 82 | Полный конфиг (RAM, cache, training, GUI) |
| `eva_ai/config/fractal_model_config.json` | 14 | Подмножество (только fractal_model_manager) |

### 6.2 Сравнение

fractal_model_config.json является ПОДМНОЖЕСТВОМ optimal_config.json (секция fractal_model_manager).

### 6.3 Рекомендация: **УДАЛИТЬ fractal_model_config.json**

**Приоритет:** НИЗКИЙ (3/10)

---

## 7. DUALGENERATOR - 2 ВЕРСИИ

### 7.1 Найденные файлы

| Путь | Класс | Строк |
|------|-------|-------|
| `eva_ai/memory/fractal_graph_v2/dual_generator.py` | DualGenerator, CondensedGenerator, ExtendedGenerator | 1031 |
| `eva_ai/memory/fractal_graph_v2/dual_generator_pie.py` | DualGeneratorPie | 346 |

### 7.2 Рекомендация: **ОБЪЕДИНИТЬ В ОДИН**

**Приоритет:** СРЕДНИЙ (5/10)

---

## 8. МИГРАЦИОННЫЕ СКРИПТЫ - СЛОМАНЫ

### 8.1 Найденные скрипты

| Скрипт | Статус | Проблема |
|--------|--------|----------|
| `scripts/load_gguf_to_fg.py` | BROKEN | Зависит от fg_gguf_architecture_mapper (не существует) |
| `scripts/migrate_kg_to_fg.py` | BROKEN | Вызывает несуществующий kg_to_fg_migration |
| `scripts/simple_test.py` | WORKING | Простой тест |
| `scripts/migrate_to_optimized.py` | BROKEN | Вызывает OptimizedFractalModelManager (не существует) |
| `scripts/migrate_events.py` | DOCS ONLY | Только логирует миграцию, не выполняет |
| `scripts/export_qwen.py` | BROKEN | Неверный путь eva\mlearning\ |
| `scripts/activate_max_cache.py` | BROKEN | Зависит от unified_fractal_manager |
| `scripts/complete_fractal_solution.py` | UNKNOWN | Не проверен |

### 8.2 Рекомендация: **УДАЛИТЬ СЛОМАННЫЕ**

**Приоритет:** ВЫСОКИЙ (8/10)

---

## 9. ИТОГОВАЯ ТАБЛИЦА ДУБЛИРОВАНИЯ

| Компонент | Количество версий | Оценка серьёзности | Рекомендуемое действие |
|-----------|------------------|-------------------|----------------------|
| AdaptationManager | 2 | 6/10 | ОБЪЕДИНИТЬ |
| RecoveryManager | 3 | 7/10 | УДАЛИТЬ ЗАГЛУШКУ + ОБЪЕДИНИТЬ |
| FractalStore/Storage | 4 | 5/10 | УДАЛИТЬ НЕИСПОЛЬЗУЕМОЕ |
| EntityExtractor | 3 | 7/10 | ОБЪЕДИНИТЬ В ОДИН |
| GenerationCoordinator | 1 (дублирует) | 9/10 | УДАЛИТЬ |
| Config файлы | 2 (дублируют) | 3/10 | УДАЛИТЬ fractal_model_config |
| DualGenerator | 2 | 5/10 | ОБЪЕДИНИТЬ |
| Миграционные скрипты | 5 из 8 сломаны | 8/10 | УДАЛИТЬ СЛОМАННЫЕ |

---

## 10. ФИНАЛЬНАЯ ОЦЕНКА

| Метрика | Оценка |
|---------|--------|
| **Общая оценка дублирования** | **3.5/10** |
| Критические (немедленно) | GenerationCoordinator, миграции |
| Высокий приоритет | AdaptationManager, EntityExtractor, RecoveryManager |
| Средний приоритет | FractalStorage, DualGenerator |
| Низкий приоритет | Config файлы |

---

## 11. ПЛАН ДЕЙСТВИЙ

### ФАЗА 1: Критические (немедленно)
1. **Удалить** `generation/generation_coordinator.py`
2. **Удалить** 5 сломанных миграционных скриптов
3. **Обновить** `sre_context.py` использовать `brain.two_model_pipeline`

### ФАЗА 2: Высокий приоритет
4. Объединить AdaptationManager (2 -> 1)
5. Удалить заглушку RecoveryManager + объединить (3 -> 1)
6. Объединить EntityExtractor (3 -> 1)

### ФАЗА 3: Средний приоритет
7. Удалить `storage/fractal_storage.py` (не используется)
8. Объединить DualGenerator (2 -> 1)
9. Удалить `fractal_model_config.json`

---

## 12. ФАЙЛЫ ОТЧЁТА

Данный отчёт сохранён в: `eva_ai/analysis/audit_duplicate_systems.md`

---

*Отчёт подготовлен File Search Specialist*
*14 апреля 2026*