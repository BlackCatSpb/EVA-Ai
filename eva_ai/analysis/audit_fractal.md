# АУДИТ FRACTAL ПОДСИСТЕМЫ EVA AI
**Дата:** 14.04.2026  
**Аудитор:** EVA AI System Audit  
**Версия EVA:** Fractal Subsystems Analysis  

---

## СОДЕРЖАНИЕ
1. [Общая структура fractal](#1-общая-структура-fractal)
2. [Файлы в eva_ai/fractal/](#2-файлы-в-eva_aifractal)
3. [Анализ использования](#3-анализ-использования)
4. [Дубликаты компонентов](#4-дубликаты-компонентов)
5. [EventBus интеграция](#5-eventbus-интеграция)
6. [fractal_graph_v2 vs fractal/](#6-fractal_graph_v2-vs-fractal)
7. [Оценка по 10-балльной шкале](#7-оценка-по-10-балльной-шкале)
8. [Рекомендации](#8-рекомендации)

---

## 1. ОБЩАЯ СТРУКТУРА FRACTAL

### 1.1 Расположение fractal компонентов

| Директория | Описание | Статус |
|------------|----------|--------|
| eva_ai/fractal/ | Основной fractal модуль | **НЕ ИСПОЛЬЗУЕТСЯ** |
| eva_ai/memory/fractal_graph_v2/ | FGv2 хранилище знаний | **АКТИВНО ИСПОЛЬЗУЕТСЯ** |
| eva_ai/memory/unified_fractal_memory.py | UnifiedFractalMemory | **ИСПОЛЬЗУЕТСЯ** |
| eva_ai/mlearning/storage/ | FractalWeightStore дубликаты | **ЧАСТИЧНО ИСПОЛЬЗУЕТСЯ** |
| eva_ai/learning/fractal_store.py | Дубликат FractalWeightStore | **НЕ ИСПОЛЬЗУЕТСЯ** |

---

## 2. ФАЙЛЫ В eva_ai/fractal/

### 2.1 fractal/__init__.py

`python
from eva_ai.fractal.fractal_store import FractalStore
__all__ = [FractalStore]
`

**Проблема:** Модуль экспортирует FractalStore, который НИГДЕ НЕ ИСПОЛЬЗУЕТСЯ.

### 2.2 fractal/fractal_store.py (708 строк)

**Классы:**
- FractalContainer - контейнер для фрактальных данных
- FractalStore - унифицированный интерфейс фрактального хранилища

**Методы FractalStore:**
- pack_model_weights() - упаковка весов модели
- pack_state_dict() - упаковка state_dict
- get_container() - получение контейнера
- save_to_disk() / load_from_disk() - персистентность
- _build_fractal_hierarchy() - построение фрактальной иерархии

**Статус: НЕ ИСПОЛЬЗУЕТСЯ**

### 2.3 fractal/entity_fractal_store.py (427 строк)

**Классы:**
- EntityLevelData - данные на уровне сущности
- EntityFractalStore - 5-уровневое хранилище сущностей

**Уровни:**
- Level 0: Raw tokens
- Level 1: Ambiguous terms
- Level 2: Clarified meanings
- Level 3: Concept definitions
- Level 4: Full understanding

**Используется в:** eva_ai/core/graph_ml_core.py (строка 18)

---

## 3. АНАЛИЗ ИСПОЛЬЗОВАНИЯ

### 3.1 FractalStore (eva_ai/fractal/)

| Метрика | Значение |
|---------|----------|
| Импортов в коде | **0** |
| Экспорт в __init__ | Да |
| Реальное использование | **НЕТ** |

**КРИТИЧЕСКАЯ ПРОБЛЕМА:** FractalStore определён, экспортирован, но НИКОГДА не импортируется.

### 3.2 EntityFractalStore

| Метрика | Значение |
|---------|----------|
| Импортов в коде | 1 (graph_ml_core.py) |
| Реальное использование | Через graph_ml_core.py |

### 3.3 fractal_graph_v2

| Метрика | Значение |
|---------|----------|
| Упоминаний в коде | **212+** |
| Использование в brain_components | Да |
| Использование в dialog_core | Да |
| Использование в brain_query | Да |

**Вывод:** fractal_graph_v2 является ОСНОВНЫМ фрактальным хранилищем системы.

---

## 4. ДУБЛИКАТЫ КОМПОНЕНТОВ

### 4.1 FractalWeightStore - 4 дубликата

| Файл | Размер | Статус |
|------|--------|--------|
| eva_ai/mlearning/storage/store_core.py | 463 строки | Активен |
| eva_ai/mlearning/storage/fractal_store_core.py | 237 строк | Дубликат |
| eva_ai/mlearning/storage/fractal_weight_store.py | 157 строк | Дубликат |
| eva_ai/learning/fractal_store.py | 124 строки | **НЕ ИСПОЛЬЗУЕТСЯ** |

### 4.2 FractalContainer - 3+ дубликата

| Файл | Класс | Строк |
|------|-------|-------|
| eva_ai/fractal/fractal_store.py | FractalContainer | ~50 |
| eva_ai/mlearning/storage/fractal_store_core.py | FractalContainer | ~20 |
| eva_ai/learning/fractal_store.py | FractalContainer | ~20 |

---

## 5. EVENTBUS ИНТЕГРАЦИЯ

### 5.1 fractal/ модуль

| Компонент | EventBus | Статус |
|-----------|----------|--------|
| eva_ai/fractal/fractal_store.py | **НЕТ** | КРИТИЧЕСКАЯ ПРОБЛЕМА |
| eva_ai/fractal/entity_fractal_store.py | **НЕТ** | КРИТИЧЕСКАЯ ПРОБЛЕМА |

**Вывод:** Модуль eva_ai/fractal/ полностью изолирован от EventBus.

### 5.2 unified_fractal_memory.py

`python
from eva_ai.core.event_bus import Event, EventPriority
# Публикация событий миграции
def _publish_migration_event(self, from_tier: str, to_tier: str, node_ids: List[str]):
    event = self._Event(event_type=EventTypes.MEMORY_TIER_MIGRATED, ...)
    self._event_bus.publish(event)
`

**EventBus события:**
- memory.tier_migrated - миграция между hot/warm/cold tier

### 5.3 fractal_graph_v2/

| Компонент | EventBus | Статус |
|-----------|----------|--------|
| FractalMemoryGraph wrapper | **НЕТ** | Нет прямой интеграции |
| FractalGraphV2 storage | **НЕТ** | Нет прямой интеграции |

---

## 6. FRACTAL_GRAPH_V2 VS FRACTAL/

### 6.1 Сравнение

| Характеристика | fractal/ | fractal_graph_v2 |
|----------------|----------|-----------------|
| **Использование** | НЕ ИСПОЛЬЗУЕТСЯ | АКТИВНО ИСПОЛЬЗУЕТСЯ |
| **EventBus** | НЕТ | НЕТ (косвенно) |
| **SQLite** | НЕТ | ДА |
| **Embedding** | НЕТ | ДА |
| **Кластеризация** | НЕТ | ДА |
| **Интеграция с brain** | НЕТ | ДА |

---

## 7. ОЦЕНКА ПО 10-БАЛЛЬНОЙ ШКАЛЕ

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Функциональность fractal/ | 2/10 | Модуль не используется |
| Функциональность FGv2 | 8/10 | Полнофункциональное хранилище |
| Использование | 3/10 | fractal/ не используется |
| Дубликаты | 2/10 | 4+ FractalWeightStore |
| EventBus интеграция fractal/ | 0/10 | **ПОЛНОСТЬЮ ОТСУТСТВУЕТ** |
| Архитектура | 4/10 | Изолированный мёртвый код |
| Поддерживаемость | 3/10 | Дубликаты запутывают |
| **ИТОГО fractal/** | **2.3/10** | **КРИТИЧЕСКИ НИЗКАЯ** |
| **ИТОГО FGv2** | **6.5/10** | **ХОРОШО** |

### Общая оценка системы (с учётом мёртвого кода)

| Аспект | Оценка |
|--------|--------|
| fractal/ (изолированный модуль) | 2.3/10 |
| fractal_graph_v2 (основное хранилище) | 6.5/10 |
| UnifiedFractalMemory (вторичное хранилище) | 6.0/10 |
| **СРЕДНЯЯ** | **4.5/10** |

---

## 8. РЕКОМЕНДАЦИИ

### 8.1 КРИТИЧЕСКИЕ

#### 1. УДАЛИТЬ или ИНТЕГРИРОВАТЬ eva_ai/fractal/

**Вариант A: Удаление мёртвого кода**
`ash
rm eva_ai/fractal/fractal_store.py
rm eva_ai/fractal/entity_fractal_store.py
`

**Вариант B: Интеграция с EventBus**
`python
# В fractal_store.py добавить:
from eva_ai.core.event_bus import Event, EventPriority

class FractalStore:
    def _publish_event(self, event_type: str, data: dict):
        from eva_ai.core.event_bus import get_event_bus
        eb = get_event_bus()
        event = Event(event_type=event_type, source='fractal', data=data)
        eb.publish(event)
`

#### 2. Устранить дубликаты FractalWeightStore

**Рекомендация:** Оставить один источник истины:
- **Оставить:** eva_ai/mlearning/storage/store_core.py
- **Удалить:** остальные дубликаты
- **Обновить импорты** во всех файлах

#### 3. Добавить EventBus в fractal_graph_v2

`python
# В FractalMemoryGraph:
from eva_ai.core.event_bus import Event, EventTypes

def add_node(self, ...):
    # ... existing logic ...
    self._publish_event('node.added', {'node_id': node_id})
`

### 8.2 ВЫСОКИЙ ПРИОРИТЕТ

#### 4. Унифицировать FractalContainer

Создать единый класс в eva_ai/memory/fractal_graph_v2/types.py

#### 5. Интегрировать EntityFractalStore в FGv2 или удалить

---

## ВЫВОДЫ

### Главные проблемы:

1. **Мёртвый код:** eva_ai/fractal/ не используется системой
2. **Дубликаты:** 4+ версии FractalWeightStore
3. **Нет EventBus:** fractal/ полностью изолирован
4. **Архитектурная несогласованность**

### Что работает:

1. **fractal_graph_v2** - полнофункциональное хранилище (6.5/10)
2. **UnifiedFractalMemory** - вторичное хранилище с EventBus (6.0/10)

### Рекомендуемое действие:

**УДАЛИТЬ eva_ai/fractal/ как мёртвый код**, оставив только fractal_graph_v2 и UnifiedFractalMemory.

---

**Дата создания:** 14.04.2026  
**Следующий аудит:** При изменении fractal архитектуры
