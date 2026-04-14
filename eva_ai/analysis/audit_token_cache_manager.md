# Аудит TokenCacheManager системы EVA AI

**Дата аудита:** 2026-04-14  
**Аудитор:** EVA AI Audit System  
**Версия EVA:** DualGenerator/FractalGraph v2  

---

## Содержание

1. [Резюме](#1-резюме)
2. [Архитектура TokenCacheManager](#2-архитектура-tokencachemanager)
3. [Связь с HybridCache](#3-связь-с-hybridcache)
4. [Дубликаты и конфликты реализаций](#4-дубликаты-и-конфликты-реализаций)
5. [Интеграция в системе](#5-интеграция-в-системе)
6. [Проблемы и уязвимости](#6-проблемы-и-уязвимости)
7. [Оценка по 10-балльной шкале](#7-оценка-по-10-балльной-шкале)
8. [Рекомендации](#8-рекомендации)

---

## 1. Резюме

### 1.1 Ключевые находки

| Категория | Статус | Критичность |
|-----------|--------|-------------|
| Архитектура | Требует улучшения | Средняя |
| Дубликаты | КРИТИЧЕСКАЯ | Высокая |
| Интеграция | Частичная | Средняя |
| Безопасность Pickle | Опасная | Критическая |
| Производительность | Удовлетворительная | Низкая |

### 1.2 Основные проблемы

1. **Две реализации HybridTokenCache** - дублирование кода
2. **Две реализации TokenDiskCache** - конфликт при импорте
3. **Pickle используется без валидации** - риск выполнения кода
4. **Нет единой точки управления** - размытая ответственность

---

## 2. Архитектура TokenCacheManager

### 2.1 Компоненты системы

`
TokenCacheManager (НЕ СУЩЕСТВУЕТ как класс)
    |
    +-- HybridTokenCache (memory/cache_core.py) [ОСНОВНОЙ]
    |   +-- LRUCache (memory/cache_ram.py) - VRAM/RAM
    |   +-- TokenDiskCache (memory/cache_disk.py) - SSD
    |   +-- Metadata storage (JSON)
    |   +-- Memory pressure worker
    |
    +-- HybridTokenCache (core/hybrid_token_cache.py) [ДУБЛИКАТ - НЕ ИСПОЛЬЗУЕТСЯ]
        +-- Простая реализация
        +-- Не связан с основной системой
`

### 2.2 Иерархия уровней кэширования

| Уровень | Устройство | Лимит по умолчанию | Реализация |
|---------|------------|-------------------|------------|
| L1 | GPU (VRAM) | ~1.5 GB (262144 токенов) | LRUCache |
| L2 | RAM | ~1 GB (262144 токенов) | LRUCache |
| L3 | SSD/Disk | 50 GB | TokenDiskCache |

### 2.3 Ключевые файлы

| Файл | Класс | Строк | Назначение |
|------|-------|-------|------------|
| memory/cache_core.py | HybridTokenCache | 410 | ОСНОВНОЙ менеджер кэша |
| memory/cache_ram.py | LRUCache | 48 | LRU для RAM/VRAM |
| memory/cache_disk.py | TokenDiskCache | 275 | Дисковое хранение |
| memory/cache_eviction.py | N/A (функции) | 354 | Политики вытеснения |
| memory/cache_router.py | CacheRouter | 98 | Маршрутизация сегментов |
| memory/cache_index.py | CacheIndex | 266 | SQLite индекс |
| core/hybrid_token_cache.py | HybridTokenCache | 346 | ДУБЛИКАТ |
| memory/token_disk_cache.py | TokenDiskCache | 130 | ДУБЛИКАТ |

---

## 3. Связь с HybridCache

### 3.1 Инициализация через get_shared_cache

memory/cache_core.py:17-22 - Синглтон registry для HybridTokenCache

### 3.2 Регистрация в CoreBrain

core/init_factories.py:132-181 - create_hybrid_cache()

**КРИТИЧЕСКАЯ ПРОБЛЕМА:**
init_factories.py:171-172 создаёт АЛИАСЫ:
`python
initializer.core_brain.token_cache = hybrid_cache  # Алиас!
initializer.core_brain.hybrid_cache = hybrid_cache  # Тот же объект!
`

### 3.3 Использование brain.token_cache

| Компонент | Файл | Использование |
|-----------|------|--------------|
| TokenProcessor | core/token_processor.py | tokenize_query(), prewarm_tokens_async() |
| UnifiedCacheBridge | core/unified_cache_bridge.py | preload_graph_context(), cache_generation() |
| BrainMemory | core/brain_memory.py | _evict_vram_to_ram(), _evict_ram_to_ssd() |
| CacheRouter | memory/cache_router.py | addressable routing |
| SystemOptimizer | core/system_optimizer.py | clear_inactive() |

---

## 4. Дубликаты и конфликты реализаций

### 4.1 КРИТИЧЕСКИЕ ДУБЛИКАТЫ

#### HybridTokenCache - ДВЕ РЕАЛИЗАЦИИ

| Реализация | Файл | Статус | Используется? |
|------------|------|--------|--------------|
| Полная | memory/cache_core.py | АКТИВНА | ДА |
| Упрощённая | core/hybrid_token_cache.py | НЕИСПОЛЬЗУЕТСЯ | НЕТ |

#### TokenDiskCache - ДВЕ РЕАЛИЗАЦИИ

| Реализация | Файл | Статус |
|------------|------|--------|
| Активная | memory/cache_disk.py | Используется |
| Дубликат | memory/token_disk_cache.py | НЕИСПОЛЬЗУЕТСЯ |

### 4.2 Импорт из memory/hybrid_token_cache.py

memory/hybrid_token_cache.py - это реэкспорт из cache_core (ПРАВИЛЬНЫЙ).

Дубликат в core/hybrid_token_cache.py НЕ импортируется и НЕ используется.

---

## 5. Интеграция в системе

### 5.1 Точки интеграции

CoreBrain.token_cache подключается к:
- TokenProcessor - токенизация запросов
- UnifiedCacheBridge - предзагрузка контекста графа
- BrainMemory - вытеснение при давлении памяти
- CacheRouter - маршрутизация сегментов
- SystemOptimizer - очистка неактивных элементов
- BrainMonitoring - статистика

### 5.2 Жизненный цикл

1. Инициализация через create_hybrid_cache()
2. Работа - кэширование токенов и контекста
3. Очистка - memory pressure worker + clear_inactive()

### 5.3 Интеграция с EventBus

- memory_pressure - для обработки давления памяти
- vram_to_ram_eviction - логирование
- ram_to_ssd_eviction - логирование

---

## 6. Проблемы и уязвимости

### 6.1 КРИТИЧЕСКИЕ

#### Pickle Deserialization (ОПАСНО)

memory/cache_disk.py:97 - pickle.loads БЕЗ валидации

**Риск:** pickle.loads может выполнить произвольный код

**Рекомендация:** Использовать json или msgpack

### 6.2 СРЕДНИЕ

#### TTL не применяется глобально

cache_ttl задаётся но не используется в базовых операциях

#### Нет единой точки управления

TokenCacheManager как концепция НЕ СУЩЕСТВУЕТ

---

## 7. Оценка по 10-балльной шкале

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Функциональность | 8/10 | Полная: VRAM+RAM+Disk, вытеснение, мониторинг |
| Архитектура | 5/10 | Дублирование, нет единого TokenCacheManager |
| Безопасность | 3/10 | Pickle без валидации - критический риск |
| Производительность | 7/10 | Многоуровневое кэширование |
| Интеграция | 7/10 | Интегрирован в основные компоненты |
| Поддерживаемость | 4/10 | Дубликаты усложняют код |
| Надёжность | 6/10 | Есть error handling, но есть утечки |
| Документация | 5/10 | Частичная |

### Итоговая оценка: 5.7/10

---

## 8. Рекомендации

### КРИТИЧЕСКИЕ (немедленно)

1. Удалить дубликат HybridTokenCache (core/hybrid_token_cache.py)
2. Удалить дубликат TokenDiskCache (memory/token_disk_cache.py)
3. Заменить Pickle на безопасный сериализатор

### ВАЖНЫЕ (на этой неделе)

4. Создать класс TokenCacheManager - единая точка входа
5. Унифицировать терминологию - выбрать ОДНО имя
6. Применить TTL глобально

### ЖЕЛАТЕЛЬНЫЕ

7. Добавить метрики cache_hit, cache_miss в EventBus
8. Улучшить документацию

---

## Приложение: Карта файлов

eva_ai/core/
  + hybrid_token_cache.py    ДУБЛИКАТ (не используется)
  + token_processor.py        Использует token_cache
  + unified_cache_bridge.py   Использует token_cache
  + brain_memory.py           Вытеснение
  + init_factories.py         Инициализация

eva_ai/memory/
  + cache_core.py             ОСНОВНОЙ HybridTokenCache
  + cache_ram.py              LRUCache
  + cache_disk.py             TokenDiskCache (активный)
  + token_disk_cache.py       ДУБЛИКАТ (не используется)
  + cache_eviction.py         Политики вытеснения
  + cache_router.py           Маршрутизация
  + cache_index.py            SQLite индекс
  + disk_cache.py             Не используется
  + hybrid_token_cache.py     Реэкспорт

---

**Дата создания отчёта:** 2026-04-14  
**Следующий аудит:** Через 2 недели