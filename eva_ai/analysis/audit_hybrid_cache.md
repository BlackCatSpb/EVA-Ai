# Аудит системы кэширования HybridCache/TokenCache в EVA AI

**Дата аудита:** 2026-04-14
**Аудитор:** EVA AI Audit System
**Версия EVA:** DualGenerator/FractalGraph v2

---

## Содержание

1. Архитектура кэширования
2. Проблемы с Pickle
3. Проблемы с TTL
4. Интеграция с memory_manager
5. Места использования кэша
6. Оценка по 10-балльной шкале
7. Конкретные рекомендации

---

## 1. Архитектура кэширования

### 1.1 Компоненты системы

```
HybridTokenCache (cache_core.py)
├── LRUCache (cache_ram.py) - VRAM кэш (при наличии GPU)
├── LRUCache (cache_ram.py) - RAM кэш
├── TokenDiskCache (cache_disk.py) - Дисковый кэш (до 50GB)
└── Metadata (JSON файл)
```

### 1.2 Иерархия уровней

| Уровень | Устройство | Лимит | Реализация |
|---------|------------|-------|------------|
| L1 | GPU (VRAM) | ~1.5 GB | LRUCache |
| L2 | RAM | ~1 GB | LRUCache |
| L3 | SSD/Disk | 50 GB | TokenDiskCache |

### 1.3 Ключевые файлы

| Файл | Назначение | Строк |
|------|------------|-------|
| memory/cache_core.py | Основной класс HybridTokenCache | 410 |
| memory/cache_ram.py | LRU реализация для RAM/VRAM | 48 |
| memory/cache_disk.py | Дисковое хранение с pickle | 275 |
| memory/cache_eviction.py | Политики вытеснения и TTL | 354 |
| memory/cache_router.py | Маршрутизация по сегментам | 98 |
| memory/disk_cache.py | Альтернативный дисковый кэш с SQLite | 410 |

### 1.4 Конфигурация (cache_core.py:35-50)

- max_memory_tokens: 100000
- disk_cache_dir: token_cache
- target_memory_gb: 50.0
- max_disk_cache_gb: 50.0
- dynamic_memory_limit: True
- eviction_policy: hybrid

---

## 2. Проблемы с Pickle

### 2.1 Места использования Pickle

| Файл | Строки | Риск |
|------|--------|------|
| memory/cache_disk.py | 82, 97, 102, 121, 122 | ВЫСОКИЙ |
| memory/disk_cache.py | 7, 250, 344 | ВЫСОКИЙ |
| memory/fractal_torch_storage/base_storage.py | 206, 216, 229, 232 | ВЫСОКИЙ |
| storage/fractal_storage.py | 206, 208 | ВЫСОКИЙ |
| mlearning/storage/fractal_weight_store.py | 54, 63 | СРЕДНИЙ |

### 2.2 Детальный анализ TokenDiskCache

cache_disk.py:121-122 - pickle.dumps с HIGHEST_PROTOCOL
cache_disk.py:97 - pickle.loads без валидации

**Проблемы:**
1. pickle.HIGHEST_PROTOCOL создает несовместимые данные между Python версиями
2. Нет верификации данных перед десериализацией
3. pickle.loads может выполнить произвольный код

---

## 3. Проблемы с TTL

### 3.1 Где TTL ЗАДАЁТСЯ (но не используется)

cache_core.py:166 - cache_ttl: 86400 (24 часа)
**ПРОБЛЕМА:** Значение сохраняется, но НЕ используется в коде!

### 3.2 Где TTL РЕАЛЬНО работает

| Метод | Файл | Строки | TTL |
|-------|------|--------|-----|
| add_context/get_context | cache_eviction.py | 180-207 | 3600 сек |
| add_document/get_document | cache_eviction.py | 229-258 | 86400 сек |
| add_search_results/get_search_results | cache_eviction.py | 292-333 | 43200 сек |

### 3.3 Где TTL НЕ работает

- _add_token_impl() - НЕТ проверки TTL
- _get_token_impl() - НЕТ проверки TTL
- VRAM/RAM/Disk базовые токены - живут вечно

---

## 4. Интеграция с memory_manager

### 4.1 Инициализация (manager_core.py:127-145)

get_hybrid_cache() возвращает синглтон из brain.hybrid_cache
или создает новый через get_shared_cache()

### 4.2 Регистрация (init_factories.py:132-181)

- core_brain.token_cache = hybrid_cache
- core_brain.hybrid_cache = hybrid_cache
- core_brain.components[hybrid_cache] = hybrid_cache

### 4.3 Синглтон-реестр (cache_core.py:13-22)

_cache_registry хранит отдельные инстансы по имени

---

## 5. Места использования кэша

### 5.1 Прямое использование

- core/openvino_generator.py:771 - add_token()
- core/token_processor.py:32,48 - get_token/add_token
- core/brain_query.py:591 - get_context()
- preprocess/preprocessing_pipeline.py:319,332 - add/get_context()

### 5.2 Косвенное использование

- core/unified_generator.py - context для генерации
- core/pipeline_adapter.py - FractalGraph integration
- core/context_chunking.py - chunking integration

---

## 6. Оценка по 10-балльной шкале

### 6.1 Итоговые оценки

| Категория | Вес | Балл | Итог |
|-----------|-----|------|------|
| Безопасность | 40% | 3.3 | 1.32 |
| Функциональность | 30% | 5.7 | 1.71 |
| Производительность | 30% | 7.7 | 2.31 |
| **ИТОГО** | 100% | - | **5.34/10** |

### 6.2 Детализация

**Безопасность:**
- Pickle usage: 3/10
- Input validation: 4/10
- Code execution risk: 3/10

**Функциональность:**
- TTL для базовых токенов: 2/10
- TTL для высокоуровневых данных: 7/10
- Иерархия кэшей: 8/10

**Производительность:**
- LRU eviction: 8/10
- Memory pressure handling: 7/10
- I/O throttling: 7/10
- Thread safety: 8/10

---

## 7. Конкретные рекомендации

### 7.1 Критические (немедленно)

1. **Заменить Pickle на msgpack/json**
   - Файлы: cache_disk.py, disk_cache.py

2. **Реализовать TTL для базовых токенов**
   - Добавить проверку в _get_token_impl()

### 7.2 Высокий приоритет

1. Добавить валидацию данных при загрузке (pydantic)
2. Очистка просроченных TTL в фоне

### 7.3 Средний приоритет

1. Конфигурация TTL через brain_config.json

---

## Файлы для изменений

| Файл | Изменения |
|------|-----------|
| memory/cache_disk.py | Заменить pickle |
| memory/disk_cache.py | Заменить pickle |
| memory/cache_eviction.py | Добавить TTL |
| memory/cache_core.py | Активировать cache_ttl |

---

**Дата:** 2026-04-14
**Следующий аудит:** Через 30 дней
