# Аудит LRUCacheWithTTL системы в EVA AI

**Дата аудита:** 14.04.2026  
**Аудитор:** EVA AI Analysis System  
**Объект аудита:** LRUCacheWithTTL в eva_ai/memory/fractal_graph_v2/__init__.py (строки 120-177)

---

## Содержание

1. [Описание системы](#1-описание-системы)
2. [Реализация LRUCacheWithTTL](#2-реализация-lrucachewithttl)
3. [Анализ TTL механизма](#3-анализ-ttl-механизма)
4. [Анализ Thread Safety](#4-анализ-thread-safety)
5. [Анализ Eviction Policy](#5-анализ-eviction-policy)
6. [Использование в системе](#6-использование-в-системе)
7. [Проблемы и риски](#7-проблемы-и-риски)
8. [Рекомендации](#8-рекомендации)
9. [Оценка и выводы](#9-оценка-и-выводы)

---

## 1. Описание системы

### 1.1 Назначение

LRUCacheWithTTL - это кэш с вытеснением по принципу Least Recently Used (LRU) и Time-To-Live (TTL) для оптимизации производительности семантического поиска во FractalMemoryGraph V2.

### 1.2 Расположение кода

- **Основной файл:** C:\\Users\\black\\OneDrive\\Desktop\\CogniFlex\\eva_ai\\memory\\fractal_graph_v2\\__init__.py
- **Строки:** 120-177
- **Класс:** LRUCacheWithTTL

### 1.3 Конфигурация по умолчанию

`python
self._search_cache = LRUCacheWithTTL(maxsize=100, ttl_seconds=300.0)
`

| Параметр | Значение | Описание |
|----------|----------|----------|
| maxsize | 100 | Максимальное количество записей в кэше |
| ttl_seconds | 300.0 | Время жизни записи в секундах (5 минут) |

---

## 2. Реализация LRUCacheWithTTL

### 2.1 Полный исходный код

`python
class LRUCacheWithTTL:
    def __init__(self, maxsize: int = 100, ttl_seconds: float = 300.0):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            value, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                self._misses += 1
                return None
            
            self._cache.move_to_end(key)
            self._hits += 1
            return value
    
    def put(self, key: str, value: Any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            
            self._cache[key] = (value, time.time())
            
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                size: len(self._cache),
                maxsize: self.maxsize,
                hits: self._hits,
                misses: self._misses,
                hit_rate: hit_rate,
                ttl_seconds: self.ttl
            }
`

### 2.2 Структура данных

| Поле | Тип | Описание |
|------|-----|----------|
| _cache | OrderedDict[str, Tuple[Any, float]] | Основное хранилище: ключ -> (значение, timestamp) |
| _hits | int | Счётчик успешных обращений |
| _misses | int | Счётчик промахов |
| _lock | threading.RLock | Блокировка для thread safety |
| maxsize | int | Максимальный размер кэша |
| ttl | float | Время жизни записи в секундах |

---

## 3. Анализ TTL механизма

### 3.1 Принцип работы

TTL (Time-To-Live) реализован следующим образом:

1. При помещении в кэш (put()) сохраняется текущее время time.time() как timestamp
2. При извлечении из кэша (get()) проверяется условие: time.time() - timestamp > self.ttl
3. Если TTL истёк, запись удаляется и возвращается None

### 3.2 Критическая проблема: Нет фоновой очистки просроченных записей

TTL проверяется ТОЛЬКО при вызове get()

**Последствия:**
- Если запись с истёкшим TTL никогда не запрашивается, она остаётся в кэше навсегда
- Кэш может содержать maxsize просроченных записей + новые записи
- Это приводит к неэффективному использованию памяти
- Fresh записей может быть меньше чем maxsize, но статистика hit_rate будет низкой

**Пример сценария:**
`
1. Кладём 100 записей (кэш заполнен)
2. Ждём 5 минут (все записи просрочены)
3. Делаем get(ключ_1) - TTL истёк, запись удалена
4. Но все остальные 99 записей всё ещё в кэше как просроченные!
5. Кэш раздут просроченными записями
`

### 3.3 Оценка TTL механизма: 5/10

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Корректность | 7/10 | TTL вычисляется правильно, но нет автоматической очистки |
| Эффективность | 4/10 | Просроченные записи не удаляются автоматически |
| Потребление памяти | 3/10 | Кэш может содержать только просроченные записи |

---

## 4. Анализ Thread Safety

### 4.1 Используемый механизм

`python
self._lock = threading.RLock()  # Reentrant Lock
`

### 4.2 Защищённые операции

| Метод | Блокировка | Операции |
|-------|------------|----------|
| get() | with self._lock | Чтение, проверка TTL, move_to_end, инкремент счётчиков |
| put() | with self._lock | Проверка наличия, move_to_end, запись, eviction |
| clear() | with self._lock | Полная очистка кэша и счётчиков |
| stats() | with self._lock | Чтение статистики |

### 4.3 Оценка Thread Safety: 8/10

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Атомарность | 9/10 | Все операции защищены RLock |
| Deadlock protection | 9/10 | RLock предотвращает self-deadlock |
| Конкурентный доступ | 8/10 | Реализация корректна для сценариев использования |

---

## 5. Анализ Eviction Policy

### 5.1 Реализация LRU

LRU (Least Recently Used) реализован через OrderedDict:

`python
# При get() - перемещаем в конец (most recently used)
self._cache.move_to_end(key)

# При eviction - удаляем первый элемент (least recently used)  
self._cache.popitem(last=False)
`

### 5.2 Оценка Eviction Policy: 7/10

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| LRU корректность | 9/10 | Реализация соответствует LRU |
| TTL-aware eviction | 3/10 | При eviction не проверяется TTL |
| Эффективность | 6/10 | Возможно вытеснение ещё валидных записей |

---

## 6. Использование в системе

### 6.1 Инициализация

`python
# FractalMemoryGraph.__init__() (строка 210)
self._search_cache = LRUCacheWithTTL(maxsize=100, ttl_seconds=300.0)
`

### 6.2 Используемые методы в FractalMemoryGraph

| Метод | Строки | Описание |
|-------|--------|----------|
| semantic_search() | 450-515 | Кэширование результатов поиска |
| semantic_search_batch() | 517-616 | Batch кэширование для нескольких запросов |
| get_search_cache_stats() | 1205 | Получение статистики |
| clear_search_cache() | 1214 | Очистка кэша |
| invalidate_cache_for_node() | 1219 | Инвалидация при изменении узла |

### 6.3 API Endpoints

| Endpoint | Файл | Описание |
|----------|------|----------|
| GET /api/stats | server_routes.py:1868 | Возвращает search_cache статистику |
| POST /api/clear_cache | server_routes.py:1452 | Очищает кэш через get_search_cache_stats |

---

## 7. Проблемы и риски

### 7.1 Критические проблемы

| ID | Проблема | Серьёзность | Статус |
|----|----------|-------------|--------|
| 1 | Нет фоновой очистки TTL | Высокая | Требует исправления |
| 2 | Полная очистка при invalidate | Средняя | Требует улучшения |
| 3 | Eviction не учитывает TTL | Средняя | Требует улучшения |

---

## 8. Рекомендации

### 8.1 Критические рекомендации

#### 8.1.1 Добавить фоновую очистку просроченных записей

`python
import threading
import time

class LRUCacheWithTTL:
    def __init__(self, maxsize: int = 100, ttl_seconds: float = 300.0, 
                 cleanup_interval: float = 60.0):
        # ... existing init ...
        self._cleanup_interval = cleanup_interval
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
    
    def start_cleanup_thread(self):
        if self._cleanup_thread is None:
            self._stop_cleanup.clear()
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop, daemon=True
            )
            self._cleanup_thread.start()
    
    def stop_cleanup_thread(self):
        if self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join()
            self._cleanup_thread = None
    
    def _cleanup_loop(self):
        while not self._stop_cleanup.wait(self._cleanup_interval):
            self._cleanup_expired()
    
    def _cleanup_expired(self):
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, (_, ts) in self._cache.items() 
                if now - ts > self.ttl
            ]
            for k in expired_keys:
                del self._cache[k]
`

### 8.2 Рекомендуемый план действий

| Приоритет | Действие | Сложность |
|-----------|----------|-----------|
| 1 | Добавить фоновую очистку просроченных записей | Низкая |
| 2 | Реализовать умную инвалидацию по паттернам | Средняя |
| 3 | Добавить TTL-aware eviction в put() | Низкая |
| 4 | Расширить статистику (expired_count, etc) | Низкая |
| 5 | Добавить метрики для мониторинга | Средняя |

---

## 9. Оценка и выводы

### 9.1 Итоговые оценки по категориям

| Категория | Оценка | Максимум |
|-----------|--------|----------|
| TTL механизм | 5 | 10 |
| Thread Safety | 8 | 10 |
| Eviction Policy | 7 | 10 |
| Интеграция в систему | 7 | 10 |
| Управляемость | 6 | 10 |

### 9.2 Расчёт общей оценки

`
Итоговая оценка = (TTL*0.3 + ThreadSafety*0.25 + Eviction*0.2 + Integration*0.15 + Manageability*0.1)
Итоговая оценка = (5*0.3 + 8*0.25 + 7*0.2 + 7*0.15 + 6*0.1)
Итоговая оценка = 1.5 + 2.0 + 1.4 + 1.05 + 0.6 = 6.55
`

### 9.3 Оценка: 6.5/10 - УДОВЛЕТВОРИТЕЛЬНО

### 9.4 Выводы

#### Положительные аспекты:

1. **Thread Safety реализована корректно** - использование RLock обеспечивает безопасный конкурентный доступ
2. **LRU политика реализована правильно** - OrderedDict с move_to_end/popitem корректно реализует LRU
3. **Интеграция в semantic search** - кэш используется в правильных местах для оптимизации поиска
4. **Простота кода** - реализация компактна и понятна
5. **Статистика** - есть stats() метод для мониторинга

#### Недостатки:

1. **Нет автоматической очистки TTL** - просроченные записи не удаляются без обращения
2. **Eviction не учитывает TTL** - при вытеснении не проверяется актуальность
3. **Грубая инвалидация** - полная очистка кэша вместо выборочной
4. **Нет фонового потока очистки** - система полностью зависит от обращений get()

### 9.5 Заключение

LRUCacheWithTTL представляет собой базовую, но функциональную реализацию LRU кэша с TTL. Текущая реализация подходит для систем с низкой и средней нагрузкой, где просроченные записи в кэше не являются критичной проблемой.

Для production системы рекомендуется:
- Обязательно добавить фоновую очистку
- Рассмотреть возможность использования cachetools или functools.lru_cache с TTL
- Добавить метрики для мониторинга эффективности кэширования

---

**Конец отчёта**

*Отчёт подготовлен EVA AI Analysis System*
