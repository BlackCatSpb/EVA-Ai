# АУДИТ Memory Подсистемы EVA AI

**Дата:** 14.04.2026  
**Версия:** DualGenerator/FractalGraph v2  
**Аудитор:** EVA AI System Audit

---

## РЕЗЮМЕ

### Оценка системы: 4/10 (НЕУДОВЛЕТВОРИТЕЛЬНО)

| Критерий | Оценка | Проблемы |
|----------|--------|----------|
| Архитектура | 3/10 | Дубликаты, фрагментация |
| Безопасность Pickle | 2/10 | Используется без защиты |
| EventBus интеграция | 3/10 | Только 2 компонента |
| Производительность | 6/10 | Хорошая многоуровневость |
| Поддержка | 3/10 | 60+ файлов, сложно |

---

## 1. ФАЙЛОВАЯ СТРУКТУРА (60+ файлов)

### Основные компоненты кэширования:

| Файл | Класс | Назначение | Строк |
|------|-------|------------|-------|
| cache_core.py | HybridTokenCache | Multi-level (VRAM-RAM-Disk) | 410 |
| cache_ram.py | LRUCache | Базовый LRU на OrderedDict | 48 |
| cache_disk.py | TokenDiskCache | Дисковый кэш с JSON индексом | 275 |
| memory_cache.py | MemoryCache | ДУБЛИКАТ LRUCache | 93 |
| disk_cache.py | DiskCache | SQLite + pickle + zlib | 410 |
| embedding_cache.py | EmbeddingCache | SQLite для эмбеддингов | 241 |
| semantic_cache.py | SemanticCache | In-memory семантический | 132 |
| hybrid_token_cache.py | - | Реэкспорт cache_core | 9 |

### Подсистемы памяти:

| Директория | Компонент | Описание |
|------------|-----------|----------|
| fractal_graph_v2/ | FractalGraphV2 | Граф памяти с SQLite |
| fractal_cache/ | FractalCache | Кэш ответов с эмбеддингами |
| fractal_torch_storage/ | FractalWeightStorage | PyTorch веса |
| pie_integration/ | PIE adapters | Интеграция с PIE |
| ltm_core.py | LongTermMemory | Семантическая + Эпизодическая |
| working_memory.py | WorkingMemory | Краткосрочная память |
| manager_core.py | MemoryManager | Главный менеджер |

---

## 2. КРИТИЧЕСКИЕ ДУБЛИКАТЫ

### 2.1 LRUCache ДУБЛИКАТ

**Файл 1:** cache_ram.py (строки 7-48)

```python
class LRUCache:
    def __init__(self, max_size: int):
        self.max_size = max(1, max_size)
        self.cache = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
```

**Файл 2:** memory_cache.py (строки 13-48)

```python
class MemoryCache:
    def __init__(self, max_memory_tokens: int = 10000):
        self.max_memory_tokens = max_memory_tokens
        self.memory_cache = OrderedDict()
        self.memory_lock = threading.Lock()
    
    def get(self, token_hash: str) -> Optional[Dict[str, Any]]:
        with self.memory_lock:
            if token_hash in self.memory_cache:
                token_data = self.memory_cache[token_hash]
                self.memory_hits += 1
                self.memory_cache.move_to_end(token_hash)
                return token_data
        return None
```

**Проблема:** Идентичная функциональность! Отличия только в именах переменных.

**Решение:** Объединить в один класс LRUCache.

---

### 2.2 DiskCache ДУБЛИКАТ

**TokenDiskCache** (cache_disk.py):
- JSON файл индекс
- Pickle для данных
- Простая структура

**DiskCache** (disk_cache.py):
- SQLite для метаданных
- Pickle + Zlib компрессия
- Rate limiting (throttling)

**Проблема:** Переработка функциональности. DiskCache более зрелый.

**Решение:** Удалить TokenDiskCache, оставить DiskCache.

---

## 3. PICKLE ИСПОЛЬЗОВАНИЕ (ОПАСНО!)

### 3.1 Локации использования:

| Файл | Класс | Опасность | Строки |
|------|-------|-----------|--------|
| cache_disk.py | TokenDiskCache | КРИТИЧЕСКАЯ | 82,97,102,121,122 |
| disk_cache.py | DiskCache | КРИТИЧЕСКАЯ | 7,250,344,345 |
| fractal_torch_storage/base_storage.py | FractalWeightStorage | КРИТИЧЕСКАЯ | 206,216,229,232 |

### 3.2 Примеры кода:

**cache_disk.py:122** - Прямой pickle без валидации:
```python
data = pickle.dumps(token_data)  # ОПАСНО!
```

**disk_cache.py:250** - Pickle с компрессией:
```python
serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
compressed = zlib.compress(serialized)
```

**fractal_torch_storage/base_storage.py:216** - Сохранение весов:
```python
pickle.dump(data, f)  # ОПАСНО!
```

### 3.3 Безопасность:

**Проблемы:**
1. Pickle может выполнять произвольный код при десериализации
2. Нет проверки типа данных
3. Нет валидации схемы

**Рекомендации:**
- Заменить на JSON или MessagePack
- Использовать pydantic для валидации
- Добавить криптографическую подпись данных

---

## 4. EVENTBUS ИНТЕГРАЦИЯ

### 4.1 Компоненты с EventBus:

**MemoryManager** (manager_core.py:82-100):
```python
def _setup_event_connections(self):
    if self.event_bus:
        self.event_bus.subscribe("memory.optimized", self._on_memory_optimized)
        self.event_bus.subscribe("memory.warning", self._on_memory_warning)
        self.event_bus.subscribe("system.state_changed", self._on_system_state_changed)
```

**UnifiedFractalMemory** (unified_fractal_memory.py:215-237):
```python
def _publish_migration_event(self, from_tier: str, to_tier: str, node_ids: List[str]):
    if self._event_bus is None:
        from eva_ai.core.event_bus import get_event_bus
        self._event_bus = get_event_bus()
    
    event = self._Event(...)
    self._event_bus.publish(event)
```

### 4.2 Проблемы интеграции:

| Проблема | Описание |
|----------|----------|
| Слабое использование | Только 2 компонента из 60+ |
| Непоследовательность | Нет централизованного паттерна |
| Пропущенные события | MemoryManager не публикует события |

### 4.3 Рекомендации:

1. MemoryManager должен публиковать memory.updated, memory.cleared
2. Все кэши должны публиковать cache.hit, cache.miss
3. FractalGraphV2 должен публиковать graph.node_added, graph.edge_created

---

## 5. АРХИТЕКТУРНЫЕ ПРОБЛЕМЫ

### 5.1 Сложность навигации:

```
memory/
├── cache_core.py         # HybridTokenCache (main)
├── cache_ram.py          # LRUCache
├── cache_disk.py         # TokenDiskCache
├── memory_cache.py       # ДУБЛИКАТ!
├── disk_cache.py         # DiskCache
├── embedding_cache.py    # EmbeddingCache
├── semantic_cache.py     # SemanticCache
├── hybrid_token_cache.py # Реэкспорт
├── manager_core.py       # MemoryManager
├── working_memory.py     # WorkingMemory
├── ltm_core.py           # LongTermMemory
├── unified_fractal_memory.py
├── fractal_graph_v2/     # 15 файлов
├── fractal_cache/        # 5 файлов
├── fractal_torch_storage/ # 6 файлов
└── pie_integration/      # 5 файлов
```

**Итого:** ~60 файлов в одной директории!

### 5.2 Перекрёстные зависимости:

```
MemoryManager
├── HybridTokenCache (cache_core)
├── FractalGraphV2 (fractal_graph_v2)
├── LongTermMemory (ltm_core)
└── WorkingMemory (working_memory)

HybridTokenCache
├── LRUCache (cache_ram)
└── TokenDiskCache (cache_disk)
```

### 5.3 Рекомендуемая реструктуризация:

```
eva_ai/memory/
├── caches/              # Все кэши в одном месте
│   ├── __init__.py
│   ├── lru_cache.py     # ЕДИНСТВЕННЫЙ LRU
│   ├── hybrid_cache.py  # HybridTokenCache
│   ├── disk_cache.py    # DiskCache (удалить TokenDiskCache)
│   ├── embedding_cache.py
│   └── semantic_cache.py
├── memory/
│   ├── manager.py       # MemoryManager
│   ├── working.py       # WorkingMemory
│   ├── longterm.py      # LongTermMemory
│   └── tiered.py        # UnifiedFractalMemory
├── graph/
│   └── fractal_v2.py    # FractalGraphV2
└── storage/
    ├── torch_storage.py # FractalWeightStorage
    └── ...
```

---

## 6. ПОДРОБНЫЙ АНАЛИЗ КОМПОНЕНТОВ

### 6.1 HybridTokenCache (cache_core.py)

**Плюсы:**
- Многоуровневая архитектура (VRAM-RAM-Disk)
- Динамические лимиты памяти
- Статистика и метрики
- Event loop для pressure handling

**Минусы:**
- Сложная структура (410 строк)
- Зависит от дубликата LRUCache
- Использует TokenDiskCache с pickle

**Строка 129:** Проблемная связь:
```python
self.memory_cache = self.ram_cache  # Использует LRUCache!
self.disk_cache = TokenDiskCache(...)  # Pickle внутри!
```

---

### 6.2 FractalGraphV2 (fractal_graph_v2/storage.py)

**Плюсы:**
- SQLite с векторными индексами
- Семантический поиск по косинусной мере
- Фрактальные уровни L0-L3
- Детекция противоречий
- JSON сериализация (без pickle!)

**Минусы:**
- 1294 строки - слишком большой класс
- Нет EventBus интеграции
- Ручное управление соединениями

**Сериализация (строки 988-1081):**
```python
def save_to_blob(self, compression: str = "zstd") -> bytes:
    # Использует JSON + zstd/gzip - ПРАВИЛЬНО!
    json_data = json.dumps(data, ensure_ascii=False)
    json_bytes = json_data.encode('utf-8')
    
    if compression == "zstd" and HAS_ZSTD:
        compressed = zstd.compress(json_bytes, level=3)
```

---

### 6.3 EmbeddingCache (embedding_cache.py)

**Плюсы:**
- SQLite persistence
- SHA256 хеширование ключей
- WAL mode для concurrency
- LRU eviction через accessed_at

**Минусы:**
- Нет бинарного индекса для векторов
- Один pickle-free (хорошо!)

**Структура:**
```python
# ПРАВИЛЬНО - не использует pickle!
embedding = json.loads(row[0])  # JSON для векторов
```

---

## 7. СТАТИСТИКИ И МЕТРИКИ

### 7.1 Файлы по размеру:

| Файл | Строк | Назначение |
|------|-------|------------|
| fractal_graph_v2/storage.py | 1294 | Граф памяти |
| disk_cache.py | 410 | Дисковый кэш |
| cache_core.py | 410 | Hybrid cache |
| ltm_core.py | 371 | Долгосрочная память |
| embedding_cache.py | 241 | Кэш эмбеддингов |
| fractal_torch_storage/base_storage.py | 239 | PyTorch хранилище |
| cache_disk.py | 275 | Token disk cache |
| semantic_cache.py | 132 | Семантический кэш |
| memory_cache.py | 93 | ДУБЛИКАТ |
| cache_ram.py | 48 | Базовый LRU |

### 7.2 Использование сериализации:

| Компонент | Pickle | Zlib | JSON | SQLite |
|-----------|--------|------|------|--------|
| TokenDiskCache | Да | Нет | Индекс | Нет |
| DiskCache | Да | Да | Нет | Метаданные |
| FractalWeightStorage | Да | Нет | Нет | Нет |
| EmbeddingCache | Нет | Нет | Да | Да |
| FractalGraphV2 | Нет | Да* | Да | Да |

*zstd опционально

---

## 8. РЕКОМЕНДАЦИИ

### 8.1 Немедленные действия (критические):

1. **Удалить memory_cache.py** - дубликат LRUCache
2. **Удалить TokenDiskCache** (cache_disk.py) - дубликат DiskCache
3. **Заменить pickle** на MessagePack или JSON в:
   - fractal_torch_storage/base_storage.py
   - disk_cache.py (опционально с компрессией)

### 8.2 Среднесрочные (1-2 недели):

4. **Реструктурировать директории** согласно разделу 5.3
5. **Добавить EventBus** во все ключевые компоненты
6. **Создать единый интерфейс** кэширования

### 8.3 Долгосрочные:

7. **Типизация** - добавить pydantic модели
8. **Тестирование** - unit тесты для всех кэшей
9. **Документация** - ADR для архитектурных решений

---

## 9. ВЫВОДЫ

### 9.1 Что хорошо:

- Многоуровневая архитектура кэширования (VRAM-RAM-Disk)
- FractalGraphV2 с хорошим дизайном (JSON вместо pickle)
- EmbeddingCache использует SQLite + JSON правильно
- SemanticMemory и EpisodicMemory хорошо разделены

### 9.2 Что плохо:

- **Дубликаты** - 2 LRU, 2 DiskCache
- **Pickle** - небезопасная десериализация
- **EventBus** - почти не используется
- **Структура** - 60+ файлов без группировки
- **Сложность** - классы по 1000+ строк

### 9.3 Финальная оценка: 4/10

**Причина:** Критические проблемы безопасности (pickle), дублирование функциональности, слабая интеграция.

---

*Отчёт сгенерирован EVA AI System Audit*
