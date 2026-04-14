# Отчёт: Storage/Cache

## 1. Структура

### 1.1 Директория va_ai/storage/

| Файл | Назначение |
|------|------------|
| storage_types.py | Типы данных: StorageType, AccessPattern, StorageMetrics, StorageEntry |
| ractal_storage.py | Базовый класс FractalStorage - файловый JSON- keyed storage |

### 1.2 Директория va_ai/memory/

**Основные файлы кэша:**

| Файл | Назначение |
|------|------------|
| cache_core.py | HybridTokenCache - главный гибридный кэш токенов |
| cache_ram.py | LRUCache - RAM кэш на OrderedDict |
| cache_disk.py | TokenDiskCache - дисковый кэш с pickle сериализацией |
| cache_eviction.py | Политики вытеснения, мониторинг памяти |
| cache_index.py | CacheIndex - SQLite индекс для маршрутизации |
| cache_router.py | CacheRouter - маршрутизация по HybridTokenCache |
| cache_types.py | Типы: CacheLevel, CacheStrategy, CacheEntry, CacheStats |
| memory_cache.py | MemoryCache - простой LRU кэш в RAM |
| semantic_cache.py | CPU-based семантический кэш контекстов |
| mbedding_cache.py | Кэш эмбеддингов |

**Фрактальный кэш** (va_ai/memory/fractal_cache/):

| Файл | Назначение |
|------|------------|
| cache_manager.py | FractalCache - семантический кэш ответов |
| esponse_store.py | ResponseStore - хранилище ответов на диске |
| semantic_embedder.py | Семантическое кодирование |
| similarity_engine.py | Вычисление similarity |
| viction_policy.py | Политика вытеснения фрактального кэша |

---

## 2. Хранение данных

### 2.1 FractalStorage (storage/fractal_storage.py)

**Формат хранения:** JSON файлы
- Директория по умолчанию: ./data/fractal_storage
- Ключ = имя файла (safe_key с заменой / и \ на _)
- Формат файла:
`json
{
  ""key"": ""original_key"",
  ""data"": <any>,
  ""timestamp"": <time.time()>
}
`

**Методы:**
- store(key, data) → JSON файл
- etrieve(key) → данные или None
- delete(key) → bool
- save_tokenizer() / get_tokenizer() - токенизаторы

---

### 2.2 HybridTokenCache (memory/cache_core.py)

**Трёхуровневая архитектура:**

`
┌─────────────────────┐
│   GPU VRAM (1.5GB)  │  ← vram_cache (LRUCache)
├─────────────────────┤
│   RAM (~1GB)         │  ← ram_cache (LRUCache)
├─────────────────────┤
│   SSD/Disk (50GB)   │  ← TokenDiskCache (pickle .bin)
└─────────────────────┘
`

**TokenDiskCache (cache_disk.py):**
- Директория: <cache_dir>/data/
- Поддиректории по 2-символьному префиксу хеша
- Файлы: <token_id>.bin (pickle сериализация)
- Индекс: disk_cache_index.json

**Данные токена (pickle):**
`python
{
    # любые данные Dict
}
`

**Метаданные токена (JSON):**
`json
{
  ""files"": {
    ""<token_id>"": {
      ""size"": <bytes>,
      ""created"": <timestamp>,
      ""last_access"": <timestamp>,
      ""access_count"": <int>
    }
  },
  ""total_size"": <bytes>
}
`

---

### 2.3 FractalCache (memory/fractal_cache/cache_manager.py)

**Формат хранения:** JSON + numpy embeddings

**ResponseStore (esponse_store.py):**
- Индекс: store_index.json
- Данные: data/<key[:16]>.json
- Эмбеддинги не сохраняются в файле (слишком большие)

**Entry формат:**
`json
{
  ""key"": ""<hash>"",
  ""query"": ""текст запроса"",
  ""response"": ""ответ"",
  ""embedding"": [<float>, ...],
  ""metadata"": {},
  ""timestamp"": <time>,
  ""access_count"": <int>
}
`

---

### 2.4 FractalGraphV2 (memory/fractal_graph_v2/storage.py)

**SQLite база данных:**
- Путь: <storage_dir>/fractal_graph.db
- Таблицы:
  - 
odes - узлы графа
  - dges - связи
  - semantic_groups - семантические группы

**Таблица 
odes:**
`sql
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    node_type TEXT NOT NULL,
    level INTEGER DEFAULT 0,
    parent_group_id TEXT,
    embedding BLOB,          -- сериализованный numpy float32
    confidence REAL DEFAULT 0.5,
    created_at REAL,
    updated_at REAL,
    last_accessed REAL,
    metadata TEXT,           -- JSON
    access_count INTEGER DEFAULT 0,
    version INTEGER DEFAULT 1,
    is_static INTEGER DEFAULT 0,
    is_contradiction INTEGER DEFAULT 0
)
`

**Индексы:**
- idx_nodes_type - по типу узла
- idx_nodes_level - по уровню
- idx_nodes_parent_group - по группе
- idx_edges_source/target - по связям
- idx_groups_level - по уровню группы

**Сериализация эмбеддингов:**
`python
# В binary blob (numpy float32)
embedding = np.array(embedding, dtype=np.float32).tobytes()
# Обратно: np.frombuffer(blob, dtype=np.float32).tolist()
`

---

## 3. Интеграция

### 3.1 Интеграция Cache ↔ FractalGraphV2

**CacheRouter (cache_router.py):**
- Связывает кэш токенов с графом знаний
- Методы:
  - egister_batch() → регистрация батча
  - egister_segment() → сегменты
  - egister_token_nodes() → токен-узлы
  - link_nodes_to_kg() → связывание с графом
  - ank_segments() → ранжирование по весам

**CacheIndex (cache_index.py):**
- SQLite таблицы для маршрутизации:
  - atches - батчи
  - segments - сегменты  
  - 	oken_nodes - токен-узлы
  - weights - веса для ранжирования

### 3.2 SemanticContextCache ↔ FractalGraphV2

**SemanticContextCache (memory/fractal_graph_v2/semantic_context_cache.py):**
- CPU-based семантический поиск контекстов
- Интеграция с sentence_transformers
- FAISS для быстрого ANN-поиска
- Хранит тексты НЕ в графе, но релевантные для поиска

### 3.3 HybridTokenCache в Brain

Инициализация в CoreBrain:
`python
self.token_cache = HybridTokenCache(
    brain=self,
    max_memory_tokens=100000,
    disk_cache_dir="token_cache",
    target_memory_gb=50.0,
    max_disk_cache_gb=50.0
)
`

---

## 4. Оценка

### 4.1 Преимущества

| Компонент | Преимущество |
|----------|--------------|
| **HybridTokenCache** | Трёхуровневая иерархия (VRAM→RAM→Disk) |
| **FractalCache** | Семантический поиск по кэшу ответов |
| **FractalGraphV2** | SQLite + ANN-индексы + кластеризация |
| **CacheRouter** | Адресуемая маршрутизация батчей/сегментов |
| **SemanticContextCache** | FAISS-based семантический поиск |
| **Disk eviction** | LRU при давлении памяти |

### 4.2 Проблемы и риски

| Проблема | Серьёзность | Описание |
|----------|------------|----------|
| **Нет единой шины событий** | Средняя | Компоненты кэша не используют EventBus для инвалидации |
| **Ручная синхронизация индексов** | Средняя | При сбое возможна рассинхронизация |
| **Pickle security** | Высокая | TokenDiskCache использует pickle (уязвимость) |
| **Нет TTL в RAM кэше** | Средняя | Только в SemanticContextCache |
| **FractalStorage примитивен** | Низкая | Простой JSON-keyed storage, нет индексации |

### 4.3 Рекомендации

1. **Безопасность:** Заменить pickle на JSON или msgpack в TokenDiskCache
2. **Интеграция:** Подключить кэш-инвалидацию к EventBus
3. **Мониторинг:** Добавить метрики hit rate в FractalStorage
4. **TTL:** Внедрить TTL для RAM-слоя в HybridTokenCache

---

## Резюме

Система хранения данных EVA AI имеет **многоуровневую гибридную архитектуру**:
- **Storage** (ractal_storage.py) - простой файловый JSON-keyed storage
- **Cache** - три уровня: VRAM/RAM/Disk с LRU eviction
- **FractalCache** - семантический кэш ответов с FAISS
- **FractalGraphV2** - основное хранилище графа знаний (SQLite + ANN)

Интеграция между компонентами осуществляется через CacheRouter и CacheIndex, связывающие кэш токенов с графом знаний. Однако **нет единой системы событий** для синхронизации и инвалидации кэша.
