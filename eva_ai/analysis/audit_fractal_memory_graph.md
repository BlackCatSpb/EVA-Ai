## Содержание
1. [Архитектура графа](#1-архитектура-графа)
2. [Методы работы с кластерами](#2-методы-работы-с-кластерами)
3. [Проблема с get_clusters()](#3-проблема-с-get_clusters)
4. [Embedding Device настройки](#4-embedding-device-настройки)
5. [Интеграция с memory_manager](#5-интеграция-с-memory_manager)
6. [Fault Tolerance](#6-fault-tolerance)
7. [Оценка по 10-балльной шкале](#7-оценка-по-10-балльной-шкале)
8. [Конкретные рекомендации](#8-конкретные-рекомендации)
---
## 1. Архитектура графа
### 1.1 Компоненты системы
```
FractalMemoryGraph (__init__.py) - WRAPPER
├── storage: FractalGraphV2          # Хранилище (SQLite)
├── embeddings: EmbeddingsManager     # Векторизация
└── _search_cache: LRUCacheWithTTL   # Кэш поиска (100 entry, 300s TTL)
FractalGraphV2 (storage.py) - ОСНОВНОЕ ХРАНИЛИЩЕ
├── nodes: Dict[str, FractalNode]           # Словарь узлов в памяти
├── edges: Dict[str, FractalEdge]           # Словарь связей
├── semantic_groups: Dict[str, SemanticGroup] # Семантические группы
└── SQLite DB: fractal_graph.db              # Персистентное хранение
```
### 1.2 Структура данных
**FractalNode (types.py):**
- `id`: str - SHA256 хеш контента
- `content`: str - текстовое содержание
- `node_type`: str - тип узла (concept, fact, detail, query, response и др.)
- `level`: int - фрактальный уровень (0-LN, где 0 самый глубокий)
- `parent_group_id`: Optional[str] - ID семантической группы
- `embedding`: Optional[List[float]] - вектор (768d для multilingual-e5-base)
- `confidence`: float - уверенность (0-1)
- `temporal_weight`: float - временной вес для decay
- `domain_lambda`: float - коэффициент распада
- `is_contradiction`: bool - флаг противоречия
- `is_static`: bool - статичный узел (не удаляется)
**FractalEdge (types.py):**
- `id`: str - SHA256 хеш(source_id + target_id + relation)
- `source_id`: str
- `target_id`: str
- `relation_type`: str - тип связи (is_a, part_of, contradicts, related_to и др.)
- `weight`: float - вес связи
- `contradiction_flag`: bool
**SemanticGroup (types.py):**
- `id`: str - хеш группы
- `name`: str - название
- `embedding`: Optional[List[float]] - центроид группы
- `member_count`: int - количество членов
- `avg_confidence`: float - средняя уверенность
- `cluster_coherence`: float - связность кластера (0-1)
- `needs_recluster`: bool - флаг перекластеризации
### 1.3 Хранение в SQLite
**Таблица nodes:**
```sql
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    node_type TEXT NOT NULL,
    level INTEGER DEFAULT 0,
    parent_group_id TEXT,
    embedding BLOB,
    confidence REAL DEFAULT 0.5,
    created_at REAL,
    updated_at REAL,
    last_accessed REAL,
    metadata TEXT,
    access_count INTEGER DEFAULT 0,
    version INTEGER DEFAULT 1,
    is_static INTEGER DEFAULT 0,
    is_contradiction INTEGER DEFAULT 0
)
```
**Индексы:**
- `idx_nodes_type` - по типу узла
- `idx_nodes_level` - по уровню
- `idx_nodes_parent_group` - по группе
- `idx_edges_source`, `idx_edges_target` - по связям
- `idx_groups_level` - по уровню группы
### 1.4 Векторное хранилище
**ANN-поиск:** Ручная реализация косинусного сходства (NOT HNSW/FAISS)
- Нормализованные векторы кэшируются в `_normalized_embeddings: Dict[str, np.ndarray]`
- Групповые векторы в `_group_embeddings: Dict[str, np.ndarray]`
- similarity threshold = 0.5 (жестко закодирован)
---
## 2. Методы работы с кластерами
### 2.1 Доступные методы
| Метод | Класс | Описание |
|-------|-------|----------|
| `get_groups(level)` | FractalMemoryGraph | Возвращает список SemanticGroup |
| `get_group_members(group_id)` | FractalGraphV2 | Возвращает членов группы |
| `cluster_nodes(level, threshold, method)` | FractalGraphV2 | Кластеризация узлов уровня |
| `auto_cluster(level, threshold, method)` | FractalMemoryGraph | Автокластеризация с созданием групп |
| `create_semantic_group(name, member_ids)` | FractalGraphV2 | Создание группы |
| `_find_nearest_group(embedding, level, threshold)` | FractalGraphV2 | Инкрементальное присоединение |
### 2.2 Алгоритмы кластеризации
**1. agglomerative (иерархическая агломеративная):**
- Начинает с каждого узла как отдельного кластера
- Итеративно объединяет пару кластеров с максимальной близостью центроидов
- Прекращает при `similarity < threshold`
- Сложность: O(n^2) на каждой итерации
**2. dbscan (упрощённый):**
- eps = 0.5 (жёстко закодирован)
- min_samples = 2 (жёстко закодирован)
- BFS для расширения кластеров
**3. simple:**
- Присоединение к ближайшей существующей группе
- Если `similarity > threshold` - присоединяется
- Иначе создаётся новый кластер
### 2.3 Инкрементальная кластеризация
При добавлении узла (`add_node`):
```python
if auto_cluster and node.embedding:
    best_group = storage._find_nearest_group(
        node.embedding, level, cluster_threshold  # threshold=0.6
    )
    if best_group:
        node.parent_group_id = best_group
```
---
## 3. Проблема с get_clusters()
### 3.1 Констатация проблемы
**Метод `get_clusters()` ОТСУТСТВУЕТ в FractalGraphV2 и FractalMemoryGraph.**
Существующие методы:
- `get_groups(level)` - возвращает **SemanticGroup** объекты, НЕ кластеры с узлами
- `get_group_members(group_id)` - возвращает узлы конкретной группы
**Отсутствует метод для получения кластеров как словаря `{cluster_name: [node_ids]}`**
### 3.2 Как компоненты обходят эту проблему
**ConceptMiner._get_clusters() (concept_miner.py:442-507):**
Концепт майнер делает прямое обращение к storage.semantic_groups и выполняет O(n^2) кластеризацию на лету как fallback. Это создаёт дублирование логики и проблемы с производительностью.
### 3.3 Последствия
1. **ConceptMiner не может эффективно получать кластеры**
2. **Дублирование логики кластеризации** между `cluster_nodes()` и `_get_clusters()`
3. **Несоответствие API** - разные компоненты ожидают разные структуры
---
## 4. Embedding Device настройки
### 4.1 EmbeddingsManager
```python
class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "eva_ai/core/hf_cache/multilingual-e5-base",
        device: str = "cuda",  # ПО УМОЛЧАНИЮ CUDA!
        cache_dir: str = None,
        batch_size: int = 32,
        max_length: int = 512
    ):
```
### 4.2 FractalMemoryGraph
```python
class FractalMemoryGraph:
    def __init__(
        self,
        storage_dir: str = None,
        embedding_model: str = "eva_ai/core/hf_cache/multilingual-e5-base",
        embedding_device: str = "cuda",  # ПО УМОЛЧАНИЮ CUDA!
        embedding_dim: int = 768
    ):
```
### 4.3 Проблемы
1. **CUDA по умолчанию** - если GPU недоступен,fallback на CPU (медленно)
2. **Нет автоопределения device** - torch.cuda.is_available() не используется
3. **Batch size фиксирован** - 32, нет адаптации под память GPU
4. **Кэш эмбеддингов** - только in-memory, нет персистентности
### 4.4 Fallback механизм
При ошибке векторизации используются случайные векторы, что полностью ломает семантический поиск.
---
## 5. Интеграция с memory_manager
### 5.1 Интеграционная схема
```
CoreBrain
├── memory_manager: MemoryManager
│   └── fractal_graph_v2: FractalMemoryGraph (если enabled)
└── fractal_graph_v2: FractalMemoryGraph (прямой доступ)
```
### 5.2 MemoryManager (manager_core.py)
Инициализация происходит в 3 этапа:
1. Проверка: передан ли уже инстанс
2. Проверка: есть ли fractal_graph_v2 в brain
3. Создание нового инстанса если не найден
### 5.3 Доступные методы в MemoryManager
| Метод | Описание |
|-------|----------|
| `search_knowledge(query, top_k)` | Семантический поиск |
| `add_fact(subject, relation, object_)` | Добавить S-P-O факт |
| `get_graph_stats()` | Статистика графа |
| `verify_knowledge(knowledge)` | Проверка через self_dialogue |
### 5.4 Проблемы интеграции
1. **Двойное хранение:** FractalMemoryGraph создаётся и в CoreBrain и в MemoryManager
2. **Нет синхронизации** между двумя инстансами
3. **MemoryManager не управляет жизненным циклом** FractalGraphV2
4. **Конфигурация разная:** в brain_config.json и в manager_core.py
---
## 6. Fault Tolerance
### 6.1 Существующие механизмы
**A. SnapshotManager**
- Неизменяемые снимки для консистентности генерации
- Создание снимков по запросу
- cleanup_expired() удаляет снимки старше TTL (300s)
Проблемы:
- Не используется в основном цикле FractalGraphV2
- Только для генерации, не для восстановления после сбоев
- Нет персистентности снимков на диск
**B. Сериализация в файл**
- save_to_file / load_from_file
- Формат: [4 bytes header_len][header JSON][compressed blob]
- checksum для проверки целостности
Проблемы:
- Нет автоматических бэкапов
- Нет инкрементного сохранения
- Нет проверки целостности при записи
- load_from_file теряет связи если blob повреждён
**C. Exception handling**
- Generic Exception handling
- Частичное восстановление после ошибок не предусмотрено
- Нет rollback механизма для транзакций
### 6.2 Отсутствующие механизмы
| Механизм | Статус |
|----------|--------|
| Автоматические бэкапы | OTL |
| Репликация | OTL |
| Write-ahead logging (WAL) | OTL |
| Точка восстановления (checkpoint) | OTL |
| Целостность данных (checksum) | Частично |
| Деградация при сбое | OTL |
| Таймауты для операций | OTL |
### 6.3 Уязвимости
1. **SQLite без WAL** - блокировка записи при чтении
2. **Нет транзакционности** - частичная запись при сбое
3. **In-memory словари** - потеря данных при краше процесса
4. **embedding кэш** - теряется при перезапуске
5. **Хрупкое восстановление** - один повреждённый blob = потеря всего графа
---
## 7. Оценка по 10-балльной шкале
| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| **Функциональность** | 8/10 | Полный набор операций с графом, но нет get_clusters() |
| **Архитектура** | 7/10 | Хорошее разделение storage/wrapper, но дублирование |
| **Кластеризация** | 6/10 | 3 алгоритма, но инкрементальная работает плохо |
| **Embedding** | 5/10 | CUDA по умолчанию,fallback на случайные векторы |
| **Интеграция** | 6/10 | MemoryManager knows FGv2, но есть дублирование |
| **Fault Tolerance** | 3/10 | Нет автобэкапов, нет WAL, уязвим к сбоям |
| **Performance** | 6/10 | LRU cache, batch operations, но O(n^2) кластеризация |
| **API полнота** | 5/10 | Нет get_clusters(), приходится обходить |
| **Документация** | 6/10 | Есть docstrings, но нет диаграмм |
| **Тестирование** | 4/10 | Нет явных тестов в коде |
| **ИТОГО: 5.6/10** |
---
## 8. Конкретные рекомендации
### Критические (влияют на работоспособность)
#### 1. Реализовать get_clusters() метод
```python
def get_clusters(self, level: int = None, min_members: int = 2) -> Dict[str, List[str]]:
    clusters = {}
    for group_id, group in self.semantic_groups.items():
        if level and group.level != level:
            continue
        member_ids = self.nodes_by_group.get(group_id, [])
        if len(member_ids) >= min_members:
            clusters[group.name or group_id] = member_ids
    if not clusters:
        clusters = self.cluster_nodes(level=level, threshold=0.5, method="agglomerative")
    return clusters
```
#### 2. Исправить embedding device определение
```python
def _get_device(self) -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```
#### 3. Включить SQLite WAL mode
```python
def _init_database(self):
    conn = sqlite3.connect(self.db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
```
### Высокий приоритет (улучшают стабильность)
#### 4. Автоматические бэкапы
- Создание бэкапа при достижении N изменений
- Ротация бэкапов (keep последние 5)
#### 5. Инкрементное сохранение узлов
- UPDATE вместо INSERT OR REPLACE дляminor изменений
#### 6. Интеграция SnapshotManager в основной цикл
- Создание снимков при семантическом поиске
### Средний приоритет (улучшают maintainability)
#### 7. Унифицировать доступ к кластерам
- get_cluster_for_node()
- get_cluster_stats()
#### 8. Добавить метрики мониторинга
- nodes_count, edges_count, groups_count
- avg_degree, clustering_coefficient
- orphaned_nodes, db_size_mb
---
## Резюме
FractalMemoryGraph (FractalGraphV2) является центральным хранилищем знаний EVA AI системы. Архитектура в целом зрелая с хорошим разделением на storage/wrapper слои.
**Главные проблемы:**
1. **Отсутствие get_clusters()** - вынуждает компоненты делать обходные пути
2. **Fault tolerance** - нет автобэкапов, уязвим к сбоям
3. **Embedding fallback** - случайные векторы ломают семантический поиск
**Оценка: 5.6/10** - требует доработки в первую очередь по fault tolerance и API полноте.
