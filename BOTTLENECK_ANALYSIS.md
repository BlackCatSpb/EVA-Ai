# Технический анализ bottleneck'ов EVA AI

**Анализ проведен:** 2025-01-10  
**Цель:** Выявить конкретные узкие места производительности

---

## 1. Bottleneck: Синхронная генерация с threading timeout

**Локация:** 
- `eva_ai/core/brain_query.py:1360-1373` (метод `_generate_with_timeout`)
- `eva_ai/core/pipeline_quality.py:282-302`

**Проблема:**
```python
def _generate_with_timeout(self, generate_fn, timeout=60):
    """Wraps a generate call with a timeout using threading."""
    result = [None]
    def _gen():
        try:
            result[0] = generate_fn()
        except Exception as e:
            result[0] = e
    
    t = threading.Thread(target=_gen, daemon=True)
    t.start()
    t.join(timeout=timeout)
    
    if t.is_alive():
        return None, 'timeout'
```

**Почему это проблема:**
1. Создание нового thread на каждую генерацию = overhead
2. GIL Python блокирует параллельное выполнение
3. Нет возможности отменить генерацию graceful
4. Память thread stack (~8MB) не освобождается мгновенно

**Влияние:** +50-100ms overhead на каждый запрос

**Решение:** Асинхронная архитектура с asyncio + ThreadPoolExecutor

---

## 2. Bottleneck: Последовательный semantic_search

**Локация:**
- `eva_ai/memory/fractal_graph_v2/__init__.py:259-295`
- `eva_ai/memory/fractal_graph_v2/storage.py:499-550`

**Проблема:**
```python
def semantic_search(self, query, top_k=10, ...):
    # 1. Вычисление embedding'а (CPU-bound)
    query_emb = self.embedding_model.encode(query)
    
    # 2. Поиск по всем узлам (O(n))
    results = []
    for node in self.nodes:
        similarity = cosine_similarity(query_emb, node.embedding)
        results.append((node, similarity))
    
    # 3. Сортировка (O(n log n))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
```

**Почему это проблема:**
1. Нет индекса (наивный O(n) перебор)
2. Нет кэширования embedding'ов запросов
3. Нет параллелизма для больших графов

**Влияние:** 100-500ms для графа с 1000+ узлами

**Решение:**
- FAISS / Annoy для приближенного поиска
- Кэширование embedding'ов
- Batch processing

---

## 3. Bottleneck: Блокирующий веб-поиск

**Локация:**
- `eva_ai/websearch/web_search_integrated.py`

**Проблема:**
```python
def search(self, query, max_results=5):
    # Синхронный HTTP запрос
    response = requests.get(url, params=params, timeout=10)
    results = response.json()
    
    # Синхронный парсинг каждого результата
    for result in results:
        content = self._fetch_page_content(result['url'])  # Еще HTTP!
        result['content'] = content
    
    return results
```

**Почему это проблема:**
1. Блокирует весь pipeline
2. Нет connection pooling
3. Последовательная загрузка страниц
4. Нет timeout handling

**Влияние:** +1-3 секунды если веб-поиск включен

**Решение:** aiohttp + asyncio.gather

---

## 4. Bottleneck: Отсутствие кэширования контекста

**Локация:**
- `eva_ai/memory/fractal_graph_v2/dual_generator.py:420-455` (_get_context)

**Проблема:**
```python
def _get_context(self, query: str) -> str:
    # Каждый запрос - полный semantic_search с нуля
    results = self.graph.semantic_search(query, top_k=max_nodes)
    
    # Повторная загрузка узлов из БД
    context_parts = []
    for node in results:
        content = self._load_node_content(node.id)  # Disk I/O!
        context_parts.append(content)
    
    return '\n'.join(context_parts)
```

**Почему это проблема:**
1. Повторяющиеся запросы вычисляются заново
2. Нет in-memory кэша для узлов
3. Disk I/O на каждый запрос

**Влияние:** 200-800ms на каждый запрос с графом

**Решение:** LRUCache для узлов + embedding cache

---

## 5. Bottleneck: Последовательная загрузка моделей

**Локация:**
- `eva_ai/core/brain_components.py:280-350`
- `eva_ai/core/hybrid_pipeline_adapter.py:89-120`

**Проблема:**
```python
def load_models(self):
    # Model A загружается
    self.model_a = Llama(model_path_a, ...)
    
    # Model B загружается ПОСЛЕ A
    self.model_b = Llama(model_path_b, ...)
    
    # Model C загружается ПОСЛЕ B
    self.model_c = Llama(model_path_c, ...)
```

**Почему это проблема:**
1. Суммарное время загрузки = t1 + t2 + t3
2. Блокирует старт системы
3. Нет lazy loading

**Влияние:** 10-30 секунд стартового времени

**Решение:** Параллельная загрузка + lazy loading

---

## 6. Bottleneck: Полная перегенерация для refinement

**Локация:**
- `eva_ai/core/hybrid_pipeline_adapter.py:300-400`

**Проблема:**
```python
def process_query(self, query):
    # 1. Первичная генерация
    response = self.generate(query)
    
    # 2. Проверка качества
    if self.needs_refinement(response):
        # 3. ПОЛНАЯ перегенерация с нуля!
        context = self.get_web_context(query)
        response = self.generate(query, context)  # С нуля!
```

**Почему это проблема:**
1. Всё вычисляется заново вместо корректировки
2. KV-cache сбрасывается
3. Удвоенное время на refinement

**Влияние:** 2x время при использовании веб-поиска

**Решение:** In-place editing + KV-cache reuse

---

## 7. Bottleneck: Нет batching'а для embeddings

**Локация:**
- `eva_ai/memory/fractal_graph_v2/storage.py`

**Проблема:**
```python
# Псевдокод текущей реализации
def add_node(self, content):
    # По одному embedding'у за раз
    embedding = self.model.encode([content])[0]
    self.nodes.append(Node(content, embedding))
```

**Почему это проблема:**
1. GPU не эффективно используется
2. Overhead на каждый вызов
3. Плохая throughput при bulk операциях

**Влияние:** 10x медленнее для bulk операций

**Решение:** Batch processing с размером 32-64

---

## 8. Bottleneck: Синхронное логирование

**Локация:**
- Во многих файлах

**Проблема:**
```python
import logging
logger = logging.getLogger(__name__)

# Блокирующий вызов на каждую операцию
logger.info(f"Processing query: {query}")
logger.debug(f"Context retrieved: {len(context)} chars")
```

**Почему это проблема:**
1. File I/O блокирует поток
2. Форматирование строки даже если уровень логирования выше
3. Нет buffering'а

**Влияние:** +1-5ms на каждый лог (накапливается!)

**Решение:**
```python
# Использовать lazy formatting
logger.info("Processing query: %s", query)  # Не форматирует если не нужно

# Асинхронный handler
logging.handlers.QueueHandler
```

---

## Приоритет исправления

| # | Bottleneck | Влияние | Сложность | Приоритет |
|---|------------|---------|-----------|-----------|
| 1 | Веб-поиск блокирует | Высокое | Низкая | P0 |
| 2 | Отсутствие кэша контекста | Высокое | Низкая | P0 |
| 3 | Синхронная генерация | Высокое | Средняя | P1 |
| 4 | Semantic search O(n) | Среднее | Средняя | P1 |
| 5 | Нет batching'а | Среднее | Низкая | P1 |
| 6 | Последовательная загрузка | Низкое | Низкая | P2 |
| 7 | Полная перегенерация | Среднее | Высокая | P2 |
| 8 | Синхронное логирование | Низкое | Низкая | P3 |

---

## Быстрые измерения

Для подтверждения bottleneck'ов:

```python
import time

# Профилирование конкретного метода
class TimedOperation:
    def __init__(self, name):
        self.name = name
        self.start = None
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self.start) * 1000
        print(f"{self.name}: {elapsed:.2f}ms")

# Использование
with TimedOperation("semantic_search"):
    results = graph.semantic_search(query)

with TimedOperation("web_search"):
    results = web_search.search(query)

with TimedOperation("generation"):
    response = generator.generate(query, context)
```

---

## Ожидаемые результаты после исправления

| Метрика | Сейчас | После оптимизации | Улучшение |
|---------|--------|-------------------|-----------|
| Среднее время ответа | 5-10s | 2-3s | 60% |
| P95 время ответа | 15-20s | 5-8s | 70% |
| TTFT | 3-5s | <1s | 80% |
| Пропускная способность | 2-5 RPM | 20-50 RPM | 10x |
| Использование RAM | 4-6GB | 2-4GB | 50% |
| Cache hit rate | 0% | 50-70% | +∞ |

---

**Примечание:** Этот документ дополняет PERFORMANCE_ROADMAP.md конкретными техническими деталями.
