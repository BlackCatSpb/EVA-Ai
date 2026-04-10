# EVA AI Performance Optimization Roadmap
# Руководство по оптимизации производительности CogniFlex

**Версия:** 1.0  
**Дата:** 2025-01-10  
**Приоритет:** Высокий (медленные ответы)  

---

## Executive Summary

Система EVA AI имеет архитектурные bottleneck'ы, которые приводят к задержкам в обработке запросов. Этот документ содержит пошаговый план устранения проблем производительности с приоритетом на быстрые победы и стратегические улучшения.

**Ключевые метрики для отслеживания:**
- Время ответа (цель: < 3 сек для простых запросов)
- Время до первого токена (TTFT, цель: < 1 сек)
- Пропускная способность (запросы/мин)
- Использование CPU/RAM

---

## Phase 1: Quick Wins (1-2 недели)
### Приоритет: Критический | Влияние: Высокое | Сложность: Низкая

### 1.1 Кэширование Semantic Search
**Проблема:** Каждый запрос выполняет semantic_search с нуля даже для одинаковых запросов.

**Решение:**
```python
# В eva_ai/memory/fractal_graph_v2/__init__.py
class FractalMemoryGraph:
    def __init__(self, ...):
        # Добавить кэш
        self._search_cache = LRUCache(maxsize=100, ttl=300)  # 5 минут
    
    def semantic_search(self, query, top_k=10, ...):
        cache_key = f"{query}:{top_k}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
        
        results = self._perform_search(query, top_k, ...)
        self._search_cache[cache_key] = results
        return results
```

**Ожидаемый эффект:** 30-50% ускорение для повторяющихся запросов  
**Файлы:** `eva_ai/memory/fractal_graph_v2/__init__.py`, `eva_ai/memory/fractal_graph_v2/storage.py`

### 1.2 Batch Processing для Embeddings
**Проблема:** Embeddings вычисляются по одному, что неэффективно.

**Решение:**
```python
# Оптимизировать _compute_embeddings в storage.py
async def compute_embeddings_batch(self, texts: List[str], batch_size=32):
    """Вычисление эмбеддингов батчами."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = await self._embed_async(batch)
        results.extend(batch_embeddings)
    return results
```

**Ожидаемый эффект:** 2-3x ускорение для bulk операций  
**Файлы:** `eva_ai/memory/fractal_graph_v2/storage.py`

### 1.3 Оптимизация Timeout-ов
**Проблема:** Жесткие timeout'ы в 60 секунд без graceful degradation.

**Решение:**
```python
# В hybrid_pipeline_adapter.py
TIERED_TIMEOUTS = {
    'condensed': 10,    # Быстрые ответы
    'extended': 30,     # Развернутые
    'large': 60,        # Большие генерации
    'streaming_chunk': 5  # На чанк при стриминге
}
```

**Ожидаемый эффект:** Более предсказуемое время ответа  
**Файлы:** `eva_ai/core/hybrid_pipeline_adapter.py`, `eva_ai/core/brain_query.py`

### 1.4 Lazy Loading для Веб-Поиска
**Проблема:** Веб-поиск выполняется синхронно и блокирует всё.

**Решение:**
```python
# Запускать веб-поиск параллельно с генерацией
async def process_with_optional_web_search(query):
    generation_task = asyncio.create_task(generate_response(query))
    web_search_task = asyncio.create_task(web_search(query))
    
    # Ждем генерацию, веб-поиск может подтянуться позже
    response = await generation_task
    web_results = await web_search_task
    
    if web_results and needs_refinement(response, web_results):
        response = await refine_with_web_context(response, web_results)
    
    return response
```

**Ожидаемый эффект:** -1-2 секунды на запросы с веб-поиском  
**Файлы:** `eva_ai/websearch/web_search_integrated.py`, `eva_ai/core/hybrid_pipeline_adapter.py`

---

## Phase 2: Async Architecture (2-4 недели)
### Приоритет: Высокий | Влияние: Высокое | Сложность: Средняя

### 2.1 Асинхронный Pipeline
**Проблема:** Блокирующие вызовы моделей останавливают всё.

**Архитектура:**
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Request   │────▶│ Async Queue  │────▶│  Workers    │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                    ┌──────────────┐           │
                    │   Results    │◀──────────┘
                    └──────────────┘
```

**Implementation:**
```python
# Новый файл: eva_ai/core/async_pipeline.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncGenerationPipeline:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.queue = asyncio.Queue(maxsize=100)
        self._start_workers()
    
    async def generate(self, query, context):
        loop = asyncio.get_event_loop()
        # Выполняем блокирующую генерацию в пуле потоков
        return await loop.run_in_executor(
            self.executor, 
            self._sync_generate, 
            query, 
            context
        )
```

**Файлы:** Новый: `eva_ai/core/async_pipeline.py`, Изменения: `eva_ai/core/hybrid_pipeline_adapter.py`

### 2.2 Streaming Generation
**Проблема:** Пользователь ждёт полного ответа перед показом.

**Решение:**
```python
async def generate_streaming(self, query, context):
    """Генерация с потоковой отдачей токенов."""
    buffer = ""
    async for token in self.model.generate_async(query, context, stream=True):
        buffer += token
        if len(buffer) >= 10 or token in {'.', '!', '?', '\n'}:
            yield buffer
            buffer = ""
    if buffer:
        yield buffer
```

**Ожидаемый эффект:** TTFT < 1 сек, улучшение UX  
**Файлы:** `eva_ai/memory/fractal_graph_v2/eva_generator.py`, `eva_ai/gui/web_gui/server_main.py` (SSE)

### 2.3 Connection Pooling для Внешних API
**Проблема:** Новое TCP-соединение на каждый веб-поиск.

**Решение:**
```python
import aiohttp

class WebSearchClient:
    def __init__(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
        )
    
    async def search(self, query):
        async with self.session.get(url, params=params) as resp:
            return await resp.json()
```

**Ожидаемый эффект:** -100-200ms на веб-поиск  
**Файлы:** `eva_ai/websearch/web_search_integrated.py`

---

## Phase 3: Smart Caching & Pre-computation (3-4 недели)
### Приоритет: Средний | Влияние: Высокое | Сложность: Средняя

### 3.1 Response Cache с Semantic Key
**Идея:** Кэшировать ответы на семантически похожие запросы.

```python
class SemanticResponseCache:
    def __init__(self):
        self.cache = {}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_cache_key(self, query):
        """Ключ на основе embedding'а запроса."""
        embedding = self.embedding_model.encode(query)
        # Кластеризация embedding'ов
        return self._quantize_embedding(embedding)
    
    async def get_or_generate(self, query, generator_fn):
        key = self.get_cache_key(query)
        if key in self.cache:
            cached = self.cache[key]
            if self._semantic_similarity(query, cached['query']) > 0.95:
                return cached['response']
        
        response = await generator_fn(query)
        self.cache[key] = {'query': query, 'response': response}
        return response
```

**Ожидаемый эффект:** 50-70% cache hit для типовых вопросов  
**Файлы:** Новый: `eva_ai/core/semantic_cache.py`

### 3.2 Pre-warming Cache
**Идея:** Предварительно загружать популярные данные.

```python
class CacheWarmer:
    """Предзагрузка часто используемых данных."""
    
    async def warm_on_startup(self):
        # Загрузить популярные узлы графа
        popular_nodes = await self.get_popular_nodes(limit=100)
        for node in popular_nodes:
            await self.cache.preload(node.id)
    
    async def warm_embeddings(self, texts: List[str]):
        """Предвычисление эмбеддингов."""
        await self.embedding_service.compute_batch(texts)
```

**Файлы:** `eva_ai/core/unified_cache_bridge.py`

### 3.3 Incremental Context Updates
**Проблема:** Контекст перестраивается полностью каждый раз.

**Решение:**
```python
class IncrementalContextManager:
    def __init__(self):
        self._context_hash = None
        self._context_data = None
    
    def get_context(self, query, new_nodes):
        # Проверить, что изменилось
        new_hash = self._hash_nodes(new_nodes)
        if new_hash == self._context_hash:
            return self._context_data
        
        # Обновить только изменившиеся части
        delta = self._compute_delta(self._context_data, new_nodes)
        self._context_data = self._merge_context(self._context_data, delta)
        self._context_hash = new_hash
        return self._context_data
```

---

## Phase 4: Model Optimization (4-6 недель)
### Приоритет: Средний | Влияние: Среднее | Сложность: Высокая

### 4.1 Model Quantization
**Идея:** Использовать INT4/INT8 вместо FP16 для ускорения.

```python
# В конфигурации загрузки модели
model = Llama(
    model_path="model.gguf",
    n_gpu_layers=-1,  # Все слои на GPU если доступно
    n_batch=512,      # Увеличить batch size
    quantization='q4_0'  # 4-bit quantization
)
```

**Ожидаемый эффект:** 2-4x ускорение генерации, -50% памяти  
**Файлы:** `eva_ai/core/brain_components.py`

### 4.2 Speculative Decoding
**Идея:** Маленькая модель генерирует черновик, большая проверяет.

```python
class SpeculativeDecoder:
    def __init__(self, draft_model, target_model):
        self.draft = draft_model  # Быстрая маленькая модель
        self.target = target_model  # Основная модель
    
    async def generate(self, prompt, max_tokens):
        tokens = []
        while len(tokens) < max_tokens:
            # Генерация черновика
            draft_tokens = self.draft.generate(prompt, n_tokens=5)
            
            # Проверка основной моделью
            accepted = self.target.verify(prompt, draft_tokens)
            tokens.extend(accepted)
            
            if len(accepted) < len(draft_tokens):
                # Перегенерация с корректировкой
                break
```

**Ожидаемый эффект:** 2-3x ускорение генерации  
**Файлы:** Новый: `eva_ai/core/speculative_decoder.py`

### 4.3 Dynamic Batching
**Идея:** Объединять несколько запросов в один batch.

```python
class DynamicBatcher:
    def __init__(self, model, max_batch_size=4, max_wait_ms=50):
        self.model = model
        self.queue = []
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000
    
    async def generate(self, query):
        future = asyncio.Future()
        self.queue.append((query, future))
        
        if len(self.queue) >= self.max_batch_size:
            await self._process_batch()
        
        return await future
    
    async def _process_batch(self):
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        
        results = self.model.generate_batch([q for q, _ in batch])
        for (_, future), result in zip(batch, results):
            future.set_result(result)
```

---

## Phase 5: Monitoring & Observability (Параллельно)
### Приоритет: Высокий | Влияние: Среднее | Сложность: Низкая

### 5.1 Performance Metrics Collection
```python
import time
from dataclasses import dataclass
from typing import Dict, List
import statistics

@dataclass
class PerformanceMetrics:
    query: str
    total_time: float
    context_retrieval_time: float
    generation_time: float
    tokens_generated: int
    cache_hits: int
    cache_misses: int

class MetricsCollector:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
    
    def record(self, metric: PerformanceMetrics):
        self.metrics.append(metric)
        
        # Keep only last 1000
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    def get_stats(self) -> Dict:
        if not self.metrics:
            return {}
        
        total_times = [m.total_time for m in self.metrics]
        return {
            'avg_response_time': statistics.mean(total_times),
            'p95_response_time': statistics.quantiles(total_times, n=20)[18],
            'p99_response_time': statistics.quantiles(total_times, n=100)[98],
            'throughput_qpm': len(self.metrics) / (sum(total_times) / 60),
            'cache_hit_rate': self._compute_cache_hit_rate()
        }
```

### 5.2 Distributed Tracing
```python
import uuid
from contextvars import ContextVar

trace_id_var: ContextVar[str] = ContextVar('trace_id')

class TracedOperation:
    def __init__(self, name: str):
        self.name = name
        self.trace_id = trace_id_var.get(str(uuid.uuid4()))
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        logger.info(f"[{self.trace_id}] Starting {self.name}")
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.start_time
        logger.info(f"[{self.trace_id}] Completed {self.name} in {elapsed:.3f}s")
```

### 5.3 Dashboard
Создать простой веб-дашборд на `/admin/performance`:
- realtime графики времени ответа
- heatmap медленных запросов
- cache hit/miss ratio
- активные генерации

---

## Implementation Timeline

```
Week 1-2:  Phase 1 (Quick Wins)
Week 3-4:  Phase 2.1-2.2 (Async Pipeline + Streaming)
Week 5-6:  Phase 2.3 + Phase 3 (Connection Pooling + Smart Cache)
Week 7-8:  Phase 4.1 (Quantization)
Week 9-10: Phase 4.2-4.3 (Speculative + Batching)
Ongoing:   Phase 5 (Monitoring)
```

---

## Success Criteria

**Phase 1 Complete:**
- [ ] Semantic search cache < 100ms для повторных запросов
- [ ] Время ответа снижено на 20%

**Phase 2 Complete:**
- [ ] TTFT < 1 секунды
- [ ] Система обрабатывает 10+ параллельных запросов

**Phase 3 Complete:**
- [ ] Cache hit rate > 50%
- [ ] Уменьшение CPU usage на 30%

**Phase 4 Complete:**
- [ ] Generation throughput увеличен в 2x
- [ ] Memory usage снижен на 40%

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking changes | Medium | High | Feature flags, gradual rollout |
| Memory leaks in async | Medium | High | Strict resource cleanup, monitoring |
| Model quality degradation | Low | High | A/B testing, quality gates |
| Complexity increase | High | Medium | Documentation, code reviews |

---

## Testing Strategy

### Load Testing
```bash
# Инструмент: locust
locust -f load_test.py --host=http://localhost:5555
```

### Benchmarks
- Baseline измерить ДО изменений
- После каждой фазы сравнить с baseline
- Регрессионные тесты на каждом PR

### Profiling
```python
# Использовать для поиска bottleneck'ов
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... код ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

---

## Next Steps

1. **Создать ветку** `performance-optimization`
2. **Настроить мониторинг** (Phase 5.1) - чтобы видеть метрики до и после
3. **Начать с Phase 1.1** (Semantic Search Cache) - быстрая победа
4. **Code review** каждого изменения
5. **Load test** после каждой фазы

---

## Appendix: Quick Reference

### Быстрые команды для проверки производительности

```bash
# Профилирование Python
python -m cProfile -o profile.stats run.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Мониторинг в реальном времени
watch -n 1 'curl -s http://localhost:5555/api/metrics | python -m json.tool'

# Нагрузочное тестирование
ab -n 100 -c 10 http://localhost:5555/api/status
```

### Key Metrics to Watch
- `query_latency_p95` < 3s
- `generation_tokens_per_second` > 50
- `cache_hit_rate` > 50%
- `memory_usage_mb` < 4096
- `cpu_usage_percent` < 70%

---

**Document Owner:** OpenCode  
**Review Schedule:** Weekly  
**Last Updated:** 2025-01-10
