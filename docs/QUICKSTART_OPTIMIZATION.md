# Performance Optimization - Quick Start Checklist
# Быстрый старт оптимизации производительности EVA AI

**Начни отсюда!** Этот документ содержит конкретные действия которые можно сделать прямо сейчас.

---

## Что сделать сегодня (30 минут)

### 1. Включить базовое логирование времени (5 минут)

```python
# Добавить в начало eva_ai/core/hybrid_pipeline_adapter.py
import time
from functools import wraps

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"⏱️ {func.__name__}: {elapsed:.1f}ms")
        return result
    return wrapper

# Применить к ключевым методам:
# @timed
# def process_query(...)
# @timed  
# def semantic_search(...)
```

### 2. Добавить простой кэш для semantic_search (10 минут)

```python
# В eva_ai/memory/fractal_graph_v2/__init__.py

class FractalMemoryGraph:
    def __init__(self, ...):
        ...
        # Добавить простой кэш
        self._search_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def semantic_search(self, query, top_k=10, ...):
        cache_key = f"{query}:{top_k}"
        
        if cache_key in self._search_cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit! ({self._cache_hits}/{self._cache_hits + self._cache_misses})")
            return self._search_cache[cache_key]
        
        self._cache_misses += 1
        results = self._perform_search(query, top_k, ...)
        
        # Кэшировать только если граф небольшой
        if len(self._search_cache) < 100:
            self._search_cache[cache_key] = results
        
        return results
```

### 3. Оптимизировать timeout'ы (5 минут)

```python
# В eva_ai/core/hybrid_pipeline_adapter.py
# Изменить DEFAULT_TIMEOUTS:

DEFAULT_TIMEOUTS = {
    'condensed': 8,     # Было: 30-60
    'extended': 20,     # Было: 60
    'large': 45,        # Было: 120
}
```

### 4. Добавить быстрый health check endpoint (10 минут)

```python
# В eva_ai/gui/web_gui/server_routes.py добавить:

@app.route('/api/health/performance')
def api_health_performance():
    """Быстрая проверка производительности системы."""
    import psutil
    import time
    
    start = time.perf_counter()
    
    # Тестовый запрос
    if web_gui_instance and hasattr(web_gui_instance.brain, 'fractal_graph_v2'):
        fg = web_gui_instance.brain.fractal_graph_v2
        test_results = fg.semantic_search("test query", top_k=1)
        search_time = (time.perf_counter() - start) * 1000
    else:
        search_time = None
    
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'test_search_time_ms': search_time,
        'cache_stats': getattr(fg, '_cache_hits', 0) if fg else None
    })
```

---

## Что сделать на этой неделе

### День 1-2: Semantic Search Cache
- [ ] Реализовать LRU cache для semantic_search
- [ ] Добавить метрики cache hit/miss
- [ ] Протестировать на типовых запросах
- [ ] **Цель:** Cache hit rate > 30%

### День 3-4: Асинхронный веб-поиск  
- [ ] Заменить requests на aiohttp
- [ ] Добавить connection pooling
- [ ] Сделать веб-поиск неблокирующим
- [ ] **Цель:** -1 секунда на запросы с веб-поиском

### День 5: Оптимизация моделей
- [ ] Включить batch processing для embeddings
- [ ] Настроить n_gpu_layers если есть GPU
- [ ] Проверить n_batch и n_threads
- [ ] **Цель:** +20% к скорости генерации

---

## Быстрые проверки

### Как измерить baseline?

```bash
# 1. Запустить сервер
python run.py --web

# 2. В другом терминале - измерить время ответа
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:5555/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"test","session_id":"test123"}'

# curl-format.txt:
# time_namelookup:  %{time_namelookup}\n
# time_connect:  %{time_connect}\n
# time_appconnect:  %{time_appconnect}\n
# time_pretransfer:  %{time_pretransfer}\n
# time_redirect:  %{time_redirect}\n
# time_starttransfer:  %{time_starttransfer}\n
# time_total:  %{time_total}\n
```

### Как найти самые медленные запросы?

```python
# Добавить в server_main.py process_message
import json
import os

# Логирование медленных запросов
PERFORMANCE_LOG = "performance.log"

def log_slow_query(query, duration, source):
    if duration > 5.0:  # Логировать если > 5 секунд
        with open(PERFORMANCE_LOG, "a") as f:
            f.write(json.dumps({
                'timestamp': time.time(),
                'query': query[:100],
                'duration': duration,
                'source': source
            }) + "\n")
```

---

## Измерения до и после

Заполни эту таблицу до начала оптимизации:

| Метрика | До | После Phase 1 | После Phase 2 | Цель |
|---------|----|---------------|---------------|------|
| Среднее время ответа (простой запрос) | ___s | ___s | ___s | <3s |
| Среднее время ответа (с веб-поиском) | ___s | ___s | ___s | <5s |
| TTFT (time to first token) | ___s | ___s | ___s | <1s |
| Cache hit rate | ___% | ___% | ___% | >50% |
| Пропускная способность | ___ RPM | ___ RPM | ___ RPM | >20 RPM |
| Использование RAM | ___ MB | ___ MB | ___ MB | <4GB |

---

## Красные флаги (что проверить сразу)

Если ты видишь это - срочно исправляй:

- [ ] **CPU 100%** постоянно - возможно бесконечный цикл
- [ ] **RAM растёт** без остановки - memory leak
- [ ] **Запросы > 30 сек** без ответа - deadlock или тяжёлая операция
- [ ] **"timeout" в логах** каждый запрос - перегрузка или bottleneck
- [ ] **Диск I/O 100%** - слишком много чтения/записи

---

## Полезные команды

```bash
# Мониторинг Python процесса
watch -n 1 'ps aux | grep python | grep -v grep'

# Профилирование памяti
python -m memory_profiler run.py

# Профилирование CPU (cProfile)
python -m cProfile -o profile.stats -s cumulative run.py &
# Подождать минуту, остановить
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('time').print_stats(20)"

# Сетевая активность
netstat -tulpn | grep :5555

# Дисковая активность
iotop -p $(pgrep -f "python run.py")

# Мониторинг в реальном времени
htop
```

---

## Где получить помощь

- **PERFORMANCE_ROADMAP.md** - полный план с приоритетами
- **BOTTLENECK_ANALYSIS.md** - технические детали bottleneck'ов
- **GitHub Issues** - создавай issue для обсуждения конкретных оптимизаций

---

## Success Criteria

✅ **Phase 1 завершена когда:**
- Среднее время ответа снижено на 20%
- Cache hit rate > 30%
- Нет запросов > 30 секунд

✅ **Phase 2 завершена когда:**
- TTFT < 1 секунды
- Система выдерживает 10 параллельных запросов
- Среднее время ответа < 3 секунд

---

**Начни с первого пункта прямо сейчас!** ⏱️
