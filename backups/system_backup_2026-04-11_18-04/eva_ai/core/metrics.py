"""
Performance Metrics - Система метрик для мониторинга EVA AI

Собирает и предоставляет метрики:
- Latency (p50, p95, p99)
- Throughput (RPM - requests per minute)
- Cache hit rates
- Resource usage (CPU, RAM)
- Custom business metrics
"""

import time
import threading
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import contextmanager
import logging

logger = logging.getLogger("eva_ai.metrics")


@dataclass
class MetricValue:
    """Значение метрики с timestamp."""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class Histogram:
    """Гистограмма для измерения распределения значений (latency и т.д.)."""
    
    def __init__(self, buckets: List[float] = None, max_samples: int = 10000):
        """
        Args:
            buckets: Границы бакетов в секундах (default: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])
            max_samples: Максимальное количество samples
        """
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        self.max_samples = max_samples
        self._samples: deque = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        self._sum = 0.0
        self._count = 0
    
    def observe(self, value: float):
        """Добавить наблюдение."""
        with self._lock:
            self._samples.append(value)
            self._sum += value
            self._count += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Получить статистику."""
        with self._lock:
            if not self._samples:
                return {
                    'count': 0,
                    'sum': 0.0,
                    'avg': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'p50': 0.0,
                    'p95': 0.0,
                    'p99': 0.0
                }
            
            sorted_samples = sorted(self._samples)
            n = len(sorted_samples)
            
            return {
                'count': self._count,
                'sum': self._sum,
                'avg': self._sum / self._count if self._count > 0 else 0.0,
                'min': sorted_samples[0],
                'max': sorted_samples[-1],
                'p50': self._percentile(sorted_samples, 0.50),
                'p95': self._percentile(sorted_samples, 0.95),
                'p99': self._percentile(sorted_samples, 0.99)
            }
    
    def _percentile(self, sorted_data: List[float], p: float) -> float:
        """Вычислить перцентиль."""
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        
        if f == c:
            return sorted_data[f]
        
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
    
    def get_buckets(self) -> Dict[str, int]:
        """Получить распределение по бакетам."""
        with self._lock:
            buckets_count = defaultdict(int)
            
            for sample in self._samples:
                for bucket in self.buckets:
                    if sample <= bucket:
                        buckets_count[f"le_{bucket}"] += 1
                        break
                else:
                    buckets_count["le_inf"] += 1
            
            return dict(buckets_count)


class Counter:
    """Счетчик (монотонно растущий)."""
    
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def inc(self, amount: float = 1.0):
        """Увеличить счетчик."""
        with self._lock:
            self._value += amount
    
    def get(self) -> float:
        """Получить значение."""
        with self._lock:
            return self._value


class Gauge:
    """Gauge (может расти и уменьшаться)."""
    
    def __init__(self):
        self._value = 0.0
        self._lock = threading.Lock()
    
    def set(self, value: float):
        """Установить значение."""
        with self._lock:
            self._value = value
    
    def inc(self, amount: float = 1.0):
        """Увеличить."""
        with self._lock:
            self._value += amount
    
    def dec(self, amount: float = 1.0):
        """Уменьшить."""
        with self._lock:
            self._value -= amount
    
    def get(self) -> float:
        """Получить значение."""
        with self._lock:
            return self._value


class MetricsRegistry:
    """Реестр всех метрик системы."""
    
    def __init__(self):
        self._histograms: Dict[str, Histogram] = {}
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._lock = threading.Lock()
    
    def histogram(self, name: str, buckets: List[float] = None) -> Histogram:
        """Получить или создать гистограмму."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(buckets=buckets)
            return self._histograms[name]
    
    def counter(self, name: str) -> Counter:
        """Получить или создать счетчик."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter()
            return self._counters[name]
    
    def gauge(self, name: str) -> Gauge:
        """Получить или создать gauge."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge()
            return self._gauges[name]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Получить все метрики."""
        with self._lock:
            return {
                'histograms': {name: h.get_stats() for name, h in self._histograms.items()},
                'counters': {name: c.get() for name, c in self._counters.items()},
                'gauges': {name: g.get() for name, g in self._gauges.items()}
            }
    
    def export_prometheus(self) -> str:
        """Экспорт в Prometheus format."""
        lines = []
        
        with self._lock:
            # Counters
            for name, counter in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {counter.get()}")
            
            # Gauges
            for name, gauge in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {gauge.get()}")
            
            # Histograms
            for name, histogram in self._histograms.items():
                stats = histogram.get_stats()
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_sum {stats['sum']}")
                
                buckets = histogram.get_buckets()
                for bucket_name, count in sorted(buckets.items()):
                    bucket_value = bucket_name.replace('le_', '').replace('inf', '+Inf')
                    lines.append(f'{name}_bucket{{le="{bucket_value}"}} {count}')
        
        return '\n'.join(lines)


# Глобальный реестр метрик
_metrics_registry = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    """Получить глобальный реестр метрик."""
    return _metrics_registry


@contextmanager
def timed_metric(name: str, labels: Dict[str, str] = None):
    """Контекстный менеджер для измерения времени выполнения."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        registry = get_metrics_registry()
        histogram = registry.histogram(f"{name}_duration_seconds")
        histogram.observe(elapsed)


def record_counter(name: str, amount: float = 1.0):
    """Записать значение в counter."""
    registry = get_metrics_registry()
    registry.counter(name).inc(amount)


def set_gauge(name: str, value: float):
    """Установить значение gauge."""
    registry = get_metrics_registry()
    registry.gauge(name).set(value)


class PerformanceMonitor:
    """Монитор производительности с автоматическим сбором метрик."""
    
    def __init__(self, interval: float = 10.0):
        """
        Args:
            interval: Интервал сбора метрик в секундах
        """
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._registry = get_metrics_registry()
    
    def start(self):
        """Запустить мониторинг."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"PerformanceMonitor запущен (interval={self.interval}s)")
    
    def stop(self):
        """Остановить мониторинг."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("PerformanceMonitor остановлен")
    
    def _monitor_loop(self):
        """Цикл сбора метрик."""
        while self._running:
            try:
                self._collect_system_metrics()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Ошибка сбора метрик: {e}")
    
    def _collect_system_metrics(self):
        """Собрать системные метрики."""
        try:
            import psutil
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self._registry.gauge("system_cpu_percent").set(cpu_percent)
            
            # Memory
            memory = psutil.virtual_memory()
            self._registry.gauge("system_memory_used_bytes").set(memory.used)
            self._registry.gauge("system_memory_percent").set(memory.percent)
            
            # Disk
            disk = psutil.disk_usage('/')
            self._registry.gauge("system_disk_used_bytes").set(disk.used)
            self._registry.gauge("system_disk_percent").set(disk.percent)
            
        except ImportError:
            logger.debug("psutil не установлен, системные метрики недоступны")
        except Exception as e:
            logger.debug(f"Ошибка сбора системных метрик: {e}")


# Предопределенные метрики для EVA AI

class EVAMetrics:
    """Предопределенные метрики для EVA AI."""
    
    def __init__(self):
        self.registry = get_metrics_registry()
    
    # Request metrics
    def record_request(self, duration: float, endpoint: str = "unknown"):
        """Записать метрику запроса."""
        self.registry.histogram("eva_request_duration_seconds").observe(duration)
        self.registry.counter("eva_requests_total").inc()
    
    def record_error(self, error_type: str = "unknown"):
        """Записать ошибку."""
        self.registry.counter("eva_errors_total").inc()
    
    # Generation metrics
    def record_generation(self, duration: float, mode: str = "unknown", tokens: int = 0):
        """Записать метрику генерации."""
        self.registry.histogram("eva_generation_duration_seconds").observe(duration)
        self.registry.counter("eva_generations_total").inc()
        self.registry.counter("eva_generation_tokens_total").inc(tokens)
        
        if tokens > 0:
            tokens_per_sec = tokens / duration if duration > 0 else 0
            self.registry.gauge("eva_generation_tokens_per_second").set(tokens_per_sec)
    
    # Cache metrics
    def record_cache_hit(self, cache_name: str = "default"):
        """Записать cache hit."""
        self.registry.counter(f"eva_cache_hits_total").inc()
    
    def record_cache_miss(self, cache_name: str = "default"):
        """Записать cache miss."""
        self.registry.counter(f"eva_cache_misses_total").inc()
    
    # Search metrics
    def record_search(self, duration: float, search_type: str = "semantic"):
        """Записать метрику поиска."""
        self.registry.histogram("eva_search_duration_seconds").observe(duration)
        self.registry.counter("eva_searches_total").inc()
    
    # Web search metrics
    def record_web_search(self, duration: float, results_count: int = 0):
        """Записать метрику веб-поиска."""
        self.registry.histogram("eva_web_search_duration_seconds").observe(duration)
        self.registry.counter("eva_web_searches_total").inc()
        self.registry.counter("eva_web_search_results_total").inc(results_count)
    
    def get_cache_hit_rate(self) -> float:
        """Получить cache hit rate."""
        hits = self.registry.counter("eva_cache_hits_total").get()
        misses = self.registry.counter("eva_cache_misses_total").get()
        total = hits + misses
        return hits / total if total > 0 else 0.0
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Получить статистику генераций."""
        hist = self.registry.histogram("eva_generation_duration_seconds")
        return hist.get_stats()


# Глобальный экземпляр
_eva_metrics = EVAMetrics()


def get_eva_metrics() -> EVAMetrics:
    """Получить глобальный экземпляр EVAMetrics."""
    return _eva_metrics
