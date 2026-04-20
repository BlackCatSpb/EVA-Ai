"""Core module for HybridTokenCache - main class, initialization, lifecycle."""
import os
import json
import time
import threading
from typing import Dict, List, Optional, Any
import logging

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

_cache_registry: Dict[str, 'HybridTokenCache'] = {}
_registry_lock = threading.RLock()


def get_shared_cache(brain, cache_name: str = "default") -> 'HybridTokenCache':
    """Returns a singleton HybridTokenCache for the given name."""
    with _registry_lock:
        if cache_name not in _cache_registry:
            _cache_registry[cache_name] = HybridTokenCache(brain, _cache_name=cache_name)
        return _cache_registry[cache_name]


from .cache_ram import LRUCache
from .cache_disk import TokenDiskCache


class HybridTokenCache:
    """
    Гибридная система кэширования токенов для ЕВА.
    Многоуровневая архитектура: GPU (VRAM) → RAM → SSD
    """

    def __new__(cls, brain, _cache_name: str = "default", **kwargs):
        with _registry_lock:
            if _cache_name in _cache_registry:
                return _cache_registry[_cache_name]
            instance = super().__new__(cls)
            _cache_registry[_cache_name] = instance
            return instance

    def __init__(
        self,
        brain,
        max_memory_tokens: int = 100000,
        disk_cache_dir: str = "token_cache",
        target_memory_gb: float = 50.0,
        max_disk_cache_gb: float = 50.0,
        dynamic_memory_limit: bool = True,
        max_ram_usage_percent: float = 70.0,
        vram_threshold: float = 0.15,
        ram_threshold: float = 0.12,
        eviction_policy: str = "hybrid",
        hot_threshold: int = 5,
        _cache_name: str = "default",
        **kwargs
    ):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        self.brain = brain
        self.gpu_available = torch is not None and torch.cuda.is_available() if torch else False

        cfg = getattr(self.brain, 'config', {}) or {}
        hc = cfg.get('hybrid_cache', {}) or {}
        model_cfg = cfg.get('model', {}) or {}

        config_device = hc.get('device') or model_cfg.get('device')
        if config_device:
            self.device = config_device
        elif self.gpu_available:
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        try:
            base_cache_dir = None
            if brain is not None:
                base_cache_dir = getattr(brain, 'cache_dir', None)
            if not base_cache_dir:
                base_cache_dir = os.environ.get('COGNIFLEX_CACHE_DIR')
            if not base_cache_dir:
                base_cache_dir = os.path.join(os.getcwd(), 'ml_cache')
            os.makedirs(base_cache_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Error setting up cache directory: {e}")
            base_cache_dir = os.path.join(os.getcwd(), 'ml_cache')
            os.makedirs(base_cache_dir, exist_ok=True)

        self.disk_cache_dir = os.path.join(base_cache_dir, disk_cache_dir)
        os.makedirs(self.disk_cache_dir, exist_ok=True)

        try:
            cfg = getattr(self.brain, 'config', {}) or {}
            hc = cfg.get('hybrid_cache') or {}
        except Exception as e:
            logger.warning(f"Error loading hybrid_cache config: {e}")
            hc = {}

        target_memory_gb = float(hc.get('target_memory_gb', target_memory_gb))
        dynamic_memory_limit = bool(hc.get('dynamic_memory_limit', dynamic_memory_limit))
        max_ram_usage_percent = float(hc.get('max_ram_usage_percent', max_ram_usage_percent))
        hot_threshold = int(hc.get('hot_threshold', hot_threshold))
        eviction_policy = hc.get('eviction_policy', eviction_policy)

        self.dynamic_memory_limit = dynamic_memory_limit
        self.target_memory_bytes = int(target_memory_gb * 1024 ** 3)
        self.avg_token_size_bytes = 4096

        if psutil and dynamic_memory_limit:
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            self.target_memory_bytes = min(
                self.target_memory_bytes,
                int(total_ram_gb * 1024 ** 3 * (max_ram_usage_percent / 100.0))
            )
            logger.debug(f"Динамический лимит памяти: {target_memory_gb:.2f}GB ({max_ram_usage_percent}% RAM)")

        self.max_memory_tokens = (
            max(1, int(self.target_memory_bytes / self.avg_token_size_bytes))
            if dynamic_memory_limit else max(1, max_memory_tokens)
        )

        if self.gpu_available:
            vram_limit_bytes = int(1.5 * 1024 ** 3)
            vram_tokens_limit = max(1, int(vram_limit_bytes / self.avg_token_size_bytes))
            ram_limit_bytes = int(1.0 * 1024 ** 3)
            ram_tokens_limit = max(1, int(ram_limit_bytes / self.avg_token_size_bytes))
            self.vram_cache = LRUCache(vram_tokens_limit)
            self.ram_cache = LRUCache(ram_tokens_limit)
            logger.debug(f"VRAM кэш: {vram_tokens_limit} токенов (~1.5GB)")
            logger.debug(f"RAM кэш: {ram_tokens_limit} токенов (~1GB)")
        else:
            ram_limit_bytes = int(1.0 * 1024 ** 3)
            ram_tokens_limit = max(1, int(ram_limit_bytes / self.avg_token_size_bytes))
            self.ram_cache = LRUCache(ram_tokens_limit)
            self.vram_cache = LRUCache(1)
            logger.debug(f"RAM кэш: {ram_tokens_limit} токенов (~1GB)")

        self.memory_cache = self.ram_cache

        self.disk_cache = TokenDiskCache(self.disk_cache_dir, max_disk_cache_gb)

        self.token_metadata = {}
        self.metadata_lock = threading.RLock()
        self.stats_lock = threading.RLock()
        self._lock = threading.RLock()

        self.vram_threshold = vram_threshold
        self.ram_threshold = ram_threshold
        self.eviction_policy = eviction_policy
        self.hot_threshold = hot_threshold

        self.cache_stats = {
            'vram_hits': 0,
            'ram_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }

        self.usage_stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "total_accesses": 0,
            "cache_efficiency": 0.0,
            "avg_access_time": 0.0,
            "memory_usage_mb": 0.0
        }

        self.cache_settings = {
            "max_memory_size": self.max_memory_tokens,
            "disk_cache_threshold": max(1, self.max_memory_tokens // 2),
            "eviction_policy": eviction_policy,
            "cache_ttl": int(hc.get("cache_ttl", 86400)),
            "min_relevance_score": float(hc.get("min_relevance_score", 0.3)),
            "max_context_tokens": int(hc.get("max_context_tokens", 1000)),
        }

        self.system_free_mem_threshold = float(hc.get('system_free_mem_threshold', 0.1))
        self.memory_pressure_interval_s = float(hc.get('memory_pressure_interval_s', 2.0))
        self.pressure_offload_batch = int(hc.get('pressure_offload_batch', 32))

        self._load_metadata()
        self._start_memory_pressure_worker()

        approx_mem_gb = (self.max_memory_tokens * self.avg_token_size_bytes) / (1024 ** 3)
        logger.info(
            f"HybridTokenCache инициализирован: {self.device}, "
            f"память~{self.max_memory_tokens} токенов (~{approx_mem_gb:.2f}GB), "
            f"диск=50.0GB"
        )

    def get_token(self, token_id: str) -> Optional[Dict]:
        from .cache_eviction import _get_token_impl
        return _get_token_impl(self, token_id)

    def add_token(self, token_id: str, token_data: Any) -> None:
        from .cache_eviction import _add_token_impl
        _add_token_impl(self, token_id, token_data)

    def put(self, key: str, value: Any) -> None:
        self.add_token(key, value)

    def get(self, key: str) -> Optional[Any]:
        return self.get_token(key)

    def set(self, key: str, value: Any) -> None:
        self.add_token(key, value)

    def clear(self) -> None:
        with self.stats_lock:
            with self.metadata_lock:
                self.vram_cache.clear()
                self.ram_cache.clear()
                self.disk_cache.clear()

                self.cache_stats = {
                    'vram_hits': 0, 'ram_hits': 0, 'disk_hits': 0,
                    'misses': 0, 'evictions': 0, 'total_requests': 0
                }

                self.token_metadata = {}

        logger.info("Все кэши очищены")

    def _load_token_from_disk(self, token_id: str) -> Optional[Dict]:
        with self._lock:
            return self.disk_cache.get(token_id)

    def _save_token_to_disk(self, token_id: str, token_data: Dict) -> None:
        with self._lock:
            self.disk_cache.put(token_id, token_data)

    def _move_token_to_memory(self, token_id: str, token_data: Optional[Dict] = None) -> None:
        from .cache_eviction import _move_token_to_memory_impl
        _move_token_to_memory_impl(self, token_id, token_data)

    def _move_token_to_disk(self, token_id: str) -> None:
        with self._lock:
            token_data = self.memory_cache.get(token_id)
            if not token_data:
                return

            self.memory_cache.remove(token_id)
            self.disk_cache.put(token_id, token_data)

            with self.metadata_lock:
                if token_id in self.token_metadata:
                    self.token_metadata[token_id]["in_memory"] = False
                    self.token_metadata[token_id]["mem_size"] = 0

    def _update_token_access(self, token_id: str) -> None:
        with self.metadata_lock:
            if token_id in self.token_metadata:
                self.token_metadata[token_id]["access_count"] += 1
                self.token_metadata[token_id]["last_access"] = time.time()

    def _estimate_size_bytes(self, value: Any) -> int:
        try:
            if isinstance(value, (dict, list)):
                return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
            if isinstance(value, str):
                return len(value.encode("utf-8"))
            return len(str(value).encode("utf-8"))
        except Exception as e:
            logger.warning(f"Error estimating size: {e}")
            return 0

    def _evict_one_lru(self) -> None:
        from .cache_eviction import _evict_one_lru_impl
        _evict_one_lru_impl(self)

    def _load_metadata(self) -> None:
        metadata_path = os.path.join(self.disk_cache_dir, "token_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.token_metadata = json.load(f)
                logger.info(f"Загружены метаданные для {len(self.token_metadata)} токенов")
            except Exception as e:
                logger.error(f"Ошибка загрузки метаданных: {e}")
                self.token_metadata = {}
        else:
            self.token_metadata = {}

    def _save_metadata(self) -> None:
        try:
            metadata_path = os.path.join(self.disk_cache_dir, "token_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.token_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения метаданных: {e}")

    def _start_memory_pressure_worker(self) -> None:
        from .cache_eviction import _start_memory_pressure_worker_impl
        _start_memory_pressure_worker_impl(self)

    def _offload_under_pressure(self) -> None:
        from .cache_eviction import _offload_under_pressure_impl
        _offload_under_pressure_impl(self)

    def _calculate_token_priority(self, token_id: str, metadata: Dict) -> float:
        access_count = metadata.get("access_count", 0)
        last_access = metadata.get("last_access", 0)
        priority = metadata.get("priority", 0.5)

        time_factor = 1.0 if time.time() - last_access < 3600 else 0.3
        return (access_count * 0.6) + (priority * 0.3) + (time_factor * 0.1)

    def get_cache_stats(self) -> Dict[str, Any]:
        with self.stats_lock:
            stats = {
                "vram_tokens": len(self.vram_cache),
                "ram_tokens": len(self.ram_cache),
                "disk_tokens": len(self.disk_cache),
                "vram_hits": self.cache_stats.get('vram_hits', 0),
                "ram_hits": self.cache_stats.get('ram_hits', 0),
                "disk_hits": self.cache_stats.get('disk_hits', 0),
                "misses": self.cache_stats.get('misses', 0),
                "evictions": self.cache_stats.get('evictions', 0),
                "total_requests": self.cache_stats.get('total_requests', 0),
                "memory_usage_mb": self.usage_stats.get("memory_usage_mb", 0.0),
                "cache_efficiency": self.usage_stats.get("cache_efficiency", 0.0),
                "avg_access_time": self.usage_stats.get("avg_access_time", 0.0),
            }

            total = stats['total_requests']
            if total > 0:
                stats['hit_rate'] = (stats['vram_hits'] + stats['ram_hits'] + stats['disk_hits']) / total
            else:
                stats['hit_rate'] = 0.0

            return stats

    def cleanup(self) -> None:
        try:
            try:
                if hasattr(self, "_local_stop_event"):
                    self._local_stop_event.set()
            except Exception as e:
                logger.warning(f"Error setting stop event: {e}")

            try:
                stop_event = getattr(self.brain, 'stop_event', None)
                if isinstance(stop_event, threading.Event):
                    stop_event.set()
            except Exception as e:
                logger.warning(f"Error setting brain stop event: {e}")

            try:
                t = getattr(self, "_memory_pressure_thread", None)
                if t and isinstance(t, threading.Thread) and t.is_alive():
                    t.join(timeout=2.0)
            except Exception as e:
                logger.warning(f"Error joining memory pressure thread: {e}")

            self._save_metadata()

        except Exception as e:
            logger.error(f"Ошибка при очистке HybridTokenCache: {e}")

    def __len__(self) -> int:
        with self._lock:
            return len(self.memory_cache)

    def __bool__(self) -> bool:
        return True

    def __contains__(self, token_hash: str) -> bool:
        with self._lock:
            return token_hash in self.memory_cache

    def add_context(self, session_id: str, query: str, entities: List[str],
                    raw_text: str, ttl: int = 3600) -> None:
        from .cache_eviction import _add_context_impl
        _add_context_impl(self, session_id, query, entities, raw_text, ttl)

    def get_context(self, session_id: str) -> Optional[Dict]:
        from .cache_eviction import _get_context_impl
        return _get_context_impl(self, session_id)

    def get_recent_contexts(self, limit: int = 10) -> List[Dict]:
        from .cache_eviction import _get_recent_contexts_impl
        return _get_recent_contexts_impl(self, limit)

    def add_document(self, session_id: str, doc_id: str, filename: str,
                    extracted_text: str, entities: List[str],
                    doc_type: str = "unknown", ttl: int = 86400) -> None:
        from .cache_eviction import _add_document_impl
        _add_document_impl(self, session_id, doc_id, filename, extracted_text, entities, doc_type, ttl)

    def get_document(self, session_id: str, doc_id: str) -> Optional[Dict]:
        from .cache_eviction import _get_document_impl
        return _get_document_impl(self, session_id, doc_id)

    def get_session_documents(self, session_id: str) -> List[Dict]:
        from .cache_eviction import _get_session_documents_impl
        return _get_session_documents_impl(self, session_id)

    def delete_document(self, session_id: str, doc_id: str) -> bool:
        from .cache_eviction import _delete_document_impl
        return _delete_document_impl(self, session_id, doc_id)

    def add_search_results(self, query_hash: str, query: str,
                          results: List[Dict], ttl: int = 43200) -> None:
        from .cache_eviction import _add_search_results_impl
        _add_search_results_impl(self, query_hash, query, results, ttl)

    def get_search_results(self, query_hash: str) -> Optional[Dict]:
        from .cache_eviction import _get_search_results_impl
        return _get_search_results_impl(self, query_hash)

    def get_search_cache_stats(self) -> Dict:
        from .cache_eviction import _get_search_cache_stats_impl
        return _get_search_cache_stats_impl(self)


__all__ = ['HybridTokenCache', 'LRUCache', 'get_shared_cache']
