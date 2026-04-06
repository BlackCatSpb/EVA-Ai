"""Cache management strategies for MemoryManager."""
import os
import logging
import time
import shutil
from typing import Dict, Any

try:
    import psutil
except Exception:
    psutil = None

logger = logging.getLogger("eva.memory.manager")


def get_memory_statistics(manager) -> Dict[str, Any]:
    try:
        total_nodes = 0
        working_mem = getattr(manager, 'working_memory', None)
        if isinstance(working_mem, dict):
            total_nodes += len(working_mem)
        semantic_mem = getattr(manager, 'semantic_memory', None)
        if isinstance(semantic_mem, dict):
            total_nodes += len(semantic_mem)
        episodic_mem = getattr(manager, 'episodic_memory', None)
        if isinstance(episodic_mem, list):
            total_nodes += len(episodic_mem)

        total_gb: float
        used_gb: float
        free_gb: float
        cache_gb: float = 0.0

        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                gb = 1024.0 ** 3
                total_gb = float(vm.total) / gb
                free_gb = float(vm.available) / gb
                used_gb = max(0.0, total_gb - free_gb)
                cache_gb = float(getattr(vm, 'cached', 0) or 0) / gb
            except Exception:
                total_gb = 2.0
                used_gb = min(2.0, 0.5 + total_nodes * 0.001)
                free_gb = max(0.0, total_gb - used_gb)
                cache_gb = 0.2
        else:
            total_gb = 2.0
            used_gb = min(2.0, 0.5 + total_nodes * 0.001)
            free_gb = max(0.0, total_gb - used_gb)
            cache_gb = 0.2

        stats = {
            "total_memory": round(total_gb, 3),
            "used_memory": round(used_gb, 3),
            "free_memory": round(free_gb, 3),
            "cache_memory": round(cache_gb, 3),
            "total_nodes": total_nodes,
            "active_nodes": total_nodes,
            "cached_nodes": 0,
            "memory_efficiency": 0.8,
            "cache_hits": 0,
            "cache_hit_ratio": 0.0,
            "last_update": time.time()
        }
        return stats
    except Exception as e:
        logger.error(f"Ошибка формирования статистики памяти: {e}", exc_info=True)
        return {
            "total_memory": 2.0,
            "used_memory": 1.2,
            "free_memory": 0.8,
            "cache_memory": 0.2,
            "total_nodes": 0,
            "active_nodes": 0,
            "cached_nodes": 0,
            "memory_efficiency": 0.0,
            "cache_hits": 0,
            "cache_hit_ratio": 0.0,
            "last_update": time.time()
        }


def analyze_memory_usage(manager) -> Dict[str, Any]:
    try:
        domain_distribution: Dict[str, int] = {}
        for entry in getattr(manager, 'semantic_memory', {}).values():
            domain = entry.get('metadata', {}).get('domain', 'unknown') if isinstance(entry, dict) else 'unknown'
            domain_distribution[domain] = domain_distribution.get(domain, 0) + 1

        cache_stats = {}
        hybrid_cache = getattr(manager, 'hybrid_cache', None)
        if hybrid_cache and hasattr(hybrid_cache, 'get_stats'):
            try:
                cache_stats = hybrid_cache.get_stats()
            except Exception:
                pass

        cache_hits = cache_stats.get('cache_hits', 0)
        cache_misses = cache_stats.get('cache_misses', 1)
        total_requests = cache_hits + cache_misses
        cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0

        efficiency_score = min(1.0, cache_hit_rate + 0.3)

        fragmentation_level = 0.2
        if hasattr(manager, 'semantic_memory') and manager.semantic_memory:
            fragmentation_level = min(0.8, len(manager.semantic_memory) / 1000 * 0.1)

        recommendations = []
        if cache_hit_rate < 0.3:
            recommendations.append("Низкий cache hit rate - рассмотрите увеличение кэша")
        if len(domain_distribution) > 20:
            recommendations.append("Много доменов - используйте категоризацию")
        if not recommendations:
            recommendations.append("Система работает оптимально")

        return {
            "efficiency_score": efficiency_score,
            "fragmentation_level": fragmentation_level,
            "cache_hit_rate": cache_hit_rate,
            "recommendations": recommendations,
            "memory_trends": {
                "usage_trend": "stable",
                "efficiency_trend": "improving" if cache_hit_rate > 0.5 else "stable"
            },
            "domain_distribution": domain_distribution
        }
    except Exception as e:
        logger.error(f"Ошибка анализа памяти: {e}", exc_info=True)
        return {
            "efficiency_score": 0.5,
            "fragmentation_level": 0.3,
            "cache_hit_rate": 0.0,
            "recommendations": [],
            "memory_trends": {"usage_trend": "unknown", "efficiency_trend": "unknown"},
            "domain_distribution": {}
        }


def set_cache_size(manager, cache_size: int) -> None:
    manager.cache_size = cache_size
    logger.info(f"Установлен размер кэша: {cache_size}")


def clear_cache(manager) -> None:
    try:
        if manager.hybrid_cache:
            if hasattr(manager.hybrid_cache, "clear"):
                manager.hybrid_cache.clear()
        cache_dir = os.path.join(os.getcwd(), "hybrid_cache")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        logger.info("Кэш памяти очищен")
    except Exception as e:
        logger.error(f"Ошибка очистки кэша памяти: {e}")
        raise


def optimize_cache(manager) -> None:
    try:
        if manager.hybrid_cache and hasattr(manager.hybrid_cache, 'cleanup'):
            manager.hybrid_cache.cleanup()
        _optimize_memory_lists(manager)
        logger.info("Кэш оптимизирован")
    except Exception as e:
        logger.error(f"Ошибка оптимизации кэша: {e}")


def _optimize_memory_lists(manager) -> None:
    try:
        cutoff = time.time() - 7 * 24 * 3600
        for mem_type in ("working_memory", "semantic_memory", "episodic_memory"):
            mem_obj = getattr(manager, mem_type, None)
            if mem_obj is None:
                continue
            original_len = len(mem_obj)
            if isinstance(mem_obj, list):
                mem_obj[:] = [e for e in mem_obj if isinstance(e, dict) and e.get('timestamp', 0) > cutoff]
            elif isinstance(mem_obj, dict):
                keys_to_remove = [k for k, v in mem_obj.items() if isinstance(v, dict) and v.get('timestamp', 0) <= cutoff]
                for k in keys_to_remove:
                    del mem_obj[k]
            if len(mem_obj) < original_len:
                from .manager_operations import _save_memory
                _save_memory(manager, mem_type.replace("_memory", ""))
    except Exception as e:
        logger.debug(f"Ошибка оптимизации списков памяти: {e}")


def clear_inactive_caches(manager, max_age_days: int = 30) -> None:
    try:
        cutoff = time.time() - max_age_days * 24 * 3600
        for mem_type in ("working_memory", "semantic_memory", "episodic_memory"):
            mem_obj = getattr(manager, mem_type, None)
            if mem_obj is None:
                continue
            original_len = len(mem_obj)
            if isinstance(mem_obj, list):
                mem_obj[:] = [entry for entry in mem_obj if isinstance(entry, dict) and entry.get('timestamp', 0) > cutoff]
            elif isinstance(mem_obj, dict):
                keys_to_remove = [k for k, v in mem_obj.items() if isinstance(v, dict) and v.get('timestamp', 0) <= cutoff]
                for k in keys_to_remove:
                    del mem_obj[k]
            if len(mem_obj) < original_len:
                from .manager_operations import _save_memory
                _save_memory(manager, mem_type.replace("_memory", ""))
        logger.info(f"Очищены неактивные кэши старше {max_age_days} дней")
    except Exception as e:
        logger.error(f"Ошибка очистки неактивных кэшей: {e}")


def compress_data(manager) -> None:
    try:
        for mem_type in ("working_memory", "semantic_memory", "episodic_memory"):
            mem_obj = getattr(manager, mem_type, None)
            if mem_obj is None:
                continue
            seen = set()
            if isinstance(mem_obj, dict):
                keys_to_remove = []
                for key, entry in mem_obj.items():
                    content_hash = hash(str(entry.get('content', '')) if isinstance(entry, dict) else str(entry))
                    if content_hash in seen:
                        keys_to_remove.append(key)
                    else:
                        seen.add(content_hash)
                for key in keys_to_remove:
                    del mem_obj[key]
                if keys_to_remove:
                    from .manager_operations import _save_memory
                    _save_memory(manager, mem_type.replace("_memory", ""))
            elif isinstance(mem_obj, list):
                compressed = []
                for entry in mem_obj:
                    content_hash = hash(str(entry.get('content', '')) if isinstance(entry, dict) else str(entry))
                    if content_hash not in seen:
                        seen.add(content_hash)
                        compressed.append(entry)
                if len(compressed) < len(mem_obj):
                    mem_obj[:] = compressed
                    from .manager_operations import _save_memory
                    _save_memory(manager, mem_type.replace("_memory", ""))
        logger.info("Данные в памяти сжаты")
    except Exception as e:
        logger.error(f"Ошибка сжатия данных: {e}")
