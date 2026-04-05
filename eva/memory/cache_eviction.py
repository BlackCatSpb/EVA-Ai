"""Eviction policies, cleanup, size management for HybridTokenCache."""
import time
import threading
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def _get_token_impl(cache, token_id: str) -> Optional[Dict]:
    start_time = time.time()

    with cache._lock:
        with cache.stats_lock:
            cache.usage_stats["total_accesses"] += 1
            cache.cache_stats['total_requests'] += 1

        if cache.gpu_available and token_id in cache.vram_cache:
            token_data = cache.vram_cache.get(token_id)
            if token_data:
                with cache.stats_lock:
                    cache.cache_stats['vram_hits'] += 1
                    cache.usage_stats["memory_hits"] += 1
                cache._update_token_access(token_id)
                return token_data

        if token_id in cache.ram_cache:
            token_data = cache.ram_cache.get(token_id)
            if token_data:
                with cache.stats_lock:
                    cache.cache_stats['ram_hits'] += 1
                    cache.usage_stats["memory_hits"] += 1

                if cache.gpu_available and len(cache.vram_cache) < cache.vram_cache.max_size:
                    cache.vram_cache.put(token_id, token_data)

                cache._update_token_access(token_id)
                return token_data

        disk_token = cache._load_token_from_disk(token_id)
        if disk_token:
            with cache.stats_lock:
                cache.cache_stats['disk_hits'] += 1
                cache.usage_stats["disk_hits"] += 1

            with cache.metadata_lock:
                metadata = cache.token_metadata.get(token_id, {})
                if metadata.get("access_count", 0) > cache.hot_threshold:
                    threading.Thread(
                        target=cache._move_token_to_memory,
                        args=(token_id, disk_token),
                        daemon=True
                    ).start()

            cache._update_token_access(token_id)
            return disk_token

        with cache.stats_lock:
            cache.cache_stats['misses'] += 1
            cache.usage_stats["misses"] += 1

    return None


def _add_token_impl(cache, token_id: str, token_data: Any) -> None:
    with cache._lock:
        with cache.metadata_lock:
            if isinstance(token_data, dict):
                priority = token_data.get("priority", 0.5)
                actual_data = token_data
            else:
                priority = 0.5
                actual_data = token_data

            if token_id not in cache.token_metadata:
                cache.token_metadata[token_id] = {
                    "access_count": 0,
                    "last_access": time.time(),
                    "priority": priority,
                    "in_memory": False,
                    "mem_size": 0,
                }

            cache._save_token_to_disk(token_id, actual_data)
            cache.token_metadata[token_id]["in_memory"] = False

            if len(cache.ram_cache) < cache.ram_cache.max_size:
                cache.ram_cache.put(token_id, actual_data)
                cache.token_metadata[token_id]["in_memory"] = True


def _move_token_to_memory_impl(cache, token_id: str, token_data: Optional[Dict] = None) -> None:
    with cache._lock:
        if token_data is None:
            token_data = cache.disk_cache.get(token_id)
        if not token_data:
            return

        size_bytes = cache._estimate_size_bytes(token_data)
        if len(cache.memory_cache) >= cache.max_memory_tokens:
            cache._evict_one_lru()

        cache.memory_cache.put(token_id, token_data)

        with cache.metadata_lock:
            if token_id in cache.token_metadata:
                cache.token_metadata[token_id]["in_memory"] = True
                cache.token_metadata[token_id]["mem_size"] = size_bytes


def _evict_one_lru_impl(cache) -> None:
    try:
        if len(cache.memory_cache) == 0:
            return
        first_key = next(iter(cache.memory_cache.cache))
        cache._move_token_to_disk(first_key)
    except Exception as e:
        logger.warning(f"Error in _evict_one_lru: {e}")


def _start_memory_pressure_worker_impl(cache) -> None:
    import psutil

    if psutil is None:
        logger.info("psutil недоступен: мониторинг давления памяти отключён")
        return

    stop_event = getattr(cache.brain, 'stop_event', None)
    if stop_event is None:
        cache._local_stop_event = threading.Event()
        stop_event = cache._local_stop_event

    def worker():
        while not stop_event.is_set():
            try:
                vm = psutil.virtual_memory()
                free_ratio = float(vm.available) / float(max(1, vm.total))
                if free_ratio < cache.system_free_mem_threshold:
                    cache._offload_under_pressure()
                time.sleep(cache.memory_pressure_interval_s)
            except Exception as e:
                logger.error(f"Ошибка монитора памяти: {e}")
                time.sleep(cache.memory_pressure_interval_s)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    cache._memory_pressure_thread = t


def _offload_under_pressure_impl(cache) -> None:
    try:
        if len(cache.memory_cache) == 0:
            return

        candidates = []
        for key in list(cache.memory_cache.keys()):
            meta = cache.token_metadata.get(key, {})
            pr = cache._calculate_token_priority(key, meta)
            candidates.append((key, pr))

        candidates.sort(key=lambda x: x[1])
        to_offload = min(cache.pressure_offload_batch, len(candidates))

        offloaded = 0
        for i in range(to_offload):
            key = candidates[i][0]
            if key in cache.memory_cache:
                cache._move_token_to_disk(key)
                offloaded += 1

        if offloaded > 0:
            logger.warning(
                f"Memory pressure: выгружено {offloaded} токенов "
                f"(batch={cache.pressure_offload_batch})"
            )
    except Exception as e:
        logger.error(f"Ошибка offload при давлении памяти: {e}")


def _add_context_impl(cache, session_id: str, query: str, entities: List[str],
                      raw_text: str, ttl: int = 3600) -> None:
    context_key = f"context:{session_id}"

    context_data = {
        "query": query,
        "entities": entities,
        "raw_text": raw_text,
        "timestamp": time.time(),
        "ttl": ttl
    }

    cache.add_token(context_key, context_data)
    logger.debug(f"Сохранен raw context для сессии {session_id}")


def _get_context_impl(cache, session_id: str) -> Optional[Dict]:
    context_key = f"context:{session_id}"
    data = cache.get_token(context_key)

    if data and isinstance(data, dict):
        timestamp = data.get("timestamp", 0)
        ttl = data.get("ttl", 3600)

        if time.time() - timestamp < ttl:
            return data

    return None


def _get_recent_contexts_impl(cache, limit: int = 10) -> List[Dict]:
    contexts = []

    with cache._lock:
        for token_id, metadata in cache.token_metadata.items():
            if token_id.startswith("context:"):
                data = cache.get_token(token_id)
                if data and isinstance(data, dict):
                    contexts.append({
                        "session_id": token_id.replace("context:", ""),
                        "query": data.get("query", ""),
                        "entities": data.get("entities", []),
                        "timestamp": data.get("timestamp", 0)
                    })

    contexts.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return contexts[:limit]


def _add_document_impl(cache, session_id: str, doc_id: str, filename: str,
                       extracted_text: str, entities: List[str],
                       doc_type: str = "unknown", ttl: int = 86400) -> None:
    doc_key = f"doc:{session_id}:{doc_id}"

    doc_data = {
        "filename": filename,
        "extracted_text": extracted_text,
        "entities": entities,
        "doc_type": doc_type,
        "timestamp": time.time(),
        "ttl": ttl,
        "size": len(extracted_text)
    }

    cache.add_token(doc_key, doc_data)


def _get_document_impl(cache, session_id: str, doc_id: str) -> Optional[Dict]:
    doc_key = f"doc:{session_id}:{doc_id}"
    data = cache.get_token(doc_key)

    if data and isinstance(data, dict):
        timestamp = data.get("timestamp", 0)
        ttl = data.get("ttl", 86400)

        if time.time() - timestamp < ttl:
            return data

    return None


def _get_session_documents_impl(cache, session_id: str) -> List[Dict]:
    docs = []

    with cache._lock:
        for token_id in cache.token_metadata.keys():
            if token_id.startswith(f"doc:{session_id}:"):
                data = cache.get_token(token_id)
                if data and isinstance(data, dict):
                    docs.append({
                        "doc_id": token_id.split(":")[-1],
                        "filename": data.get("filename", ""),
                        "doc_type": data.get("doc_type", ""),
                        "size": data.get("size", 0),
                        "entities": data.get("entities", []),
                        "timestamp": data.get("timestamp", 0)
                    })

    return docs


def _delete_document_impl(cache, session_id: str, doc_id: str) -> bool:
    doc_key = f"doc:{session_id}:{doc_id}"

    with cache._lock:
        if doc_key in cache.token_metadata:
            del cache.token_metadata[doc_key]
            return True

    return False


def _add_search_results_impl(cache, query_hash: str, query: str,
                             results: List[Dict], ttl: int = 43200) -> None:
    import re

    search_key = f"search:{query_hash}"

    tokenized_results = []
    for r in results:
        snippet = r.get("snippet", "")
        tokenized = {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": snippet,
            "source": r.get("source", ""),
            "tokens": re.findall(r'\b\w+\b', snippet.lower())[:100],
            "entities": re.findall(r'\b[A-ZА-Я][a-zа-я]+\b', snippet)[:20]
        }
        tokenized_results.append(tokenized)

    search_data = {
        "query": query,
        "results": tokenized_results,
        "timestamp": time.time(),
        "ttl": ttl,
        "count": len(results)
    }

    cache.add_token(search_key, search_data)


def _get_search_results_impl(cache, query_hash: str) -> Optional[Dict]:
    search_key = f"search:{query_hash}"
    data = cache.get_token(search_key)

    if data and isinstance(data, dict):
        timestamp = data.get("timestamp", 0)
        ttl = data.get("ttl", 43200)

        if time.time() - timestamp < ttl:
            return data

    return None


def _get_search_cache_stats_impl(cache) -> Dict:
    search_count = 0
    doc_count = 0
    context_count = 0

    with cache._lock:
        for token_id in cache.token_metadata.keys():
            if token_id.startswith("search:"):
                search_count += 1
            elif token_id.startswith("doc:"):
                doc_count += 1
            elif token_id.startswith("context:"):
                context_count += 1

    return {
        "search_entries": search_count,
        "document_entries": doc_count,
        "context_entries": context_count
    }
