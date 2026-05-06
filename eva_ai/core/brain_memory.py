"""
Memory pressure detection and cache eviction for CoreBrain.
Split from brain_monitoring to keep modules under 500 lines.
"""
import time
import logging
from typing import Dict, Any

query_logger = logging.getLogger("eva_ai.core_brain.query_processing")
logger = logging.getLogger("eva_ai.core_brain")


class MemoryMixin:
    """Mixin providing memory pressure and cache eviction to CoreBrain."""

    def _check_memory_pressure(self):
        try:
            if not hasattr(self, 'token_cache') or not self.token_cache:
                return
            cache_stats = self.token_cache.get_cache_stats()
            import psutil
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            vram_pressure = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    vram_used = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
                    vram_pressure = vram_used
            except ImportError:
                pass
            needs_eviction = False
            eviction_source = None
            if memory_percent > 0.85:
                needs_eviction = True; eviction_source = 'ram'
            elif vram_pressure > 0.9:
                needs_eviction = True; eviction_source = 'vram'
            if needs_eviction:
                query_logger.info(f"Обнаружено давление памяти: {eviction_source}={memory_percent:.2f}, VRAM={vram_pressure:.2f}")
                if self.events:
                    self.events.trigger('memory_pressure', {'source': eviction_source, 'memory_percent': memory_percent, 'vram_pressure': vram_pressure, 'cache_stats': cache_stats})
                self._perform_smart_eviction(eviction_source, memory_percent, vram_pressure)
        except Exception as e:
            query_logger.error(f"Ошибка проверки давления памяти: {e}")

    def _handle_memory_pressure(self, event_data):
        try:
            source = event_data.get('source', 'unknown')
            memory_percent = event_data.get('memory_percent', 0.0)
            vram_pressure = event_data.get('vram_pressure', 0.0)
            query_logger.info(f"Обработка давления памяти из {source}: RAM={memory_percent:.2f}, VRAM={vram_pressure:.2f}")
            self._perform_smart_eviction(source, memory_percent, vram_pressure)
        except Exception as e:
            query_logger.error(f"Ошибка обработки давления памяти: {e}")

    def _handle_cache_eviction(self, event_data):
        try:
            eviction_type = event_data.get('type', 'lru')
            target_tokens = event_data.get('target_tokens', 100)
            query_logger.info(f"Выполнение вытеснения кэша: {eviction_type}, токенов: {target_tokens}")
            if eviction_type == 'smart':
                self._perform_smart_eviction('system', 0.9, 0.9)
            else:
                self._perform_basic_eviction(target_tokens)
        except Exception as e:
            query_logger.error(f"Ошибка обработки вытеснения кэша: {e}")

    def _perform_smart_eviction(self, source, memory_percent, vram_pressure):
        try:
            if not hasattr(self, 'token_cache') or not self.token_cache:
                return
            if source == 'vram':
                self._evict_vram_to_ram()
            elif source == 'ram':
                self._evict_ram_to_ssd()
            else:
                if vram_pressure > 0.8:
                    self._evict_vram_to_ram()
                if memory_percent > 0.8:
                    self._evict_ram_to_ssd()
            self._update_cache_metrics()
        except Exception as e:
            query_logger.error(f"Ошибка умного вытеснения: {e}")

    def _evict_vram_to_ram(self):
        try:
            if not hasattr(self.token_cache, 'vram_cache') or not hasattr(self.token_cache, 'ram_cache'):
                return
            vram_cache = self.token_cache.vram_cache
            ram_cache = self.token_cache.ram_cache
            tokens_to_evict = []
            for token_id, token_data in vram_cache.items():
                metadata = self.token_cache.token_metadata.get(token_id, {})
                tokens_to_evict.append({'id': token_id, 'data': token_data, 'last_access': metadata.get('last_access', 0), 'access_count': metadata.get('access_count', 0), 'relevance': metadata.get('relevance_score', 0.5)})
            tokens_to_evict.sort(key=lambda x: (x['relevance'], x['last_access']))
            evict_count = max(1, len(tokens_to_evict) // 4)
            evicted = 0
            for token_info in tokens_to_evict[:evict_count]:
                ram_cache.put(token_info['id'], token_info['data'])
                vram_cache.pop(token_info['id'], None)
                self.token_cache.token_metadata[token_info['id']] = {**self.token_cache.token_metadata.get(token_info['id'], {}), 'location': 'ram', 'evicted_from': 'vram', 'eviction_time': time.time()}
                evicted += 1
            query_logger.info(f"Вытеснено {evicted} токенов из VRAM в RAM")
            if self.events:
                self.events.trigger('vram_to_ram_eviction', {'evicted_count': evicted, 'timestamp': time.time()})
        except Exception as e:
            query_logger.error(f"Ошибка вытеснения VRAM->RAM: {e}")

    def _evict_ram_to_ssd(self):
        try:
            if not hasattr(self.token_cache, 'ram_cache') or not hasattr(self.token_cache, 'disk_cache'):
                return
            ram_cache = self.token_cache.ram_cache
            disk_cache = self.token_cache.disk_cache
            tokens_to_evict = []
            for token_id, token_data in ram_cache.items():
                metadata = self.token_cache.token_metadata.get(token_id, {})
                tokens_to_evict.append({'id': token_id, 'data': token_data, 'last_access': metadata.get('last_access', 0), 'access_count': metadata.get('access_count', 0), 'relevance': metadata.get('relevance_score', 0.5)})
            tokens_to_evict.sort(key=lambda x: (x['relevance'], x['last_access']))
            evict_count = max(1, len(tokens_to_evict) * 3 // 10)
            evicted = 0
            for token_info in tokens_to_evict[:evict_count]:
                if disk_cache.save_token(token_info['id'], token_info['data']):
                    ram_cache.pop(token_info['id'], None)
                    self.token_cache.token_metadata[token_info['id']] = {**self.token_cache.token_metadata.get(token_info['id'], {}), 'location': 'ssd', 'evicted_from': 'ram', 'eviction_time': time.time()}
                    evicted += 1
            query_logger.info(f"Вытеснено {evicted} токенов из RAM в SSD")
            if self.events:
                self.events.trigger('ram_to_ssd_eviction', {'evicted_count': evicted, 'timestamp': time.time()})
        except Exception as e:
            query_logger.error(f"Ошибка вытеснения RAM->SSD: {e}")

    def _perform_basic_eviction(self, target_tokens):
        try:
            if hasattr(self, 'token_cache') and self.token_cache and hasattr(self.token_cache, '_evict_one_lru'):
                evicted = 0
                for _ in range(min(target_tokens, 100)):
                    self.token_cache._evict_one_lru()
                    evicted += 1
                query_logger.info(f"Выполнено базовое вытеснение: {evicted} токенов")
        except Exception as e:
            query_logger.error(f"Ошибка базового вытеснения: {e}")

    def _update_cache_metrics(self):
        try:
            if hasattr(self, 'metrics_manager') and self.metrics_manager and hasattr(self.token_cache, 'get_cache_stats'):
                stats = self.token_cache.get_cache_stats()
                self.metrics_manager.record_query_metrics(cache_vram_hits=stats.get('vram_hits', 0), cache_ram_hits=stats.get('ram_hits', 0), cache_disk_hits=stats.get('disk_hits', 0), cache_evictions=stats.get('evictions', 0), cache_efficiency=stats.get('cache_efficiency', 0.0))
        except Exception as e:
            query_logger.debug(f"Ошибка обновления метрик кэша: {e}")
