"""
Гибридная система кэширования токенов для CogniFlex.
Оптимизирует обработку запросов за счет переноса части токенов в кеш жесткого диска.
"""

import os
import json
import time
import zlib
import threading
import hashlib
from typing import Dict, List, Optional, Any
from collections import OrderedDict
import logging
from .disk_cache import DiskCache

logger = logging.getLogger(__name__)

class LRUCache:
    """Простая реализация LRU кэша."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Добавляет элемент в кэш."""
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Проверяет наличие ключа в кэше."""
        return key in self.cache
    
    def keys(self):
        """Возвращает ключи кэша."""
        return self.cache.keys()
    
    def remove(self, key: str):
        """Удаляет элемент из кэша."""
        if key in self.cache:
            del self.cache[key]
    
    def __len__(self):
        """Возвращает размер кэша."""
        return len(self.cache)

class HybridTokenCache:
    """Гибридная система кэширования токенов с использованием оперативной памяти и дискового кэша."""
    
    def __init__(
        self,
        brain,
        max_memory_tokens: int = 10000,
        disk_cache_dir: str = "token_cache",
        target_memory_gb: float = 50.0,
        dynamic_memory_limit: bool = True,
    ):
        self.brain = brain
        self.disk_cache_dir = os.path.join(brain.cache_dir, disk_cache_dir)
        os.makedirs(self.disk_cache_dir, exist_ok=True)

        # Целевой размер оперативного кэша в байтах (по умолчанию 50 ГБ, по запросу пользователя)
        self.dynamic_memory_limit = dynamic_memory_limit
        self.target_memory_bytes = int(target_memory_gb * 1024 ** 3)
        # Стартовая оценка среднего размера токена (4 КБ)
        self.avg_token_size_bytes = 4096
        # Если включена динамика — рассчитываем число токенов из целевого объёма,
        # иначе используем переданное значение max_memory_tokens
        self.max_memory_tokens = (
            max(1, int(self.target_memory_bytes / self.avg_token_size_bytes))
            if self.dynamic_memory_limit else max_memory_tokens
        )
        
        # Оперативный кэш с улучшенной производительностью
        self.memory_cache = LRUCache(self.max_memory_tokens)
        
        # Дисковый кэш с полным использованием 50GB
        self.disk_cache = DiskCache(os.path.join(self.disk_cache_dir, "disk_storage"), max_size_gb=50.0)
        
        # Агрессивная стратегия кэширования для полного использования 50GB
        self.aggressive_caching = True
        self.cache_utilization_target = 0.95  # Стремимся к 95% использованию
        
        # Метаданные о токенах с thread-safe доступом
        self.token_metadata = {}
        self.metadata_lock = threading.RLock()
        self._load_metadata()
        
        # Расширенная статистика использования
        self.usage_stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "total_accesses": 0,
            "cache_efficiency": 0.0,
            "avg_access_time": 0.0,
            "memory_usage_mb": 0.0
        }

        # Прочие параметры и структуры
        self.stats_lock = threading.RLock()
        
        # Оптимизированные параметры
        self.hot_threshold = 3  # Количество обращений для перемещения в память
        self.eviction_threshold = 0.85  # Порог заполнения для начала вытеснения
        self.batch_size = 100  # Размер пакета для массовых операций
        
        # Кэш для быстрого поиска
        self.token_index = {}  # token_hash -> metadata
        
        # Отключаем фоновую оптимизацию для стабильности
        # self._start_optimization_worker()
        
        approx_mem_gb = (self.max_memory_tokens * self.avg_token_size_bytes) / (1024 ** 3)
        logger.info(
            f"HybridTokenCache инициализирован: память~{self.max_memory_tokens} токенов (~{approx_mem_gb:.2f}GB), диск=50.0GB"
        )

        # Настройки гибридного кэша (расширяемые)
        self.cache_settings: Dict[str, Any] = {
            "max_memory_size": self.max_memory_tokens,
            "disk_cache_threshold": max(1, self.max_memory_tokens // 2),
            "eviction_policy": "lru",  # lru, lfu, fifo, hybrid
            "cache_ttl": 86400,
            "disk_cache_size": 50000,
            "min_relevance_score": 0.3,
            "max_context_tokens": 1000,
        }

    # ===== Механизм расширения и приоритизации контекста =====
    def expand_context(self, query: str, current_context: str, task_type: str = "general") -> str:
        """Расширяет контекст на основе динамических весов узлов графа знаний."""
        kg = getattr(self.brain, 'knowledge_graph', None)
        if kg is None:
            return current_context

        # Получаем релевантные узлы
        relevant_nodes = kg.get_relevant_nodes(query)
        prioritized = kg.prioritize_nodes(query, relevant_nodes)

        expanded = current_context or ""
        tokens_used = self._count_tokens(expanded)
        max_tokens = int(self.cache_settings.get("max_context_tokens", 1000))
        min_relevance = float(self.cache_settings.get("min_relevance_score", 0.3))

        for node, weight in prioritized:
            if tokens_used >= max_tokens * 0.9:
                break
            if weight < min_relevance:
                continue
            node_text = self._format_node_for_context(node)
            node_tokens = self._count_tokens(node_text)
            if tokens_used + node_tokens <= max_tokens:
                expanded += ("\n\n" if expanded else "") + node_text
                tokens_used += node_tokens
        return expanded

    def get_optimal_cache_strategy(self, task_type: str) -> Dict[str, Any]:
        """Возвращает оптимальные параметры кэширования для указанного типа задачи."""
        strategies = {
            "text-generation": {
                "max_context_tokens": 1000,
                "min_relevance_score": 0.35,
                "temporal_weight": 0.15,
                "relevance_weight": 0.5,
            },
            "summarization": {
                "max_context_tokens": 800,
                "min_relevance_score": 0.4,
                "temporal_weight": 0.2,
                "relevance_weight": 0.6,
            },
            "translation": {
                "max_context_tokens": 700,
                "min_relevance_score": 0.25,
                "temporal_weight": 0.1,
                "relevance_weight": 0.4,
            },
            "question-answering": {
                "max_context_tokens": 900,
                "min_relevance_score": 0.45,
                "temporal_weight": 0.25,
                "relevance_weight": 0.65,
            },
        }
        return strategies.get(task_type, strategies["text-generation"])

    def prioritize_context(self, query: str, context: str, task_type: str = "general") -> str:
        """Динамически управляет контекстом, приоритизируя наиболее релевантные части."""
        strategy = self.get_optimal_cache_strategy(task_type)
        cache_key = f"context:{hashlib.md5((query + '|' + task_type).encode()).hexdigest()}"
        if self.exists(cache_key):
            cached = self.get(cache_key)
            if isinstance(cached, str):
                return cached
        # Расширяем при необходимости
        if self._count_tokens(context) < strategy["max_context_tokens"] * 0.7:
            context = self.expand_context(query, context, task_type)
        optimized = self._optimize_context(query, context, strategy)
        self.set(cache_key, optimized)
        return optimized

    def _optimize_context(self, query: str, context: str, strategy: Dict[str, Any]) -> str:
        """Оптимизирует контекст, сохраняя наиболее релевантные сегменты в пределах лимита токенов."""
        segments = self._split_into_segments(context)
        scored: List[tuple[str, float]] = []
        for seg in segments:
            rel = self._calculate_segment_relevance(query, seg, strategy)
            scored.append((seg, rel))
        scored.sort(key=lambda x: x[1], reverse=True)
        max_tokens = strategy["max_context_tokens"]
        min_rel = strategy["min_relevance_score"]
        selected: List[str] = []
        used = 0
        for seg, score in scored:
            if score < min_rel:
                continue
            seg_tokens = self._count_tokens(seg)
            if used + seg_tokens <= max_tokens:
                selected.append(seg)
                used += seg_tokens
            elif used >= max_tokens * 0.8:
                break
        return "\n\n".join(selected) if selected else context

    # ===== Служебные методы для контекста =====
    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        # Приближённо: 1 токен ~ 4 символа для латиницы, здесь используем универсальную оценку
        return max(1, len(text) // 4)

    def _format_node_for_context(self, node: Any) -> str:
        name = getattr(node, 'name', '')
        description = getattr(node, 'description', '')
        domain = getattr(node, 'domain', 'general')
        return f"[{domain}] {name}: {description}".strip()

    def _calculate_segment_relevance(self, query: str, segment: str, strategy: Dict[str, Any]) -> float:
        try:
            q = (query or '').lower().split()
            s = (segment or '').lower()
            if not q:
                return 0.0
            matches = sum(1 for t in set(q) if len(t) > 2 and t in s)
            base_rel = matches / max(1, len(set(q)))
            return float(min(1.0, max(0.0, base_rel)))
        except Exception:
            return 0.0

    def _split_into_segments(self, context: str) -> List[str]:
        if not context:
            return []
        # Простое разбиение по двойным переводам строк, затем по точкам как запасной вариант
        parts = [p.strip() for p in context.split("\n\n") if p.strip()]
        if len(parts) <= 1:
            # Разбиение по предложениям (ограниченно)
            import re
            parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", context) if p.strip()]
        return parts
    
    def get_token(self, token_id: str) -> Optional[Dict]:
        """Получает токен по ID из гибридного кэша с оптимизированной производительностью."""
        start_time = time.time()
        
        with self.stats_lock:
            self.usage_stats["total_accesses"] += 1
        
        # Быстрая проверка оперативного кэша
        if token_id in self.memory_cache:
            token_data = self.memory_cache.get(token_id)
            if token_data:
                with self.stats_lock:
                    self.usage_stats["memory_hits"] += 1
                    access_time = time.time() - start_time
                    self._update_avg_access_time(access_time)
                self._update_token_access(token_id)
                return token_data
        
        # Проверяем дисковый кэш
        disk_token = self._load_token_from_disk(token_id)
        if disk_token:
            with self.stats_lock:
                self.usage_stats["disk_hits"] += 1
                access_time = time.time() - start_time
                self._update_avg_access_time(access_time)
            
            self._update_token_access(token_id)
            
            # Асинхронно проверяем, нужно ли перемещать в память
            with self.metadata_lock:
                metadata = self.token_metadata.get(token_id, {})
                if metadata.get("access_count", 0) > self.hot_threshold:
                    # Перемещаем в память в фоновом режиме
                    threading.Thread(
                        target=self._move_token_to_memory, 
                        args=(token_id, disk_token),
                        daemon=True
                    ).start()
            
            return disk_token
        
        with self.stats_lock:
            self.usage_stats["misses"] += 1
        return None
    
    def add_token(self, token_id: str, token_data: Any) -> None:
        """Добавляет токен в кэш."""
        with self.metadata_lock:
            # Обрабатываем различные форматы token_data
            if isinstance(token_data, dict):
                priority = token_data.get("priority", 0.5)
                actual_data = token_data
            else:
                priority = 0.5
                actual_data = token_data
            
            # Обновляем метаданные
            if token_id not in self.token_metadata:
                self.token_metadata[token_id] = {
                    "access_count": 0,
                    "last_access": time.time(),
                    "priority": priority,
                    "in_memory": False
                }      
        # Динамически пересчитываем целевой лимит оперативной памяти исходя из среднего размера токена
        if self.dynamic_memory_limit:
            try:
                # Оценка размера токена в байтах
                if isinstance(actual_data, (dict, list)):
                    serialized = json.dumps(actual_data, ensure_ascii=False).encode("utf-8")
                    size_bytes = len(serialized)
                elif isinstance(actual_data, str):
                    size_bytes = len(actual_data.encode("utf-8"))
                else:
                    size_bytes = len(str(actual_data).encode("utf-8"))

                # EMA для сглаживания средней величины
                alpha = 0.1
                self.avg_token_size_bytes = int(self.avg_token_size_bytes * (1 - alpha) + size_bytes * alpha)
                # Пересчёт лимита токенов
                new_limit = max(1, int(self.target_memory_bytes / max(1, self.avg_token_size_bytes)))
                if new_limit != self.max_memory_tokens:
                    self.max_memory_tokens = new_limit
                    self.memory_cache.max_size = new_limit
                    # При необходимости вытеснить избыточные записи
                    while len(self.memory_cache) > self.max_memory_tokens:
                        # Вытеснение по LRU
                        first_key = next(iter(self.memory_cache.cache))
                        self._move_token_to_disk(first_key)
            except Exception as _:
                # Не мешаем основному потоку работы кэша
                pass
        # Агрессивная стратегия кэширования - всегда сохраняем на диск для максимального использования 50GB
        if self.aggressive_caching:
            # Сохраняем на диск в любом случае для полного использования кэша
            self._save_token_to_disk(token_id, actual_data)
            self.token_metadata[token_id]["in_memory"] = False
            
            # Если есть место в памяти, дублируем туда для быстрого доступа
            if len(self.memory_cache) < self.max_memory_tokens:
                self.memory_cache.put(token_id, actual_data)
                self.token_metadata[token_id]["in_memory"] = True
        else:
            # Стандартная логика
            if len(self.memory_cache) < self.max_memory_tokens:
                self.memory_cache.put(token_id, actual_data)
                self.token_metadata[token_id]["in_memory"] = True
            else:
                self._save_token_to_disk(token_id, actual_data)
                self.token_metadata[token_id]["in_memory"] = False

    # ===== Простой KV-интерфейс для совместимости с контекст-менеджером =====
    def exists(self, key: str) -> bool:
        return key in self.memory_cache or bool(self._load_token_from_disk(key))

    def get(self, key: str) -> Optional[Any]:
        return self.get_token(key)

    def set(self, key: str, value: Any) -> None:
        self.add_token(key, value)
    
    def _generate_token_id(self, token: str) -> str:
        """Генерирует уникальный ID для токена."""
        return hashlib.md5(token.encode('utf-8')).hexdigest()
    
    def _load_token_from_disk(self, token_id: str) -> Optional[Dict]:
        """Загружает токен с диска."""
        return self.disk_cache.get(token_id)
    
    def _save_token_to_disk(self, token_id: str, token_data: Dict):
        """Сохраняет токен на диск."""
        self.disk_cache.put(token_id, token_data)
    
    def _move_token_to_memory(self, token_id: str, token_data: Optional[Dict] = None):
        """Перемещает токен из дискового кэша в память."""
        if token_data is None:
            token_data = self._load_token_from_disk(token_id)
            if not token_data:
                return
        
        # Просто перемещаем в память (дисковый кэш сам управляет своими файлами)
        
        # Добавляем в память
        self.memory_cache.put(token_id, token_data)
        if token_id in self.token_metadata:
            self.token_metadata[token_id]["in_memory"] = True
    
    def _move_token_to_disk(self, token_id: str):
        """Перемещает токен из памяти на диск."""
        token_data = self.memory_cache.get(token_id)
        if not token_data:
            return
        
        self.memory_cache.remove(token_id)
        self._save_token_to_disk(token_id, token_data)
        
        if token_id in self.token_metadata:
            self.token_metadata[token_id]["in_memory"] = False
    
    def _update_token_access(self, token_id: str):
        """Обновляет статистику доступа к токену thread-safe."""
        with self.metadata_lock:
            if token_id in self.token_metadata:
                self.token_metadata[token_id]["access_count"] += 1
                self.token_metadata[token_id]["last_access"] = time.time()
    
    def _update_avg_access_time(self, access_time: float):
        """Обновляет среднее время доступа."""
        current_avg = self.usage_stats.get("avg_access_time", 0.0)
        total_accesses = self.usage_stats.get("total_accesses", 1)
        
        # Экспоненциальное скользящее среднее
        alpha = 0.1  # Коэффициент сглаживания
        self.usage_stats["avg_access_time"] = current_avg * (1 - alpha) + access_time * alpha
    
    def _evict_least_important(self):
        """Вытесняет наименее важные токены из памяти."""
        if len(self.memory_cache) <= self.max_memory_tokens * 0.7:
            return
        
        # Получаем токены с приоритетами
        tokens = []
        for token_id in self.memory_cache.keys():
            metadata = self.token_metadata.get(token_id, {})
            priority = self._calculate_token_priority(token_id, metadata)
            tokens.append((token_id, priority))
        
        # Сортируем по приоритету (низкий к высокому)
        tokens.sort(key=lambda x: x[1])
        
        # Вытесняем до целевого размера
        target_size = int(self.max_memory_tokens * 0.7)
        evicted = 0
        
        for token_id, _ in tokens:
            if len(self.memory_cache) <= target_size:
                break
            self._move_token_to_disk(token_id)
            evicted += 1
        
        logger.info(f"Вытеснено {evicted} токенов в дисковый кэш")

    # Новый адаптивный алгоритм вытеснения, учитывающий веса
    def _evict_items(self):
        """Вытесняет элементы из кэша в соответствии с self.cache_settings['eviction_policy']."""
        if len(self.memory_cache) <= self.cache_settings.get("max_memory_size", self.max_memory_tokens):
            return
        # Собираем веса элементов
        items = [(key, self._calculate_item_weight(key)) for key in list(self.memory_cache.keys())]
        items.sort(key=lambda x: x[1])  # меньше вес — раньше вытеснение
        # Оцениваем, сколько освободить
        space_needed = len(self.memory_cache) - int(self.cache_settings.get("max_memory_size", self.max_memory_tokens))
        for key, _ in items:
            if space_needed <= 0:
                break
            self._move_token_to_disk(key)
            space_needed -= 1

    def _calculate_item_weight(self, key: str) -> float:
        """Рассчитывает вес элемента для решения о вытеснении."""
        meta = self.token_metadata.get(key, {})
        access_frequency = meta.get('access_count', 1)
        last_access = meta.get('last_access', 0)
        temporal_factor = 1.0 if time.time() - last_access < 3600 else 0.5
        policy = self.cache_settings.get("eviction_policy", "lru")
        if policy == "lru":
            return last_access
        elif policy == "lfu":
            return 1.0 / max(1, access_frequency)
        elif policy == "hybrid":
            lru_weight = last_access / (time.time() + 1)
            lfu_weight = 1.0 / (access_frequency + 1)
            return (lru_weight * 0.6) + (lfu_weight * 0.4)
        else:
            return last_access
    
    def _calculate_token_priority(self, token_id: str, metadata: Dict) -> float:
        """Рассчитывает приоритет токена."""
        access_count = metadata.get("access_count", 0)
        last_access = metadata.get("last_access", 0)
        priority = metadata.get("priority", 0.5)
        
        # Временной фактор
        time_factor = 1.0 if time.time() - last_access < 3600 else 0.3
        
        # Общий приоритет
        return (access_count * 0.6) + (priority * 0.3) + (time_factor * 0.1)
    
    def _load_metadata(self):
        """Загружает метаданные токенов."""
        try:
            metadata_path = os.path.join(self.disk_cache_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.token_metadata = json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки метаданных: {e}")
    
    def _save_metadata(self):
        """Сохраняет метаданные токенов."""
        try:
            metadata_path = os.path.join(self.disk_cache_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.token_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения метаданных: {e}")
    
    def _optimize_memory_task(self):
        """Фоновая задача оптимизации памяти."""
        try:
            # Проверяем уровень использования памяти
            memory_usage = len(self.memory_cache) / self.max_memory_tokens
            
            if memory_usage > self.eviction_threshold:
                self._evict_least_important()
            
            # Сохраняем метаданные
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Ошибка в фоновой оптимизации: {e}")
    
    def _start_optimization_worker(self):
        """Запускает фоновый процесс оптимизации."""
        def worker():
            while not getattr(self.brain, 'stop_event', threading.Event()).is_set():
                try:
                    self._optimize_memory_task()
                    time.sleep(30)  # Проверяем каждые 30 секунд
                except Exception as e:
                    logger.error(f"Ошибка в фоновом процессе: {e}")
                    time.sleep(60)
        
        self.optimization_thread = threading.Thread(target=worker, daemon=True)
        self.optimization_thread.start()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша."""
        disk_stats = self.disk_cache.get_stats()
        cache_utilization = disk_stats.get("size_bytes", 0) / (50 * 1024**3)  # 50GB в байтах
        
        return {
            "memory_tokens": len(self.memory_cache),
            "memory_limit_tokens": self.max_memory_tokens,
            "avg_token_size_bytes": self.avg_token_size_bytes,
            "target_memory_gb": round(self.target_memory_bytes / (1024 ** 3), 2),
            "disk_stats": disk_stats,
            "usage_stats": self.usage_stats.copy(),
            "hit_rate": (self.usage_stats["memory_hits"] + self.usage_stats["disk_hits"]) / max(1, self.usage_stats["total_accesses"]),
            "cache_utilization_percent": cache_utilization * 100,
            "aggressive_caching": self.aggressive_caching,
            "target_utilization_percent": self.cache_utilization_target * 100
        }
    
    def clear_cache(self):
        """Очищает весь кэш."""
        self.memory_cache = LRUCache(self.max_memory_tokens)
        self.disk_cache.close()
        self.disk_cache = DiskCache(os.path.join(self.disk_cache_dir, "disk_storage"))
        self.token_metadata = {}
        self.usage_stats = {"memory_hits": 0, "disk_hits": 0, "misses": 0, "total_accesses": 0}