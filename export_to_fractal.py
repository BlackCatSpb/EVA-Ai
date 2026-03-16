"""
Гибридная система кэширования токенов для CogniFlex.
Оптимизирует обработку запросов за счет многоуровневой архитектуры:
GPU (VRAM) → RAM → SSD
"""
import os
import json
import time
import threading
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import logging
import torch
import pickle

# Опционально используем psutil для мониторинга памяти
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

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
    
    def clear(self):
        """Очищает кэш."""
        self.cache.clear()


class HybridTokenCache:
    """
    Гибридная система кэширования токенов с загрузкой тензоров на GPU.
    Архитектура: GPU (VRAM) → RAM → SSD
    """
    
    def __init__(
        self,
        brain,
        max_memory_tokens: int = 10000,
        disk_cache_dir: str = "token_cache",
        target_memory_gb: float = 2.0,
        dynamic_memory_limit: bool = True,
        max_ram_usage_percent: float = 80.0,
        gpu_acceleration: bool = True,
        **kwargs
    ):
        self.brain = brain
        
        # Проверка доступности GPU
        self.gpu_available = gpu_acceleration and torch.cuda.is_available()
        self.device = "cuda:0" if self.gpu_available else "cpu"
        
        if self.gpu_available:
            logger.info(f"🎮 GPU активирован: {torch.cuda.get_device_name(0)}")
            logger.info(f"   Доступно VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            logger.info("💻 Работа в CPU режиме")
        
        # Определение базовой директории кэша
        try:
            base_cache_dir = None
            if brain is not None:
                base_cache_dir = getattr(brain, 'cache_dir', None)
            if not base_cache_dir:
                base_cache_dir = os.environ.get('COGNIFLEX_CACHE_DIR')
            if not base_cache_dir:
                base_cache_dir = os.path.join(os.getcwd(), 'ml_cache')
            os.makedirs(base_cache_dir, exist_ok=True)
        except Exception:
            base_cache_dir = os.path.join(os.getcwd(), 'ml_cache')
            os.makedirs(base_cache_dir, exist_ok=True)
        
        self.disk_cache_dir = os.path.join(base_cache_dir, disk_cache_dir)
        os.makedirs(self.disk_cache_dir, exist_ok=True)
        
        # Загрузка конфигурации из brain.config
        try:
            cfg = getattr(self.brain, 'config', {}) or {}
            hc = cfg.get('hybrid_cache') or {}
        except Exception:
            hc = {}
        
        # Применение настроек из конфигурации
        target_memory_gb = float(hc.get('target_memory_gb', target_memory_gb))
        dynamic_memory_limit = bool(hc.get('dynamic_memory_limit', dynamic_memory_limit))
        max_ram_usage_percent = float(hc.get('max_ram_usage_percent', max_ram_usage_percent))
        
        # Динамический расчет лимита памяти
        self.dynamic_memory_limit = dynamic_memory_limit
        self.target_memory_bytes = int(target_memory_gb * 1024 ** 3)
        
        if psutil and dynamic_memory_limit:
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            self.target_memory_bytes = min(
                self.target_memory_bytes,
                int(total_ram_gb * 1024 ** 3 * (max_ram_usage_percent / 100.0))
            )
            logger.info(f"Динамический лимит памяти: {target_memory_gb:.2f}GB ({max_ram_usage_percent}% RAM)")
        
        # Стартовая оценка среднего размера токена (4 КБ)
        self.avg_token_size_bytes = 4096
        
        # Расчет максимального количества токенов
        self.max_memory_tokens = (
            max(1, int(self.target_memory_bytes / self.avg_token_size_bytes))
            if dynamic_memory_limit else max_memory_tokens
        )
        
        # Инициализация кэшей для разных уровней памяти
        # GPU кэш (горячее окно)
        self.gpu_cache = OrderedDict() if self.gpu_available else None
        self.gpu_cache_size = 0
        self.max_gpu_memory = int(self.target_memory_bytes * 0.7) if self.gpu_available else 0
        
        # RAM кэш
        self.ram_cache = LRUCache(max(1, int(self.max_memory_tokens * 0.3)))
        
        # Для обратной совместимости
        self.memory_cache = self.ram_cache
        
        # Дисковый кэш (простой dict для надежности)
        self.disk_cache = {}
        
        # Метаданные токенов
        self.token_metadata = {}
        self.metadata_lock = threading.RLock()
        self.stats_lock = threading.RLock()
        
        # Статистика
        self.cache_stats = {
            'gpu_hits': 0,
            'ram_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'gpu_to_ram_transfers': 0,
            'ram_to_gpu_transfers': 0
        }
        
        # Расширенная статистика
        self.usage_stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "total_accesses": 0,
            "cache_efficiency": 0.0,
            "avg_access_time": 0.0,
            "memory_usage_mb": 0.0
        }
        
        # Параметры вытеснения
        self.hot_threshold = int(hc.get('hot_threshold', 3))
        self.vram_threshold = float(hc.get('vram_threshold', 0.2))
        self.ram_threshold = float(hc.get('ram_threshold', 0.15))
        
        # Загрузка метаданных
        self._load_metadata()
        
        # Запуск монитора памяти
        self._start_memory_pressure_worker()
        
        approx_mem_gb = (self.max_memory_tokens * self.avg_token_size_bytes) / (1024 ** 3)
        logger.info(
            f"HybridTokenCache инициализирован: {self.device}, "
            f"память~{self.max_memory_tokens} токенов (~{approx_mem_gb:.2f}GB), "
            f"диск={self.disk_cache_dir}"
        )
    
    def get_token(self, token_id: str) -> Optional[Dict]:
        """
        Получает токен по ID из гибридного кэша.
        Приоритет: GPU → RAM → Disk
        """
        start_time = time.time()
        
        with self.stats_lock:
            self.usage_stats["total_accesses"] += 1
            self.cache_stats['total_requests'] += 1
        
        # 1. Проверка GPU кэша
        if self.gpu_available and self.gpu_cache is not None:
            if token_id in self.gpu_cache:
                token_data = self.gpu_cache[token_id]
                self.gpu_cache.move_to_end(token_id)
                
                with self.stats_lock:
                    self.cache_stats['gpu_hits'] += 1
                    self.usage_stats["memory_hits"] += 1
                
                access_time = time.time() - start_time
                self._update_avg_access_time(access_time)
                self._update_token_access(token_id)
                
                # Возвращаем тензор на GPU
                if isinstance(token_data, dict) and 'tensor' in token_data:
                    if not token_data['tensor'].is_cuda:
                        token_data['tensor'] = token_data['tensor'].to(self.device)
                
                logger.debug(f"GPU hit для {token_id}")
                return token_data
        
        # 2. Проверка RAM кэша
        if token_id in self.ram_cache:
            token_data = self.ram_cache.get(token_id)
            
            with self.stats_lock:
                self.cache_stats['ram_hits'] += 1
                self.usage_stats["memory_hits"] += 1
            
            access_time = time.time() - start_time
            self._update_avg_access_time(access_time)
            self._update_token_access(token_id)
            
            # Перемещаем в GPU кэш если есть место и данные часто используются
            if self.gpu_available and self.gpu_cache is not None:
                metadata = self.token_metadata.get(token_id, {})
                if metadata.get("access_count", 0) > self.hot_threshold:
                    self._move_to_gpu(token_id, token_data)
            
            logger.debug(f"RAM hit для {token_id}")
            return token_data
        
        # 3. Проверка дискового кэша
        disk_token = self._load_token_from_disk(token_id)
        if disk_token:
            with self.stats_lock:
                self.cache_stats['disk_hits'] += 1
                self.usage_stats["disk_hits"] += 1
            
            access_time = time.time() - start_time
            self._update_avg_access_time(access_time)
            self._update_token_access(token_id)
            
            # Асинхронно перемещаем в RAM/GPU при частом доступе
            with self.metadata_lock:
                metadata = self.token_metadata.get(token_id, {})
                if metadata.get("access_count", 0) > self.hot_threshold:
                    threading.Thread(
                        target=self._move_to_memory,
                        args=(token_id, disk_token),
                        daemon=True
                    ).start()
            
            logger.debug(f"Disk hit для {token_id}")
            return disk_token
        
        # 4. Cache miss
        with self.stats_lock:
            self.usage_stats["misses"] += 1
            self.cache_stats['misses'] += 1
        
        logger.debug(f"Cache miss для {token_id}")
        return None
    
    def add_token(self, token_id: str, token_data: Any) -> None:
        """
        Добавляет токен в кэш с приоритетным размещением на GPU.
        """
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
                    "in_memory": False,
                    "on_gpu": False,
                    "mem_size": 0,
                }
            
            # Динамически пересчитываем целевой лимит
            if self.dynamic_memory_limit:
                try:
                    size_bytes = self._estimate_size_bytes(actual_data)
                    alpha = 0.1
                    self.avg_token_size_bytes = int(
                        self.avg_token_size_bytes * (1 - alpha) + size_bytes * alpha
                    )
                    new_limit = max(1, int(self.target_memory_bytes / max(1, self.avg_token_size_bytes)))
                    if new_limit != self.max_memory_tokens:
                        self.max_memory_tokens = new_limit
                        self.ram_cache.max_size = int(self.max_memory_tokens * 0.3)
                        
                        # Вытесняем избыточные записи
                        while len(self.ram_cache) > self.ram_cache.max_size:
                            first_key = next(iter(self.ram_cache.cache))
                            self._move_to_disk(first_key)
                except Exception:
                    pass
            
            # Сохраняем на диск в любом случае
            self._save_token_to_disk(token_id, actual_data)
            self.token_metadata[token_id]["in_memory"] = False
            self.token_metadata[token_id]["on_gpu"] = False
            
            # Попытка размещения в GPU кэше
            if self.gpu_available and self.gpu_cache is not None:
                if self._can_fit_in_gpu(token_id, actual_data):
                    self._move_to_gpu(token_id, actual_data)
                    self.token_metadata[token_id]["on_gpu"] = True
                    self.token_metadata[token_id]["in_memory"] = True
                    logger.debug(f"Токен {token_id} размещен на GPU")
                    return
            
            # Размещение в RAM кэше
            if len(self.ram_cache) < self.ram_cache.max_size:
                if len(self.ram_cache) >= self.ram_cache.max_size:
                    self._evict_one_lru()
                self.ram_cache.put(token_id, actual_data)
                self.token_metadata[token_id]["in_memory"] = True
                self.token_metadata[token_id]["mem_size"] = self._estimate_size_bytes(actual_data)
                logger.debug(f"Токен {token_id} размещен в RAM")
            else:
                # Только на диске
                self.token_metadata[token_id]["in_memory"] = False
    
    def _can_fit_in_gpu(self, token_id: str, token_data: Any) -> bool:
        """Проверяет, поместится ли токен в GPU кэш."""
        if not self.gpu_available or self.gpu_cache is None:
            return False
        
        size_bytes = self._estimate_size_bytes(token_data)
        
        # Проверяем доступную VRAM
        try:
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            free = total - allocated
            
            # Оставляем 20% VRAM для вычислений
            available_for_cache = free * 0.8
            
            return size_bytes < available_for_cache
        except Exception:
            return False
    
    def _move_to_gpu(self, token_id: str, token_data: Any) -> None:
        """Перемещает токен в GPU кэш."""
        if not self.gpu_available or self.gpu_cache is None:
            return
        
        try:
            # Вытесняем старые токены если нужно
            while self.gpu_cache_size >= self.max_gpu_memory * 0.9:
                if self.gpu_cache:
                    oldest_key = next(iter(self.gpu_cache))
                    oldest_data = self.gpu_cache.pop(oldest_key)
                    self.gpu_cache_size -= self._estimate_size_bytes(oldest_data)
                    
                    # Перемещаем в RAM
                    if len(self.ram_cache) >= self.ram_cache.max_size:
                        self._evict_one_lru()
                    self.ram_cache.put(oldest_key, oldest_data)
                    
                    with self.stats_lock:
                        self.cache_stats['gpu_to_ram_transfers'] += 1
                    
                    if token_id in self.token_metadata:
                        self.token_metadata[token_id]["on_gpu"] = False
            
            # Добавляем токен на GPU
            if isinstance(token_data, dict) and 'tensor' in token_data:
                # Переносим тензор на GPU
                token_data['tensor'] = token_data['tensor'].to(self.device)
            
            self.gpu_cache[token_id] = token_data
            self.gpu_cache_size += self._estimate_size_bytes(token_data)
            
            if token_id in self.token_metadata:
                self.token_metadata[token_id]["on_gpu"] = True
                self.token_metadata[token_id]["in_memory"] = True
            
            with self.stats_lock:
                self.cache_stats['ram_to_gpu_transfers'] += 1
            
            logger.debug(f"Токен {token_id} перемещен на GPU")
            
        except Exception as e:
            logger.error(f"Ошибка перемещения на GPU: {e}")
    
    def _move_to_memory(self, token_id: str, token_data: Any = None) -> None:
        """Перемещает токен из диска в память (RAM или GPU)."""
        if token_data is None:
            token_data = self._load_token_from_disk(token_id)
        
        if not token_data:
            return
        
        # Попытка размещения на GPU
        if self.gpu_available and self.gpu_cache is not None:
            if self._can_fit_in_gpu(token_id, token_data):
                self._move_to_gpu(token_id, token_data)
                return
        
        # Размещение в RAM
        size_bytes = self._estimate_size_bytes(token_data)
        if len(self.ram_cache) < self.ram_cache.max_size:
            if len(self.ram_cache) >= self.ram_cache.max_size:
                self._evict_one_lru()
            self.ram_cache.put(token_id, token_data)
            
            if token_id in self.token_metadata:
                self.token_metadata[token_id]["in_memory"] = True
                self.token_metadata[token_id]["mem_size"] = size_bytes
    
    def _move_to_disk(self, token_id: str) -> None:
        """Перемещает токен из памяти на диск."""
        # Проверяем GPU кэш
        if self.gpu_available and self.gpu_cache is not None:
            if token_id in self.gpu_cache:
                token_data = self.gpu_cache.pop(token_id)
                self.gpu_cache_size -= self._estimate_size_bytes(token_data)
                self._save_token_to_disk(token_id, token_data)
                
                if token_id in self.token_metadata:
                    self.token_metadata[token_id]["on_gpu"] = False
                    self.token_metadata[token_id]["in_memory"] = False
                return
        
        # Проверяем RAM кэш
        token_data = self.ram_cache.get(token_id)
        if token_data:
            self.ram_cache.remove(token_id)
            self._save_token_to_disk(token_id, token_data)
            
            if token_id in self.token_metadata:
                self.token_metadata[token_id]["in_memory"] = False
                self.token_metadata[token_id]["mem_size"] = 0
    
    def _load_token_from_disk(self, token_id: str) -> Optional[Dict]:
        """Загружает токен с диска."""
        try:
            return self.disk_cache.get(token_id)
        except Exception as e:
            logger.error(f"Ошибка загрузки токена {token_id} с диска: {e}")
            return None
    
    def _save_token_to_disk(self, token_id: str, token_data: Dict) -> None:
        """Сохраняет токен на диск."""
        try:
            # Конвертируем тензоры в CPU перед сохранением
            if isinstance(token_data, dict):
                for key, value in token_data.items():
                    if isinstance(value, torch.Tensor):
                        token_data[key] = value.cpu().detach()
            
            self.disk_cache[token_id] = token_data
        except Exception as e:
            logger.error(f"Ошибка сохранения токена {token_id} на диск: {e}")
    
    def _update_token_access(self, token_id: str) -> None:
        """Обновляет статистику доступа к токену."""
        with self.metadata_lock:
            if token_id in self.token_metadata:
                self.token_metadata[token_id]["access_count"] += 1
                self.token_metadata[token_id]["last_access"] = time.time()
    
    def _update_avg_access_time(self, access_time: float) -> None:
        """Обновляет среднее время доступа."""
        current_avg = self.usage_stats.get("avg_access_time", 0.0)
        total_accesses = self.usage_stats.get("total_accesses", 1)
        alpha = 0.1
        self.usage_stats["avg_access_time"] = current_avg * (1 - alpha) + access_time * alpha
    
    def _estimate_size_bytes(self, value: Any) -> int:
        """Оценивает размер объекта в байтах."""
        try:
            if isinstance(value, torch.Tensor):
                return value.element_size() * value.nelement()
            if isinstance(value, (dict, list)):
                return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
            if isinstance(value, str):
                return len(value.encode("utf-8"))
            return len(str(value).encode("utf-8"))
        except Exception:
            return 0
    
    def _evict_one_lru(self) -> None:
        """Вытесняет один наименее недавно использованный элемент."""
        try:
            if len(self.ram_cache) == 0:
                return
            first_key = next(iter(self.ram_cache.cache))
            self._move_to_disk(first_key)
        except Exception:
            pass
    
    def _load_metadata(self) -> None:
        """Загружает метаданные о токенах с диска."""
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
        """Сохраняет метаданные токенов."""
        try:
            metadata_path = os.path.join(self.disk_cache_dir, "token_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.token_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения метаданных: {e}")
    
    def _start_memory_pressure_worker(self) -> None:
        """Запускает монитор давления памяти."""
        if psutil is None:
            logger.info("psutil недоступен: мониторинг давления памяти отключён")
            return
        
        stop_event = getattr(self.brain, 'stop_event', None)
        if stop_event is None:
            self._local_stop_event = threading.Event()
            stop_event = self._local_stop_event
        
        def worker():
            while not stop_event.is_set():
                try:
                    vm = psutil.virtual_memory()
                    free_ratio = float(vm.available) / float(max(1, vm.total))
                    if free_ratio < 0.1:  # 10% свободной памяти
                        self._offload_under_pressure()
                    time.sleep(2.0)
                except Exception as e:
                    logger.error(f"Ошибка монитора памяти: {e}")
                    time.sleep(2.0)
        
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self._memory_pressure_thread = t
    
    def _offload_under_pressure(self) -> None:
        """Выгружает токены при низком уровне свободной памяти."""
        try:
            if len(self.ram_cache) == 0:
                return
            
            # Вытесняем 20% наименее используемых токенов
            to_offload = max(1, len(self.ram_cache) // 5)
            offloaded = 0
            
            for _ in range(to_offload):
                if len(self.ram_cache) == 0:
                    break
                first_key = next(iter(self.ram_cache.cache))
                self._move_to_disk(first_key)
                offloaded += 1
            
            if offloaded > 0:
                logger.warning(f"Memory pressure: выгружено {offloaded} токенов на диск")
        except Exception as e:
            logger.error(f"Ошибка offload при давлении памяти: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша."""
        with self.stats_lock:
            stats = {
                "gpu_tokens": len(self.gpu_cache) if self.gpu_cache else 0,
                "ram_tokens": len(self.ram_cache),
                "disk_tokens": len(self.disk_cache),
                "gpu_hits": self.cache_stats.get('gpu_hits', 0),
                "ram_hits": self.cache_stats.get('ram_hits', 0),
                "disk_hits": self.cache_stats.get('disk_hits', 0),
                "misses": self.cache_stats.get('misses', 0),
                "evictions": self.cache_stats.get('evictions', 0),
                "total_requests": self.cache_stats.get('total_requests', 0),
                "gpu_to_ram_transfers": self.cache_stats.get('gpu_to_ram_transfers', 0),
                "ram_to_gpu_transfers": self.cache_stats.get('ram_to_gpu_transfers', 0),
                "memory_usage_mb": self.usage_stats.get("memory_usage_mb", 0.0),
                "cache_efficiency": self.usage_stats.get("cache_efficiency", 0.0),
                "avg_access_time": self.usage_stats.get("avg_access_time", 0.0),
            }
            
            # Расчет hit rate
            total = stats['total_requests']
            if total > 0:
                stats['hit_rate'] = (
                    stats['gpu_hits'] + stats['ram_hits'] + stats['disk_hits']
                ) / total
            else:
                stats['hit_rate'] = 0.0
            
            return stats
    
    def clear(self) -> None:
        """Очищает все кэши."""
        with self.stats_lock:
            # Очищаем GPU кэш
            if self.gpu_available and self.gpu_cache is not None:
                self.gpu_cache.clear()
                self.gpu_cache_size = 0
                
                # Очищаем VRAM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Очищаем RAM кэш
            self.ram_cache.clear()
            
            # Очищаем дисковый кэш
            self.disk_cache.clear()
            
            # Сбрасываем статистику
            self.cache_stats = {
                'gpu_hits': 0,
                'ram_hits': 0,
                'disk_hits': 0,
                'misses': 0,
                'evictions': 0,
                'total_requests': 0,
                'gpu_to_ram_transfers': 0,
                'ram_to_gpu_transfers': 0
            }
            
            # Очищаем метаданные
            with self.metadata_lock:
                self.token_metadata = {}
            
            logger.info("Все кэши очищены")
    
    def cleanup(self) -> None:
        """Очищает ресурсы гибридного кэша."""
        try:
            # Остановка монитора памяти
            try:
                if hasattr(self, "_local_stop_event"):
                    self._local_stop_event.set()
            except Exception:
                pass
            
            try:
                stop_event = getattr(self.brain, 'stop_event', None)
                if isinstance(stop_event, threading.Event):
                    stop_event.set()
            except Exception:
                pass
            
            # Ожидание завершения потоков
            try:
                t = getattr(self, "_memory_pressure_thread", None)
                if t and isinstance(t, threading.Thread) and t.is_alive():
                    t.join(timeout=2.0)
            except Exception:
                pass
            
            # Сохранение метаданных
            self._save_metadata()
            
            # Очищаем GPU память
            if self.gpu_available and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Ошибка при очистке HybridTokenCache: {e}")
    
    # ===== KV-интерфейс для совместимости =====
    def exists(self, key: str) -> bool:
        return (
            (self.gpu_cache is not None and key in self.gpu_cache) or
            key in self.ram_cache or
            key in self.disk_cache
        )
    
    def get(self, key: str, default=None) -> Any:
        return self.get_token(key) or default
    
    def set(self, key: str, value: Any) -> None:
        self.add_token(key, value)
    
    def __len__(self) -> int:
        gpu_len = len(self.gpu_cache) if self.gpu_cache else 0
        return gpu_len + len(self.ram_cache) + len(self.disk_cache)
    
    def __contains__(self, key: str) -> bool:
        return self.exists(key)