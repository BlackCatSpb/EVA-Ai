"""
Гибридная система кэширования токенов для CogniFlex.
Оптимизирует обработку запросов за счет многоуровневой архитектуры:
GPU (VRAM) → RAM → SSD

Модульная архитектура с разделением ответственности между:
- MemoryCache: оперативная память (LRU)
- TokenDiskCache: дисковое хранилище
- MetadataManager: метаданные токенов
"""
import os
import json
import time
import threading
import hashlib
from typing import Dict, List, Optional, Any
from collections import OrderedDict
import logging
import torch

# Опционально используем psutil для мониторинга памяти
import psutil  # type: ignore

logger = logging.getLogger(__name__)

_cache_registry: Dict[str, 'HybridTokenCache'] = {}
_registry_lock = threading.RLock()


def get_shared_cache(brain, cache_name: str = "default") -> 'HybridTokenCache':
    """Returns a singleton HybridTokenCache for the given name."""
    with _registry_lock:
        if cache_name not in _cache_registry:
            _cache_registry[cache_name] = HybridTokenCache(brain, _cache_name=cache_name)
        return _cache_registry[cache_name]


# ============================================================================
# Disk Cache Implementation
# ============================================================================

class TokenDiskCache:
    """Дисковый кэш токенов с поддержкой больших объемов (до 50 ГБ)."""
    
    def __init__(self, cache_dir: str, max_size_gb: float = 50.0):
        self.cache_dir = cache_dir
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)
        self.index_file = os.path.join(cache_dir, "disk_cache_index.json")
        self.data_dir = os.path.join(cache_dir, "data")
        
        # Создаем директории
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Индекс файлов
        self.file_index = {}
        self.current_size_bytes = 0
        self._lock = threading.RLock()
        
        # Загружаем индекс
        self._load_index()
        
        logger.info(f"TokenDiskCache инициализирован: {cache_dir}, лимит={max_size_gb}GB")
    
    def _load_index(self):
        """Загружает индекс дискового кэша."""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.file_index = data.get('files', {})
                    self.current_size_bytes = data.get('total_size', 0)
                logger.debug(f"Загружен индекс дискового кэша: {len(self.file_index)} файлов")
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса дискового кэша: {e}")
            self.file_index = {}
            self.current_size_bytes = 0
    
    def _save_index(self):
        """Сохраняет индекс дискового кэша."""
        try:
            data = {
                'files': self.file_index,
                'total_size': self.current_size_bytes,
                'last_updated': time.time()
            }
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения индекса дискового кэша: {e}")
    
    def _get_file_path(self, token_id: str) -> str:
        """Генерирует путь к файлу токена."""
        # Используем хэш для распределения файлов по поддиректориям
        hash_prefix = token_id[:2]
        subdir = os.path.join(self.data_dir, hash_prefix)
        os.makedirs(subdir, exist_ok=True)
        return os.path.join(subdir, f"{token_id}.bin")
    
    def get(self, token_id: str) -> Optional[Dict]:
        """Получает токен с диска."""
        with self._lock:
            if token_id not in self.file_index:
                return None
            
            file_path = self._get_file_path(token_id)
            if not os.path.exists(file_path):
                # Файл удален, удаляем из индекса
                del self.file_index[token_id]
                self._save_index()
                return None
            
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # Десериализация
                import pickle
                token_data = pickle.loads(data)
                
                # Обновляем метаданные
                self.file_index[token_id]['last_access'] = time.time()
                self.file_index[token_id]['access_count'] = self.file_index[token_id].get('access_count', 0) + 1
                self._save_index()
                
                return token_data
                
            except Exception as e:
                logger.error(f"Ошибка загрузки токена {token_id} с диска: {e}")
                # Удаляем поврежденный файл
                self._remove_file(token_id)
                return None
    
    def put(self, token_id: str, token_data: Dict) -> bool:
        """Сохраняет токен на диск."""
        with self._lock:
            try:
                # Сериализация
                import pickle
                data = pickle.dumps(token_data)
                data_size = len(data)
                
                # Проверяем размер
                if data_size > 100 * 1024 * 1024:  # Максимум 100 МБ на токен
                    logger.warning(f"Токен {token_id} слишком большой: {data_size / 1024 / 1024:.2f} MB")
                    return False
                
                # Проверяем лимит размера
                while (self.current_size_bytes + data_size > self.max_size_bytes and 
                       len(self.file_index) > 0):
                    self._evict_lru()
                
                # Сохраняем файл
                file_path = self._get_file_path(token_id)
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # Обновляем индекс
                old_size = 0
                if token_id in self.file_index:
                    old_size = self.file_index[token_id].get('size', 0)
                
                self.file_index[token_id] = {
                    'size': data_size,
                    'created': time.time(),
                    'last_access': time.time(),
                    'access_count': 1
                }
                
                self.current_size_bytes += data_size - old_size
                self._save_index()
                
                return True
                
            except Exception as e:
                logger.error(f"Ошибка сохранения токена {token_id} на диск: {e}")
                return False
    
    def _evict_lru(self):
        """Удаляет наименее используемый токен."""
        if not self.file_index:
            return
        
        # Находим LRU токен
        lru_token_id = min(self.file_index.keys(), 
                          key=lambda k: self.file_index[k].get('last_access', 0))
        
        self._remove_file(lru_token_id)
    
    def _remove_file(self, token_id: str):
        """Удаляет файл токена."""
        if token_id not in self.file_index:
            return
        
        file_path = self._get_file_path(token_id)
        file_size = self.file_index[token_id].get('size', 0)
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Ошибка удаления файла {file_path}: {e}")
        
        # Обновляем индекс
        del self.file_index[token_id]
        self.current_size_bytes -= file_size
        self._save_index()
    
    def remove(self, token_id: str) -> bool:
        """Удаляет токен из дискового кэша."""
        with self._lock:
            if token_id not in self.file_index:
                return False
            
            self._remove_file(token_id)
            return True
    
    def clear(self):
        """Очищает дисковый кэш."""
        with self._lock:
            try:
                # Удаляем все файлы данных
                for subdir in os.listdir(self.data_dir):
                    subdir_path = os.path.join(self.data_dir, subdir)
                    if os.path.isdir(subdir_path):
                        for file_name in os.listdir(subdir_path):
                            file_path = os.path.join(subdir_path, file_name)
                            os.remove(file_path)
                
                # Очищаем индекс
                self.file_index.clear()
                self.current_size_bytes = 0
                self._save_index()
                
                logger.info("Дисковый кэш очищен")
                
            except Exception as e:
                logger.error(f"Ошибка очистки дискового кэша: {e}")
    
    def get_stats(self) -> Dict:
        """Возвращает статистику дискового кэша."""
        with self._lock:
            return {
                'total_files': len(self.file_index),
                'total_size_bytes': self.current_size_bytes,
                'total_size_mb': self.current_size_bytes / (1024 * 1024),
                'total_size_gb': self.current_size_bytes / (1024 * 1024 * 1024),
                'max_size_gb': self.max_size_bytes / (1024 * 1024 * 1024),
                'usage_percent': (self.current_size_bytes / self.max_size_bytes) * 100
            }
    
    def __contains__(self, token_id: str) -> bool:
        """Проверяет наличие токена в кэше."""
        with self._lock:
            return token_id in self.file_index
    
    def __len__(self) -> int:
        """Возвращает количество токенов в кэше."""
        with self._lock:
            return len(self.file_index)


# ============================================================================
# LRU Cache Implementation
# ============================================================================

class LRUCache:
    """Простая реализация LRU кэша на основе OrderedDict."""
    
    def __init__(self, max_size: int):
        self.max_size = max(1, max_size)
        self.cache = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Получает элемент из кэша."""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Добавляет элемент в кэш."""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def remove(self, key: str) -> bool:
        """Удаляет элемент из кэша."""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Очищает кэш."""
        with self._lock:
            self.cache.clear()
    
    def __contains__(self, key: str) -> bool:
        """Проверяет наличие ключа в кэше."""
        return key in self.cache
    
    def __len__(self) -> int:
        """Возвращает размер кэша."""
        return len(self.cache)
    
    def keys(self):
        """Возвращает ключи кэша."""
        return list(self.cache.keys())


# ============================================================================
# Hybrid Token Cache
# ============================================================================

class HybridTokenCache:
    """
    Гибридная система кэширования токенов для CogniFlex.
    Многоуровневая архитектура: GPU (VRAM) → RAM → SSD
    
    Особенности:
    - Динамическое управление памятью
    - Мониторинг давления памяти ОС
    - Адаптивная стратегия вытеснения
    - Потокобезопасность
    - Модульная архитектура
    """
    
    def __init__(
        self,
        brain,
        max_memory_tokens: int = 25000,  # Уменьшаем для оптимизации
        disk_cache_dir: str = "token_cache",
        target_memory_gb: float = 50.0,  # Увеличиваем до 50 ГБ для лучшей производительности
        max_disk_cache_gb: float = 50.0,  # Расширяем до 50 ГБ на SSD
        dynamic_memory_limit: bool = True,
        max_ram_usage_percent: float = 70.0,  # Снижаем до 70% для безопасности
        vram_threshold: float = 0.15,  # Более агрессивная выгрузка VRAM
        ram_threshold: float = 0.12,  # Более агрессивная выгрузка RAM
        eviction_policy: str = "hybrid",
        hot_threshold: int = 5,  # Увеличиваем порог для горячих токенов
        _cache_name: str = "default",
        **kwargs
    ):
        """
        Инициализирует гибридный кэш токенов.
        
        Args:
            brain: Ссылка на ядро CogniFlex
            max_memory_tokens: Максимальное количество токенов в памяти
            disk_cache_dir: Директория для дискового кэша
            target_memory_gb: Целевой объем памяти в GB
            dynamic_memory_limit: Включить динамическое управление памятью
            max_ram_usage_percent: Максимальный процент использования RAM
            vram_threshold: Порог использования VRAM для вытеснения
            ram_threshold: Порог использования RAM для вытеснения
            eviction_policy: Стратегия вытеснения (lru, lfu, hybrid)
            hot_threshold: Количество обращений для перемещения в горячий кэш
        """
        self.brain = brain
        self.gpu_available = torch.cuda.is_available()
        
        # Get device from brain config if available, otherwise use auto-detection
        cfg = getattr(self.brain, 'config', {}) or {}
        hc = cfg.get('hybrid_cache', {}) or {}
        model_cfg = cfg.get('model', {}) or {}
        
        # Use device from config, fallback to cuda:0 if available, else cpu
        config_device = hc.get('device') or model_cfg.get('device')
        if config_device:
            self.device = config_device
        elif self.gpu_available:
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        
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
        except Exception as e:
            logger.warning(f"Error setting up cache directory: {e}")
            base_cache_dir = os.path.join(os.getcwd(), 'ml_cache')
            os.makedirs(base_cache_dir, exist_ok=True)
        
        self.disk_cache_dir = os.path.join(base_cache_dir, disk_cache_dir)
        os.makedirs(self.disk_cache_dir, exist_ok=True)
        
        # Загрузка конфигурации из brain.config
        try:
            cfg = getattr(self.brain, 'config', {}) or {}
            hc = cfg.get('hybrid_cache') or {}
        except Exception as e:
            logger.warning(f"Error loading hybrid_cache config: {e}")
            hc = {}
        
        # Применение настроек из конфигурации
        target_memory_gb = float(hc.get('target_memory_gb', target_memory_gb))
        dynamic_memory_limit = bool(hc.get('dynamic_memory_limit', dynamic_memory_limit))
        max_ram_usage_percent = float(hc.get('max_ram_usage_percent', max_ram_usage_percent))
        hot_threshold = int(hc.get('hot_threshold', hot_threshold))
        eviction_policy = hc.get('eviction_policy', eviction_policy)
        
        # Динамический расчет лимита памяти
        self.dynamic_memory_limit = dynamic_memory_limit
        self.target_memory_bytes = int(target_memory_gb * 1024 ** 3)
        self.avg_token_size_bytes = 4096  # Стартовая оценка (4 КБ на токен)
        
        if psutil and dynamic_memory_limit:
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            self.target_memory_bytes = min(
                self.target_memory_bytes,
                int(total_ram_gb * 1024 ** 3 * (max_ram_usage_percent / 100.0))
            )
            logger.debug(f"Динамический лимит памяти: {target_memory_gb:.2f}GB ({max_ram_usage_percent}% RAM)")
        
        # Расчет максимального количества токенов
        self.max_memory_tokens = (
            max(1, int(self.target_memory_bytes / self.avg_token_size_bytes))
            if dynamic_memory_limit else max(1, max_memory_tokens)
        )
        
        # Инициализация кэшей с оптимизированными лимитами
        # VRAM: максимум 1.5GB (если GPU доступен)
        # RAM: максимум 1GB
        # SSD: 50GB (уже зарезервировано)
        
        if self.gpu_available:
            # Рассчитываем лимит VRAM: 1.5GB / средний размер токена
            vram_limit_bytes = int(1.5 * 1024 ** 3)  # 1.5GB
            vram_tokens_limit = max(1, int(vram_limit_bytes / self.avg_token_size_bytes))
            
            # RAM лимит: 1GB
            ram_limit_bytes = int(1.0 * 1024 ** 3)  # 1GB
            ram_tokens_limit = max(1, int(ram_limit_bytes / self.avg_token_size_bytes))
            
            # Применяем лимиты
            self.vram_cache = LRUCache(vram_tokens_limit)
            self.ram_cache = LRUCache(ram_tokens_limit)
            
            logger.debug(f"VRAM кэш: {vram_tokens_limit} токенов (~1.5GB)")
            logger.debug(f"RAM кэш: {ram_tokens_limit} токенов (~1GB)")
        else:
            # Без GPU весь кэш в RAM (1GB)
            ram_limit_bytes = int(1.0 * 1024 ** 3)  # 1GB
            ram_tokens_limit = max(1, int(ram_limit_bytes / self.avg_token_size_bytes))
            self.ram_cache = LRUCache(ram_tokens_limit)
            self.vram_cache = LRUCache(1)  # Пустой кэш
            logger.debug(f"RAM кэш: {ram_tokens_limit} токенов (~1GB)")
        
        self.memory_cache = self.ram_cache  # Для обратной совместимости
        
        # Дисковый кэш с поддержкой 50 ГБ
        self.disk_cache = TokenDiskCache(self.disk_cache_dir, max_disk_cache_gb)
        
        # Метаданные токенов
        self.token_metadata = {}
        self.metadata_lock = threading.RLock()
        self.stats_lock = threading.RLock()
        self._lock = threading.RLock()  # Добавляем _lock для thread-safe операций
        
        # Пороги и настройки
        self.vram_threshold = vram_threshold
        self.ram_threshold = ram_threshold
        self.eviction_policy = eviction_policy
        self.hot_threshold = hot_threshold
        
        # Статистика
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
        
        # Настройки кэширования
        self.cache_settings = {
            "max_memory_size": self.max_memory_tokens,
            "disk_cache_threshold": max(1, self.max_memory_tokens // 2),
            "eviction_policy": eviction_policy,
            "cache_ttl": int(hc.get("cache_ttl", 86400)),
            "min_relevance_score": float(hc.get("min_relevance_score", 0.3)),
            "max_context_tokens": int(hc.get("max_context_tokens", 1000)),
        }
        
        # Мониторинг давления памяти
        self.system_free_mem_threshold = float(hc.get('system_free_mem_threshold', 0.1))
        self.memory_pressure_interval_s = float(hc.get('memory_pressure_interval_s', 2.0))
        self.pressure_offload_batch = int(hc.get('pressure_offload_batch', 32))
        
        # Загрузка метаданных
        self._load_metadata()
        
        # Запуск монитора давления памяти
        self._start_memory_pressure_worker()
        
        approx_mem_gb = (self.max_memory_tokens * self.avg_token_size_bytes) / (1024 ** 3)
        logger.info(
            f"HybridTokenCache инициализирован: {self.device}, "
            f"память~{self.max_memory_tokens} токенов (~{approx_mem_gb:.2f}GB), "
            f"диск=50.0GB"
        )
    
    # ========================================================================
    # Основные методы кэширования
    # ========================================================================
    
    def get_token(self, token_id: str) -> Optional[Dict]:
        """
        Получает токен по ID из гибридного кэша.
        
        Args:
            token_id: Уникальный идентификатор токена
            
        Returns:
            Dict с данными токена или None
        """
        start_time = time.time()
        
        with self.stats_lock:
            self.usage_stats["total_accesses"] += 1
            self.cache_stats['total_requests'] += 1
        
        # 1. Проверка VRAM кэша
        if self.gpu_available and token_id in self.vram_cache:
            token_data = self.vram_cache.get(token_id)
            if token_data:
                with self.stats_lock:
                    self.cache_stats['vram_hits'] += 1
                    self.usage_stats["memory_hits"] += 1
                self._update_token_access(token_id)
                return token_data
        
        # 2. Проверка RAM кэша
        if token_id in self.ram_cache:
            token_data = self.ram_cache.get(token_id)
            if token_data:
                with self.stats_lock:
                    self.cache_stats['ram_hits'] += 1
                    self.usage_stats["memory_hits"] += 1
                
                # Перемещение в VRAM если возможно
                if self.gpu_available and len(self.vram_cache) < self.vram_cache.max_size:
                    self.vram_cache.put(token_id, token_data)
                
                self._update_token_access(token_id)
                return token_data
        
        # 3. Проверка дискового кэша
        disk_token = self._load_token_from_disk(token_id)
        if disk_token:
            with self.stats_lock:
                self.cache_stats['disk_hits'] += 1
                self.usage_stats["disk_hits"] += 1
            
            # Асинхронное перемещение в память при частом доступе
            with self.metadata_lock:
                metadata = self.token_metadata.get(token_id, {})
                if metadata.get("access_count", 0) > self.hot_threshold:
                    threading.Thread(
                        target=self._move_token_to_memory,
                        args=(token_id, disk_token),
                        daemon=True
                    ).start()
            
            self._update_token_access(token_id)
            return disk_token
        
        # 4. Cache miss
        with self.stats_lock:
            self.cache_stats['misses'] += 1
            self.usage_stats["misses"] += 1
        
        return None
    
    def add_token(self, token_id: str, token_data: Any) -> None:
        """
        Добавляет токен в кэш.
        
        Args:
            token_id: Уникальный идентификатор токена
            token_data: Данные токена
        """
        with self.metadata_lock:
            # Обработка различных форматов token_data
            if isinstance(token_data, dict):
                priority = token_data.get("priority", 0.5)
                actual_data = token_data
            else:
                priority = 0.5
                actual_data = token_data
            
            # Инициализация метаданных
            if token_id not in self.token_metadata:
                self.token_metadata[token_id] = {
                    "access_count": 0,
                    "last_access": time.time(),
                    "priority": priority,
                    "in_memory": False,
                    "mem_size": 0,
                }
            
            # Сохранение на диск
            self._save_token_to_disk(token_id, actual_data)
            self.token_metadata[token_id]["in_memory"] = False
            
            # Дублирование в память если есть место
            if len(self.ram_cache) < self.ram_cache.max_size:
                self.ram_cache.put(token_id, actual_data)
                self.token_metadata[token_id]["in_memory"] = True
    
    def put(self, key: str, value: Any) -> None:
        """
        Добавляет элемент в кэш (метод совместимости).
        
        Args:
            key: Ключ элемента
            value: Значение элемента
        """
        self.add_token(key, value)
    
    def get(self, key: str) -> Optional[Any]:
        """Получает значение из кэша (KV-интерфейс)."""
        return self.get_token(key)
    
    def set(self, key: str, value: Any) -> None:
        """Устанавливает значение в кэш (KV-интерфейс)."""
        self.add_token(key, value)
    
    def clear(self) -> None:
        """Очищает все кэши."""
        # Используем consistent lock ordering: stats_lock first, then metadata_lock
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
        """Загружает токен с диска."""
        return self.disk_cache.get(token_id)
    
    def _save_token_to_disk(self, token_id: str, token_data: Dict) -> None:
        """Сохраняет токен на диск."""
        self.disk_cache.put(token_id, token_data)
    
    def _move_token_to_memory(self, token_id: str, token_data: Optional[Dict] = None) -> None:
        """Перемещает токен из диска в память."""
        if token_data is None:
            token_data = self._load_token_from_disk(token_id)
        if not token_data:
            return
        
        size_bytes = self._estimate_size_bytes(token_data)
        if len(self.memory_cache) >= self.max_memory_tokens:
            self._evict_one_lru()
        
        self.memory_cache.put(token_id, token_data)
        
        with self.metadata_lock:
            if token_id in self.token_metadata:
                self.token_metadata[token_id]["in_memory"] = True
                self.token_metadata[token_id]["mem_size"] = size_bytes
    
    def _move_token_to_disk(self, token_id: str) -> None:
        """Перемещает токен из памяти на диск."""
        token_data = self.memory_cache.get(token_id)
        if not token_data:
            return
        
        self.memory_cache.remove(token_id)
        self._save_token_to_disk(token_id, token_data)
        
        with self.metadata_lock:
            if token_id in self.token_metadata:
                self.token_metadata[token_id]["in_memory"] = False
                self.token_metadata[token_id]["mem_size"] = 0
    
    def _update_token_access(self, token_id: str) -> None:
        """Обновляет статистику доступа к токену."""
        with self.metadata_lock:
            if token_id in self.token_metadata:
                self.token_metadata[token_id]["access_count"] += 1
                self.token_metadata[token_id]["last_access"] = time.time()
    
    def _estimate_size_bytes(self, value: Any) -> int:
        """Оценивает размер объекта в байтах."""
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
        """Вытесняет один наименее недавно использованный элемент."""
        try:
            if len(self.memory_cache) == 0:
                return
            first_key = next(iter(self.memory_cache.cache))
            self._move_token_to_disk(first_key)
        except Exception as e:
            logger.warning(f"Error in _evict_one_lru: {e}")
    
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
    
    # ========================================================================
    # Мониторинг памяти
    # ========================================================================
    
    def _start_memory_pressure_worker(self) -> None:
        """Запускает монитор давления памяти ОС."""
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
                    if free_ratio < self.system_free_mem_threshold:
                        self._offload_under_pressure()
                    time.sleep(self.memory_pressure_interval_s)
                except Exception as e:
                    logger.error(f"Ошибка монитора памяти: {e}")
                    time.sleep(self.memory_pressure_interval_s)
        
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self._memory_pressure_thread = t
    
    def _offload_under_pressure(self) -> None:
        """Агрессивно выгружает токены при низком уровне свободной памяти."""
        try:
            if len(self.memory_cache) == 0:
                return
            
            candidates = []
            for key in list(self.memory_cache.keys()):
                meta = self.token_metadata.get(key, {})
                pr = self._calculate_token_priority(key, meta)
                candidates.append((key, pr))
            
            candidates.sort(key=lambda x: x[1])
            to_offload = min(self.pressure_offload_batch, len(candidates))
            
            offloaded = 0
            for i in range(to_offload):
                key = candidates[i][0]
                if key in self.memory_cache:
                    self._move_token_to_disk(key)
                    offloaded += 1
            
            if offloaded > 0:
                logger.warning(
                    f"Memory pressure: выгружено {offloaded} токенов "
                    f"(batch={self.pressure_offload_batch})"
                )
        except Exception as e:
            logger.error(f"Ошибка offload при давлении памяти: {e}")
    
    def _calculate_token_priority(self, token_id: str, metadata: Dict) -> float:
        """Рассчитывает приоритет токена для вытеснения."""
        access_count = metadata.get("access_count", 0)
        last_access = metadata.get("last_access", 0)
        priority = metadata.get("priority", 0.5)
        
        time_factor = 1.0 if time.time() - last_access < 3600 else 0.3
        return (access_count * 0.6) + (priority * 0.3) + (time_factor * 0.1)
    
    # ========================================================================
    # Статистика
    # ========================================================================
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша."""
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
            
            # Расчет эффективности
            total = stats['total_requests']
            if total > 0:
                stats['hit_rate'] = (stats['vram_hits'] + stats['ram_hits'] + stats['disk_hits']) / total
            else:
                stats['hit_rate'] = 0.0
            
            return stats
    
    def cleanup(self) -> None:
        """Очищает ресурсы кэша."""
        try:
            # Остановка монитора памяти
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
            
            # Ожидание завершения потоков
            try:
                t = getattr(self, "_memory_pressure_thread", None)
                if t and isinstance(t, threading.Thread) and t.is_alive():
                    t.join(timeout=2.0)
            except Exception as e:
                logger.warning(f"Error joining memory pressure thread: {e}")
            
            # Сохранение метаданных
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Ошибка при очистке HybridTokenCache: {e}")
    
    def __len__(self) -> int:
        """Возвращает количество токенов в памяти."""
        with self._lock:
            return len(self.memory_cache)
    
    def __bool__(self) -> bool:
        """Возвращает True если кэш работает (даже если пустой)."""
        return True
    
    def __contains__(self, token_hash: str) -> bool:
        """Проверяет наличие токена в кэше."""
        return token_hash in self.memory_cache


# Экспорт для совместимости
__all__ = ['HybridTokenCache', 'LRUCache', 'get_shared_cache']