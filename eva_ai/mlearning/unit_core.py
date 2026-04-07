#!/usr/bin/env python3
"""
Исправленная версия MLUnit с правильными отступами
"""

import os
import sys
import json
import logging
import time
import threading
import queue
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

# Добавляем корневую директорию проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("eva_ai.ml_unit")


def _load_brain_config() -> Dict[str, Any]:
    """Loads brain configuration from brain_config.json."""
    config_path = os.path.join(_get_project_root(), 'brain_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load brain_config.json: {e}")
    return {}


def _get_hybrid_cache_config() -> Dict[str, Any]:
    """Returns hybrid_cache config from brain_config.json."""
    config = _load_brain_config()
    return config.get('hybrid_cache', {})


def _get_project_root() -> str:
    """Возвращает корневую директорию проекта"""
    import sys
    
    possible_roots = []
    
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    possible_roots.append(os.path.dirname(os.path.dirname(current_dir)))
    possible_roots.append(os.path.dirname(current_dir))
    
    if sys.argv and sys.argv[0]:
        argv_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        possible_roots.append(argv_dir)
        possible_roots.append(os.path.dirname(argv_dir))
    
    for root in possible_roots:
        if root and os.path.exists(root):
            eva_marker = os.path.join(root, 'eva')
            if os.path.exists(eva_marker):
                return root
    
    drive = os.path.splitdrive(os.getcwd())[0] or 'C:'
    username = os.environ.get('USERNAME', 'user')
    
    onedrive_path = os.path.join(drive, 'Users', username, 'OneDrive', 'Desktop', 'ЕВА')
    if os.path.exists(onedrive_path) and os.path.exists(os.path.join(onedrive_path, 'eva')):
        return onedrive_path
    
    possible_locations = [
        os.path.join(drive, 'Users', username, 'OneDrive', 'Desktop', 'ЕВА'),
        os.path.join(drive, 'Users', username, 'Desktop', 'ЕВА'),
        os.path.join(drive, 'ЕВА'),
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            if os.path.exists(os.path.join(loc, 'eva')):
                return os.path.abspath(loc)
    
    return os.getcwd()

# Пытаемся импортировать torch
try:
    import torch
except ImportError as e:
    logger.error(f"Не удалось импортировать torch: {e}")
    torch = None

ModelManager = None

class MLUnit:
    """
    Исправленная версия MLUnit с правильными отступами
    """
    
    def __init__(self, brain=None, cache_dir="cache/ml_unit", use_gpu=True, 
                 max_workers=4, hybrid_cache_size=None, safe_test_mode=False):
        """
        Инициализирует MLUnit с улучшенной архитектурой.
        
        Args:
            brain: Экземпляр CoreBrain для интеграции
            cache_dir: Директория для кэша
            use_gpu: Использовать GPU если доступно
            max_workers: Максимальное количество воркеров
            hybrid_cache_size: Размер гибридного кэша (если None - берется из config)
            safe_test_mode: Режим безопасного тестирования
        """
        self.brain = brain
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu
        self.max_workers = max_workers
        self.safe_test_mode = safe_test_mode
        
        # Get hybrid_cache_size from config if not provided
        hc_config = _get_hybrid_cache_config()
        if hybrid_cache_size is None:
            max_hot_tokens = hc_config.get('max_hot_tokens', 8192)
            token_size_multiplier = hc_config.get('token_size_multiplier', 4096)
            available_memory_mb = hc_config.get('available_memory_mb')
            if available_memory_mb is None:
                available_memory_mb = 512
            available_memory = available_memory_mb * 1024 * 1024
            hybrid_cache_size = min(max_hot_tokens * token_size_multiplier, available_memory // 2)
        
        self.hybrid_cache_size = hybrid_cache_size
        logger.info(f"MLUnit hybrid_cache_size: {self.hybrid_cache_size} (from config: max_hot_tokens={hc_config.get('max_hot_tokens', 8192)})")
        
        # Компоненты MLUnit
        self.ml_core = None
        self.model_manager = None
        self.text_processor = None
        self.response_generator = None
        self.training_orchestrator = None
        self.token_streamer = None
        self.hybrid_cache = None
        
        # Состояние
        self.running = False
        self.models_ready = False
        self.training_mode = False
        
        # Статистика
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "last_request_time": 0.0
        }
        
        self.model = None
        self._model_initialized = False
        
        # Memory management
        self._last_cleanup_time = 0.0
        self._cleanup_interval = 60.0  # seconds
        
        # Очередь для GUI
        self.gui_queue = queue.Queue()
        
        logger.info("MLUnit инициализирован")
    
    def _maybe_cleanup_memory(self):
        """Cleanup memory periodically to prevent spikes."""
        current_time = time.time()
        if current_time - self._last_cleanup_time > self._cleanup_interval:
            self._last_cleanup_time = current_time
            try:
                if self.hybrid_cache and hasattr(self.hybrid_cache, 'cleanup'):
                    self.hybrid_cache.cleanup()
            except Exception as e:
                logger.debug(f"Memory cleanup warning: {e}")


from .unit_components import (
    _init_ml_core, _init_text_processor, _init_model_manager,
    _init_response_generator, _init_hybrid_cache, _link_components,
    _verify_basic_functionality, _init_training_orchestrator, _is_training_mode,
    get_system_health, process_query, _create_fallback_response,
    _update_statistics, initialize, start, stop, generate_response,
    get_model_statistics, get_text_processor, get_response_generator,
    get_model_manager, get_ml_core, process_text,
)

MLUnit._init_ml_core = _init_ml_core
MLUnit._init_text_processor = _init_text_processor
MLUnit._init_model_manager = _init_model_manager
MLUnit._init_response_generator = _init_response_generator
MLUnit._init_hybrid_cache = _init_hybrid_cache
MLUnit._link_components = _link_components
MLUnit._verify_basic_functionality = _verify_basic_functionality
MLUnit._init_training_orchestrator = _init_training_orchestrator
MLUnit._is_training_mode = _is_training_mode
MLUnit.get_system_health = get_system_health
MLUnit.process_query = process_query
MLUnit._create_fallback_response = _create_fallback_response
MLUnit._update_statistics = _update_statistics
MLUnit.initialize = initialize
MLUnit.start = start
MLUnit.stop = stop
MLUnit.generate_response = generate_response
MLUnit.get_model_statistics = get_model_statistics
MLUnit.get_text_processor = get_text_processor
MLUnit.get_response_generator = get_response_generator
MLUnit.get_model_manager = get_model_manager
MLUnit.get_ml_core = get_ml_core
MLUnit.process_text = process_text
MLUnit._get_project_root = staticmethod(_get_project_root)
