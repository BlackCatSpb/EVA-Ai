"""
Обновленный FractalModelManager с оптимальной конфигурацией
"""
from __future__ import annotations

import os
import json
import time
import logging
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from safetensors.torch import load_file

# Импорты для проверки качества текста
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.text_quality import check_and_fix_response, TextQualityChecker
    TEXT_QUALITY_AVAILABLE = True
except ImportError:
    TEXT_QUALITY_AVAILABLE = False

# Импорты для генерации через transformers
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger = logging.getLogger("eva.fractal_model_manager")
    logger.warning("Transformers не доступны, генерация будет ограничена")

logger = logging.getLogger("eva.fractal_model_manager_optimized")


class OptimizedFractalModelManager:
    """Оптимизированный FractalModelManager с оптимальной конфигурацией"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Инициализация с оптимальными параметрами"""
        
        self.config = self._load_optimal_config(config_path)
        
        self.model_path = self.config.get("model_path")
        self.config_path = self.config.get("config_path")
        
        self.device = torch.device(self.config.get("device", "cpu"))
        self.max_memory_tokens = self.config.get("max_memory_tokens", 50000)
        self.target_memory_gb = self.config.get("target_memory_gb", 4.0)
        
        self.cache_tokenization = self.config.get("cache_tokenization", True)
        self.parallel_tokenization = self.config.get("parallel_tokenization", True)
        self.tokenization_workers = self.config.get("tokenization_workers", 4)
        self.memory_optimization = self.config.get("memory_optimization", True)
        self.use_uint16 = self.config.get("use_uint16", True)
        self.tensor_pool_size = self.config.get("tensor_pool_size", 1000)
        
        self.batch_size = self.config.get("batch_size", 4)
        self.max_length = self.config.get("max_length", 32768)
        self.overlap_tokens = self.config.get("overlap_tokens", 64)
        
        self.model = None
        self.tokenizer = None
        self.state_dict = None
        self.initialized = False
        self.model_name = "fractal_gpt2_optimized"
        
        self.tensor_pool = []
        self.tokenization_cache = {}
        self.tokenization_executor = ThreadPoolExecutor(max_workers=self.tokenization_workers)
        
        self.performance_stats = {
            "tokenization_time": 0.0,
            "generation_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_saved_mb": 0.0
        }
        
        self._initialize_components()
        
        logger.info(f"OptimizedFractalModelManager инициализирован с {self.max_memory_tokens} токенов")
    
    def _load_optimal_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Загружает оптимальную конфигурацию"""
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        optimal_config_path = os.path.join(
            os.getcwd(), "eva", "config", "fractal_model_config.json"
        )
        
        if os.path.exists(optimal_config_path):
            with open(optimal_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {
            "model_path": None,
            "config_path": None,
            "device": "cpu",
            "max_memory_tokens": 50000,
            "target_memory_gb": 4.0,
            "cache_tokenization": True,
            "parallel_tokenization": True,
            "tokenization_workers": 4,
            "memory_optimization": True,
            "use_uint16": True,
            "tensor_pool_size": 1000,
            "batch_size": 4,
            "max_length": 32768,
            "overlap_tokens": 64,
            "auto_improvement": True,
            "quality_threshold": 0.7,
            "check_interval_seconds": 30
        }
    
    def __del__(self):
        """Очистка при удалении"""
        
        try:
            if hasattr(self, 'tokenization_executor'):
                if not self.tokenization_executor._shutdown:
                    self.tokenization_executor.shutdown(wait=False)
        except Exception:
            pass
        
        try:
            if hasattr(self, 'background_executor'):
                if not self.background_executor._shutdown:
                    self.background_executor.shutdown(wait=False)
        except Exception:
            pass
        
        logger.info("OptimizedFractalModelManager очищен")


from .opt_models import (
    _initialize_components, _initialize_model, _create_optimized_model,
    _load_optimized_tokenizer, _load_fallback_tokenizer, _find_tokenizer_in_fractal_storage,
    optimized_tokenize, generate_response, generate_text, generate_response_optimized,
    _clean_response, _get_fallback_response, get_performance_stats,
    get_quality_metrics, improve_quality, generate_response_with_web_search,
    generate_training_texts_from_web, get_web_search_stats, configure_web_search,
    clear_web_search_cache,
)
from .opt_cache import optimize_memory, clear_gpu_cache

OptimizedFractalModelManager._initialize_components = _initialize_components
OptimizedFractalModelManager._initialize_model = _initialize_model
OptimizedFractalModelManager._create_optimized_model = _create_optimized_model
OptimizedFractalModelManager._load_optimized_tokenizer = _load_optimized_tokenizer
OptimizedFractalModelManager._load_fallback_tokenizer = _load_fallback_tokenizer
OptimizedFractalModelManager._find_tokenizer_in_fractal_storage = _find_tokenizer_in_fractal_storage
OptimizedFractalModelManager.optimized_tokenize = optimized_tokenize
OptimizedFractalModelManager.generate_response = generate_response
OptimizedFractalModelManager.generate_text = generate_text
OptimizedFractalModelManager.generate_response_optimized = generate_response_optimized
OptimizedFractalModelManager._clean_response = _clean_response
OptimizedFractalModelManager._get_fallback_response = _get_fallback_response
OptimizedFractalModelManager.get_performance_stats = get_performance_stats
OptimizedFractalModelManager.get_quality_metrics = get_quality_metrics
OptimizedFractalModelManager.improve_quality = improve_quality
OptimizedFractalModelManager.generate_response_with_web_search = generate_response_with_web_search
OptimizedFractalModelManager.generate_training_texts_from_web = generate_training_texts_from_web
OptimizedFractalModelManager.get_web_search_stats = get_web_search_stats
OptimizedFractalModelManager.configure_web_search = configure_web_search
OptimizedFractalModelManager.clear_web_search_cache = clear_web_search_cache
OptimizedFractalModelManager.optimize_memory = optimize_memory
OptimizedFractalModelManager.clear_gpu_cache = clear_gpu_cache
