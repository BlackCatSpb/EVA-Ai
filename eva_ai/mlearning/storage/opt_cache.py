"""Caching and optimization strategies for OptimizedFractalModelManager"""
from __future__ import annotations

import logging
import torch

logger = logging.getLogger("eva_ai.fractal_model_manager_optimized")


def optimize_memory(self):
    """Оптимизирует использование памяти - улучшенная версия"""
    
    if not self.memory_optimization:
        return
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Очищен кэш GPU CUDA")
        
        import gc
        gc.collect()
        
        if len(self.tokenization_cache) > 1000:
            items = list(self.tokenization_cache.items())
            self.tokenization_cache = dict(items[-500:])
            saved_memory = len(items) - 500
            self.performance_stats["memory_saved_mb"] += saved_memory * 0.001
            logger.info(f"Оптимизирован кэш токенизации: освобождено {saved_memory} записей")
        
        if len(self.tensor_pool) > self.tensor_pool_size:
            old_size = len(self.tensor_pool)
            self.tensor_pool = self.tensor_pool[-self.tensor_pool_size//2:]
            logger.info(f"Оптимизирован пул тензоров: {old_size} -> {len(self.tensor_pool)}")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU память: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
            
    except Exception as e:
        logger.error(f"Ошибка оптимизации памяти: {e}")


def clear_gpu_cache(self):
    """Явная очистка GPU кэша после генерации"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logger.debug(f"Ошибка очистки GPU кэша: {e}")
