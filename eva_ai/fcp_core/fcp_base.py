"""
FCP Base - Базовые компоненты и конфигурация
Часть FCPipeline - вынесена для модульности
"""
import logging
from typing import Dict, Any, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eva_ai.core.fcp_pipeline import FCPipeline

logger = logging.getLogger("eva_ai.fcp_base")


def get_generation_config(fcp: 'FCPipeline', max_new_tokens: int = None) -> Any:
    """Получить конфигурацию генерации"""
    try:
        import openvino_genai as ov_genai
        
        config = ov_genai.GenerationConfig()
        
        gen_config = fcp.generation_config.copy()
        
        if max_new_tokens:
            gen_config['max_new_tokens'] = max_new_tokens
        
        for key, value in gen_config.items():
            setattr(config, key, value)
        
        return config
    except ImportError:
        return fcp.generation_config.copy()
    except Exception as e:
        logger.debug(f"[FCP] Get generation config failed: {e}")
        return fcp.generation_config.copy()


def update_generation_config(fcp: 'FCPipeline', **kwargs):
    """Обновить конфигурацию генерации"""
    for key, value in kwargs.items():
        if key in fcp.generation_config:
            fcp.generation_config[key] = value
    
    logger.info(f"[FCP] Generation config updated: {list(kwargs.keys())}")


def get_generation_config_summary(fcp: 'FCPipeline') -> Dict:
    """Получить сводку конфигурации генерации"""
    return {
        "max_new_tokens": fcp.generation_config.get('max_new_tokens', 8192),
        "temperature": fcp.generation_config.get('temperature', 0.15),
        "top_p": fcp.generation_config.get('top_p', 0.85),
        "top_k": fcp.generation_config.get('top_k', 40),
        "repetition_penalty": fcp.generation_config.get('repetition_penalty', 1.1)
    }


def get_kv_cache_config(fcp: 'FCPipeline') -> Dict:
    """Получить конфигурацию KV кэша"""
    if not hasattr(fcp, 'kv_cache_config'):
        fcp.kv_cache_config = {
            "max_memory_tokens": 50000,
            "max_gpu_tokens": 45000,
            "eviction_threshold": 0.3,
            "prefetch_enabled": True
        }
    return fcp.kv_cache_config.copy()


def update_kv_cache_config(fcp: 'FCPipeline', **kwargs):
    """Обновить конфигурацию KV кэша"""
    if not hasattr(fcp, 'kv_cache_config'):
        fcp.kv_cache_config = {
            "max_memory_tokens": 50000,
            "max_gpu_tokens": 45000,
            "eviction_threshold": 0.3,
            "prefetch_enabled": True
        }
    
    for key, value in kwargs.items():
        if key in fcp.kv_cache_config:
            fcp.kv_cache_config[key] = value
    
    logger.info(f"[FCP] KV cache config updated: {list(kwargs.keys())}")


def enrich_with_kca(fcp: 'FCPipeline', prompt: str, context: str = "", 
                    use_hybrid: bool = True) -> str:
    """Обогатить промпт с помощью KCA"""
    if not context:
        return prompt
    
    enriched = f"Контекст:\n{context}\n\nВопрос: {prompt}\n\nОтвет:"
    
    logger.debug(f"[FCP] Enriched prompt with KCA, context length: {len(context)}")
    return enriched


def get_statistics(fcp: 'FCPipeline') -> Dict:
    """Получить статистику FCP"""
    return fcp.stats.copy()


def get_fcp_status(fcp: 'FCPipeline') -> Dict:
    """Получить статус FCP компонентов"""
    status = {
        "pipeline": fcp.pipeline is not None,
        "tokenizer": fcp.tokenizer is not None,
        "graph": fcp.graph is not None,
        "state_injector": fcp.state_injector is not None,
        "fcp_api": fcp.fcp_api is not None and fcp.fcp_api.is_initialized(),
        "online_trainer": fcp.online_trainer is not None
    }
    
    if fcp.online_trainer:
        status["trainer_status"] = fcp.online_trainer.get_status()
    
    return status


def link_brain(fcp: 'FCPipeline', brain):
    """Связать FCP с CoreBrain"""
    fcp.brain = brain
    
    if hasattr(brain, 'hybrid_cache'):
        fcp.hybrid_cache = brain.hybrid_cache
    
    if hasattr(brain, 'fractal_graph'):
        fcp.graph = brain.fractal_graph
        if hasattr(fcp, 'graph_mgr') and fcp.graph_mgr:
            fcp.graph_mgr._fractal_graph = brain.fractal_graph
    
    logger.info("[FCP] Linked to brain")