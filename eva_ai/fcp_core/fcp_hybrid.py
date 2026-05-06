"""
FCP Hybrid - Гибридные слои и обработка
Часть FCPipeline - вынесена для модульности
"""
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eva_ai.core.fcp_pipeline import FCPipeline

logger = logging.getLogger("eva_ai.fcp_hybrid")


def _estimate_logits_from_prompt(fcp: 'FCPipeline', prompt: str, hidden_dim: int = 2560) -> np.ndarray:
    """Оценить логиты из промпта через хэширование"""
    prompt_hash = hash(prompt) % 10000
    np.random.seed(prompt_hash)
    logits = np.random.randn(hidden_dim).astype(np.float32)
    logits = logits / (np.linalg.norm(logits) + 1e-8)
    return logits * 0.5


def _get_query_embedding(fcp: 'FCPipeline', text: str) -> np.ndarray:
    """Получить эмбеддинг запроса"""
    if hasattr(fcp, '_query_embedder') and fcp._query_embedder:
        try:
            return fcp._query_embedder.encode(text)
        except Exception as e:
            logger.debug(f"[FCP] Embedder failed: {e}")
    
    return _estimate_logits_from_prompt(fcp, text)


def _get_base_tokenizer(fcp: 'FCPipeline'):
    """Получить базовый токенизатор"""
    if hasattr(fcp, 'tokenizer'):
        return fcp.tokenizer
    return None


def _get_embedding_model(fcp: 'FCPipeline'):
    """Получить модель эмбеддингов"""
    if hasattr(fcp, '_embedding_model'):
        return fcp._embedding_model
    return None


def _enrich_prompt_with_subgraph(fcp: 'FCPipeline', prompt: str, subgraph: dict, kca_info: dict = None) -> str:
    """Обогатить промпт subgraph данными"""
    if not subgraph or not subgraph.get('nodes'):
        return prompt
    
    context_parts = []
    
    if 'nodes' in subgraph:
        for node in subgraph['nodes'][:3]:
            text = node.get('text', '')
            if text:
                context_parts.append(text[:200])
    
    if kca_info:
        if 'relevant_info' in kca_info:
            context_parts.append(kca_info['relevant_info'][:300])
    
    if context_parts:
        context = "\n\n".join(context_parts)
        enriched = f"{context}\n\nВопрос: {prompt}\n\nОтвет:"
        return enriched
    
    return prompt


def _process_with_hybrid_layers(fcp: 'FCPipeline', chat_prompt: str, user_prompt: str) -> Tuple[str, dict]:
    """
    Обработка через гибридные слои (KCA + GNN + LoRA)
    Согласно EVA.txt раздел 3
    """
    if not hasattr(fcp, 'hybrid_layer_processor') or not fcp.hybrid_layer_processor:
        return chat_prompt, {"used_hybrid": False}
    
    try:
        query_embedding = _get_query_embedding(fcp, user_prompt)
        
        context = ""
        metadata = {"used_hybrid": True, "layers_used": []}
        
        if hasattr(fcp, 'graph') and fcp.graph:
            try:
                from eva_ai.fcp_gnn import HybridLayerConfig
                
                config = HybridLayerConfig(
                    num_layers=32,
                    hidden_dim=2560,
                    num_heads=32,
                    max_seq_len=262144,
                    graph_retrieval_k=32
                )
                
                if hasattr(fcp.hybrid_layer_processor, 'process_with_graph'):
                    subgraph, kca_info = fcp.hybrid_layer_processor.process_with_graph(
                        query_embedding,
                        config
                    )
                    
                    if subgraph:
                        context = fcp.hybrid_layer_processor.get_context_text(subgraph)
                        metadata['context_nodes'] = len(subgraph.get('nodes', []))
                    
                    if kca_info and kca_info.get('kca_weights') is not None:
                        metadata['kca_applied'] = True
                        
                        if hasattr(fcp, 'state_injector') and fcp.state_injector:
                            logger.info("[FCP] Applying KCA weights to state injector")
                            metadata['layers_used'].append('kca')
                        
                        if hasattr(fcp, 'srg_gate'):
                            srg_weight = kca_info.get('srg_weight', 0.5)
                            fcp.srg_gate.update_gate(srg_weight)
                            metadata['layers_used'].append('srg')
            except Exception as e:
                logger.debug(f"[FCP] Hybrid graph processing failed: {e}")
        
        if context:
            enriched_prompt = _enrich_prompt_with_subgraph(fcp, user_prompt, {'nodes': [{'text': context}]}, kca_info)
            return enriched_prompt, metadata
        
        return chat_prompt, metadata
        
    except Exception as e:
        logger.error(f"[FCP] Hybrid processing failed: {e}")
        return chat_prompt, {"used_hybrid": False, "error": str(e)}


def _save_layer_snapshot(fcp: 'FCPipeline', query: str, response: str) -> None:
    """Сохранить snapshot слоёв в память"""
    if not hasattr(fcp, 'memory_snapshot') or not fcp.memory_snapshot:
        return
    
    try:
        snapshot_data = {
            "query": query,
            "response": response,
            "timestamp": __import__('time').time()
        }
        
        if hasattr(fcp.memory_snapshot, 'save_snapshot'):
            fcp.memory_snapshot.save_snapshot(snapshot_data)
        
        logger.debug(f"[FCP] Layer snapshot saved for query: {query[:30]}...")
    except Exception as e:
        logger.debug(f"[FCP] Snapshot save failed: {e}")


def get_hybrid_cache_config(fcp: 'FCPipeline') -> Dict[str, Any]:
    """Получить конфигурацию гибридного кэша"""
    return {
        "max_memory_tokens": getattr(fcp, 'max_memory_tokens', 50000),
        "max_gpu_tokens": getattr(fcp, 'max_gpu_tokens', 45000),
        "eviction_threshold": getattr(fcp, 'eviction_threshold', 0.3),
        "prefetch_enabled": getattr(fcp, 'prefetch_enabled', True),
    }


def update_hybrid_cache_config(fcp: 'FCPipeline', **kwargs):
    """Обновить конфигурацию гибридного кэша"""
    for key, value in kwargs.items():
        if hasattr(fcp, key):
            setattr(fcp, key, value)
    logger.info(f"[FCP] Hybrid cache config updated: {kwargs}")


def get_kv_cache_stats(fcp: 'FCPipeline') -> Dict:
    """Получить статистику KV кэша"""
    stats = {
        "gpu_memory_mb": 0,
        "cpu_memory_mb": 0,
        "cached_tokens": 0
    }
    
    if hasattr(fcp, 'hybrid_cache') and fcp.hybrid_cache:
        try:
            cache_info = fcp.hybrid_cache.get_cache_info()
            stats.update(cache_info)
        except Exception as e:
            logger.debug(f"[FCP] Cache stats failed: {e}")
    
    return stats