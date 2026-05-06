"""
FCP Storage - Управление сессиями и историей
Часть FCPipeline - вынесена для модульности
"""
import os
import json
import logging
import time
import numpy as np
from typing import Optional, Dict, List, Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eva_ai.core.fcp_pipeline import FCPipeline

logger = logging.getLogger("eva_ai.fcp_storage")


def _get_session_dir(fcp: 'FCPipeline') -> str:
    """Получить директорию для сессий"""
    base_dir = getattr(fcp, 'session_dir', 'eva_ai/fcp_sessions')
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def save_session(fcp: 'FCPipeline', session_id: str = "default") -> bool:
    """Сохранить текущую сессию"""
    try:
        session_dir = _get_session_dir(fcp)
        session_file = os.path.join(session_dir, f"{session_id}.json")
        
        session_data = {
            "session_id": session_id,
            "timestamp": time.time(),
            "conversation_history": fcp.conversation_history.copy(),
            "stats": fcp.stats.copy(),
            "generation_config": fcp.generation_config.copy(),
            "kv_cache_config": getattr(fcp, 'kv_cache_config', {}).copy()
        }
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[FCP] Session saved: {session_id}")
        return True
        
    except Exception as e:
        logger.error(f"[FCP] Save session failed: {e}")
        return False


def load_session(fcp: 'FCPipeline', session_id: str = "default") -> bool:
    """Загрузить сессию"""
    try:
        session_dir = _get_session_dir(fcp)
        session_file = os.path.join(session_dir, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            logger.warning(f"[FCP] Session not found: {session_id}")
            return False
        
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        fcp.conversation_history = session_data.get("conversation_history", [])
        fcp.stats.update(session_data.get("stats", {}))
        
        if "generation_config" in session_data:
            fcp.generation_config.update(session_data["generation_config"])
        
        if "kv_cache_config" in session_data and hasattr(fcp, 'kv_cache_config'):
            fcp.kv_cache_config.update(session_data["kv_cache_config"])
        
        logger.info(f"[FCP] Session loaded: {session_id}, {len(fcp.conversation_history)} turns")
        return True
        
    except Exception as e:
        logger.error(f"[FCP] Load session failed: {e}")
        return False


def clear_session(fcp: 'FCPipeline', session_id: str = "default") -> bool:
    """Очистить сессию"""
    try:
        session_dir = _get_session_dir(fcp)
        session_file = os.path.join(session_dir, f"{session_id}.json")
        
        if os.path.exists(session_file):
            os.remove(session_file)
        
        fcp.conversation_history = []
        fcp.stats = {
            "queries": 0,
            "tokens_generated": 0,
            "injections": 0,
            "cache_hits": 0,
            "graph_retrievals": 0
        }
        
        logger.info(f"[FCP] Session cleared: {session_id}")
        return True
        
    except Exception as e:
        logger.error(f"[FCP] Clear session failed: {e}")
        return False


def get_relevant_context(fcp: 'FCPipeline', query: str, max_history: int = 5) -> str:
    """Получить релевантный контекст из истории"""
    if not fcp.conversation_history:
        return ""
    
    try:
        if hasattr(fcp, '_query_embedder') and fcp._query_embedder:
            query_emb = fcp._query_embedder.encode(query)
            
            contexts = []
            for turn in fcp.conversation_history[-max_history:]:
                if 'embedding' in turn:
                    similarity = np.dot(query_emb, turn['embedding'])
                    contexts.append((similarity, turn.get('assistant', '')))
            
            contexts.sort(key=lambda x: x[0], reverse=True)
            
            if contexts:
                return contexts[0][1]
        
        return fcp.conversation_history[-1].get('assistant', '')
        
    except Exception as e:
        logger.debug(f"[FCP] Context retrieval failed: {e}")
        return fcp.conversation_history[-1].get('assistant', '') if fcp.conversation_history else ""


def get_similar_scenarios(fcp: 'FCPipeline', query: str, max_scenarios: int = 3) -> str:
    """Найти похожие сценарии из истории"""
    if not fcp.conversation_history:
        return ""
    
    try:
        scenarios = []
        
        for i, turn in enumerate(fcp.conversation_history):
            user_text = turn.get('user', '')
            
            if any(word in user_text.lower() for word in query.lower().split()[:3]):
                scenarios.append({
                    "index": i,
                    "user": user_text[:100],
                    "assistant": turn.get('assistant', '')[:200]
                })
        
        if not scenarios and len(fcp.conversation_history) >= max_scenarios:
            scenarios = fcp.conversation_history[-max_scenarios:]
        
        if scenarios:
            result = "Похожие ситуации из истории:\n\n"
            for s in scenarios[:max_scenarios]:
                if isinstance(s, dict):
                    result += f"Вопрос: {s.get('user', '')}\nОтвет: {s.get('assistant', '')}\n\n"
                else:
                    result += f"Вопрос: {s.get('user', '')}\nОтвет: {s.get('assistant', '')}\n\n"
            return result
        
        return ""
        
    except Exception as e:
        logger.debug(f"[FCP] Similar scenarios failed: {e}")
        return ""


def add_dialog_turn(fcp: 'FCPipeline', role: str, text: str, embedding: np.ndarray = None):
    """Добавить диалоговыйturn в историю"""
    turn = {
        "role": role,
        "text": text,
        "timestamp": time.time()
    }
    
    if embedding is not None:
        turn['embedding'] = embedding.tolist()
    
    if not hasattr(fcp, '_dialog_history'):
        fcp._dialog_history = []
    
    fcp._dialog_history.append(turn)
    
    max_history = getattr(fcp, 'max_history', 50)
    if len(fcp._dialog_history) > max_history:
        fcp._dialog_history = fcp._dialog_history[-max_history:]


def add_knowledge_node(fcp: 'FCPipeline', text: str, embedding: np.ndarray, metadata: dict = None) -> int:
    """Добавить узел знаний в граф"""
    if not hasattr(fcp, 'knowledge_graph') or not fcp.knowledge_graph:
        return -1
    
    try:
        node_id = fcp.knowledge_graph.add_node(
            text=text,
            embedding=embedding,
            metadata=metadata or {}
        )
        logger.debug(f"[FCP] Knowledge node added: {node_id}")
        return node_id
    except Exception as e:
        logger.error(f"[FCP] Add knowledge node failed: {e}")
        return -1


def retrieve_relevant_knowledge(fcp: 'FCPipeline', query_embedding: np.ndarray, top_k: int = 5) -> dict:
    """Извлечь релевантные знания"""
    if not hasattr(fcp, 'knowledge_graph') or not fcp.knowledge_graph:
        return {"nodes": [], "scores": []}
    
    try:
        results = fcp.knowledge_graph.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        return results
    except Exception as e:
        logger.error(f"[FCP] Knowledge retrieval failed: {e}")
        return {"nodes": [], "scores": []}