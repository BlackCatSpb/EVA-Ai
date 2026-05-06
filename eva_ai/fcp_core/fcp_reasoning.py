"""
FCP Reasoning - Рассуждения и UES (Unified Episodic System)
Часть FCPipeline - вынесена для модульности
"""
import logging
import time
import numpy as np
from typing import Optional, Dict, List, Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eva_ai.core.fcp_pipeline import FCPipeline

logger = logging.getLogger("eva_ai.fcp_reasoning")


def start_concept_mining(fcp: 'FCPipeline'):
    """Запустить майнинг концептов"""
    if not hasattr(fcp, 'concept_miner') or not fcp.concept_miner:
        logger.warning("[FCP] Concept miner not available")
        return
    
    try:
        fcp.concept_miner.start()
        logger.info("[FCP] Concept mining started")
    except Exception as e:
        logger.error(f"[FCP] Start concept mining failed: {e}")


def stop_concept_mining(fcp: 'FCPipeline'):
    """Остановить майнинг концептов"""
    if not hasattr(fcp, 'concept_miner') or not fcp.concept_miner:
        return
    
    try:
        fcp.concept_miner.stop()
        logger.info("[FCP] Concept mining stopped")
    except Exception as e:
        logger.error(f"[FCP] Stop concept mining failed: {e}")


def get_mined_concepts(fcp: 'FCPipeline') -> List[Dict]:
    """Получить добытые концепты"""
    if not hasattr(fcp, 'concept_miner') or not fcp.concept_miner:
        return []
    
    try:
        if hasattr(fcp.concept_miner, 'get_concepts'):
            return fcp.concept_miner.get_concepts()
        return []
    except Exception as e:
        logger.debug(f"[FCP] Get mined concepts failed: {e}")
        return []


def detect_contradictions(fcp: 'FCPipeline', concept: str = None) -> List[Dict]:
    """Обнаружить противоречия"""
    if not hasattr(fcp, 'contradiction_detector') or not fcp.contradiction_detector:
        return []
    
    try:
        if concept:
            return fcp.contradiction_detector.check_concept(concept)
        return fcp.contradiction_detector.get_all_contradictions()
    except Exception as e:
        logger.debug(f"[FCP] Contradiction detection failed: {e}")
        return []


def get_contradiction_stats(fcp: 'FCPipeline') -> Dict:
    """Получить статистику противоречий"""
    if not hasattr(fcp, 'contradiction_detector') or not fcp.contradiction_detector:
        return {"total": 0, "active": 0, "resolved": 0}
    
    try:
        stats = fcp.contradiction_detector.get_stats()
        return stats
    except Exception as e:
        logger.debug(f"[FCP] Contradiction stats failed: {e}")
        return {"total": 0, "active": 0, "resolved": 0}


def optimize_with_ues(fcp: 'FCPipeline', benchmark_fn=None) -> Dict:
    """Оптимизировать параметры с помощью UES"""
    if not hasattr(fcp, 'ues_optimizer') or not fcp.ues_optimizer:
        return {"status": "not_available"}
    
    try:
        if benchmark_fn is None:
            benchmark_fn = _default_benchmark
        
        result = fcp.ues_optimizer.optimize(benchmark_fn)
        logger.info(f"[FCP] UES optimization completed: {result}")
        return result
    except Exception as e:
        logger.error(f"[FCP] UES optimization failed: {e}")
        return {"status": "failed", "error": str(e)}


def _default_benchmark(fcp: 'FCPipeline', params: Dict[str, int]) -> float:
    """Бенчмарк по умолчанию для UES"""
    try:
        test_prompts = [
            "Привет",
            "Как дела?",
            "Расскажи о себе"
        ]
        
        total_time = 0
        for prompt in test_prompts:
            start = time.time()
            fcp.generate(prompt, max_new_tokens=20)
            total_time += time.time() - start
        
        avg_time = total_time / len(test_prompts)
        score = 1.0 / (avg_time + 1.0)
        
        return score
    except Exception:
        return 0.5


def get_ues_topology(fcp: 'FCPipeline') -> Dict:
    """Получить топологию UES"""
    if not hasattr(fcp, 'ues_optimizer') or not fcp.ues_optimizer:
        return {}
    
    try:
        if hasattr(fcp.ues_optimizer, 'get_topology'):
            return fcp.ues_optimizer.get_topology()
        return {}
    except Exception as e:
        logger.debug(f"[FCP] UES topology failed: {e}")
        return {}


def pin_gnn_to_e_cores(fcp: 'FCPipeline') -> Dict:
    """Закрепить GNN на E-ядрах (энергоэффективные)"""
    import platform
    
    result = {
        "status": "not_supported",
        "platform": platform.system()
    }
    
    if platform.system() == "Windows":
        try:
            import psutil
            process = psutil.Process()
            
            affinity = psutil.cpu_affinity()
            e_cores = [i for i in range(psutil.cpu_count()) if i % 2 == 1]
            
            if e_cores:
                process.cpu_affinity(e_cores[:4])
                result = {
                    "status": "success",
                    "pinned_cores": e_cores[:4],
                    "note": "GNN pinned to efficiency cores"
                }
                logger.info(f"[FCP] GNN pinned to E-cores: {e_cores[:4]}")
        except Exception as e:
            logger.debug(f"[FCP] Pin GNN failed: {e}")
    
    return result


def pin_llm_to_p_cores(fcp: 'FCPipeline') -> Dict:
    """Закрепить LLM на P-ядрах (производительные)"""
    import platform
    
    result = {
        "status": "not_supported",
        "platform": platform.system()
    }
    
    if platform.system() == "Windows":
        try:
            import psutil
            process = psutil.Process()
            
            p_cores = [i for i in range(psutil.cpu_count()) if i % 2 == 0]
            
            if p_cores:
                process.cpu_affinity(p_cores[:6])
                result = {
                    "status": "success",
                    "pinned_cores": p_cores[:6],
                    "note": "LLM pinned to performance cores"
                }
                logger.info(f"[FCP] LLM pinned to P-cores: {p_cores[:6]}")
        except Exception as e:
            logger.debug(f"[FCP] Pin LLM failed: {e}")
    
    return result


def start_reasoning_session(fcp: 'FCPipeline', session_id: str = None) -> str:
    """Начать сессию рассуждений"""
    if session_id is None:
        session_id = f"reasoning_{int(time.time())}"
    
    if not hasattr(fcp, '_reasoning_sessions'):
        fcp._reasoning_sessions = {}
    
    fcp._reasoning_sessions[session_id] = {
        "steps": [],
        "start_time": time.time(),
        "status": "active"
    }
    
    logger.info(f"[FCP] Reasoning session started: {session_id}")
    return session_id


def end_reasoning_session(fcp: 'FCPipeline', save_to_tcm: bool = True) -> Dict:
    """Завершить сессию рассуждений"""
    if not hasattr(fcp, '_reasoning_sessions') or not fcp._reasoning_sessions:
        return {"status": "no_active_session"}
    
    current_session = None
    for sid, session in fcp._reasoning_sessions.items():
        if session.get("status") == "active":
            current_session = sid
            break
    
    if not current_session:
        return {"status": "no_active_session"}
    
    session = fcp._reasoning_sessions[current_session]
    session["status"] = "completed"
    session["end_time"] = time.time()
    session["duration"] = session["end_time"] - session["start_time"]
    
    if save_to_tcm:
        if not hasattr(fcp, 'tcm_store'):
            fcp.tcm_store = {}
        fcp.tcm_store[current_session] = session.copy()
    
    result = {
        "session_id": current_session,
        "steps_count": len(session.get("steps", [])),
        "duration": session.get("duration", 0)
    }
    
    logger.info(f"[FCP] Reasoning session ended: {current_session}")
    return result


def add_reasoning_step(fcp: 'FCPipeline', step_type: str, content: str, metadata: Dict = None):
    """Добавить шаг рассуждения"""
    if not hasattr(fcp, '_reasoning_sessions'):
        fcp._reasoning_sessions = {}
    
    current_session = None
    for sid, session in fcp._reasoning_sessions.items():
        if session.get("status") == "active":
            current_session = sid
            break
    
    if not current_session:
        logger.warning("[FCP] No active reasoning session")
        return
    
    step = {
        "type": step_type,
        "content": content,
        "timestamp": time.time(),
        "metadata": metadata or {}
    }
    
    fcp._reasoning_sessions[current_session]["steps"].append(step)


def get_reasoning_context(fcp: 'FCPipeline', max_steps: int = 5) -> str:
    """Получить контекст рассуждений"""
    if not hasattr(fcp, '_reasoning_sessions'):
        return ""
    
    context_parts = []
    
    for sid, session in fcp._reasoning_sessions.items():
        if session.get("status") == "active":
            steps = session.get("steps", [])[-max_steps:]
            for step in steps:
                context_parts.append(f"[{step.get('type', 'step')}] {step.get('content', '')}")
            break
    
    return "\n".join(context_parts)


def get_reasoning_summary(fcp: 'FCPipeline') -> str:
    """Получить сводку рассуждений"""
    if not hasattr(fcp, '_reasoning_sessions') or not fcp._reasoning_sessions:
        return "Нет активных рассуждений"
    
    summaries = []
    for sid, session in fcp._reasoning_sessions.items():
        if session.get("status") == "active":
            steps_count = len(session.get("steps", []))
            duration = session.get("end_time", time.time()) - session.get("start_time", time.time())
            summaries.append(f"Сессия {sid}: {steps_count} шагов, {duration:.1f}с")
    
    return "\n".join(summaries) if summaries else "Нет активных рассуждений"


def analyze_reasoning_consistency(fcp: 'FCPipeline') -> Dict:
    """Проанализировать согласованность рассуждений"""
    if not hasattr(fcp, '_reasoning_sessions'):
        return {"score": 1.0, "issues": []}
    
    issues = []
    total_steps = 0
    
    for sid, session in fcp._reasoning_sessions.items():
        steps = session.get("steps", [])
        total_steps += len(steps)
        
        for i in range(len(steps) - 1):
            if steps[i].get("type") != steps[i+1].get("type"):
                issues.append({
                    "session": sid,
                    "step": i,
                    "issue": "type_mismatch"
                })
    
    score = 1.0 - (len(issues) / max(total_steps, 1))
    
    return {
        "score": score,
        "issues": issues,
        "total_steps": total_steps
    }


def clear_reasoning_chain(fcp: 'FCPipeline'):
    """Очистить цепочку рассуждений"""
    if hasattr(fcp, '_reasoning_sessions'):
        fcp._reasoning_sessions.clear()
    logger.info("[FCP] Reasoning chain cleared")


def get_reasoning_state(fcp: 'FCPipeline') -> Dict:
    """Получить состояние рассуждений"""
    if not hasattr(fcp, '_reasoning_sessions'):
        return {"active_sessions": 0, "total_steps": 0}
    
    active = sum(1 for s in fcp._reasoning_sessions.values() if s.get("status") == "active")
    total = sum(len(s.get("steps", [])) for s in fcp._reasoning_sessions.values())
    
    return {
        "active_sessions": active,
        "total_steps": total,
        "sessions": list(fcp._reasoning_sessions.keys())
    }


def restore_reasoning_state(fcp: 'FCPipeline', state: Dict):
    """Восстановить состояние рассуждений"""
    if 'sessions' in state:
        fcp._reasoning_sessions = state['sessions']
    logger.info("[FCP] Reasoning state restored")