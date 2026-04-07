"""
Интеграция Self-Reasoning Engine с CoreBrain
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

STANDARD_ERROR_RESPONSE = {
    "response": "",
    "status": "error",
    "confidence": 0.0,
    "error_type": "unknown",
    "error_detail": "",
}


def _error_response(message: str, error_type: str = "unknown") -> Dict[str, Any]:
    """Возвращает стандартный формат ошибки."""
    resp = dict(STANDARD_ERROR_RESPONSE)
    resp["response"] = message
    resp["error_type"] = error_type
    resp["error_detail"] = message
    return resp


class ReasoningIntegration:
    """
    Класс интеграции Self-Reasoning Engine с CoreBrain
    """
    
    def __init__(self, brain):
        self.brain = brain
        self.reasoning_engine = None
        self.enabled = False
        
        # Конфигурация
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из brain_config"""
        try:
            if hasattr(self.brain, 'config'):
                reasoning_config = self.brain.config.get('reasoning', {})
                return {
                    'enabled': reasoning_config.get('enabled', True),
                    'max_iterations': reasoning_config.get('max_iterations', 5),
                    'confidence_threshold': reasoning_config.get('confidence_threshold', 0.75),
                    'store_reasoning_chains': reasoning_config.get('store_reasoning_chains', True)
                }
        except Exception as e:
            logger.warning(f"Не удалось загрузить конфигурацию: {e}")
        
        return {
            'enabled': True,
            'max_iterations': 5,
            'confidence_threshold': 0.75,
            'store_reasoning_chains': True
        }
    
    def integrate_with_brain(self) -> bool:
        """
        Интегрировать Self-Reasoning Engine с CoreBrain
        """
        if not self.config.get('enabled', True):
            logger.info("Reasoning Engine отключён в конфигурации")
            return False
        
        try:
            from eva_ai.reasoning.self_reasoning_engine import SelfReasoningEngine
            
            # Get two_model_pipeline directly from brain
            two_model_pipeline = getattr(self.brain, 'two_model_pipeline', None)
            logger.info(f"ReasoningIntegration: brain.two_model_pipeline = {two_model_pipeline is not None}")
            logger.info(f"ReasoningIntegration: brain.two_model_pipeline type = {type(two_model_pipeline)}")
            if two_model_pipeline:
                logger.info(f"ReasoningIntegration: pipeline.model_a = {two_model_pipeline.model_a is not None}")
                logger.info(f"ReasoningIntegration: pipeline.model_b = {two_model_pipeline.model_b is not None}")
            
            logger.info(f"DEBUG Integration: Creating SRE with two_model_pipeline = {two_model_pipeline}")
            self.reasoning_engine = SelfReasoningEngine(
                brain=self.brain,
                two_model_pipeline=two_model_pipeline,
                config={
                    'max_iterations': self.config.get('max_iterations', 5),
                    'confidence_threshold': self.config.get('confidence_threshold', 0.75)
                }
            )
            logger.info(f"DEBUG Integration: SRE created, SRE.two_model_pipeline = {self.reasoning_engine.two_model_pipeline}")
            
            # Добавляем в brain как компонент
            self.brain.self_reasoning_engine = self.reasoning_engine
            
            self.enabled = True
            logger.info("✅ Self-Reasoning Engine интегрирован с CoreBrain")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка интеграции Self-Reasoning Engine: {e}")
            return False
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Обработка запроса через Self-Reasoning Engine
        """
        if not self.enabled or self.reasoning_engine is None:
            return _error_response("Reasoning Engine не инициализирован", "not_initialized")
        
        try:
            return self.reasoning_engine.process_query(query, context)
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return _error_response(f"Ошибка: {e}", "processing_error")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику"""
        if self.reasoning_engine:
            return self.reasoning_engine.get_stats()
        return {}
    
    def enable(self) -> bool:
        """Включить Self-Reasoning Engine"""
        if self.reasoning_engine is None:
            return self.integrate_with_brain()
        self.enabled = True
        self.brain.self_reasoning_engine = self.reasoning_engine
        logger.info("Self-Reasoning Engine включён")
        return True
    
    def disable(self):
        """Отключить Self-Reasoning Engine"""
        self.enabled = False
        logger.info("Self-Reasoning Engine отключён")
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус reasoning engine"""
        return {
            "enabled": self.enabled,
            "engine_ready": self.reasoning_engine is not None,
            "stats": self.get_stats()
        }


def integrate_reasoning(brain) -> ReasoningIntegration:
    """
    Фабричная функция для интеграции
    """
    integration = ReasoningIntegration(brain)
    integration.integrate_with_brain()
    return integration
