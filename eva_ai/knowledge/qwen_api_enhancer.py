"""
Qwen API Enhancer - расширение возможностей Qwen через API
Заглушка для обратной совместимости
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class QwenAPIEnhancer:
    """
    Расширение возможностей Qwen через внешний API.
    Используется как fallback для генерации когда локальные модели недоступны.
    """
    
    def __init__(self, brain=None, api_key: str = None):
        self.brain = brain
        self.api_key = api_key
        self.enabled = False
        logger.info("QwenAPIEnhancer инициализирован (отключен)")
    
    def is_available(self) -> bool:
        """Проверить доступность API."""
        return False
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Сгенерировать текст через API."""
        logger.warning("QwenAPIEnhancer отключен - используйте локальные модели")
        return None
    
    def enhance_response(self, response: str, context: Dict[str, Any] = None) -> str:
        """Улучшить ответ через API."""
        return response
