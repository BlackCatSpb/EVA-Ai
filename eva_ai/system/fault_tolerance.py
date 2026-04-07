"""
Модуль отказоустойчивости для ЕВА
"""
import logging
import time
from typing import Dict, Any, Optional, List, Callable

logger = logging.getLogger("eva_ai.fault_tolerance")

class FaultTolerance:
    """Система обеспечения отказоустойчивости ЕВА."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует систему отказоустойчивости.
        
        Args:
            brain: Ссылка на ядро ЕВА
            cache_dir: Путь к директории кэша
        """
        self.brain = brain
        self.cache_dir = cache_dir
        self.initialized = False
        self.running = False
        
        # Внутренние структуры
        self.fault_handlers: Dict[str, Callable] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.fault_history: List[Dict[str, Any]] = []
        
        self.initialized = True
        logger.info("FaultTolerance инициализирован")
    
    def is_ready(self) -> bool:
        """Проверяет готовность системы отказоустойчивости."""
        return self.initialized
    
    def start(self):
        """Запускает систему отказоустойчивости."""
        self.running = True
        logger.info("FaultTolerance запущен")
    
    def stop(self):
        """Останавливает систему отказоустойчивости."""
        self.running = False
        logger.info("FaultTolerance остановлен")
    
    def register_fault_handler(self, fault_type: str, handler: Callable):
        """Регистрирует обработчик ошибок."""
        self.fault_handlers[fault_type] = handler
        logger.debug(f"Зарегистрирован обработчик для {fault_type}")
    
    def handle_fault(self, fault_type: str, error: Exception, context: Optional[Dict] = None):
        """Обрабатывает ошибку."""
        fault_record = {
            "type": fault_type,
            "error": str(error),
            "timestamp": time.time(),
            "context": context or {}
        }
        
        self.fault_history.append(fault_record)
        
        # Вызываем обработчик, если есть
        if fault_type in self.fault_handlers:
            try:
                self.fault_handlers[fault_type](error, context)
            except Exception as e:
                logger.error(f"Ошибка в обработчике {fault_type}: {e}")
        
        logger.warning(f"Обработана ошибка {fault_type}: {error}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Возвращает информацию о здоровье системы отказоустойчивости."""
        recent_faults = [f for f in self.fault_history if time.time() - f["timestamp"] < 3600]
        
        health_score = 100.0
        if len(recent_faults) > 10:
            health_score -= 30
        elif len(recent_faults) > 5:
            health_score -= 15
        
        status = "healthy" if health_score > 80 else "warning" if health_score > 50 else "critical"
        
        return {
            "health_score": health_score,
            "status": status,
            "total_faults": len(self.fault_history),
            "recent_faults": len(recent_faults),
            "registered_handlers": len(self.fault_handlers),
            "timestamp": time.time()
        }