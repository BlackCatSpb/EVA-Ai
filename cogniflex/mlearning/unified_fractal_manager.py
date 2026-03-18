"""
UnifiedFractalManager - автоматический выбор между оптимизированным и стандартным менеджерами
"""
import os
import logging
from typing import Optional, Dict, Any, List

from .fractal_model_manager import FractalModelManager

logger = logging.getLogger("cogniflex.unified_manager")

class UnifiedFractalManager:
    """Унифицированный менеджер с автоматическим выбором оптимизаций"""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None, 
                 force_optimized: bool = False):
        """Инициализация с автоматическим выбором менеджера"""
        
        self.model_path = model_path
        self.config_path = config_path
        self.force_optimized = force_optimized
        self.manager = None
        self.is_optimized = False
        
        # Автоматический выбор менеджера
        self._select_manager()
    
    def _select_manager(self):
        """Автоматически выбирает лучший менеджер"""
        
        try:
            # Проверяем доступность оптимизированного менеджера
            if self.force_optimized or self._should_use_optimized():
                try:
                    from .optimized_fractal_model_manager import OptimizedFractalModelManager
                    
                    # Инициализируем менеджер
                    use_optimized = True
                    if use_optimized and OptimizedFractalModelManager:
                        self.manager = OptimizedFractalModelManager(config_path=self.config_path)
                        self.is_optimized = True
                        logger.info("Используется OptimizedFractalModelManager")
                    else:
                        self.manager = FractalModelManager(config_path=self.config_path)
                    self.is_optimized = False
                    logger.info("Используется FractalModelManager")
                    
                    # Инициализируем улучшенную систему обучения без конфликтов
                    try:
                        from .enhanced_learning_integration import EnhancedLearningIntegration
                        self.enhanced_learning = EnhancedLearningIntegration(self.manager, self.config_path)
                        logger.info("EnhancedLearningIntegration инициализирована")
                    except Exception as e:
                        logger.warning(f"Не удалось инициализировать EnhancedLearningIntegration: {e}")
                        self.enhanced_learning = None
                    
                    # Инициализируем комплексную систему обучения
                    try:
                        from .comprehensive_learning_system import ComprehensiveLearningSystem
                        self.learning_system = ComprehensiveLearningSystem(self.manager, self.config_path)
                        logger.info("ComprehensiveLearningSystem инициализирована")
                    except Exception as e:
                        logger.warning(f"Не удалось инициализировать ComprehensiveLearningSystem: {e}")
                        self.learning_system = None
                except Exception as e:
                    logger.warning(f"Не удалось загрузить оптимизированный менеджер: {e}")
            
            # Fallback на стандартный менеджер
            else:
                self.manager = FractalModelManager(model_path=self.model_path, config_path=self.config_path)
                self.is_optimized = False
                logger.info("Используется стандартный FractalModelManager")
                
                # Инициализируем комплексную систему обучения
                try:
                    from .comprehensive_learning_system import ComprehensiveLearningSystem
                    self.learning_system = ComprehensiveLearningSystem(self.manager, self.config_path)
                    logger.info("ComprehensiveLearningSystem инициализирована")
                except Exception as e:
                    logger.warning(f"Не удалось инициализировать ComprehensiveLearningSystem: {e}")
                    self.learning_system = None

        except Exception as e:
            logger.error(f"Критическая ошибка при выборе менеджера: {e}")
            raise
    
    def _should_use_optimized(self) -> bool:
        """Определяет, следует ли использовать оптимизированный менеджер"""
        
        # Проверяем наличие конфигурации
        config_path = os.path.join(os.getcwd(), "cogniflex", "config", "max_cache_config.json")
        
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                return config.get("use_optimized", True)
            except Exception as e:
                logger.warning(f"Ошибка загрузки конфигурации: {e}")
        
        # По умолчанию используем оптимизированный
        return True
    
    def generate_response(self, query: str, max_tokens: int = 100) -> str:
        """Генерирует ответ"""
        return self.manager.generate_response(query, max_tokens)
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Возвращает метрики качества"""
        if hasattr(self.manager, 'get_quality_metrics'):
            return self.manager.get_quality_metrics()
        return {}
    
    def improve_quality(self, training_texts=None, save_path=None):
        """Улучшает качество модели"""
        if hasattr(self.manager, 'improve_quality'):
            return self.manager.improve_quality(training_texts, save_path)
        return {"status": "error", "message": "Метод улучшения недоступен"}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности"""
        if self.is_optimized and hasattr(self.manager, 'get_performance_stats'):
            return self.manager.get_performance_stats()
        else:
            return {
                "is_optimized": self.is_optimized,
                "manager_type": type(self.manager).__name__
            }
    
    # Методы улучшенной системы обучения
    def start_enhanced_learning_session(self, topics: Optional[List[str]] = None, 
                                       session_name: Optional[str] = None) -> str:
        """Запускает улучшенную сессию обучения"""
        if self.enhanced_learning:
            return self.enhanced_learning.start_enhanced_learning_session(topics, session_name)
        else:
            logger.warning("EnhancedLearningIntegration недоступна")
            return ""
    
    def generate_enhanced_response(self, query: str, max_tokens: int = 100, 
                                 use_web_search: bool = True) -> Dict[str, Any]:
        """Генерирует улучшенный ответ с использованием веб-поиска"""
        if self.enhanced_learning:
            return self.enhanced_learning.generate_enhanced_response(query, max_tokens, use_web_search)
        else:
            # Fallback на базовую генерацию
            response = self.generate_response(query, max_tokens)
            return {
                "status": "completed",
                "response": response,
                "web_search_used": False,
                "message": "EnhancedLearningIntegration недоступна"
            }
    
    def get_enhanced_system_status(self) -> Dict[str, Any]:
        """Возвращает статус улучшенной системы"""
        if self.enhanced_learning:
            return self.enhanced_learning.get_enhanced_system_status()
        else:
            return {
                "enhanced_system": {
                    "available": False,
                    "message": "EnhancedLearningIntegration недоступна"
                }
            }
    
    def add_enhanced_topics(self, topics: List[str]):
        """Добавляет улучшенные темы для обучения"""
        if self.enhanced_learning:
            self.enhanced_learning.add_enhanced_topics(topics)
        else:
            logger.warning("EnhancedLearningIntegration недоступна")
    
    def configure_enhanced_learning(self, **settings):
        """Настраивает параметры улучшенного обучения"""
        if self.enhanced_learning:
            return self.enhanced_learning.configure_enhanced_learning(**settings)
        else:
            logger.warning("EnhancedLearningIntegration недоступна")
            return False
    
    def __getattr__(self, name):
        """Делегирует остальные атрибуты менеджеру"""
        return getattr(self.manager, name)
    
    def __del__(self):
        """Очистка"""
        if hasattr(self.manager, '__del__') and callable(getattr(self.manager, '__del__')):
            self.manager.__del__()
