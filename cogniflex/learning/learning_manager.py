"""
Learning Manager для CogniFlex - управление процессом обучения
"""
import logging
import time
from typing import Dict, Any, Optional, List, Callable

logger = logging.getLogger("cogniflex.learning_manager")

class LearningManager:
    """Менеджер обучения, управляющий процессом обучения и дообучения моделей"""
    
    def __init__(self, brain=None, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация менеджера обучения
        
        Args:
            brain: Ссылка на основной объект CogniFlex
            config: Конфигурация менеджера обучения
        """
        self.brain = brain
        self.config = config or {}
        self.is_initialized = False
        self.active_models = {}
        
    def initialize(self):
        """Инициализация менеджера обучения"""
        if self.is_initialized:
            return True
            
        try:
            # Инициализация компонентов обучения
            logger.info("Инициализация LearningManager...")
            
            # Здесь может быть загрузка сохраненных состояний обучения
            # и инициализация необходимых компонентов
            
            self.is_initialized = True
            logger.info("LearningManager успешно инициализирован")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации LearningManager: {e}", exc_info=True)
            self.is_initialized = False
            return False
    
    def train_model(self, model_id: str, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Обучение модели
        
        Args:
            model_id: Идентификатор модели
            data: Данные для обучения
            **kwargs: Дополнительные параметры обучения
            
        Returns:
            Словарь с результатами обучения
        """
        try:
            logger.info(f"Начало обучения модели {model_id}")
            
            # Проверяем доступность TrainingOrchestrator
            training_orchestrator = self._get_training_orchestrator()
            if not training_orchestrator:
                return self._create_error_result(model_id, "training_orchestrator_unavailable")
            
            # Определяем тип данных и используем соответствующий метод обучения
            if hasattr(data, 'iter_segments'):
                # Документ с сегментами - используем TrainingOrchestrator
                result = training_orchestrator.train_from_document(
                    imported_doc=data,
                    model_id=model_id,
                    use_fractal=kwargs.get('use_fractal', False),
                    fractal_config=kwargs.get('fractal_config', {})
                )
                
                logger.info(f"Обучение модели {model_id} через TrainingOrchestrator завершено")
                return result
                
            elif isinstance(data, (list, tuple)):
                # Список текстовых сегментов
                return self._train_from_segments(model_id, data, **kwargs)
                
            elif isinstance(data, str):
                # Один текстовый сегмент
                return self._train_from_text(model_id, data, **kwargs)
                
            else:
                # Неизвестный формат данных
                error_msg = f"Неподдерживаемый формат данных для обучения: {type(data)}"
                logger.error(error_msg)
                return self._create_error_result(model_id, error_msg)
            
        except Exception as e:
            error_msg = f"Ошибка при обучении модели {model_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_result(model_id, error_msg)
    
    def _get_training_orchestrator(self):
        """Получает TrainingOrchestrator из различных источников."""
        try:
            # Проверяем в brain
            if self.brain and hasattr(self.brain, 'training_orchestrator'):
                return self.brain.training_orchestrator
            
            # Проверяем в ml_unit
            if self.brain and hasattr(self.brain, 'ml_unit'):
                ml_unit = self.brain.ml_unit
                if hasattr(ml_unit, 'training_orchestrator'):
                    return ml_unit.training_orchestrator
            
            # Пробуем создать новый экземпляр
            if self.brain:
                from cogniflex.mlearning.training_orchestrator import TrainingOrchestrator
                return TrainingOrchestrator(brain=self.brain)
                
        except Exception as e:
            logger.error(f"Ошибка получения TrainingOrchestrator: {e}")
        
        return None
    
    def _train_from_segments(self, model_id: str, segments: List[str], **kwargs) -> Dict[str, Any]:
        """Обучение из списка текстовых сегментов."""
        try:
            # Создаем временный документ из сегментов
            class TempDocument:
                def __init__(self, segments_list):
                    self.segments = segments_list
                    self.source_path = f"temp_training_{model_id}"
                    self.title = f"Training Document {model_id}"
                
                def iter_segments(self):
                    return iter(self.segments)
                
                @property
                def metadata(self):
                    return {
                        "model_id": model_id,
                        "source": "learning_manager",
                        "segments_count": len(self.segments)
                    }
            
            temp_doc = TempDocument(segments)
            
            # Используем TrainingOrchestrator
            training_orchestrator = self._get_training_orchestrator()
            if training_orchestrator:
                return training_orchestrator.train_from_document(
                    imported_doc=temp_doc,
                    model_id=model_id,
                    use_fractal=kwargs.get('use_fractal', False)
                )
            else:
                return self._create_error_result(model_id, "training_orchestrator_unavailable")
                
        except Exception as e:
            error_msg = f"Ошибка обучения из сегментов: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_result(model_id, error_msg)
    
    def _train_from_text(self, model_id: str, text: str, **kwargs) -> Dict[str, Any]:
        """Обучение из одного текстового сегмента."""
        return self._train_from_segments(model_id, [text], **kwargs)
    
    def _create_error_result(self, model_id: str, error: str) -> Dict[str, Any]:
        """Создает результат с ошибкой."""
        return {
            "model_id": model_id,
            "status": "error",
            "error": error,
            "timestamp": time.time()
        }
    
    def evaluate_model(self, model_id: str, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Оценка модели
        
        Args:
            model_id: Идентификатор модели
            data: Данные для оценки
            **kwargs: Дополнительные параметры
            
        Returns:
            Словарь с метриками оценки
        """
        try:
            logger.info(f"Оценка модели {model_id}")
            
            # Здесь должна быть логика оценки модели
            # ...
            
            metrics = {
                "accuracy": 0.0,
                "loss": 0.0,
                # Другие метрики...
            }
            
            return {
                "model_id": model_id,
                "status": "success",
                "metrics": metrics
            }
            
        except Exception as e:
            error_msg = f"Ошибка при оценке модели {model_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "model_id": model_id,
                "status": "error",
                "error": error_msg
            }

    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """
        Получение статуса модели
        
        Args:
            model_id: Идентификатор модели
            
        Returns:
            Словарь со статусом модели
        """
        return {
            "model_id": model_id,
            "status": "ready",  # или 'training', 'error', 'not_found'
            "progress": 1.0,    # прогресс обучения (0.0 - 1.0)
            "metrics": {}       # последние метрики
        }

    def stop_training(self, model_id: str) -> bool:
        """
        Остановка обучения модели
        
        Args:
            model_id: Идентификатор модели
            
        Returns:
            True, если остановка прошла успешно, иначе False
        """
        logger.info(f"Остановка обучения модели {model_id}")
        # Логика остановки обучения
        return True
