"""
Событийная система для CogniFlex - управление уведомлениями о готовности компонентов
"""
import logging
from typing import Dict, List, Callable, Any
from collections import defaultdict

logger = logging.getLogger("cogniflex.events")

class EventSystem:
    """
    Событийная система для уведомления компонентов о готовности зависимостей.
    
    Поддерживаемые события:
    - 'memory_manager_ready': MemoryManager инициализирован
    - 'text_processor_ready': UnifiedTextProcessor готов к работе
    - 'model_manager_ready': ModelManager загружен и готов
    - 'response_generator_ready': ResponseGenerator инициализирован
    - 'ethics_framework_ready': EthicsFramework готов к работе
    """
    
    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._triggered_events: Dict[str, Any] = {}
    
    def subscribe(self, event_name: str, callback: Callable):
        """
        Подписывается на событие.
        
        Args:
            event_name: Имя события
            callback: Функция обратного вызова
        """
        self.listeners[event_name].append(callback)
        
        # Если событие уже было вызвано, немедленно вызываем callback
        if event_name in self._triggered_events:
            try:
                callback(self._triggered_events[event_name])
            except Exception as e:
                logger.error(f"Ошибка в обработчике события {event_name}: {e}")
    
    def trigger(self, event_name: str, data: Any = None):
        """
        Вызывает событие и уведомляет всех подписчиков.
        
        Args:
            event_name: Имя события
            data: Данные для передачи подписчикам
        """
        self._triggered_events[event_name] = data
        
        if event_name in self.listeners:
            for callback in self.listeners[event_name]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Ошибка в обработчике события {event_name}: {e}")
    
    def is_triggered(self, event_name: str) -> bool:
        """Проверяет, было ли событие уже вызвано."""
        return event_name in self._triggered_events
    
    def get_event_data(self, event_name: str) -> Any:
        """Получает данные события, если оно было вызвано."""
        return self._triggered_events.get(event_name)