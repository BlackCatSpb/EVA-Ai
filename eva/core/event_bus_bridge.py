"""
EventBus Bridge — адаптер между старой EventSystem и новой EventBus.

Обеспечивает двунаправленную совместимость:
- Старые trigger() → новые publish()
- Новые publish() → старые trigger()
"""
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("eva.events.bridge")


# Маппинг старых имён событий на новые EventTypes
OLD_TO_NEW_EVENT_MAP = {
    'query_received': 'pipeline.start',
    'tokenize_request': 'pipeline.model_a.start',
    'tokens_ready': 'pipeline.model_a.complete',
    'hot_window_ready': 'pipeline.model_a.complete',
    'response_generated': 'pipeline.complete',
    'contradiction_detected': 'contradiction.detected',
    'system_health_check': 'system.ready',
    'self_dialog_request': 'learning.started',
    'learning_opportunity': 'learning.progress',
    'ethical_check_request': 'pipeline.ethics.check_complete',
    'memory_optimization': 'memory.optimized',
    'component_initialization_requested': 'component.started',
    'component_initialized': 'component.initialized',
    'component_initialization_failed': 'component.error',
    'training_progress': 'learning.progress',
    'reasoning_step': 'analytics.insight',
    'system_error': 'system.error',
}

# Обратный маппинг
NEW_TO_OLD_EVENT_MAP = {v: k for k, v in OLD_TO_NEW_EVENT_MAP.items()}


class EventBusBridge:
    """
    Мост между старой EventSystem (event_system.py) и новой EventBus (event_bus.py).
    
    При вызове old.trigger() → вызывает new.publish()
    При вызове new.publish() → вызывает old.trigger()
    """
    
    def __init__(self, old_event_system, new_event_bus):
        """
        Args:
            old_event_system: Экземпляр EventSystem из event_system.py
            new_event_bus: Экземпляр EventBus из event_bus.py
        """
        self.old_system = old_event_system
        self.new_bus = new_event_bus
        self._old_to_new_subscriptions = {}
        self._new_to_old_subscriptions = {}
        self._old_trigger_original = None
        self._new_publish_original = None
        self._patched = False
        
        self._setup_bridges()
        logger.info("EventBusBridge инициализирован")
    
    def _setup_bridges(self):
        """Настроить двустороннюю связь"""
        self._bridge_old_to_new()
        self._bridge_new_to_old()
    
    def _bridge_old_to_new(self):
        """Перенаправление событий из old EventSystem в EventBus"""
        events_to_bridge = [
            'ethical_check_request',
            'pipeline.start',
            'pipeline.complete',
            'learning.started',
            'memory.updated'
        ]
        
        for event_type in events_to_bridge:
            self._register_bridge(event_type)
    
    def _bridge_new_to_old(self):
        """Перенаправление событий из EventBus в old EventSystem"""
        new_events = [
            'pipeline.start',
            'pipeline.complete', 
            'pipeline.failed',
            'command.completed',
            'command.failed'
        ]
        
        for event_type in new_events:
            self.new_bus.subscribe(event_type, self._on_new_event)
    
    def _on_new_event(self, event=None):
        """Обработка нового события для old системы"""
        if event is None:
            logger.warning("EventBusBridge._on_new_event received None event")
            return
        if self.old_system:
            old_event = self._convert_to_old_format(event)
            self.old_system.trigger(old_event['type'], old_event['data'])
    
    def _convert_to_old_format(self, event) -> Dict:
        """Конвертация события в формат old EventSystem"""
        return {
            'type': event.event_type,
            'source': event.source,
            'data': event.data,
            'timestamp': event.timestamp
        }
    
    def _register_bridge(self, event_type: str):
        """Регистрация моста для события"""
        pass
    
    def link(self):
        """Связывает старую и новую шины событий."""
        if self._patched:
            logger.warning("EventBusBridge уже связан")
            return
        
        # Патчим old.trigger() чтобы форвардить в new.publish()
        if hasattr(self.old_system, 'trigger'):
            self._old_trigger_original = self.old_system.trigger
            self.old_system.trigger = self._forward_old_to_new
        
        # Патчим new.publish() чтобы форвардить в old.trigger()
        if hasattr(self.new_bus, 'publish_sync'):
            self._new_publish_original = self.new_bus.publish_sync
            self.new_bus.publish_sync = self._forward_new_to_old_sync
        
        self._patched = True
        logger.info("EventBusBridge связан")
    
    def _forward_old_to_new(self, event_name: str, data: Any = None, priority_override: Optional[int] = None):
        """Вызывается при old.trigger() — форвардит в new.publish()."""
        # Сначала вызываем оригинальный trigger
        try:
            self._old_trigger_original(event_name, data, priority_override)
        except Exception as e:
            logger.error(f"Ошибка в original trigger для {event_name}: {e}")
        
        # Затем публикуем в новую шину
        try:
            new_event_type = OLD_TO_NEW_EVENT_MAP.get(event_name, f"legacy.{event_name}")
            
            from .event_bus import Event, EventPriority
            
            priority = EventPriority.NORMAL
            if priority_override is not None:
                if priority_override >= 8:
                    priority = EventPriority.CRITICAL
                elif priority_override >= 6:
                    priority = EventPriority.HIGH
                elif priority_override <= 0:
                    priority = EventPriority.LOW
            
            event = Event(
                event_type=new_event_type,
                source="legacy_event_system",
                data={'event_name': event_name, 'payload': data} if data else {'event_name': event_name},
                priority=priority
            )
            
            self.new_bus.publish(event)
        except Exception as e:
            logger.error(f"Ошибка форварда {event_name} в новую шину: {e}")
    
    def _forward_new_to_old_sync(self, event) -> int:
        """Вызывается при new.publish_sync() — форвардит в old.trigger()."""
        # Сначала обрабатываем в новой шине
        try:
            processed = self._new_publish_original(event)
        except Exception as e:
            logger.error(f"Ошибка в original publish_sync для {event.event_type}: {e}")
            processed = 0
        
        # Затем триггерим в старой шине
        try:
            old_event_name = NEW_TO_OLD_EVENT_MAP.get(event.event_type)
            if old_event_name is None:
                # Пытаемся извлечь legacy имя из data
                old_event_name = event.data.get('event_name')
            
            if old_event_name and hasattr(self.old_system, 'trigger'):
                payload = event.data.get('payload', event.data)
                self._old_trigger_original(old_event_name, payload)
        except Exception as e:
            logger.error(f"Ошибка форварда {event.event_type} в старую шину: {e}")
        
        return processed
    
    def map_event_type(self, old_event_name: str) -> str:
        """Маппит старое имя события на новое."""
        return OLD_TO_NEW_EVENT_MAP.get(old_event_name, f"legacy.{old_event_name}")
    
    def map_to_old_event(self, new_event_type: str) -> Optional[str]:
        """Маппит новое имя события на старое."""
        return NEW_TO_OLD_EVENT_MAP.get(new_event_type)
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику моста."""
        return {
            'patched': self._patched,
            'old_to_new_mappings': len(OLD_TO_NEW_EVENT_MAP),
            'new_to_old_mappings': len(NEW_TO_OLD_EVENT_MAP),
        }
