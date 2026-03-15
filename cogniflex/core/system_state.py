"""
Менеджер состояния системы для CogniFlex
"""

import time
import threading
from enum import Enum
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """Состояния системы CogniFlex."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"

class SystemStateManager:
    """Управляет состоянием системы CogniFlex."""
    
    def __init__(self):
        """Инициализирует менеджер состояния."""
        self.current_state = SystemState.INITIALIZING
        self.state_history = []
        self.state_lock = threading.RLock()
        self.state_listeners = []
        
        # Информация о компонентах
        self.component_states = {}
        self.active_tasks = set()
        self.error_count = 0
        self.last_error = None
        
        # Временные метки
        self.startup_time = time.time()
        self.last_state_change = time.time()
        
        logger.info("SystemStateManager инициализирован")
    
    def set_state(self, new_state: SystemState, reason: str = ""):
        """Устанавливает новое состояние системы.
        
        Args:
            new_state: Новое состояние
            reason: Причина изменения состояния
        """
        with self.state_lock:
            if self.current_state != new_state:
                old_state = self.current_state
                self.current_state = new_state
                self.last_state_change = time.time()
                
                # Записываем в историю
                self.state_history.append({
                    "from": old_state.value,
                    "to": new_state.value,
                    "timestamp": self.last_state_change,
                    "reason": reason
                })
                
                # Ограничиваем размер истории
                if len(self.state_history) > 100:
                    self.state_history = self.state_history[-50:]
                
                logger.info(f"Состояние изменено: {old_state.value} -> {new_state.value} ({reason})")
                
                # Уведомляем слушателей
                self._notify_listeners(old_state, new_state, reason)
    
    def get_state(self) -> SystemState:
        """Возвращает текущее состояние системы."""
        return self.current_state
    
    def is_ready(self) -> bool:
        """Проверяет, готова ли система к работе."""
        return self.current_state == SystemState.READY
    
    def is_processing(self) -> bool:
        """Проверяет, обрабатывает ли система запросы."""
        return self.current_state == SystemState.PROCESSING
    
    def has_errors(self) -> bool:
        """Проверяет, есть ли ошибки в системе."""
        return self.current_state == SystemState.ERROR or self.error_count > 0
    
    def set_component_state(self, component: str, state: str, details: str = ""):
        """Устанавливает состояние компонента.
        
        Args:
            component: Название компонента
            state: Состояние компонента
            details: Дополнительная информация
        """
        with self.state_lock:
            self.component_states[component] = {
                "state": state,
                "details": details,
                "timestamp": time.time()
            }
            
            logger.debug(f"Компонент {component}: {state} ({details})")
    
    def get_component_state(self, component: str) -> Optional[Dict[str, Any]]:
        """Получает состояние компонента.
        
        Args:
            component: Название компонента
            
        Returns:
            Информация о состоянии компонента или None
        """
        return self.component_states.get(component)
    
    def add_active_task(self, task_id: str):
        """Добавляет активную задачу.
        
        Args:
            task_id: Идентификатор задачи
        """
        with self.state_lock:
            self.active_tasks.add(task_id)
            if self.current_state == SystemState.READY:
                self.set_state(SystemState.PROCESSING, f"Начата задача {task_id}")
    
    def remove_active_task(self, task_id: str):
        """Удаляет активную задачу.
        
        Args:
            task_id: Идентификатор задачи
        """
        with self.state_lock:
            self.active_tasks.discard(task_id)
            if not self.active_tasks and self.current_state == SystemState.PROCESSING:
                self.set_state(SystemState.READY, f"Завершена задача {task_id}")
    
    def record_error(self, error: Exception, component: str = ""):
        """Записывает ошибку в систему.
        
        Args:
            error: Объект ошибки
            component: Компонент, в котором произошла ошибка
        """
        with self.state_lock:
            self.error_count += 1
            self.last_error = {
                "error": str(error),
                "component": component,
                "timestamp": time.time(),
                "type": type(error).__name__
            }
            
            # Если критическая ошибка, меняем состояние
            if self.error_count > 5 or isinstance(error, (SystemError, MemoryError)):
                self.set_state(SystemState.ERROR, f"Критическая ошибка в {component}")
            
            logger.error(f"Ошибка в {component}: {error}")
    
    def clear_errors(self):
        """Очищает счетчик ошибок."""
        with self.state_lock:
            self.error_count = 0
            self.last_error = None
            if self.current_state == SystemState.ERROR:
                self.set_state(SystemState.READY, "Ошибки устранены")
    
    def add_state_listener(self, callback):
        """Добавляет слушателя изменений состояния.
        
        Args:
            callback: Функция обратного вызова (old_state, new_state, reason)
        """
        self.state_listeners.append(callback)
    
    def remove_state_listener(self, callback):
        """Удаляет слушателя изменений состояния.
        
        Args:
            callback: Функция обратного вызова
        """
        if callback in self.state_listeners:
            self.state_listeners.remove(callback)
    
    def _notify_listeners(self, old_state: SystemState, new_state: SystemState, reason: str):
        """Уведомляет слушателей об изменении состояния."""
        for callback in self.state_listeners:
            try:
                callback(old_state, new_state, reason)
            except Exception as e:
                logger.error(f"Ошибка в слушателе состояния: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Возвращает полную информацию о состоянии системы.
        
        Returns:
            Словарь с информацией о системе
        """
        with self.state_lock:
            uptime = time.time() - self.startup_time
            
            return {
                "current_state": self.current_state.value,
                "uptime": uptime,
                "last_state_change": self.last_state_change,
                "active_tasks": len(self.active_tasks),
                "active_task_ids": list(self.active_tasks),
                "error_count": self.error_count,
                "last_error": self.last_error,
                "component_count": len(self.component_states),
                "components": self.component_states.copy(),
                "state_history": self.state_history[-10:],  # Последние 10 изменений
                "is_ready": self.is_ready(),
                "has_errors": self.has_errors()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Возвращает статус здоровья системы.
        
        Returns:
            Словарь со статусом здоровья
        """
        with self.state_lock:
            # Определяем общее здоровье
            if self.current_state == SystemState.ERROR:
                health = "critical"
            elif self.error_count > 3:
                health = "warning"
            elif self.current_state == SystemState.READY:
                health = "healthy"
            else:
                health = "unknown"
            
            # Проверяем компоненты
            component_health = {}
            for comp, state in self.component_states.items():
                if "error" in state["state"].lower() or "failed" in state["state"].lower():
                    component_health[comp] = "unhealthy"
                elif "ready" in state["state"].lower() or "ok" in state["state"].lower():
                    component_health[comp] = "healthy"
                else:
                    component_health[comp] = "unknown"
            
            return {
                "overall_health": health,
                "current_state": self.current_state.value,
                "error_count": self.error_count,
                "active_tasks": len(self.active_tasks),
                "component_health": component_health,
                "uptime": time.time() - self.startup_time,
                "last_error": self.last_error
            }
