"""Модуль сбора метрик для системы ЕВА"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Сборщик метрик для мониторинга производительности и состояния системы"""
    
    def __init__(self, brain=None):
        """
        Инициализация сборщика метрик
        
        Args:
            brain: Ссылка на ядро системы
        """
        self.brain = brain
        self.metrics = {}
        self.start_time = time.time()
        self.component_stats = {}
        self.errors = []
        
    def record_metric(self, name: str, value: Any, timestamp: Optional[datetime] = None):
        """
        Записывает метрику
        
        Args:
            name: Название метрики
            value: Значение метрики
            timestamp: Временная метка (опционально)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if name not in self.metrics:
            self.metrics[name] = []
            
        self.metrics[name].append({
            'value': value,
            'timestamp': timestamp
        })
        
    def record_error(self, component: str, error: str, severity: str = 'error'):
        """
        Записывает ошибку
        
        Args:
            component: Компонент, в котором произошла ошибка
            error: Описание ошибки
            severity: Уровень критичности
        """
        self.errors.append({
            'component': component,
            'error': error,
            'severity': severity,
            'timestamp': datetime.now()
        })
        
    def get_component_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику по компонентам
        
        Returns:
            Dict с статистикой компонентов
        """
        stats = {}
        if self.brain and hasattr(self.brain, 'components'):
            for name, component in self.brain.components.items():
                stats[name] = {
                    'initialized': hasattr(component, 'is_initialized') and component.is_initialized,
                    'errors': len([e for e in self.errors if e['component'] == name])
                }
        return stats
        
    def get_system_health(self) -> Dict[str, Any]:
        """
        Возвращает общее состояние системы
        
        Returns:
            Dict с показателями здоровья системы
        """
        total_components = len(self.brain.components) if self.brain else 0
        initialized_components = len([c for c in self.brain.components.values() 
                                   if hasattr(c, 'is_initialized') and c.is_initialized]) if self.brain else 0
                                   
        return {
            'uptime': time.time() - self.start_time,
            'total_components': total_components,
            'initialized_components': initialized_components,
            'total_errors': len(self.errors),
            'critical_errors': len([e for e in self.errors if e['severity'] == 'critical'])
        }
        
    def get_performance_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Возвращает метрики производительности
        
        Returns:
            Dict с метриками производительности
        """
        return self.metrics
