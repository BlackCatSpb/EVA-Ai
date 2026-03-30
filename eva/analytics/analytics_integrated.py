"""
Интегрированный аналитический менеджер ЕВА
Поддерживает BaseComponent и EventBus
"""

import logging
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger("eva.analytics")

from eva.core.base_component import BaseComponent, ComponentState
from eva.core.event_bus import get_event_bus, Event, EventTypes

# Импортируем оригинальный AnalyticsManager
try:
    from eva.analytics.analytics_manager import AnalyticsManager
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    logger.warning("Оригинальный AnalyticsManager недоступен")


class IntegratedAnalyticsManager(BaseComponent):
    """Интегрированный менеджер аналитики с поддержкой событий"""
    
    def __init__(self, event_bus=None, brain=None, cache_dir: Optional[str] = None):
        super().__init__("analytics_manager", event_bus)
        
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'analytics_cache')
        
        # Инициализируем оригинальный менеджер если доступен
        self._original_manager = None
        if ORIGINAL_AVAILABLE:
            try:
                self._original_manager = AnalyticsManager(brain, cache_dir)
                logger.info("Оригинальный AnalyticsManager инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации оригинального менеджера: {e}")
        
        # Метрики
        self.metrics = defaultdict(deque)
        self.performance_data = deque(maxlen=1000)
        
        logger.info(f"IntegratedAnalyticsManager {self.name} инициализирован")
    
    def _do_initialize(self) -> bool:
        """Инициализация компонента"""
        try:
            logger.info("Инициализация аналитического менеджера...")
            
            # Инициализируем оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'initialize'):
                self._original_manager.initialize()
            
            # Создаем директорию кэша
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Публикуем событие инициализации
            self._emit_event("analytics_manager.initialized", {
                'component': self.name,
                'cache_dir': self.cache_dir
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации аналитического менеджера: {e}")
            return False
    
    def _do_start(self) -> bool:
        """Запуск компонента"""
        try:
            logger.info("Запуск аналитического менеджера...")
            
            # Запускаем оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'start'):
                self._original_manager.start()
            
            # Публикуем событие запуска
            self._emit_event("analytics_manager.started", {
                'component': self.name
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска аналитического менеджера: {e}")
            return False
    
    def _do_stop(self) -> bool:
        """Остановка компонента"""
        try:
            logger.info("Остановка аналитического менеджера...")
            
            # Останавливаем оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'stop'):
                self._original_manager.stop()
            
            # Сохраняем метрики
            self._save_metrics()
            
            # Публикуем событие остановки
            self._emit_event("analytics_manager.stopped", {
                'component': self.name,
                'metrics_count': len(self.metrics)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка остановки аналитического менеджера: {e}")
            return False
    
    def track_query(self, query: str, response_time: float, success: bool = True):
        """Отслеживает выполнение запроса"""
        try:
            timestamp = datetime.now()
            
            # Сохраняем метрику
            self.metrics['queries'].append({
                'timestamp': timestamp,
                'query_length': len(query),
                'response_time': response_time,
                'success': success
            })
            
            # Сохраняем данные производительности
            self.performance_data.append({
                'timestamp': timestamp,
                'response_time': response_time,
                'success': success
            })
            
            # Публикуем событие
            self._emit_event("analytics_manager.query_tracked", {
                'query_length': len(query),
                'response_time': response_time,
                'success': success
            })
            
            # Передаем в оригинальный менеджер
            if self._original_manager and hasattr(self._original_manager, 'track_query'):
                self._original_manager.track_query(query, response_time, success)
                
        except Exception as e:
            logger.error(f"Ошибка отслеживания запроса: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Возвращает метрики производительности"""
        try:
            if not self.performance_data:
                return {}
            
            response_times = [d['response_time'] for d in self.performance_data]
            success_count = sum(1 for d in self.performance_data if d['success'])
            
            metrics = {
                'total_queries': len(self.performance_data),
                'success_rate': success_count / len(self.performance_data),
                'avg_response_time': sum(response_times) / len(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times)
            }
            
            # Добавляем метрики из оригинального менеджера
            if self._original_manager and hasattr(self._original_manager, 'get_performance_metrics'):
                original_metrics = self._original_manager.get_performance_metrics()
                metrics.update(original_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка получения метрик: {e}")
            return {}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Возвращает информацию о здоровье системы"""
        try:
            health = {
                'component_status': 'healthy',
                'metrics_count': len(self.metrics),
                'performance_data_count': len(self.performance_data),
                'cache_dir_exists': os.path.exists(self.cache_dir)
            }
            
            # Добавляем информацию из оригинального менеджера
            if self._original_manager and hasattr(self._original_manager, 'get_system_health'):
                original_health = self._original_manager.get_system_health()
                health.update(original_health)
            
            return health
            
        except Exception as e:
            logger.error(f"Ошибка получения здоровья системы: {e}")
            return {'component_status': 'error', 'error': str(e)}
    
    def _save_metrics(self):
        """Сохраняет метрики в файл"""
        try:
            import json
            
            metrics_file = os.path.join(self.cache_dir, 'metrics.json')
            
            # Конвертируем deque в списки для сериализации
            serializable_metrics = {
                key: list(deque_data) for key, deque_data in self.metrics.items()
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, default=str)
                
        except Exception as e:
            logger.error(f"Ошибка сохранения метрик: {e}")
    
    def generate_report(self, period_hours: int = 24) -> Dict[str, Any]:
        """Генерирует отчет за указанный период"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=period_hours)
            
            # Фильтруем данные за период
            period_data = [
                d for d in self.performance_data 
                if d['timestamp'] > cutoff_time
            ]
            
            if not period_data:
                return {'period_hours': period_hours, 'data_points': 0}
            
            response_times = [d['response_time'] for d in period_data]
            success_count = sum(1 for d in period_data if d['success'])
            
            report = {
                'period_hours': period_hours,
                'data_points': len(period_data),
                'success_rate': success_count / len(period_data),
                'avg_response_time': sum(response_times) / len(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times)
            }
            
            # Публикуем событие генерации отчета
            self._emit_event("analytics_manager.report_generated", {
                'period_hours': period_hours,
                'data_points': len(period_data)
            })
            
            return report
            
        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")
            return {'error': str(e)}
