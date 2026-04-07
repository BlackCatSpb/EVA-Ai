"""
Data Processor - Сбор данных от всех модулей системы
"""

import time
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import logging


class DataProcessor:
    """Компонент для сбора и обработки данных от всех модулей системы"""
    
    def __init__(self, brain=None):
        self.brain = brain
        self.logger = logging.getLogger(__name__)
        self.data_store = {}
        self.data_lock = threading.Lock()
        self.collection_interval = 300  # 5 минут
        self.max_data_age = 86400  # 24 часа
        self.is_running = False
        self.collection_thread = None
        self.last_collection = None
        self.data_collection_stop_event = threading.Event()
        
        # Модули для сбора данных
        self.data_sources = {
            'adaptation_analytics': self._collect_adaptation_analytics,
            'knowledge_analytics': self._collect_knowledge_analytics,
            'ethics_framework': self._collect_ethics_data,
            'ethical_situations': self._collect_ethical_situations,
            'web_search_engine': self._collect_web_search_data,
            'search_engines': self._collect_search_engines_data,
            'knowledge_graph_new': self._collect_knowledge_graph_data,
            'knowledge_analytics': self._collect_knowledge_analytics,
            'contradiction_integrated': self._collect_contradiction_data,
            'contradiction_manager': self._collect_contradiction_manager_data,
            'adaptation_manager': self._collect_adaptation_data,
            'memory_manager': self._collect_memory_data,
            'ml_unit': self._collect_ml_unit_data,
            'query_processor': self._collect_query_data
        }
    
    def initialize(self) -> bool:
        """Инициализация процессора данных"""
        try:
            self.logger.info("Инициализация DataProcessor...")
            
            # Очистка старых данных
            self._cleanup_old_data()
            
            # Запуск автоматического сбора данных
            self.start_collection()
            
            self.logger.info(f"DataProcessor инициализирован")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации DataProcessor: {e}", exc_info=True)
            return False
    
    def start_collection(self) -> None:
        """Запуск автоматического сбора данных"""
        if self.is_running:
            return
            
        self.data_collection_stop_event.clear()
        self.is_running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        self.logger.info("Автоматический сбор данных запущен")
    
    def stop_collection(self) -> None:
        """Остановка автоматического сбора данных"""
        self.data_collection_stop_event.set()
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        self.logger.info("Автоматический сбор данных остановлен")
    
    def _collection_loop(self) -> None:
        """Основной цикл сбора данных"""
        while self.is_running and not self.data_collection_stop_event.is_set():
            try:
                self.collect_all_data()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Ошибка в цикле сбора данных: {e}")
                time.sleep(60)  # Пауза при ошибке
    
    def collect_all_data(self) -> Dict[str, Any]:
        """Сбор данных от всех модулей"""
        collected_data = {}
        timestamp = datetime.now()
        
        with self.data_lock:
            for source_name, collector in self.data_sources.items():
                try:
                    data = collector()
                    if data:
                        collected_data[source_name] = {
                            'data': data,
                            'timestamp': timestamp.isoformat(),
                            'source': source_name
                        }
                except Exception as e:
                    self.logger.warning(f"Ошибка сбора данных из {source_name}: {e}")
            
            # Сохранение собранных данных
            self.data_store['latest_collection'] = {
                'timestamp': timestamp.isoformat(),
                'sources': collected_data
            }
            
            # Обновление истории
            if 'history' not in self.data_store:
                self.data_store['history'] = []
            
            self.data_store['history'].append({
                'timestamp': timestamp.isoformat(),
                'data_count': len(collected_data),
                'sources': list(collected_data.keys())
            })
            
            # Ограничение истории
            if len(self.data_store['history']) > 100:
                self.data_store['history'] = self.data_store['history'][-50:]
            
            self.last_collection = timestamp
        
        self.logger.info(f"Собрано данных из {len(collected_data)} источников")
        return collected_data
    
    def get_recent_data(self, hours: int = 1) -> Dict[str, Any]:
        """Получение данных за последние часы"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.data_lock:
            latest = self.data_store.get('latest_collection', {})
            collection_time = datetime.fromisoformat(latest.get('timestamp', '1970-01-01'))
            
            if collection_time >= cutoff_time:
                return latest.get('sources', {})
        
        return {}
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Получение статистики данных"""
        with self.data_lock:
            stats = {
                'total_collections': len(self.data_store.get('history', [])),
                'last_collection': self.last_collection.isoformat() if self.last_collection else None,
                'active_sources': 0,
                'source_statistics': {},
                'collection_status': 'running' if self.is_running else 'stopped'
            }
            
            latest = self.data_store.get('latest_collection', {})
            sources = latest.get('sources', {})
            
            stats['active_sources'] = len(sources)
            
            for source_name, source_data in sources.items():
                data = source_data.get('data', {})
                stats['source_statistics'][source_name] = {
                    'data_size': len(str(data)),
                    'timestamp': source_data.get('timestamp'),
                    'data_type': type(data).__name__
                }
        
        return stats
    
    def _cleanup_old_data(self) -> None:
        """Очистка старых данных"""
        cutoff_time = datetime.now() - timedelta(seconds=self.max_data_age)
        
        with self.data_lock:
            if 'history' in self.data_store:
                self.data_store['history'] = [
                    entry for entry in self.data_store['history']
                    if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
                ]
    
    # Методы сбора данных от различных модулей
    def _collect_adaptation_analytics(self) -> Optional[Dict]:
        """Сбор данных из аналитики адаптации"""
        if not self.brain or not hasattr(self.brain, 'adaptation_analytics'):
            return None
        try:
            analytics = self.brain.adaptation_analytics
            return {
                'adaptation_count': getattr(analytics, 'adaptation_count', 0),
                'success_rate': getattr(analytics, 'success_rate', 0.0),
                'recent_adaptations': getattr(analytics, 'recent_adaptations', [])
            }
        except Exception as e:
            self.logger.warning(f"Ошибка сбора данных adaptation_analytics: {e}")
            return None
    
    def _collect_knowledge_analytics(self) -> Optional[Dict]:
        """Сбор данных из аналитики знаний"""
        if not self.brain or not hasattr(self.brain, 'knowledge_analytics'):
            return None
        try:
            analytics = self.brain.knowledge_analytics
            return {
                'knowledge_nodes': getattr(analytics, 'total_nodes', 0),
                'knowledge_edges': getattr(analytics, 'total_edges', 0),
                'recent_updates': getattr(analytics, 'recent_updates', [])
            }
        except Exception as e:
            self.logger.warning(f"Ошибка сбора данных knowledge_analytics: {e}")
            return None
    
    def _collect_ethics_data(self) -> Optional[Dict]:
        """Сбор данных из этического фреймворка"""
        if not self.brain or not hasattr(self.brain, 'ethics_framework'):
            return None
        try:
            ethics = self.brain.ethics_framework
            return {
                'ethical_evaluations': getattr(ethics, 'evaluation_count', 0),
                'ethical_score': getattr(ethics, 'average_score', 0.0),
                'recent_cases': getattr(ethics, 'recent_cases', [])
            }
        except Exception as e:
            self.logger.warning(f"Ошибка сбора данных ethics_framework: {e}")
            return None
    
    def _collect_ethical_situations(self) -> Optional[Dict]:
        """Сбор данных об этических ситуациях"""
        if not self.brain or not hasattr(self.brain, 'ethical_situations'):
            return None
        try:
            situations = self.brain.ethical_situations
            return {
                'total_situations': getattr(situations, 'total_situations', 0),
                'resolved_situations': getattr(situations, 'resolved_count', 0),
                'pending_situations': getattr(situations, 'pending_count', 0)
            }
        except Exception as e:
            self.logger.warning(f"Ошибка сбора данных ethical_situations: {e}")
            return None
    
    def _collect_web_search_data(self) -> Optional[Dict]:
        """Сбор данных из веб-поиска"""
        if not self.brain or not hasattr(self.brain, 'web_search_engine'):
            return None
        try:
            web_search = self.brain.web_search_engine
            return {
                'search_count': getattr(web_search, 'search_count', 0),
                'success_rate': getattr(web_search, 'success_rate', 0.0),
                'recent_queries': getattr(web_search, 'recent_queries', [])
            }
        except Exception as e:
            self.logger.warning(f"Ошибка сбора данных web_search_engine: {e}")
            return None
    
    def _collect_search_engines_data(self) -> Optional[Dict]:
        """Сбор данных из поисковых систем"""
        if not self.brain or not hasattr(self.brain, 'search_engines'):
            return None
        try:
            engines = self.brain.search_engines
            return {
                'active_engines': len(getattr(engines, 'engines', {})),
                'total_searches': getattr(engines, 'total_searches', 0),
                'engine_performance': getattr(engines, 'performance_stats', {})
            }
        except Exception as e:
            self.logger.warning(f"Ошибка сбора данных search_engines: {e}")
            return None
    
    def _collect_knowledge_graph_data(self) -> Optional[Dict]:
        """Сбор данных из графа знаний"""
        if not self.brain or not hasattr(self.brain, 'knowledge_graph'):
            return None
        try:
            kg = self.brain.knowledge_graph
            return {
                'nodes_count': getattr(kg, 'nodes_count', 0),
                'edges_count': getattr(kg, 'edges_count', 0),
                'recent_additions': getattr(kg, 'recent_additions', [])
            }
        except Exception as e:
            self.logger.warning(f"Ошибка сбора данных knowledge_graph: {e}")
            return None
    
    def _collect_contradiction_data(self) -> Optional[Dict]:
        """Сбор данных из интегрированных противоречий"""
        if not self.brain or not hasattr(self.brain, 'contradiction_integrated'):
            return None
        try:
            contradictions = self.brain.contradiction_integrated
            return {
                'detected_contradictions': getattr(contradictions, 'detected_count', 0),
                'resolved_contradictions': getattr(contradictions, 'resolved_count', 0),
                'pending_contradictions': getattr(contradictions, 'pending_count', 0)
            }
        except Exception as e:
            self.logger.warning(f"Ошибка сбора данных contradiction_integrated: {e}")
            return None
    
    def _collect_contradiction_manager_data(self) -> Optional[Dict]:
        """Сбор данных из менеджера противоречий"""
        if not self.brain or not hasattr(self.brain, 'contradiction_manager'):
            return None
        try:
            manager = self.brain.contradiction_manager
            return {
                'active_contradictions': getattr(manager, 'active_count', 0),
                'resolution_rate': getattr(manager, 'resolution_rate', 0.0),
                'recent_resolutions': getattr(manager, 'recent_resolutions', [])
            }
        except Exception as e:
            self.logger.warning(f"Ошибка сбора данных contradiction_manager: {e}")
            return None
    
    def _collect_adaptation_data(self) -> Optional[Dict]:
        """Сбор данных из менеджера адаптации"""
        if not self.brain or not hasattr(self.brain, 'adaptation_manager'):
            return None
        try:
            adaptation = self.brain.adaptation_manager
            return {
                'adaptation_cycles': getattr(adaptation, 'cycle_count', 0),
                'improvement_rate': getattr(adaptation, 'improvement_rate', 0.0),
                'recent_adaptations': getattr(adaptation, 'recent_adaptations', [])
            }
        except Exception as e:
            self.logger.warning(f"Ошибка сбора данных adaptation_manager: {e}")
            return None
    
    def _collect_memory_data(self) -> Optional[Dict]:
        """Сбор данных из менеджера памяти"""
        if not self.brain or not hasattr(self.brain, 'memory_manager'):
            return None
        try:
            memory = self.brain.memory_manager
            return {
                'memory_size': getattr(memory, 'memory_size', 0),
                'access_count': getattr(memory, 'access_count', 0),
                'recent_accesses': getattr(memory, 'recent_accesses', [])
            }
        except Exception as e:
            self.logger.warning(f"Ошибка сбора данных memory_manager: {e}")
            return None
    
    def _collect_ml_unit_data(self) -> Optional[Dict]:
        """Сбор данных из ML юнита"""
        if not self.brain or not hasattr(self.brain, 'ml_unit'):
            return None
        try:
            ml_unit = self.brain.ml_unit
            return {
                'predictions_count': getattr(ml_unit, 'predictions_count', 0),
                'accuracy': getattr(ml_unit, 'accuracy', 0.0),
                'model_performance': getattr(ml_unit, 'performance_metrics', {})
            }
        except Exception as e:
            self.logger.warning(f"Ошибка сбора данных ml_unit: {e}")
            return None
    
    def _collect_query_data(self) -> Optional[Dict]:
        """Сбор данных из процессора запросов"""
        if not self.brain or not hasattr(self.brain, 'query_processor'):
            return None
        try:
            processor = self.brain.query_processor
            return {
                'processed_queries': getattr(processor, 'query_count', 0),
                'success_rate': getattr(processor, 'success_rate', 0.0),
                'recent_queries': getattr(processor, 'recent_queries', [])
            }
        except Exception as e:
            self.logger.warning(f"Ошибка сбора данных query_processor: {e}")
            return None
