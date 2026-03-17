"""
Улучшенный аналитический менеджер для CogniFlex
Интеграция с системой самообучения и модели
"""

import os
import logging
import time
import json
import threading
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta

logger = logging.getLogger("cogniflex.analytics_manager")

class AnalyticsManager:
    """Основной менеджер аналитики для CogniFlex"""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует менеджер аналитики.
        
        Args:
            brain: Ссылка на CoreBrain
            cache_dir: Директория для кэширования
        """
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'analytics_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Инициализация компонентов
        self._init_components()
        
        # Метрики производительности
        self.performance_metrics = defaultdict(list)
        self.learning_metrics = defaultdict(list)
        self.system_metrics = defaultdict(list)
        
        # Поток сбора данных
        self.collection_thread = None
        self.stop_event = threading.Event()
        self.running = False
        
        logger.info("AnalyticsManager инициализирован")
    
    def record(self, metric_name: str, value: float, tags: Optional[Dict] = None):
        """Записывает метрику в систему мониторинга."""
        try:
            if hasattr(self, 'performance_analyzer') and self.performance_analyzer:
                self.performance_analyzer.record_metric(metric_name, value, tags)
            logger.debug(f"Metric recorded: {metric_name}={value}")
        except Exception as e:
            logger.debug(f"Error recording metric: {e}")
    
    def _init_components(self):
        """Инициализирует аналитические компоненты"""
        try:
            # PerformanceAnalyzer
            from cogniflex.learning.performance_analyzer import PerformanceAnalyzer
            self.performance_analyzer = PerformanceAnalyzer(self.brain)
            
            # Добавляем в brain для доступа извне
            if self.brain:
                self.brain.performance_analyzer = self.performance_analyzer
            
            # KnowledgeAnalytics
            from cogniflex.knowledge.knowledge_analytics import KnowledgeAnalytics
            self.knowledge_analytics = KnowledgeAnalytics(self.brain)
            
            # LearningOpportunityManager
            from cogniflex.learning.learning_opportunity_manager import LearningOpportunityManager
            self.learning_manager = LearningOpportunityManager(self.brain)
            
            logger.info("Аналитические компоненты инициализированы")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации компонентов: {e}")
            self.performance_analyzer = None
            self.knowledge_analytics = None
            self.learning_manager = None
    
    def start_monitoring(self):
        """Запускает мониторинг системы"""
        if self.running:
            logger.warning("Мониторинг уже запущен")
            return
        
        self.running = True
        self.stop_event.clear()
        
        self.collection_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.collection_thread.start()
        
        logger.info("Мониторинг аналитики запущен")
    
    def stop_monitoring(self):
        """Останавливает мониторинг"""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
        
        logger.info("Мониторинг аналитики остановлен")
    
    def _monitoring_loop(self):
        """Основной цикл мониторинга"""
        while not self.stop_event.is_set():
            try:
                # Собираем метрики
                self._collect_performance_metrics()
                self._collect_learning_metrics()
                self._collect_system_metrics()
                
                # Анализируем данные
                self._analyze_metrics()
                
                # Ожидаем следующего сбора
                self.stop_event.wait(timeout=10.0)  # Сбор каждые 10 секунд
                
            except Exception as e:
                logger.error(f"Ошибка в цикле мониторинга: {e}")
                self.stop_event.wait(timeout=5.0)
    
    def _collect_performance_metrics(self):
        """Собирает метрики производительности"""
        if not self.performance_analyzer:
            return
        
        try:
            analysis = self.performance_analyzer.analyze_performance()
            timestamp = time.time()
            
            # Сохраняем метрики
            self.performance_metrics['timestamp'].append(timestamp)
            self.performance_metrics['component_count'].append(
                len(analysis.get('performance_analysis', {}).get('component_performance', {}))
            )
            self.performance_metrics['bottlenecks'].append(
                len(analysis.get('performance_analysis', {}).get('bottlenecks', []))
            )
            self.performance_metrics['optimization_opportunities'].append(
                len(analysis.get('performance_analysis', {}).get('optimization_opportunities', []))
            )
            
            # Ограничиваем размер данных
            max_points = 1000
            for key in self.performance_metrics:
                if len(self.performance_metrics[key]) > max_points:
                    self.performance_metrics[key] = self.performance_metrics[key][-max_points:]
                    
        except Exception as e:
            logger.error(f"Ошибка сбора метрик производительности: {e}")
    
    def _collect_learning_metrics(self):
        """Собирает метрики обучения"""
        if not self.learning_manager:
            return
        
        try:
            opportunities = self.learning_manager.get_learning_opportunities()
            timestamp = time.time()
            
            self.learning_metrics['timestamp'].append(timestamp)
            self.learning_metrics['opportunities_count'].append(len(opportunities))
            
            # Считаем по типам
            types = defaultdict(int)
            for opp in opportunities:
                types[opp.get('type', 'unknown')] += 1
            
            for opp_type in ['pattern_analysis', 'knowledge_expansion', 'model_optimization']:
                self.learning_metrics[f'{opp_type}_count'].append(types.get(opp_type, 0))
            
            # Ограничиваем размер данных
            max_points = 1000
            for key in self.learning_metrics:
                if len(self.learning_metrics[key]) > max_points:
                    self.learning_metrics[key] = self.learning_metrics[key][-max_points:]
                    
        except Exception as e:
            logger.error(f"Ошибка сбора метрик обучения: {e}")
    
    def _collect_system_metrics(self):
        """Собирает системные метрики"""
        if not self.brain:
            return
        
        try:
            timestamp = time.time()
            
            # Получаем системные метрики
            system_metrics = self.brain.get_system_metrics()
            
            self.system_metrics['timestamp'].append(timestamp)
            self.system_metrics['cpu_usage'].append(system_metrics.get('cpu_usage', 0))
            self.system_metrics['memory_usage'].append(system_metrics.get('memory_usage', 0))
            self.system_metrics['active_tasks'].append(system_metrics.get('active_tasks', 0))
            self.system_metrics['request_throughput'].append(system_metrics.get('request_throughput', 0))
            self.system_metrics['response_time'].append(system_metrics.get('response_time', 0))
            self.system_metrics['error_rate'].append(system_metrics.get('error_rate', 0))
            
            # Ограничиваем размер данных
            max_points = 1000
            for key in self.system_metrics:
                if len(self.system_metrics[key]) > max_points:
                    self.system_metrics[key] = self.system_metrics[key][-max_points:]
                    
        except Exception as e:
            logger.error(f"Ошибка сбора системных метрик: {e}")
    
    def _analyze_metrics(self):
        """Анализирует собранные метрики"""
        try:
            # Анализ производительности
            self._analyze_performance_trends()
            
            # Анализ обучения
            self._analyze_learning_trends()
            
            # Анализ системы
            self._analyze_system_trends()
            
            # Генерируем рекомендации
            self._generate_recommendations()
            
        except Exception as e:
            logger.error(f"Ошибка анализа метрик: {e}")
    
    def _analyze_performance_trends(self):
        """Анализирует тренды производительности"""
        if len(self.performance_metrics.get('bottlenecks', [])) < 10:
            return
        
        # Анализируем последние 10 точек
        recent_bottlenecks = self.performance_metrics['bottlenecks'][-10:]
        avg_bottlenecks = sum(recent_bottlenecks) / len(recent_bottlenecks)
        
        if avg_bottlenecks > 3:
            logger.warning(f"Обнаружено высокое количество узких мест: {avg_bottlenecks:.1f}")
            self._trigger_performance_optimization()
    
    def _analyze_learning_trends(self):
        """Анализирует тренды обучения"""
        if len(self.learning_metrics.get('opportunities_count', [])) < 10:
            return
        
        recent_opportunities = self.learning_metrics['opportunities_count'][-10:]
        avg_opportunities = sum(recent_opportunities) / len(recent_opportunities)
        
        if avg_opportunities > 5:
            logger.info(f"Обнаружено множество возможностей для обучения: {avg_opportunities:.1f}")
            self._trigger_learning_recommendations()
    
    def _analyze_system_trends(self):
        """Анализирует системные тренды"""
        if len(self.system_metrics.get('cpu_usage', [])) < 10:
            return
        
        recent_cpu = self.system_metrics['cpu_usage'][-10:]
        avg_cpu = sum(recent_cpu) / len(recent_cpu)
        
        recent_memory = self.system_metrics['memory_usage'][-10:]
        avg_memory = sum(recent_memory) / len(recent_memory)
        
        if avg_cpu > 80:
            logger.warning(f"Высокая загрузка CPU: {avg_cpu:.1f}%")
        
        if avg_memory > 80:
            logger.warning(f"Высокое использование памяти: {avg_memory:.1f}%")
    
    def _trigger_performance_optimization(self):
        """Запускает оптимизацию производительности"""
        try:
            if self.brain and hasattr(self.brain, 'self_learning_system'):
                self.brain.self_learning_system.optimize_performance()
                logger.info("Запущена оптимизация производительности")
        except Exception as e:
            logger.error(f"Ошибка запуска оптимизации: {e}")
    
    def _trigger_learning_recommendations(self):
        """Генерирует рекомендации по обучению"""
        try:
            if self.learning_manager:
                opportunities = self.learning_manager.get_learning_opportunities()
                high_priority = [opp for opp in opportunities if opp.get('priority') == 'high']
                
                if high_priority:
                    logger.info(f"Найдено {len(high_priority)} высокоприоритетных возможностей обучения")
                    
                    # Автоматически выполняем некоторые возможности
                    for opp in high_priority[:2]:  # Выполняем до 2 возможностей
                        if opp.get('id'):
                            self.learning_manager.execute_learning_opportunity(opp['id'])
                            
        except Exception as e:
            logger.error(f"Ошибка генерации рекомендаций: {e}")
    
    def _generate_recommendations(self):
        """Генерирует общие рекомендации"""
        recommendations = []
        
        # Анализируем все метрики
        try:
            # Производительность
            if self.performance_metrics.get('bottlenecks'):
                avg_bottlenecks = sum(self.performance_metrics['bottlenecks'][-5:]) / 5
                if avg_bottlenecks > 2:
                    recommendations.append({
                        'type': 'performance',
                        'priority': 'medium',
                        'message': f'Обнаружено {avg_bottlenecks:.1f} узких мест. Рекомендуется оптимизация.',
                        'action': 'optimize_performance'
                    })
            
            # Обучение
            if self.learning_metrics.get('opportunities_count'):
                avg_opportunities = sum(self.learning_metrics['opportunities_count'][-5:]) / 5
                if avg_opportunities > 3:
                    recommendations.append({
                        'type': 'learning',
                        'priority': 'low',
                        'message': f'Доступно {avg_opportunities:.1f} возможностей для обучения.',
                        'action': 'review_learning_opportunities'
                    })
            
            # Сохраняем рекомендации
            if recommendations:
                self._save_recommendations(recommendations)
                
        except Exception as e:
            logger.error(f"Ошибка генерации рекомендаций: {e}")
    
    def _save_recommendations(self, recommendations: List[Dict[str, Any]]):
        """Сохраняет рекомендации"""
        try:
            recommendations_file = os.path.join(self.cache_dir, 'recommendations.json')
            
            # Загружаем существующие рекомендации
            existing = []
            if os.path.exists(recommendations_file):
                with open(recommendations_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            
            # Добавляем новые рекомендации
            for rec in recommendations:
                rec['timestamp'] = time.time()
                rec['id'] = f"rec_{int(time.time())}_{len(existing)}"
                existing.append(rec)
            
            # Ограничиваем количество рекомендаций
            existing = existing[-50:]  # Храним последние 50 рекомендаций
            
            # Сохраняем
            with open(recommendations_file, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Ошибка сохранения рекомендаций: {e}")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Возвращает сводку аналитики"""
        summary = {
            'timestamp': time.time(),
            'monitoring_active': self.running,
            'performance': {},
            'learning': {},
            'system': {},
            'recommendations': []
        }
        
        try:
            # Производительность
            if self.performance_metrics.get('bottlenecks'):
                summary['performance'] = {
                    'avg_bottlenecks': sum(self.performance_metrics['bottlenecks'][-10:]) / 10,
                    'avg_optimization_opportunities': sum(self.performance_metrics['optimization_opportunities'][-10:]) / 10,
                    'trend': 'stable'
                }
            
            # Обучение
            if self.learning_metrics.get('opportunities_count'):
                summary['learning'] = {
                    'avg_opportunities': sum(self.learning_metrics['opportunities_count'][-10:]) / 10,
                    'pattern_analysis': sum(self.learning_metrics.get('pattern_analysis_count', [])[-10:]) / 10,
                    'knowledge_expansion': sum(self.learning_metrics.get('knowledge_expansion_count', [])[-10:]) / 10,
                    'trend': 'stable'
                }
            
            # Система
            if self.system_metrics.get('cpu_usage'):
                summary['system'] = {
                    'avg_cpu_usage': sum(self.system_metrics['cpu_usage'][-10:]) / 10,
                    'avg_memory_usage': sum(self.system_metrics['memory_usage'][-10:]) / 10,
                    'avg_response_time': sum(self.system_metrics['response_time'][-10:]) / 10,
                    'avg_error_rate': sum(self.system_metrics['error_rate'][-10:]) / 10,
                    'trend': 'stable'
                }
            
            # Рекомендации
            recommendations_file = os.path.join(self.cache_dir, 'recommendations.json')
            if os.path.exists(recommendations_file):
                with open(recommendations_file, 'r', encoding='utf-8') as f:
                    all_recommendations = json.load(f)
                    # Возвращаем только последние 5 рекомендаций
                    summary['recommendations'] = all_recommendations[-5:]
                    
        except Exception as e:
            logger.error(f"Ошибка формирования сводки: {e}")
        
        return summary
    
    def execute_recommendation(self, recommendation_id: str) -> bool:
        """Выполняет рекомендацию"""
        try:
            recommendations_file = os.path.join(self.cache_dir, 'recommendations.json')
            if not os.path.exists(recommendations_file):
                return False
            
            with open(recommendations_file, 'r', encoding='utf-8') as f:
                recommendations = json.load(f)
            
            # Находим рекомендацию
            target_rec = None
            for rec in recommendations:
                if rec.get('id') == recommendation_id:
                    target_rec = rec
                    break
            
            if not target_rec:
                logger.warning(f"Рекомендация {recommendation_id} не найдена")
                return False
            
            # Выполняем действие
            action = target_rec.get('action')
            if action == 'optimize_performance':
                self._trigger_performance_optimization()
            elif action == 'review_learning_opportunities':
                self._trigger_learning_recommendations()
            else:
                logger.warning(f"Неизвестное действие: {action}")
                return False
            
            # Помечаем как выполненную
            target_rec['executed'] = True
            target_rec['executed_at'] = time.time()
            
            # Сохраняем
            with open(recommendations_file, 'w', encoding='utf-8') as f:
                json.dump(recommendations, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Рекомендация {recommendation_id} выполнена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка выполнения рекомендации: {e}")
            return False
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Возвращает информацию об обучении"""
        insights = {
            'timestamp': time.time(),
            'opportunities': [],
            'trends': {},
            'recommendations': []
        }
        
        try:
            if self.learning_manager:
                # Получаем возможности
                opportunities = self.learning_manager.get_learning_opportunities()
                insights['opportunities'] = opportunities[:10]  # Последние 10
                
                # Анализируем тренды
                if self.learning_metrics.get('opportunities_count'):
                    recent = self.learning_metrics['opportunities_count'][-30:]  # Последние 30 точек
                    if len(recent) > 1:
                        trend = 'increasing' if recent[-1] > recent[0] else 'decreasing'
                        insights['trends'] = {
                            'opportunity_trend': trend,
                            'avg_opportunities': sum(recent) / len(recent),
                            'peak_opportunities': max(recent),
                            'min_opportunities': min(recent)
                        }
                
        except Exception as e:
            logger.error(f"Ошибка получения информации об обучении: {e}")
        
        return insights
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Возвращает отчет о производительности"""
        report = {
            'timestamp': time.time(),
            'current_metrics': {},
            'trends': {},
            'bottlenecks': [],
            'optimization_suggestions': []
        }
        
        try:
            if self.performance_analyzer:
                # Текущий анализ
                analysis = self.performance_analyzer.analyze_performance()
                perf_analysis = analysis.get('performance_analysis', {})
                
                report['current_metrics'] = {
                    'component_count': len(perf_analysis.get('component_performance', {})),
                    'bottleneck_count': len(perf_analysis.get('bottlenecks', [])),
                    'optimization_count': len(perf_analysis.get('optimization_opportunities', []))
                }
                
                report['bottlenecks'] = perf_analysis.get('bottlenecks', [])[:5]
                report['optimization_suggestions'] = perf_analysis.get('optimization_opportunities', [])[:5]
                
                # Тренды
                if self.performance_metrics.get('bottlenecks'):
                    recent = self.performance_metrics['bottlenecks'][-30:]
                    if len(recent) > 1:
                        trend = 'improving' if recent[-1] < recent[0] else 'degrading'
                        report['trends'] = {
                            'bottleneck_trend': trend,
                            'avg_bottlenecks': sum(recent) / len(recent),
                            'peak_bottlenecks': max(recent),
                            'min_bottlenecks': min(recent)
                        }
                
        except Exception as e:
            logger.error(f"Ошибка формирования отчета о производительности: {e}")
        
        return report
