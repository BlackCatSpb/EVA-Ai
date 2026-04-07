"""
Интегратор аналитики с системой самообучения ЕВА
"""

import os
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger("eva_ai.analytics_learning_integration")

class AnalyticsLearningIntegration:
    """Интеграция аналитики с системой самообучения"""
    
    def __init__(self, brain=None, analytics_manager=None):
        """
        Инициализирует интегратор.
        
        Args:
            brain: Ссылка на CoreBrain
            analytics_manager: Ссылка на AnalyticsManager
        """
        self.brain = brain
        self.analytics_manager = analytics_manager
        
        # Метрики обучения
        self.learning_metrics = {
            'model_performance': [],
            'knowledge_growth': [],
            'contradiction_resolution': [],
            'user_satisfaction': []
        }
        
        # История обучения
        self.learning_history = []
        
        logger.info("AnalyticsLearningIntegration инициализирован")
    
    def analyze_learning_effectiveness(self) -> Dict[str, Any]:
        """Анализирует эффективность обучения"""
        analysis = {
            'timestamp': time.time(),
            'overall_effectiveness': 0.0,
            'model_improvements': {},
            'knowledge_expansion': {},
            'contradiction_resolution': {},
            'recommendations': []
        }
        
        try:
            # Анализ производительности моделей
            model_analysis = self._analyze_model_performance()
            analysis['model_improvements'] = model_analysis
            
            # Анализ расширения знаний
            knowledge_analysis = self._analyze_knowledge_expansion()
            analysis['knowledge_expansion'] = knowledge_analysis
            
            # Анализ разрешения противоречий
            contradiction_analysis = self._analyze_contradiction_resolution()
            analysis['contradiction_resolution'] = contradiction_analysis
            
            # Общая эффективность
            effectiveness_scores = []
            
            if isinstance(model_analysis, dict) and model_analysis.get('improvement_rate', 0) > 0:
                effectiveness_scores.append(model_analysis['improvement_rate'])
            
            if isinstance(knowledge_analysis, dict) and knowledge_analysis.get('growth_rate', 0) > 0:
                effectiveness_scores.append(knowledge_analysis['growth_rate'])
            
            if isinstance(contradiction_analysis, dict) and contradiction_analysis.get('resolution_rate', 0) > 0:
                effectiveness_scores.append(contradiction_analysis['resolution_rate'])
            
            if effectiveness_scores:
                analysis['overall_effectiveness'] = sum(effectiveness_scores) / len(effectiveness_scores)
            
            # Генерируем рекомендации
            analysis['recommendations'] = self._generate_learning_recommendations(analysis)
            
        except Exception as e:
            logger.error(f"Ошибка анализа эффективности обучения: {e}")
        
        return analysis
    
    def _analyze_model_performance(self) -> Dict[str, Any]:
        """Анализирует производительность моделей"""
        analysis = {
            'timestamp': time.time(),
            'models': {},
            'improvement_rate': 0.0,
            'trend': 'stable'
        }
        
        try:
            if self.brain and hasattr(self.brain, 'model_manager'):
                model_manager = self.brain.model_manager
                
                if not hasattr(model_manager, 'models') or not isinstance(model_manager.models, dict):
                    logger.warning("model_manager.models не является словарем")
                    return analysis
                
                # Анализируем каждую модель
                for model_name in model_manager.models:
                    model_info = self._get_model_info(model_name)
                    if model_info:
                        analysis['models'][model_name] = model_info
                
                # Рассчитываем общую тенденцию
                if analysis['models']:
                    improvements = []
                    for model_data in analysis['models'].values():
                        if model_data.get('performance_trend'):
                            improvements.append(model_data['performance_trend'])
                    
                    if improvements:
                        analysis['improvement_rate'] = sum(improvements) / len(improvements)
                        
                        if analysis['improvement_rate'] > 0.1:
                            analysis['trend'] = 'improving'
                        elif analysis['improvement_rate'] < -0.1:
                            analysis['trend'] = 'degrading'
            
        except Exception as e:
            logger.error(f"Ошибка анализа производительности моделей: {e}")
        
        return analysis
    
    def _get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Получает информацию о модели"""
        try:
            if self.brain and hasattr(self.brain, 'model_manager'):
                model_manager = self.brain.model_manager
                
                if model_name in model_manager.models:
                    model = model_manager.models[model_name]
                    
                    info = {
                        'name': model_name,
                        'loaded': True,
                        'performance_score': getattr(model, 'performance_score', 0.5),
                        'last_updated': getattr(model, 'last_updated', time.time()),
                        'accuracy': getattr(model, 'accuracy', 0.0),
                        'loss': getattr(model, 'loss', 0.0)
                    }
                    
                    # Анализируем тренд производительности
                    historical_performance = getattr(model, 'performance_history', [])
                    if len(historical_performance) > 1:
                        recent = historical_performance[-5:]
                        if len(recent) > 1:
                            trend = (recent[-1] - recent[0]) / len(recent)
                            info['performance_trend'] = trend
                    
                    return info
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о модели {model_name}: {e}")
        
        return None
    
    def _analyze_knowledge_expansion(self) -> Dict[str, Any]:
        """Анализирует расширение знаний"""
        analysis = {
            'timestamp': time.time(),
            'total_concepts': 0,
            'new_concepts_today': 0,
            'growth_rate': 0.0,
            'domain_distribution': {},
            'trend': 'stable'
        }
        
        try:
            if self.brain and hasattr(self.brain, 'knowledge_graph'):
                kg = self.brain.knowledge_graph
                
                # Общая статистика
                if hasattr(kg, 'get_statistics'):
                    stats = kg.get_statistics()
                    analysis['total_concepts'] = stats.get('total_concepts', 0)
                    analysis['domain_distribution'] = stats.get('domain_distribution', {})
                
                # Новые концепции за сегодня
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                today_timestamp = today_start.timestamp()
                
                if hasattr(kg, 'get_concepts_by_date'):
                    new_concepts = kg.get_concepts_by_date(today_timestamp)
                    analysis['new_concepts_today'] = len(new_concepts)
                
                # Тренд роста
                if hasattr(kg, 'get_growth_trend'):
                    growth_data = kg.get_growth_trend(days=7)
                    if growth_data and len(growth_data) > 1:
                        recent_growth = growth_data[-3:]  # Последние 3 дня
                        if len(recent_growth) > 1:
                            analysis['growth_rate'] = (recent_growth[-1] - recent_growth[0]) / len(recent_growth)
                            
                            if analysis['growth_rate'] > 0.1:
                                analysis['trend'] = 'growing'
                            elif analysis['growth_rate'] < -0.1:
                                analysis['trend'] = 'shrinking'
            
        except Exception as e:
            logger.error(f"Ошибка анализа расширения знаний: {e}")
        
        return analysis
    
    def _analyze_contradiction_resolution(self) -> Dict[str, Any]:
        """Анализирует разрешение противоречий"""
        analysis = {
            'timestamp': time.time(),
            'total_contradictions': 0,
            'resolved_today': 0,
            'resolution_rate': 0.0,
            'resolution_methods': {},
            'trend': 'stable'
        }
        
        try:
            # Получаем статистику противоречий
            contradiction_stats = self._get_contradiction_stats()
            
            if contradiction_stats:
                analysis['total_contradictions'] = contradiction_stats.get('total', 0)
                analysis['resolved_today'] = contradiction_stats.get('resolved_today', 0)
                analysis['resolution_methods'] = contradiction_stats.get('resolution_methods', {})
                
                # Рассчитываем коэффициент разрешения
                total_resolved = contradiction_stats.get('resolved', 0)
                total_detected = contradiction_stats.get('total', 0)
                
                if total_detected > 0:
                    analysis['resolution_rate'] = total_resolved / total_detected
                    
                    if analysis['resolution_rate'] > 0.8:
                        analysis['trend'] = 'excellent'
                    elif analysis['resolution_rate'] > 0.5:
                        analysis['trend'] = 'good'
                    elif analysis['resolution_rate'] > 0.3:
                        analysis['trend'] = 'moderate'
                    else:
                        analysis['trend'] = 'poor'
            
        except Exception as e:
            logger.error(f"Ошибка анализа разрешения противоречий: {e}")
        
        return analysis
    
    def _get_contradiction_stats(self) -> Optional[Dict[str, Any]]:
        """Получает статистику противоречий"""
        try:
            if self.brain and hasattr(self.brain, 'get_contradiction_statistics'):
                stats = self.brain.get_contradiction_statistics()
                return stats
            elif self.brain and hasattr(self.brain, 'contradiction_manager'):
                cm = self.brain.contradiction_manager
                if hasattr(cm, 'get_statistics'):
                    return cm.get_statistics()
        except Exception as e:
            logger.error(f"Ошибка получения статистики противоречий: {e}")
        
        return None
    
    def _generate_learning_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерирует рекомендации по обучению"""
        recommendations = []
        
        try:
            # Рекомендации по моделям
            model_improvements = analysis.get('model_improvements', {})
            if model_improvements.get('trend') == 'degrading':
                recommendations.append({
                    'type': 'model_optimization',
                    'priority': 'high',
                    'message': 'Производительность моделей ухудшается. Рекомендуется дообучение.',
                    'action': 'retrain_models',
                    'target_models': list(model_improvements.get('models', {}).keys())
                })
            
            # Рекомендации по знаниям
            knowledge_expansion = analysis.get('knowledge_expansion', {})
            if knowledge_expansion.get('growth_rate', 0) < 0.05:
                recommendations.append({
                    'type': 'knowledge_expansion',
                    'priority': 'medium',
                    'message': 'Рост знаний замедлился. Рекомендуется расширение базы знаний.',
                    'action': 'expand_knowledge',
                    'domains': list(knowledge_expansion.get('domain_distribution', {}).keys())
                })
            
            # Рекомендации по противоречиям
            contradiction_resolution = analysis.get('contradiction_resolution', {})
            if contradiction_resolution.get('trend') in ['poor', 'moderate']:
                recommendations.append({
                    'type': 'contradiction_resolution',
                    'priority': 'medium',
                    'message': 'Низкая эффективность разрешения противоречий.',
                    'action': 'improve_resolution',
                    'methods': list(contradiction_resolution.get('resolution_methods', {}).keys())
                })
            
            # Общая эффективность
            overall_effectiveness = analysis.get('overall_effectiveness', 0)
            if overall_effectiveness < 0.3:
                recommendations.append({
                    'type': 'system_optimization',
                    'priority': 'high',
                    'message': 'Низкая общая эффективность обучения. Требуется комплексная оптимизация.',
                    'action': 'comprehensive_optimization'
                })
            
        except Exception as e:
            logger.error(f"Ошибка генерации рекомендаций: {e}")
        
        return recommendations
    
    def execute_learning_recommendation(self, recommendation: Dict[str, Any]) -> bool:
        """Выполняет рекомендацию по обучению"""
        try:
            action = recommendation.get('action')
            
            if action == 'retrain_models':
                return self._retrain_models(recommendation.get('target_models', []))
            elif action == 'expand_knowledge':
                return self._expand_knowledge(recommendation.get('domains', []))
            elif action == 'improve_resolution':
                return self._improve_contradiction_resolution(recommendation.get('methods', []))
            elif action == 'comprehensive_optimization':
                return self._comprehensive_optimization()
            else:
                logger.warning(f"Неизвестное действие: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка выполнения рекомендации: {e}")
            return False
    
    def _retrain_models(self, models: List[str]) -> bool:
        """Переобучает модели"""
        try:
            if self.brain and hasattr(self.brain, 'self_learning_system'):
                for model_name in models:
                    self.brain.self_learning_system.retrain_model(model_name)
                logger.info(f"Запущено переобучение моделей: {models}")
                return True
        except Exception as e:
            logger.error(f"Ошибка переобучения моделей: {e}")
        
        return False
    
    def _expand_knowledge(self, domains: List[str]) -> bool:
        """Расширяет знания в указанных доменах"""
        try:
            if self.analytics_manager and self.analytics_manager.learning_manager:
                for domain in domains:
                    # Создаем возможность для расширения знаний
                    opportunities = self.analytics_manager.learning_manager.get_learning_opportunities()
                    
                    # Ищем возможности по домену
                    domain_opportunities = [
                        opp for opp in opportunities 
                        if domain.lower() in opp.get('description', '').lower()
                    ]
                    
                    # Выполняем до 2 возможностей
                    for opp in domain_opportunities[:2]:
                        if opp.get('id'):
                            self.analytics_manager.learning_manager.execute_learning_opportunity(opp['id'])
                
                logger.info(f"Запущено расширение знаний в доменах: {domains}")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка расширения знаний: {e}")
        
        return False
    
    def _improve_contradiction_resolution(self, methods: List[str]) -> bool:
        """Улучшает разрешение противоречий"""
        try:
            if self.brain and hasattr(self.brain, 'contradiction_manager'):
                # Анализируем текущие методы разрешения
                current_methods = self.brain.contradiction_manager.get_resolution_methods()
                
                # Добавляем новые методы при необходимости
                for method in methods:
                    if method not in current_methods:
                        self.brain.contradiction_manager.add_resolution_method(method)
                
                logger.info(f"Улучшены методы разрешения противоречий: {methods}")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка улучшения разрешения противоречий: {e}")
        
        return False
    
    def _comprehensive_optimization(self) -> bool:
        """Комплексная оптимизация"""
        try:
            if self.brain and hasattr(self.brain, 'self_learning_system'):
                # Запускаем комплексную оптимизацию
                self.brain.self_learning_system.comprehensive_optimization()
                logger.info("Запущена комплексная оптимизация системы")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка комплексной оптимизации: {e}")
        
        return False
    
    def get_learning_dashboard_data(self) -> Dict[str, Any]:
        """Возвращает данные для дашборда обучения"""
        dashboard = {
            'timestamp': time.time(),
            'effectiveness': {},
            'trends': {},
            'recommendations': [],
            'active_learning': []
        }
        
        try:
            # Анализ эффективности
            effectiveness = self.analyze_learning_effectiveness()
            dashboard['effectiveness'] = effectiveness
            
            # Тренды
            dashboard['trends'] = {
                'model_performance_trend': effectiveness.get('model_improvements', {}).get('trend', 'stable'),
                'knowledge_growth_trend': effectiveness.get('knowledge_expansion', {}).get('trend', 'stable'),
                'contradiction_resolution_trend': effectiveness.get('contradiction_resolution', {}).get('trend', 'stable')
            }
            
            # Рекомендации
            dashboard['recommendations'] = effectiveness.get('recommendations', [])
            
            # Активное обучение
            if self.analytics_manager and self.analytics_manager.learning_manager:
                opportunities = self.analytics_manager.learning_manager.get_learning_opportunities()
                dashboard['active_learning'] = opportunities[:5]  # Последние 5 возможностей
            
        except Exception as e:
            logger.error(f"Ошибка получения данных дашборда: {e}")
        
        return dashboard
    
    def track_learning_progress(self, learning_session_id: str) -> Dict[str, Any]:
        """Отслеживает прогресс обучения"""
        progress = {
            'session_id': learning_session_id,
            'timestamp': time.time(),
            'status': 'in_progress',
            'progress_percentage': 0.0,
            'metrics': {},
            'issues': []
        }
        
        try:
            # Получаем информацию о сессии обучения
            session_info = self._get_learning_session_info(learning_session_id)
            
            if session_info:
                progress['metrics'] = session_info.get('metrics', {})
                progress['progress_percentage'] = session_info.get('progress', 0.0)
                progress['status'] = session_info.get('status', 'in_progress')
                
                # Проверяем проблемы
                issues = session_info.get('issues', [])
                if issues:
                    progress['issues'] = issues
            
        except Exception as e:
            logger.error(f"Ошибка отслеживания прогресса обучения: {e}")
        
        return progress
    
    def _get_learning_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Получает информацию о сессии обучения"""
        try:
            if self.brain and hasattr(self.brain, 'self_learning_system'):
                return self.brain.self_learning_system.get_session_info(session_id)
        except Exception as e:
            logger.error(f"Ошибка получения информации о сессии {session_id}: {e}")
        
        return None
