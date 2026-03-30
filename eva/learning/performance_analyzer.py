"""Модуль анализа производительности для ЕВА"""
import os
import logging
import time
import re
import sqlite3
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict
from eva.learning import AnalyzerCore

logger = logging.getLogger("eva.performance_analyzer")

class PerformanceAnalyzer:
    """Модуль анализа производительности системы для ЕВА."""
    
    def __init__(self, brain=None, analyzer_core=None):
        """
        Инициализирует анализатор производительности.
        
        Args:
            brain: Ссылка на ядро ЕВА (опционально)
            analyzer_core: Ссылка на ядро самоанализа (опционально)
        """
        self.brain = brain
        self.analyzer_core = analyzer_core or AnalyzerCore(brain)
        logger.info("PerformanceAnalyzer инициализирован")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Анализирует производительность системы."""
        logger.info("Анализ производительности системы...")
        try:
            # Проверяем, доступны ли компоненты
            if not self.brain:
                logger.warning("Ядро системы недоступно для анализа производительности")
                return {
                    "status": "error",
                    "message": "Ядро системы недоступно",
                    "performance_analysis": {}
                }
            
            analysis = {
                "component_performance": {},
                "bottlenecks": [],
                "optimization_opportunities": [],
                "timestamp": time.time()
            }
            
            # Анализируем компоненты системы
            for component_name, component in self._get_components().items():
                component_analysis = self._analyze_component_performance(component_name, component)
                analysis["component_performance"][component_name] = component_analysis
                
                # Собираем бутылочные горлышки
                if component_analysis["bottlenecks"]:
                    analysis["bottlenecks"].extend([
                        f"{component_name}: {bottleneck}" 
                        for bottleneck in component_analysis["bottlenecks"]
                    ])
                
                # Собираем возможности оптимизации
                for opportunity in component_analysis["optimization_opportunities"]:
                    analysis["optimization_opportunities"].append({
                        "component": component_name,
                        "description": opportunity["description"],
                        "priority": opportunity["priority"]
                    })
            
            # Сортируем возможности оптимизации по приоритету
            analysis["optimization_opportunities"].sort(
                key=lambda x: x["priority"], 
                reverse=True
            )
            
            # Добавляем возможности для обучения на основе анализа
            self._add_learning_opportunities(analysis)
            
            logger.info("Анализ производительности завершен")
            return analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа производительности: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _get_components(self) -> Dict[str, Any]:
        """Возвращает словарь компонентов системы для анализа."""
        components = {}
        
        # Добавляем компоненты, если они доступны
        if hasattr(self.brain, 'ml_core'):
            components["ml_core"] = self.brain.ml_core
            
        if hasattr(self.brain, 'model_manager'):
            components["model_manager"] = self.brain.model_manager
            
        if hasattr(self.brain, 'text_processor'):
            components["text_processor"] = self.brain.text_processor
            
        if hasattr(self.brain, 'response_generator'):
            components["response_generator"] = self.brain.response_generator
            
        if hasattr(self.brain, 'knowledge_graph'):
            components["knowledge_graph"] = self.brain.knowledge_graph
            
        if hasattr(self.brain, 'memory_manager'):
            components["memory_manager"] = self.brain.memory_manager
            
        if hasattr(self.brain, 'adaptation_manager'):
            components["adaptation_manager"] = self.brain.adaptation_manager
            
        if hasattr(self.brain, 'ethical_framework'):
            components["ethical_framework"] = self.brain.ethical_framework
            
        if hasattr(self.brain, 'neuromorphic_simulator'):
            components["neuromorphic_simulator"] = self.brain.neuromorphic_simulator
            
        if hasattr(self.brain, 'web_search_engine'):
            components["web_search_engine"] = self.brain.web_search_engine
            
        return components
    
    def _analyze_component_performance(self, component_name: str, component: Any) -> Dict[str, Any]:
        """Анализирует производительность компонента."""
        analysis = {
            "health": "unknown",
            "performance_metrics": {},
            "bottlenecks": [],
            "optimization_opportunities": []
        }
        
        try:
            # Проверяем наличие метода get_system_health
            if hasattr(component, "get_system_health"):
                health = component.get_system_health()
                analysis["health"] = health["status"]
                analysis["performance_metrics"] = health.get("statistics", {})
            
            # Анализируем проблемы
            if analysis["health"] == "critical":
                analysis["bottlenecks"].append(f"Критическое состояние компонента {component_name}")
                analysis["optimization_opportunities"].append({
                    "description": f"Требуется срочное восстановление компонента {component_name}",
                    "priority": 0.9
                })
            elif analysis["health"] == "warning":
                analysis["bottlenecks"].append(f"Проблемы с компонентом {component_name}")
                analysis["optimization_opportunities"].append({
                    "description": f"Требуется оптимизация компонента {component_name}",
                    "priority": 0.6
                })
            
            # Специфический анализ для ML компонентов
            if component_name in ["ml_core", "model_manager", "text_processor", "response_generator"]:
                self._analyze_ml_component(component_name, component, analysis)
            
            # Специфический анализ для KnowledgeGraph
            elif component_name == "knowledge_graph":
                self._analyze_knowledge_graph(component, analysis)
            
            # Специфический анализ для MemoryManager
            elif component_name == "memory_manager":
                self._analyze_memory_manager(component, analysis)
                
        except Exception as e:
            logger.error(f"Ошибка анализа производительности компонента {component_name}: {e}")
            analysis["health"] = "error"
            analysis["performance_metrics"]["error"] = str(e)
        
        return analysis
    
    def _analyze_ml_component(self, component_name: str, component: Any, analysis: Dict[str, Any]):
        """Анализирует компоненты машинного обучения."""
        # Проверяем время ответа
        if "avg_response_time" in analysis["performance_metrics"]:
            if analysis["performance_metrics"]["avg_response_time"] > 2.0:
                analysis["bottlenecks"].append("Высокое время ответа")
                analysis["optimization_opportunities"].append({
                    "description": "Оптимизировать время ответа моделей",
                    "priority": 0.7
                })
        
        # Проверяем процент ошибок
        failed = analysis["performance_metrics"].get("failed_requests", 0)
        total = analysis["performance_metrics"].get("total_requests", 1)
        error_rate = failed / total if total > 0 else 0
        
        if error_rate > 0.1:
            analysis["bottlenecks"].append("Высокий процент ошибок")
            analysis["optimization_opportunities"].append({
                "description": "Снизить процент ошибок в работе моделей",
                "priority": 0.8
            })
        
        # Проверяем кэш
        if component_name == "ml_core" and hasattr(component, "response_cache"):
            cache_size = len(component.response_cache)
            cache_limit = component.cache_size
            
            if cache_size > cache_limit * 0.9:
                analysis["bottlenecks"].append("Переполненный кэш")
                analysis["optimization_opportunities"].append({
                    "description": "Оптимизировать использование кэша",
                    "priority": 0.6
                })
    
    def _analyze_knowledge_graph(self, component: Any, analysis: Dict[str, Any]):
        """Анализирует граф знаний."""
        # Проверяем размер графа
        if hasattr(component, "get_statistics"):
            stats = component.get_statistics()
            
            # Проверяем количество узлов
            if stats.get("node_count", 0) < 100:
                analysis["bottlenecks"].append("Малое количество узлов в графе знаний")
                analysis["optimization_opportunities"].append({
                    "description": "Расширить граф знаний",
                    "priority": 0.5
                })
            
            # Проверяем плотность связей
            node_count = stats.get("node_count", 1)
            edge_count = stats.get("edge_count", 0)
            density = edge_count / node_count if node_count > 0 else 0
            
            if density < 1.5:
                analysis["bottlenecks"].append("Низкая плотность связей в графе знаний")
                analysis["optimization_opportunities"].append({
                    "description": "Увеличить плотность связей в графе знаний",
                    "priority": 0.4
                })
    
    def _analyze_memory_manager(self, component: Any, analysis: Dict[str, Any]):
        """Анализирует менеджер памяти."""
        # Проверяем использование памяти
        if hasattr(component, "get_statistics"):
            stats = component.get_statistics()
            
            # Проверяем заполненность кэша
            cache_usage = stats.get("cache_usage", 0)
            if cache_usage > 0.9:
                analysis["bottlenecks"].append("Высокая заполненность кэша")
                analysis["optimization_opportunities"].append({
                    "description": "Оптимизировать использование кэша",
                    "priority": 0.6
                })
            
            # Проверяем количество промахов кэша
            cache_misses = stats.get("cache_misses", 0)
            cache_hits = stats.get("cache_hits", 1)
            miss_rate = cache_misses / (cache_misses + cache_hits)
            
            if miss_rate > 0.3:
                analysis["bottlenecks"].append("Высокий процент промахов кэша")
                analysis["optimization_opportunities"].append({
                    "description": "Улучшить стратегию кэширования",
                    "priority": 0.7
                })
    
    def _add_learning_opportunities(self, analysis: Dict[str, Any]):
        """Добавляет возможности для обучения на основе анализа производительности."""
        # Добавляем возможности для критических проблем
        for bottleneck in analysis["bottlenecks"]:
            component_match = re.search(r"(\w+):", bottleneck)
            if component_match:
                component_name = component_match.group(1)
                
                # Определяем тип проблемы
                if "Критическое состояние" in bottleneck:
                    self.analyzer_core.add_learning_opportunity(
                        f"{component_name}_optimization",
                        "updating",
                        0.9,
                        "system",
                        [bottleneck],
                        [
                            f"Оптимизировать {component_name}",
                            "Провести диагностику"
                        ]
                    )
                elif "Проблемы с" in bottleneck:
                    self.analyzer_core.add_learning_opportunity(
                        f"{component_name}_optimization",
                        "updating",
                        0.6,
                        "system",
                        [bottleneck],
                        [
                            f"Улучшить {component_name}",
                            "Провести настройку"
                        ]
                    )
        
        # Добавляем возможности для оптимизации
        for opportunity in analysis["optimization_opportunities"]:
            self.analyzer_core.add_learning_opportunity(
                f"optimization_{opportunity['component']}",
                "refinement",
                opportunity["priority"],
                "system",
                [opportunity["description"]],
                [opportunity["description"].replace("Оптимизировать", "Выполнить оптимизацию")]
            )
    
    def analyze_user_feedback(self) -> Dict[str, Any]:
        """Анализирует пользовательский фидбэк для выявления проблем."""
        logger.info("Анализ пользовательского фидбэка...")
        try:
            # Проверяем, доступен ли AdaptationManager
            if not self.brain or not hasattr(self.brain, 'adaptation_manager'):
                logger.warning("AdaptationManager недоступен для анализа фидбэка")
                return {
                    "status": "error",
                    "message": "AdaptationManager недоступен",
                    "feedback_analysis": {}
                }
            
            # Получаем фидбэк из AdaptationManager
            feedback = self.brain.adaptation_manager.get_feedback_history()
            
            if feedback is None:
                feedback = []
            
            feedback_to_analyze = [
                fb for fb in feedback
                if isinstance(fb, dict) and fb.get("timestamp") is not None and fb.get("feedback_type") is not None
                and fb["timestamp"] > time.time() - 86400 * 7
                and fb["feedback_type"] == "negative"
            ]
            
            # Группируем фидбэк по концептам
            concept_corrections = defaultdict(list)
            for corr in feedback_to_analyze:
                concept_corrections[corr["concept"]].append(corr)
            
            # Анализируем проблемы
            analysis = {
                "total_negative_feedback": len(feedback_to_analyze),
                "concepts_with_issues": len(concept_corrections),
                "issues_by_concept": {},
                "recommendations": []
            }
            
            for concept, corrections in concept_corrections.items():
                # Анализируем типы проблем
                issue_types = defaultdict(int)
                for corr in corrections:
                    issue_types[corr["correction_type"]] += 1
                
                # Определяем основные проблемы
                primary_issue = max(issue_types, key=issue_types.get) if issue_types else "unknown"
                issue_count = issue_types.get(primary_issue, 0)
                
                # Рассчитываем приоритет
                priority = min(1.0, issue_count / len(feedback_to_analyze) * 1.5)
                
                # Сохраняем анализ
                analysis["issues_by_concept"][concept] = {
                    "primary_issue": primary_issue,
                    "issue_count": issue_count,
                    "total_feedback": len(corrections),
                    "priority": priority
                }
                
                # Добавляем возможность для обучения
                self.analyzer_core.add_learning_opportunity(
                    concept,
                    "refinement",
                    priority,
                    "user_experience",
                    [f"Пользователи сообщают о проблемах с концептом {concept}"],
                    [
                        f"Улучшить объяснение концепта {concept}",
                        "Добавить примеры использования"
                    ]
                )
            
            # Генерируем рекомендации
            if analysis["concepts_with_issues"] > 0:
                analysis["recommendations"].append(
                    f"Сфокусироваться на {analysis['concepts_with_issues']} концептах с наибольшим количеством негативного фидбэка"
                )
            
            logger.info(f"Анализ пользовательского фидбэка завершен. Найдено {analysis['concepts_with_issues']} проблемных концептов")
            return {
                "status": "success",
                "feedback_analysis": analysis,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа пользовательского фидбэка: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }