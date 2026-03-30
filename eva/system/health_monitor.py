"""Модуль мониторинга здоровья для ЕВА - анализ состояния системы"""
import os
import logging
import time
import re
import sqlite3
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger("eva.health_monitor")

class HealthMonitor:
    """Модуль мониторинга здоровья системы для ЕВА."""
    
    def __init__(self, brain=None, analyzer_core=None):
        """
        Инициализирует монитор здоровья.
        
        Args:
            brain: Ссылка на ядро ЕВА (опционально)
            analyzer_core: Ссылка на ядро самоанализа (опционально)
        """
        from eva.learning import AnalyzerCore
        
        self.brain = brain
        self.analyzer_core = analyzer_core or AnalyzerCore(brain)
        logger.info("HealthMonitor инициализирован")
    
    def analyze_system_health(self) -> Dict[str, Any]:
        """Анализирует здоровье системы."""
        try:
            report = {
                "health_score": 0.0,
                "status": "unknown",
                "components": {},
                "issues": [],
                "recommendations": [],
                "timestamp": time.time()
            }
            
            # Анализируем здоровье ML
            if self.brain and hasattr(self.brain, 'ml_core'):
                ml_health = self.brain.ml_core.get_system_health()
                report["components"]["ml"] = ml_health
                report["health_score"] += ml_health["health_score"] * 0.3
                
                # Проверяем проблемы с ML
                if ml_health["status"] == "critical":
                    report["issues"].append("Критическое состояние модуля машинного обучения")
                    self._add_ml_issue(ml_health)
                elif ml_health["status"] == "warning":
                    report["issues"].append("Проблемы с модулем машинного обучения")
            
            # Анализируем здоровье KnowledgeGraph
            if self.brain and hasattr(self.brain, 'knowledge_graph'):
                kg_health = self.brain.knowledge_graph.get_system_health()
                report["components"]["knowledge_graph"] = kg_health
                report["health_score"] += kg_health["health_score"] * 0.2
            
            # Анализируем здоровье MemoryManager
            if self.brain and hasattr(self.brain, 'memory_manager'):
                mm_health = self.brain.memory_manager.get_system_health()
                report["components"]["memory_manager"] = mm_health
                report["health_score"] += mm_health["health_score"] * 0.15
            
            # Анализируем здоровье AdaptationManager
            if self.brain and hasattr(self.brain, 'adaptation_manager'):
                am_health = self.brain.adaptation_manager.get_system_health()
                report["components"]["adaptation_manager"] = am_health
                report["health_score"] += am_health["health_score"] * 0.1
            
            # Анализируем здоровье EthicalFramework
            if self.brain and hasattr(self.brain, 'ethical_framework'):
                ef_health = self.brain.ethical_framework.get_system_health()
                report["components"]["ethical_framework"] = ef_health
                report["health_score"] += ef_health["health_score"] * 0.1
            
            # Анализируем здоровье NeuromorphicSimulator
            if self.brain and hasattr(self.brain, 'neuromorphic_simulator'):
                ns_health = self.brain.neuromorphic_simulator.get_system_health()
                report["components"]["neuromorphic_simulator"] = ns_health
                report["health_score"] += ns_health["health_score"] * 0.1
            
            # Анализируем здоровье WebSearchEngine
            if self.brain and hasattr(self.brain, 'web_search_engine'):
                wse_health = self.brain.web_search_engine.get_system_health()
                report["components"]["web_search_engine"] = wse_health
                report["health_score"] += wse_health["health_score"] * 0.05
            
            # Определяем общий статус
            if report["health_score"] > 0.7:
                report["status"] = "healthy"
            elif report["health_score"] > 0.4:
                report["status"] = "warning"
            else:
                report["status"] = "critical"
            
            # Генерируем рекомендации
            self._generate_recommendations(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Ошибка анализа состояния системы: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _add_ml_issue(self, ml_health: Dict[str, Any]):
        """Добавляет проблемы с ML как возможности для обучения."""
        # Добавляем возможность улучшения
        self.analyzer_core.add_learning_opportunity(
            "ml_optimization",
            "updating",
            0.8,
            "system",
            ["Здоровье ML ухудшается"],
            ["Обновить модели ML", "Проверить источники данных"]
        )
    
    def _generate_recommendations(self, report: Dict[str, Any]):
        """Генерирует рекомендации на основе отчета о здоровье."""
        # Рекомендации для ML
        if "ml" in report["components"]:
            ml_health = report["components"]["ml"]
            if ml_health["status"] == "critical":
                report["recommendations"].append("Срочно обновите модели машинного обучения")
            elif ml_health["status"] == "warning":
                report["recommendations"].append("Проверьте здоровье моделей машинного обучения")
        
        # Рекомендации для KnowledgeGraph
        if "knowledge_graph" in report["components"]:
            kg_health = report["components"]["knowledge_graph"]
            if kg_health["status"] == "critical":
                report["recommendations"].append("Проверьте целостность графа знаний")
            elif kg_health["status"] == "warning":
                report["recommendations"].append("Проанализируйте пробелы в знаниях")
        
        # Рекомендации для MemoryManager
        if "memory_manager" in report["components"]:
            mm_health = report["components"]["memory_manager"]
            if mm_health["status"] == "critical":
                report["recommendations"].append("Очистите кэш или увеличьте объем памяти")
            elif mm_health["status"] == "warning":
                report["recommendations"].append("Оптимизируйте использование памяти")
    
    def analyze_evolution(self) -> Dict[str, Any]:
        """Анализирует эволюцию системы со временем."""
        try:
            logger.info("Анализ эволюции системы...")
            
            # Получаем историю анализа
            history = self._get_analysis_history()
            
            # Анализируем тренды
            trends = self._analyze_trends(history)
            
            # Формируем отчет
            report = {
                "trends": trends,
                "historical_health": self._calculate_historical_health(history),
                "improvement_rate": self._calculate_improvement_rate(history),
                "critical_events": self._identify_critical_events(history),
                "timestamp": time.time()
            }
            
            # Добавляем возможность улучшения, если здоровье ухудшается
            trends = report.get("trends", {})
            if isinstance(trends, dict):
                system_health = trends.get("system_health", {})
                if isinstance(system_health, dict) and system_health.get("trend") == "decreasing":
                    self.analyzer_core.add_learning_opportunity(
                        "ml_optimization",
                        "updating",
                        0.8,
                        "system",
                        ["Здоровье ML ухудшается"],
                        ["Обновить модели ML", "Проверить источники данных"]
                    )
            
            # Сохраняем отчет
            self.analyzer_core.analysis_queue.put({
                "id": f"evolution_{int(time.time())}",
                "timestamp": time.time(),
                "findings": [
                    f"Тренд здоровья: {system_health.get('description', 'не определен') if isinstance(system_health, dict) else 'не определен'}"
                ],
                "recommendations": [],
                "metrics": {"evolution": report}
            })
            
            logger.info("Анализ эволюции системы завершен")
            return report
            
        except Exception as e:
            logger.error(f"Ошибка анализа эволюции системы: {e}")
            return {}
    
    def _get_analysis_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Получает историю анализа за указанный период."""
        try:
            conn = sqlite3.connect(self.analyzer_core.db_path)
            cursor = conn.cursor()
            
            # Получаем историю анализа
            cursor.execute('''
            SELECT id, timestamp, findings, recommendations, metrics 
            FROM analysis_history 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            ''', (time.time() - days * 86400,))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "findings": json.loads(row[2]),
                    "recommendations": json.loads(row[3]),
                    "metrics": json.loads(row[4])
                })
            
            conn.close()
            return history
            
        except Exception as e:
            logger.error(f"Ошибка получения истории анализа: {e}")
            return []
    
    def _analyze_trends(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализирует тренды на основе истории анализа."""
        trends = {}
        
        if not history:
            return trends
        
        # Анализируем тренд здоровья системы
        health_scores = []
        timestamps = []
        
        for entry in history:
            if "metrics" in entry and "evolution" in entry["metrics"]:
                if "system_health" in entry["metrics"]["evolution"]:
                    health_scores.append(entry["metrics"]["evolution"]["system_health"])
                    timestamps.append(entry["timestamp"])
        
        if health_scores:
            # Определяем тренд
            start_health = health_scores[-1]
            end_health = health_scores[0]
            change = end_health - start_health
            
            trend = "stable"
            description = "Здоровье системы стабильно"
            
            if change > 0.1:
                trend = "increasing"
                description = "Здоровье системы улучшается"
            elif change < -0.1:
                trend = "decreasing"
                description = "Здоровье системы ухудшается"
            
            trends["system_health"] = {
                "trend": trend,
                "description": description,
                "change": change,
                "start_value": start_health,
                "end_value": end_health
            }
        
        return trends
    
    def _calculate_historical_health(self, history: List[Dict[str, Any]]) -> float:
        """Рассчитывает историческое здоровье системы."""
        if not history:
            return 0.0
        
        total_health = 0.0
        count = 0
        
        for entry in history:
            if "metrics" in entry and "evolution" in entry["metrics"]:
                if "system_health" in entry["metrics"]["evolution"]:
                    total_health += entry["metrics"]["evolution"]["system_health"]
                    count += 1
        
        return total_health / count if count > 0 else 0.0
    
    def _calculate_improvement_rate(self, history: List[Dict[str, Any]]) -> float:
        """Рассчитывает скорость улучшения системы."""
        if len(history) < 2:
            return 0.0
        
        start_health = None
        end_health = None
        
        # Находим первый и последний отчеты
        for entry in reversed(history):
            if "metrics" in entry and "evolution" in entry["metrics"]:
                if "system_health" in entry["metrics"]["evolution"]:
                    start_health = entry["metrics"]["evolution"]["system_health"]
                    break
        
        for entry in history:
            if "metrics" in entry and "evolution" in entry["metrics"]:
                if "system_health" in entry["metrics"]["evolution"]:
                    end_health = entry["metrics"]["evolution"]["system_health"]
                    break
        
        if start_health is not None and end_health is not None:
            time_diff = history[0]["timestamp"] - history[-1]["timestamp"]
            if time_diff > 0:
                return (end_health - start_health) / (time_diff / 86400)  # на день
        
        return 0.0
    
    def _identify_critical_events(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Идентифицирует критические события в истории."""
        critical_events = []
        
        for entry in history:
            if "findings" in entry:
                for finding in entry["findings"]:
                    if "критическая" in finding.lower() or "ошибка" in finding.lower():
                        critical_events.append({
                            "timestamp": entry["timestamp"],
                            "description": finding,
                            "severity": "high"
                        })
        
        return critical_events
    
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
            
            # Дополнительный анализ для ML компонентов
            if component_name == "ml_core" and hasattr(component, "get_ml_health_dashboard_data"):
                ml_health = component.get_ml_health_dashboard_data()
                statistics = ml_health.get("statistics", {})
                if isinstance(statistics, dict):
                    analysis["performance_metrics"].update(statistics)
                
                # Проверяем время ответа
                if isinstance(statistics, dict) and statistics.get("avg_response_time", 0) > 2.0:
                    analysis["bottlenecks"].append("Высокое время ответа ML")
                    analysis["optimization_opportunities"].append({
                        "description": "Оптимизировать время ответа моделей ML",
                        "priority": 0.7
                    })
                
                # Проверяем процент ошибок
                failed = statistics.get("failed_requests", 0) if isinstance(statistics, dict) else 0
                total = statistics.get("total_requests", 1) if isinstance(statistics, dict) else 1
                error_rate = failed / total if total > 0 else 0
                
                if error_rate > 0.1:
                    analysis["bottlenecks"].append("Высокий процент ошибок в ML")
                    analysis["optimization_opportunities"].append({
                        "description": "Снизить процент ошибок в работе моделей ML",
                        "priority": 0.8
                    })
        
        except Exception as e:
            logger.error(f"Ошибка анализа производительности компонента {component_name}: {e}")
            analysis["health"] = "error"
            analysis["performance_metrics"]["error"] = str(e)
        
        return analysis