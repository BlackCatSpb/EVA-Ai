"""
Модуль анализа графа знаний для ЕВА
Содержит функции анализа противоречий, пробелов и трендов
"""
import time
import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger("eva.knowledge_analytics")

from .knowledge_nodes import KnowledgeNode, KnowledgeEdge


class KnowledgeAnalytics:
    """Класс для анализа графа знаний."""
    
    def __init__(self, brain=None):
        """
        Инициализирует аналитическую систему.
        
        Args:
            brain: Ссылка на CoreBrain для интеграции
        """
        self.brain = brain
    
    def detect_contradictions(self, nodes: Dict[str, KnowledgeNode], 
                             edges: Dict[str, KnowledgeEdge]) -> List[Dict[str, Any]]:
        """
        Обнаруживает противоречия в графе знаний.
        
        Args:
            nodes: Словарь узлов
            edges: Словарь связей
            
        Returns:
            List[Dict[str, Any]]: Список обнаруженных противоречий
        """
        contradictions = []
        
        try:
            # Проверяем противоречия в связях
            for edge_id1, edge1 in edges.items():
                for edge_id2, edge2 in edges.items():
                    if edge_id1 >= edge_id2:  # Избегаем дублирования
                        continue
                    
                    # Проверяем противоположные отношения
                    if self._are_opposite_relations(edge1, edge2):
                        node1 = nodes.get(edge1.source_id)
                        node2 = nodes.get(edge1.target_id)
                        
                        if node1 and node2:
                            contradictions.append({
                                "type": "opposite_relations",
                                "edge1": edge_id1,
                                "edge2": edge_id2,
                                "node1": edge1.source_id,
                                "node2": edge1.target_id,
                                "relation1": edge1.relation_type,
                                "relation2": edge2.relation_type,
                                "evidence": f"Противоположные отношения: {edge1.relation_type} vs {edge2.relation_type}",
                                "severity": self._calculate_contradiction_severity(edge1, edge2)
                            })
            
            # Проверяем противоречия в узлах
            for node_id, node in nodes.items():
                if node.contradictions:
                    for contradiction in node.contradictions:
                        contradictions.append({
                            "type": "node_contradiction",
                            "node_id": node_id,
                            "contradiction": contradiction,
                            "severity": 0.7  # Средняя серьезность
                        })
            
            logger.info(f"Обнаружено {len(contradictions)} противоречий")
            
        except Exception as e:
            logger.error(f"Ошибка обнаружения противоречий: {e}", exc_info=True)
        
        return contradictions
    
    def _are_opposite_relations(self, edge1: KnowledgeEdge, edge2: KnowledgeEdge) -> bool:
        """
        Проверяет, являются ли отношения противоположными.
        
        Args:
            edge1: Первая связь
            edge2: Вторая связь
            
        Returns:
            bool: Являются ли отношения противоположными
        """
        # Определяем противоположные отношения
        opposites = {
            "supports": "contradicts",
            "causes": "prevents",
            "enables": "disables",
            "increases": "decreases",
            "improves": "worsens",
            "positive": "negative",
            "true": "false"
        }
        
        # Проверяем прямые противоположия
        if edge1.relation_type in opposites and edge2.relation_type == opposites[edge1.relation_type]:
            # Проверяем, что связи involve те же узлы
            if (edge1.source_id == edge2.source_id and edge1.target_id == edge2.target_id) or \
               (edge1.source_id == edge2.target_id and edge1.target_id == edge2.source_id):
                return True
        
        return False
    
    def _calculate_contradiction_severity(self, edge1: KnowledgeEdge, edge2: KnowledgeEdge) -> float:
        """
        Рассчитывает серьезность противоречия.
        
        Args:
            edge1: Первая связь
            edge2: Вторая связь
            
        Returns:
            float: Серьезность противоречия (0.0-1.0)
        """
        # Учитываем силу связей
        strength_factor = (edge1.strength + edge2.strength) / 2
        
        # Учитываем свежность
        time_factor = 1.0
        current_time = time.time()
        age1 = current_time - edge1.timestamp
        age2 = current_time - edge2.timestamp
        
        # Новые противоречия более серьезные
        if age1 < 86400 and age2 < 86400:  # Менее дня
            time_factor = 1.0
        elif age1 < 604800 and age2 < 604800:  # Менее недели
            time_factor = 0.8
        else:
            time_factor = 0.6
        
        return min(1.0, strength_factor * time_factor)
    
    def analyze_knowledge_gaps(self, nodes: Dict[str, KnowledgeNode], 
                              domain: str, num_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Анализирует пробелы в знаниях в указанной области.
        
        Args:
            nodes: Словарь узлов
            domain: Область знаний
            num_samples: Количество примеров для анализа
            
        Returns:
            List[Dict[str, Any]]: Выявленные пробелы в знаниях
        """
        gaps = []
        
        try:
            # Фильтруем узлы по домену
            domain_nodes = [node for node in nodes.values() if node.domain == domain]
            
            if len(domain_nodes) < num_samples:
                return gaps
            
            # Анализируем плотность связей
            weakly_connected = []
            outdated = []
            contradictory = []
            
            for node in domain_nodes[:num_samples]:
                # Проверяем слабые связи
                if node.strength < 0.3:
                    weakly_connected.append(node.id)
                
                # Проверяем устаревание
                if time.time() - node.last_updated > 365 * 86400:  # Год
                    outdated.append(node.id)
                
                # Проверяем противоречия
                if node.contradictions:
                    contradictory.append(node.id)
            
            # Формируем отчет о пробелах
            if weakly_connected:
                gaps.append({
                    "type": "weak_connections",
                    "description": f"Слабые связи в {len(weakly_connected)} концептах",
                    "affected_nodes": weakly_connected,
                    "severity": 0.5
                })
            
            if outdated:
                gaps.append({
                    "type": "outdated_knowledge",
                    "description": f"Устаревшие знания в {len(outdated)} концептах",
                    "affected_nodes": outdated,
                    "severity": 0.7
                })
            
            if contradictory:
                gaps.append({
                    "type": "contradictions",
                    "description": f"Противоречия в {len(contradictory)} концептах",
                    "affected_nodes": contradictory,
                    "severity": 0.9
                })
            
            logger.info(f"Выявлено {len(gaps)} пробелов в знаниях для домена {domain}")
            
        except Exception as e:
            logger.error(f"Ошибка анализа пробелов в знаниях: {e}", exc_info=True)
        
        return gaps
    
    def get_knowledge_trends(self, nodes: Dict[str, KnowledgeNode], 
                           domain: Optional[str] = None, 
                           period: str = "month") -> Dict[str, Any]:
        """
        Анализирует тренды в знаниях за указанный период.
        
        Args:
            nodes: Словарь узлов
            domain: Область знаний (опционально)
            period: Период анализа (day, week, month, year)
            
        Returns:
            Dict[str, Any]: Тренды в знаниях
        """
        # Определяем временные интервалы
        intervals = self._get_time_intervals(period)
        
        # Собираем данные по интервалам
        trends = {
            "intervals": intervals,
            "new_nodes": [0] * len(intervals),
            "updated_nodes": [0] * len(intervals),
            "contradictions": [0] * len(intervals),
            "sources": defaultdict(lambda: [0] * len(intervals))
        }
        
        # Анализируем узлы
        for node in nodes.values():
            if domain and node.domain != domain:
                continue
            
            # Определяем интервал для создания
            for i, (start, end) in enumerate(intervals):
                if start <= node.timestamp <= end:
                    trends["new_nodes"][i] += 1
                    
                    # Учитываем источник
                    for source in node.meta.get("sources", []):
                        source_name = source.get("source", "unknown")
                        trends["sources"][source_name][i] += 1
            
            # Определяем интервалы для обновлений
            for change in node.history:
                for i, (start, end) in enumerate(intervals):
                    if start <= change["timestamp"] <= end:
                        trends["updated_nodes"][i] += 1
            
            # Считаем противоречия
            for contradiction in node.contradictions:
                for i, (start, end) in enumerate(intervals):
                    if start <= contradiction["timestamp"] <= end:
                        trends["contradictions"][i] += 1
        
        # Преобразуем defaultdict в обычный dict
        trends["sources"] = dict(trends["sources"])
        
        return trends
    
    def _get_time_intervals(self, period: str) -> List[Tuple[float, float]]:
        """
        Возвращает временные интервалы для анализа трендов.
        
        Args:
            period: Период (day, week, month, year)
            
        Returns:
            List[Tuple[float, float]]: Список интервалов (начало, конец)
        """
        current_time = time.time()
        intervals = []
        
        if period == "day":
            # Последние 30 дней
            for i in range(29, -1, -1):
                start = current_time - (i + 1) * 86400
                end = current_time - i * 86400
                intervals.append((start, end))
        
        elif period == "week":
            # Последние 12 недель
            for i in range(11, -1, -1):
                start = current_time - (i + 1) * 7 * 86400
                end = current_time - i * 7 * 86400
                intervals.append((start, end))
        
        elif period == "month":
            # Последние 12 месяцев
            for i in range(11, -1, -1):
                # Приблизительный расчет (30 дней в месяце)
                start = current_time - (i + 1) * 30 * 86400
                end = current_time - i * 30 * 86400
                intervals.append((start, end))
        
        elif period == "year":
            # Последние 10 лет
            for i in range(9, -1, -1):
                start = current_time - (i + 1) * 365 * 86400
                end = current_time - i * 365 * 86400
                intervals.append((start, end))
        
        return intervals
    
    def get_domain_statistics(self, nodes: Dict[str, KnowledgeNode], 
                             domain: str) -> Dict[str, Any]:
        """
        Возвращает статистику по указанному домену знаний.
        
        Args:
            nodes: Словарь узлов
            domain: Домен знаний
            
        Returns:
            Dict[str, Any]: Статистика домена
        """
        domain_nodes = [node for node in nodes.values() if node.domain == domain]
        
        if not domain_nodes:
            return {"error": "domain_not_found"}
        
        # Рассчитываем статистику
        total_nodes = len(domain_nodes)
        avg_strength = sum(node.strength for node in domain_nodes) / total_nodes
        
        # Типы узлов
        node_types = defaultdict(int)
        for node in domain_nodes:
            node_types[node.node_type] += 1
        
        # Источники
        sources = defaultdict(int)
        for node in domain_nodes:
            for source in node.meta.get("sources", []):
                sources[source.get("source", "unknown")] += 1
        
        # Противоречия
        contradictory_nodes = sum(1 for node in domain_nodes if node.contradictions)
        
        # Устаревшие знания
        outdated_threshold = time.time() - 365 * 86400  # Год
        outdated_nodes = sum(1 for node in domain_nodes if node.last_updated < outdated_threshold)
        
        return {
            "domain": domain,
            "total_nodes": total_nodes,
            "avg_strength": avg_strength,
            "node_types": dict(node_types),
            "sources": dict(sources),
            "contradictory_nodes": contradictory_nodes,
            "outdated_nodes": outdated_nodes,
            "health_score": self._calculate_domain_health(
                total_nodes, contradictory_nodes, outdated_nodes, avg_strength
            )
        }
    
    def _calculate_domain_health(self, total_nodes: int, contradictory: int, 
                                outdated: int, avg_strength: float) -> float:
        """
        Рассчитывает показатель здоровья домена.
        
        Args:
            total_nodes: Общее количество узлов
            contradictory: Количество противоречивых узлов
            outdated: Количество устаревших узлов
            avg_strength: Средняя сила знаний
            
        Returns:
            float: Показатель здоровья (0.0-1.0)
        """
        if total_nodes == 0:
            return 0.0
        
        # Вес факторов
        weights = {
            "strength": 0.4,
            "consistency": 0.3,
            "freshness": 0.3
        }
        
        # Нормализованные показатели
        strength_score = avg_strength
        consistency_score = 1.0 - (contradictory / total_nodes)
        freshness_score = 1.0 - (outdated / total_nodes)
        
        # Общий показатель здоровья
        health_score = (
            strength_score * weights["strength"] +
            consistency_score * weights["consistency"] +
            freshness_score * weights["freshness"]
        )
        
        return max(0.0, min(1.0, health_score))
    
    def get_knowledge_density(self, nodes: Dict[str, KnowledgeNode], 
                             edges: Dict[str, KnowledgeEdge], 
                             domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Рассчитывает плотность знаний в графе.
        
        Args:
            nodes: Словарь узлов
            edges: Словарь связей
            domain: Домен знаний (опционально)
            
        Returns:
            Dict[str, Any]: Плотность знаний
        """
        # Фильтруем по домену
        if domain:
            domain_nodes = {nid: node for nid, node in nodes.items() if node.domain == domain}
            domain_edges = {eid: edge for eid, edge in edges.items() 
                          if nodes.get(edge.source_id, {}).domain == domain or 
                             nodes.get(edge.target_id, {}).domain == domain}
        else:
            domain_nodes = nodes
            domain_edges = edges
        
        node_count = len(domain_nodes)
        edge_count = len(domain_edges)
        
        if node_count == 0:
            return {"density": 0.0, "node_count": 0, "edge_count": 0}
        
        # Теоретическое максимальное количество связей
        max_edges = node_count * (node_count - 1) / 2
        
        # Плотность графа
        graph_density = edge_count / max_edges if max_edges > 0 else 0.0
        
        # Средняя степень связности
        avg_degree = (2 * edge_count) / node_count if node_count > 0 else 0.0
        
        return {
            "density": graph_density,
            "node_count": node_count,
            "edge_count": edge_count,
            "max_edges": max_edges,
            "avg_degree": avg_degree,
            "domain": domain or "all"
        }
    
    def generate_learning_recommendations(self, nodes: Dict[str, KnowledgeNode], 
                                        domain: str) -> List[Dict[str, Any]]:
        """
        Генерирует рекомендации для обучения на основе анализа знаний.
        
        Args:
            nodes: Словарь узлов
            domain: Домен знаний
            
        Returns:
            List[Dict[str, Any]]: Рекомендации для обучения
        """
        recommendations = []
        
        try:
            # Анализируем пробелы
            gaps = self.analyze_knowledge_gaps(nodes, domain)
            
            for gap in gaps:
                if gap["type"] == "weak_connections":
                    recommendations.append({
                        "type": "strengthen_connections",
                        "priority": "medium",
                        "description": "Укрепите слабые связи в концептах",
                        "affected_nodes": gap["affected_nodes"],
                        "actions": [
                            "Найдите дополнительные источники информации",
                            "Установите связи с родственными концептами",
                            "Проверьте актуальность информации"
                        ]
                    })
                
                elif gap["type"] == "outdated_knowledge":
                    recommendations.append({
                        "type": "update_knowledge",
                        "priority": "high",
                        "description": "Обновите устаревшие знания",
                        "affected_nodes": gap["affected_nodes"],
                        "actions": [
                            "Проверьте актуальность информации",
                            "Найдите свежие источники данных",
                            "Обновите информацию из новых источников"
                        ]
                    })
                
                elif gap["type"] == "contradictions":
                    recommendations.append({
                        "type": "resolve_contradictions",
                        "priority": "critical",
                        "description": "Разрешите противоречия в знаниях",
                        "affected_nodes": gap["affected_nodes"],
                        "actions": [
                            "Проанализируйте источники противоречий",
                            "Найдите дополнительные доказательства",
                            "Определите наиболее достоверную информацию"
                        ]
                    })
            
            # Добавляем общие рекомендации
            domain_stats = self.get_domain_statistics(nodes, domain)
            if domain_stats.get("health_score", 0) < 0.5:
                recommendations.append({
                    "type": "improve_domain_health",
                    "priority": "high",
                    "description": f"Улучшите здоровье домена {domain}",
                    "actions": [
                        "Увеличьте количество источников информации",
                        "Проверьте качество существующих знаний",
                        "Систематизируйте знания в домене"
                    ]
                })
            
            logger.info(f"Сгенерировано {len(recommendations)} рекомендаций для домена {domain}")
            
        except Exception as e:
            logger.error(f"Ошибка генерации рекомендаций: {e}", exc_info=True)
        
        return recommendations
