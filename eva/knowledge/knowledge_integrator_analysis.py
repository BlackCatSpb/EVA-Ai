"""
Методы анализа и расширения знаний для KnowledgeIntegrator
Часть модуля knowledge_integrator.py (разделение на логические компоненты)
"""
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np


class KnowledgeIntegratorAnalysis:
    """Методы анализа и расширения знаний."""
    
    def analyze_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Анализирует пробелы в знаниях."""
        gaps = []
        
        if not self.knowledge_graph:
            return gaps
        
        # Анализируем узлы с низкой связностью
        for node_id, node in self.knowledge_graph.nodes.items():
            connected_edges = []
            for edge in self.knowledge_graph.edges.values():
                if edge.source_id == node_id or edge.target_id == node_id:
                    connected_edges.append(edge)
            
            if len(connected_edges) < 2:
                gaps.append({
                    "node_id": node_id,
                    "node_name": node.name,
                    "gap_type": "low_connectivity",
                    "connected_count": len(connected_edges),
                    "suggestion": "Добавить больше связей с другими концептами"
                })
        
        return gaps
    
    def find_contradictions(self) -> List[Dict[str, Any]]:
        """Находит противоречия в знаниях."""
        contradictions = []
        
        if not self.knowledge_graph:
            return contradictions
        
        # Проверяем узлы с противоречивыми связями
        for node_id, node in self.knowledge_graph.nodes.items():
            if hasattr(node, 'contradictions') and node.contradictions:
                for contradiction in node.contradictions:
                    contradictions.append({
                        "node_id": node_id,
                        "contradicts_with": contradiction.get("node_id"),
                        "evidence": contradiction.get("evidence"),
                        "resolved": contradiction.get("resolved", False)
                    })
        
        return contradictions
    
    def calculate_consistency_score(self) -> float:
        """Рассчитывает показатель согласованности знаний."""
        if not self.knowledge_graph or not self.knowledge_graph.nodes:
            return 0.0
        
        score = 1.0
        
        # Штраф за противоречия
        contradiction_count = 0
        for node in self.knowledge_graph.nodes.values():
            if hasattr(node, 'contradictions'):
                contradiction_count += len(node.contradictions)
        
        if contradiction_count > 0:
            score -= min(0.5, contradiction_count * 0.1)
        
        # Штраф за изолированные узлы
        isolated_count = 0
        for node_id in self.knowledge_graph.nodes:
            has_edges = any(
                e.source_id == node_id or e.target_id == node_id
                for e in self.knowledge_graph.edges.values()
            )
            if not has_edges:
                isolated_count += 1
        
        total_nodes = len(self.knowledge_graph.nodes)
        if total_nodes > 0:
            isolation_ratio = isolated_count / total_nodes
            score -= isolation_ratio * 0.3
        
        return max(0.0, score)
    
    def suggest_improvements(self) -> List[Dict[str, Any]]:
        """Предлагает улучшения для базы знаний."""
        suggestions = []
        
        # Анализ пробелов
        gaps = self.analyze_knowledge_gaps()
        if gaps:
            suggestions.append({
                "type": "connectivity",
                "priority": "high" if len(gaps) > 10 else "medium",
                "description": f"Найдено {len(gaps)} узлов с низкой связностью",
                "action": "Добавить связи между изолированными концептами"
            })
        
        # Анализ противоречий
        contradictions = self.find_contradictions()
        unresolved = [c for c in contradictions if not c.get("resolved")]
        if unresolved:
            suggestions.append({
                "type": "consistency",
                "priority": "high",
                "description": f"Найдено {len(unresolved)} неразрешённых противоречий",
                "action": "Разрешить противоречия в знаниях"
            })
        
        # Показатель согласованности
        consistency = self.calculate_consistency_score()
        if consistency < 0.7:
            suggestions.append({
                "type": "overall",
                "priority": "medium",
                "description": f"Низкая согласованность знаний: {consistency:.2f}",
                "action": "Провести аудит базы знаний"
            })
        
        return suggestions
