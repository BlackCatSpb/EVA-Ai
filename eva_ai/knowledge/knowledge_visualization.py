"""
Модуль визуализации графа знаний для ЕВА
Содержит функции для визуализации и экспорта
"""
import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger("eva_ai.knowledge_visualization")

from .knowledge_nodes import KnowledgeNode, KnowledgeEdge


class KnowledgeVisualization:
    """Класс для визуализации графа знаний."""
    
    def __init__(self, nodes: Dict[str, KnowledgeNode], edges: Dict[str, KnowledgeEdge]):
        """
        Инициализирует систему визуализации.
        
        Args:
            nodes: Словарь узлов
            edges: Словарь связей
        """
        self.nodes = nodes
        self.edges = edges
    
    def generate_knowledge_graph(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """
        Генерирует граф знаний для визуализации.
        
        Args:
            concept: Концепт для построения графа
            depth: Глубина графа
            
        Returns:
            Dict[str, Any]: Граф знаний в формате для визуализации
        """
        # Ищем узел по концепту
        found_nodes = [node for node in self.nodes.values() 
                       if concept.lower() in node.name.lower() or 
                          concept.lower() in node.description.lower()]
        
        if not found_nodes:
            return {"nodes": [], "edges": []}
        
        center_node = found_nodes[0]
        
        # Получаем подграф
        subgraph = self.get_subgraph(center_node.id, depth)
        
        # Преобразуем для визуализации
        viz_nodes = []
        for node in subgraph["nodes"]:
            viz_nodes.append({
                "id": node["id"],
                "label": node["name"],
                "title": node["description"],
                "group": node["domain"],
                "shape": "ellipse",
                "font": {"size": 14},
                "color": self._get_node_color(node["node_type"]),
                "size": self._get_node_size(node["strength"])
            })
        
        viz_edges = []
        for edge in subgraph["edges"]:
            viz_edges.append({
                "from": edge["source_id"],
                "to": edge["target_id"],
                "label": edge["relation_type"],
                "arrows": "to",
                "font": {"align": "middle"},
                "color": self._get_edge_color(edge["relation_type"]),
                "width": self._get_edge_width(edge["strength"])
            })
        
        return {
            "nodes": viz_nodes,
            "edges": viz_edges,
            "center_node": center_node.id,
            "depth": depth
        }
    
    def get_subgraph(self, center_node_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Получает подграф вокруг указанного узла.
        
        Args:
            center_node_id: ID центрального узла
            depth: Глубина поиска
            
        Returns:
            Dict[str, Any]: Подграф
        """
        visited = set()
        nodes_to_visit = [center_node_id]
        subgraph_nodes = {}
        subgraph_edges = {}
        
        for current_depth in range(depth + 1):
            if not nodes_to_visit:
                break
            
            next_nodes = []
            
            for node_id in nodes_to_visit:
                if node_id in visited:
                    continue
                
                visited.add(node_id)
                
                # Добавляем узел в подграф
                if node_id in self.nodes:
                    subgraph_nodes[node_id] = self.nodes[node_id]
                
                # Находим связанные узлы
                for edge in self.edges.values():
                    if edge.source_id == node_id or edge.target_id == node_id:
                        subgraph_edges[edge.id] = edge
                        
                        # Добавляем связанные узлы для следующего уровня
                        if edge.source_id == node_id and edge.target_id not in visited:
                            next_nodes.append(edge.target_id)
                        elif edge.target_id == node_id and edge.source_id not in visited:
                            next_nodes.append(edge.source_id)
            
            nodes_to_visit = next_nodes
        
        return {
            "nodes": [node.to_dict() for node in subgraph_nodes.values()],
            "edges": [edge.to_dict() for edge in subgraph_edges.values()]
        }
    
    def _get_node_color(self, node_type: str) -> str:
        """Возвращает цвет узла в зависимости от типа."""
        colors = {
            "concept": "#e74c3c",
            "entity": "#3498db",
            "fact": "#2ecc71",
            "event": "#f39c12",
            "relation": "#9b59b6",
            "attribute": "#1abc9c",
            "process": "#34495e",
            "spatial": "#e67e22",
            "temporal": "#95a5a6",
            "other": "#7f8c8d"
        }
        return colors.get(node_type, "#95a5a6")
    
    def _get_node_size(self, strength: float) -> int:
        """Возвращает размер узла в зависимости от силы."""
        # Размер от 10 до 30
        return int(10 + strength * 20)
    
    def _get_edge_color(self, relation_type: str) -> str:
        """Возвращает цвет связи в зависимости от типа."""
        colors = {
            "is_a": "#e74c3c",
            "part_of": "#3498db",
            "has_property": "#2ecc71",
            "causes": "#f39c12",
            "supports": "#9b59b6",
            "used_for": "#1abc9c",
            "located_at": "#e67e22",
            "occurs_during": "#95a5a6",
            "precedes": "#34495e",
            "related_to": "#7f8c8d",
            "contradicts": "#c0392b",
            "similar_to": "#8e44ad",
            "depends_on": "#27ae60",
            "other": "#95a5a6"
        }
        return colors.get(relation_type, "#95a5a6")
    
    def _get_edge_width(self, strength: float) -> int:
        """Возвращает ширину связи в зависимости от силы."""
        # Ширина от 1 до 5
        return int(1 + strength * 4)
    
    def export_graph(self, format: str = "json") -> Any:
        """
        Экспортирует граф знаний в указанный формат.
        
        Args:
            format: Формат экспорта (json, graphml, gexf)
            
        Returns:
            Any: Экспортированные данные
        """
        if format == "json":
            return self._export_to_json()
        elif format == "graphml":
            return self._export_to_graphml()
        elif format == "gexf":
            return self._export_to_gexf()
        elif format == "csv":
            return self._export_to_csv()
        elif format == "dot":
            return self._export_to_dot()
        else:
            raise ValueError(f"Неподдерживаемый формат экспорта: {format}")
    
    def _export_to_json(self) -> Dict[str, Any]:
        """Экспортирует граф в формат JSON."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "metadata": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "export_timestamp": self._get_timestamp()
            }
        }
    
    def _export_to_csv(self) -> Dict[str, str]:
        """Экспортирует граф в формат CSV."""
        import csv
        import io
        
        # CSV для узлов
        nodes_csv = io.StringIO()
        nodes_writer = csv.writer(nodes_csv)
        nodes_writer.writerow([
            "id", "name", "description", "node_type", "domain", 
            "strength", "timestamp", "version"
        ])
        
        for node in self.nodes.values():
            nodes_writer.writerow([
                node.id, node.name, node.description, node.node_type,
                node.domain, node.strength, node.timestamp, node.version
            ])
        
        # CSV для связей
        edges_csv = io.StringIO()
        edges_writer = csv.writer(edges_csv)
        edges_writer.writerow([
            "id", "source_id", "target_id", "relation_type", 
            "strength", "timestamp", "version"
        ])
        
        for edge in self.edges.values():
            edges_writer.writerow([
                edge.id, edge.source_id, edge.target_id, edge.relation_type,
                edge.strength, edge.timestamp, edge.version
            ])
        
        return {
            "nodes": nodes_csv.getvalue(),
            "edges": edges_csv.getvalue()
        }
    
    def _export_to_dot(self) -> str:
        """Экспортирует граф в формат DOT (Graphviz)."""
        dot_lines = ["digraph knowledge_graph {"]
        dot_lines.append("  rankdir=LR;")
        dot_lines.append("  node [shape=ellipse];")
        dot_lines.append("")
        
        # Добавляем узлы
        for node in self.nodes.values():
            label = f"{node.name}\\n{node.domain}"
            color = self._get_node_color(node["node_type"]).replace("#", "")
            dot_lines.append(f'  "{node.id}" [label="{label}", fillcolor="#{color}", style=filled];')
        
        dot_lines.append("")
        
        # Добавляем связи
        for edge in self.edges.values():
            dot_lines.append(f'  "{edge.source_id}" -> "{edge.target_id}" [label="{edge.relation_type}"];')
        
        dot_lines.append("}")
        
        return "\n".join(dot_lines)
    
    def _export_to_graphml(self) -> str:
        """Экспортирует граф в формат GraphML."""
        graphml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
            '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns',
            '         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
            '<key id="name" for="node" attr.name="name" attr.type="string"/>',
            '<key id="description" for="node" attr.name="description" attr.type="string"/>',
            '<key id="node_type" for="node" attr.name="node_type" attr.type="string"/>',
            '<key id="domain" for="node" attr.name="domain" attr.type="string"/>',
            '<key id="strength" for="node" attr.name="strength" attr.type="double"/>',
            '<key id="relation_type" for="edge" attr.name="relation_type" attr.type="string"/>',
            '<key id="edge_strength" for="edge" attr.name="strength" attr.type="double"/>',
            '<graph id="knowledge_graph" edgedefault="directed">'
        ]
        
        # Добавляем узлы
        for node in self.nodes.values():
            graphml_lines.append(f'  <node id="{node.id}">')
            graphml_lines.append(f'    <data key="name">{node.name}</data>')
            graphml_lines.append(f'    <data key="description">{node.description}</data>')
            graphml_lines.append(f'    <data key="node_type">{node.node_type}</data>')
            graphml_lines.append(f'    <data key="domain">{node.domain}</data>')
            graphml_lines.append(f'    <data key="strength">{node.strength}</data>')
            graphml_lines.append(f'  </node>')
        
        # Добавляем связи
        for edge in self.edges.values():
            graphml_lines.append(f'  <edge source="{edge.source_id}" target="{edge.target_id}">')
            graphml_lines.append(f'    <data key="relation_type">{edge.relation_type}</data>')
            graphml_lines.append(f'    <data key="edge_strength">{edge.strength}</data>')
            graphml_lines.append(f'  </edge>')
        
        graphml_lines.extend([
            '</graph>',
            '</graphml>'
        ])
        
        return "\n".join(graphml_lines)
    
    def _export_to_gexf(self) -> str:
        """Экспортирует граф в формат GEXF."""
        gexf_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">',
            '<meta lastmodifieddate="2023-01-01">',
            '<creator>ЕВА</creator>',
            '<description>Knowledge Graph Export</description>',
            '</meta>',
            '<graph mode="static" defaultedgetype="directed">'
        ]
        
        # Атрибуты узлов
        gexf_lines.extend([
            '<attributes class="node">',
            '<attribute id="0" title="name" type="string"/>',
            '<attribute id="1" title="description" type="string"/>',
            '<attribute id="2" title="node_type" type="string"/>',
            '<attribute id="3" title="domain" type="string"/>',
            '<attribute id="4" title="strength" type="double"/>',
            '</attributes>',
            '<attributes class="edge">',
            '<attribute id="0" title="relation_type" type="string"/>',
            '<attribute id="1" title="strength" type="double"/>',
            '</attributes>'
        ])
        
        # Узлы
        gexf_lines.append('<nodes>')
        for node in self.nodes.values():
            gexf_lines.append(f'  <node id="{node.id}" label="{node.name}">')
            gexf_lines.append(f'    <attvalues>')
            gexf_lines.append(f'      <attvalue for="0" value="{node.name}"/>')
            gexf_lines.append(f'      <attvalue for="1" value="{node.description}"/>')
            gexf_lines.append(f'      <attvalue for="2" value="{node.node_type}"/>')
            gexf_lines.append(f'      <attvalue for="3" value="{node.domain}"/>')
            gexf_lines.append(f'      <attvalue for="4" value="{node.strength}"/>')
            gexf_lines.append(f'    </attvalues>')
            gexf_lines.append(f'  </node>')
        gexf_lines.append('</nodes>')
        
        # Связи
        gexf_lines.append('<edges>')
        edge_id = 0
        for edge in self.edges.values():
            gexf_lines.append(f'  <edge id="{edge_id}" source="{edge.source_id}" target="{edge.target_id}">')
            gexf_lines.append(f'    <attvalues>')
            gexf_lines.append(f'      <attvalue for="0" value="{edge.relation_type}"/>')
            gexf_lines.append(f'      <attvalue for="1" value="{edge.strength}"/>')
            gexf_lines.append(f'    </attvalues>')
            gexf_lines.append(f'  </edge>')
            edge_id += 1
        gexf_lines.append('</edges>')
        
        gexf_lines.extend([
            '</graph>',
            '</gexf>'
        ])
        
        return "\n".join(gexf_lines)
    
    def _get_timestamp(self) -> str:
        """Возвращает текущую временную метку."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def create_domain_overview(self, domain: str) -> Dict[str, Any]:
        """
        Создает обзор домена знаний.
        
        Args:
            domain: Домен знаний
            
        Returns:
            Dict[str, Any]: Обзор домена
        """
        domain_nodes = [node for node in self.nodes.values() if node.domain == domain]
        
        if not domain_nodes:
            return {"error": "domain_not_found"}
        
        # Статистика по типам узлов
        node_types = {}
        for node in domain_nodes:
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        
        # Топ узлов по силе
        top_nodes = sorted(domain_nodes, key=lambda x: x.strength, reverse=True)[:10]
        
        # Связи между узлами домена
        domain_edges = []
        for edge in self.edges.values():
            source_node = self.nodes.get(edge.source_id)
            target_node = self.nodes.get(edge.target_id)
            
            if source_node and target_node:
                if source_node.domain == domain or target_node.domain == domain:
                    domain_edges.append(edge)
        
        return {
            "domain": domain,
            "node_count": len(domain_nodes),
            "edge_count": len(domain_edges),
            "node_types": node_types,
            "top_nodes": [{"id": n.id, "name": n.name, "strength": n.strength} for n in top_nodes],
            "avg_strength": sum(n.strength for n in domain_nodes) / len(domain_nodes) if domain_nodes else 0.0,
            "visualization": self.generate_knowledge_graph(domain_nodes[0].name if domain_nodes else "", depth=2)
        }
    
    def create_timeline_visualization(self, days: int = 30) -> Dict[str, Any]:
        """
        Создает визуализацию временной шкалы знаний.
        
        Args:
            days: Количество дней для анализа
            
        Returns:
            Dict[str, Any]: Данные для визуализации временной шкалы
        """
        from datetime import datetime, timedelta
        import time
        
        current_time = time.time()
        start_time = current_time - (days * 86400)
        
        timeline_events = []
        
        # События создания узлов
        for node in self.nodes.values():
            if start_time <= node.timestamp <= current_time:
                timeline_events.append({
                    "timestamp": node.timestamp,
                    "date": datetime.fromtimestamp(node.timestamp).strftime("%Y-%m-%d"),
                    "type": "node_created",
                    "title": f"Создан узел: {node.name}",
                    "description": node.description[:100] + "..." if len(node.description) > 100 else node.description,
                    "domain": node.domain,
                    "node_type": node.node_type,
                    "strength": node.strength
                })
        
        # События обновления узлов
        for node in self.nodes.values():
            if start_time <= node.last_updated <= current_time and node.last_updated != node.timestamp:
                timeline_events.append({
                    "timestamp": node.last_updated,
                    "date": datetime.fromtimestamp(node.last_updated).strftime("%Y-%m-%d"),
                    "type": "node_updated",
                    "title": f"Обновлен узел: {node.name}",
                    "description": f"Версия {node.version}",
                    "domain": node.domain,
                    "node_type": node.node_type,
                    "strength": node.strength
                })
        
        # Сортируем по времени
        timeline_events.sort(key=lambda x: x["timestamp"])
        
        return {
            "events": timeline_events,
            "period_days": days,
            "total_events": len(timeline_events)
        }
