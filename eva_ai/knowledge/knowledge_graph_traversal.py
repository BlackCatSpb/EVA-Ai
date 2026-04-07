"""
Модуль обхода графа знаний для ЕВА
Содержит методы поиска путей, временных и пространственных отношений, экспорта
"""
import logging
import time
import math
from typing import Dict, List, Optional, Any
from collections import deque
from datetime import datetime

logger = logging.getLogger("eva_ai.knowledge_graph")


class KnowledgeGraphTraversalMixin:
    """Mixin класс с методами обхода графа для KnowledgeGraph."""
    
    def find_path(self, start_node_id: str, end_node_id: str, 
                 max_length: int = 5) -> List[List[str]]:
        """Находит пути между двумя узлами в графе."""
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            return []
        
        paths = []
        queue = deque([(start_node_id, [start_node_id])])
        visited = set([start_node_id])
        
        while queue:
            current_node, path = queue.popleft()
            
            if current_node == end_node_id:
                paths.append(path)
                if len(paths) >= 10:
                    break
                continue
            
            if len(path) >= max_length:
                continue
            
            related_nodes = self.get_related_nodes(current_node)
            for node in related_nodes:
                if node.id not in visited:
                    visited.add(node.id)
                    queue.append((node.id, path + [node.id]))
        
        return paths
    
    def get_temporal_relations(self, start_time: Optional[float] = None, 
                              end_time: Optional[float] = None,
                              limit: int = 20) -> List[Dict[str, Any]]:
        """Возвращает временные отношения в указанном временном интервале."""
        start_time = start_time or 0
        end_time = end_time or time.time()
        
        results = []
        
        for timestamp, item_id, item_type in self.temporal_index:
            if timestamp < start_time:
                continue
            if timestamp > end_time:
                break
            
            if item_type == "node":
                node = self.nodes.get(item_id)
                if node and node.temporal_info:
                    results.append({
                        "type": "node",
                        "id": item_id,
                        "name": node.name,
                        "timestamp": timestamp,
                        "temporal_info": node.temporal_info,
                        "domain": node.domain
                    })
            elif item_type == "edge":
                edge = self.edges.get(item_id)
                if edge and edge.temporal_info:
                    results.append({
                        "type": "edge",
                        "id": item_id,
                        "relation_type": edge.relation_type,
                        "timestamp": timestamp,
                        "temporal_info": edge.temporal_info
                    })
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_spatial_relations(self, location: Dict[str, float], 
                             max_distance: float = 100.0,
                             limit: int = 20) -> List[Dict[str, Any]]:
        """Возвращает пространственные отношения в указанном радиусе."""
        results = []
        
        for node in self.nodes.values():
            if not node.spatial_info or "coordinates" not in node.spatial_info:
                continue
            
            node_coords = node.spatial_info["coordinates"]
            distance = self._calculate_distance(
                location["lat"], location["lon"],
                node_coords["lat"], node_coords["lon"]
            )
            
            if distance <= max_distance:
                results.append({
                    "type": "node",
                    "id": node.id,
                    "name": node.name,
                    "distance": distance,
                    "spatial_info": node.spatial_info,
                    "domain": node.domain
                })
        
        for edge in self.edges.values():
            if not edge.spatial_info or "coordinates" not in edge.spatial_info:
                continue
            
            edge_coords = edge.spatial_info["coordinates"]
            distance = self._calculate_distance(
                location["lat"], location["lon"],
                edge_coords["lat"], edge_coords["lon"]
            )
            
            if distance <= max_distance:
                results.append({
                    "type": "edge",
                    "id": edge.id,
                    "relation_type": edge.relation_type,
                    "distance": distance,
                    "spatial_info": edge.spatial_info
                })
        
        results.sort(key=lambda x: x["distance"])
        
        return results[:limit]
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Вычисляет расстояние между двумя точками на сфере с использованием формулы Haversine."""
        R = 6371.0
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def generate_knowledge_graph(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """Генерирует граф знаний для визуализации."""
        nodes = self.search_nodes(concept, limit=1)
        if not nodes:
            return {"nodes": [], "edges": []}
        
        center_node = nodes[0]
        
        subgraph = self.get_subgraph(center_node.id, depth)
        
        viz_nodes = []
        for node in subgraph["nodes"]:
            viz_nodes.append({
                "id": node["id"],
                "label": node["name"],
                "title": node["description"],
                "group": node["domain"],
                "shape": "ellipse",
                "font": {"size": 14}
            })
        
        viz_edges = []
        for edge in subgraph["edges"]:
            viz_edges.append({
                "from": edge["source_id"],
                "to": edge["target_id"],
                "label": edge["relation_type"],
                "arrows": "to",
                "font": {"align": "middle"}
            })
        
        return {
            "nodes": viz_nodes,
            "edges": viz_edges
        }
    
    def export_graph(self, format: str = "json") -> Any:
        """Экспортирует граф знаний в указанный формат."""
        if format == "json":
            return {
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "edges": [edge.to_dict() for edge in self.edges.values()]
            }
        elif format == "graphml":
            return self._export_to_graphml()
        elif format == "gexf":
            return self._export_to_gexf()
        else:
            raise ValueError(f"Неподдерживаемый формат экспорта: {format}")
    
    def _export_to_graphml(self) -> str:
        """Экспортирует граф в формат GraphML."""
        from xml.etree.ElementTree import Element, SubElement, tostring
        import xml.dom.minidom
        
        graphml = Element('graphml', {
            'xmlns': 'http://graphml.graphdrawing.org/xmlns',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': 'http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd'
        })
        
        key_node = SubElement(graphml, 'key', {
            'id': 'd0',
            'for': 'node',
            'attr.name': 'description',
            'attr.type': 'string'
        })
        SubElement(key_node, 'default').text = ''
        
        key_domain = SubElement(graphml, 'key', {
            'id': 'd1',
            'for': 'node',
            'attr.name': 'domain',
            'attr.type': 'string'
        })
        SubElement(key_domain, 'default').text = 'general'
        
        key_strength = SubElement(graphml, 'key', {
            'id': 'd2',
            'for': 'node',
            'attr.name': 'strength',
            'attr.type': 'double'
        })
        SubElement(key_strength, 'default').text = '0.5'
        
        key_relation = SubElement(graphml, 'key', {
            'id': 'd3',
            'for': 'edge',
            'attr.name': 'relation_type',
            'attr.type': 'string'
        })
        SubElement(key_relation, 'default').text = 'related_to'
        
        graph = SubElement(graphml, 'graph', {
            'id': 'G',
            'edgedefault': 'directed'
        })
        
        for node in self.nodes.values():
            node_elem = SubElement(graph, 'node', {'id': node.id})
            SubElement(node_elem, 'data', {'key': 'd0'}).text = node.description
            SubElement(node_elem, 'data', {'key': 'd1'}).text = node.domain
            SubElement(node_elem, 'data', {'key': 'd2'}).text = str(node.strength)
        
        for edge in self.edges.values():
            edge_elem = SubElement(graph, 'edge', {
                'id': edge.id,
                'source': edge.source_id,
                'target': edge.target_id
            })
            SubElement(edge_elem, 'data', {'key': 'd3'}).text = edge.relation_type
        
        xml_str = tostring(graphml, encoding='utf-8')
        dom = xml.dom.minidom.parseString(xml_str)
        return dom.toprettyxml()
    
    def _export_to_gexf(self) -> str:
        """Экспортирует граф в формат GEXF."""
        from xml.etree.ElementTree import Element, SubElement, tostring
        import xml.dom.minidom
        
        gexf = Element('gexf', {
            'xmlns': 'http://www.gexf.net/1.2draft',
            'version': '1.2'
        })
        
        meta = SubElement(gexf, 'meta', {'lastmodified': datetime.now().isoformat()})
        SubElement(meta, 'creator').text = 'ЕВА Knowledge Graph'
        SubElement(meta, 'description').text = 'Graph of knowledge exported from ЕВА'
        
        SubElement(gexf, 'visualization')
        
        graph = SubElement(gexf, 'graph', {
            ' defaultedgetype': 'directed',
            'mode': 'static'
        })
        
        node_attributes = SubElement(graph, 'attributes', {'class': 'node'})
        SubElement(node_attributes, 'attribute', {
            'id': 'description',
            'title': 'Description',
            'type': 'string'
        })
        SubElement(node_attributes, 'attribute', {
            'id': 'domain',
            'title': 'Domain',
            'type': 'string'
        })
        SubElement(node_attributes, 'attribute', {
            'id': 'strength',
            'title': 'Strength',
            'type': 'float'
        })
        
        edge_attributes = SubElement(graph, 'attributes', {'class': 'edge'})
        SubElement(edge_attributes, 'attribute', {
            'id': 'relation_type',
            'title': 'Relation Type',
            'type': 'string'
        })
        
        nodes_elem = SubElement(graph, 'nodes')
        for node in self.nodes.values():
            node_elem = SubElement(nodes_elem, 'node', {
                'id': node.id,
                'label': node.name
            })
            attvalues = SubElement(node_elem, 'attvalues')
            SubElement(attvalues, 'attvalue', {
                'for': 'description',
                'value': node.description
            })
            SubElement(attvalues, 'attvalue', {
                'for': 'domain',
                'value': node.domain
            })
            SubElement(attvalues, 'attvalue', {
                'for': 'strength',
                'value': str(node.strength)
            })
        
        edges_elem = SubElement(graph, 'edges')
        for i, edge in enumerate(self.edges.values()):
            edge_elem = SubElement(edges_elem, 'edge', {
                'id': str(i),
                'source': edge.source_id,
                'target': edge.target_id
            })
            attvalues = SubElement(edge_elem, 'attvalues')
            SubElement(attvalues, 'attvalue', {
                'for': 'relation_type',
                'value': edge.relation_type
            })
        
        xml_str = tostring(gexf, encoding='utf-8')
        dom = xml.dom.minidom.parseString(xml_str)
        return dom.toprettyxml()
    
    def import_graph(self, data: Any, format: str = "json") -> bool:
        """Импортирует граф знаний из указанного формата."""
        try:
            if format == "json":
                return self._import_from_json(data)
            elif format == "graphml":
                return self._import_from_graphml(data)
            elif format == "gexf":
                return self._import_from_gexf(data)
            else:
                raise ValueError(f"Неподдерживаемый формат импорта: {format}")
        except Exception as e:
            logger.error(f"Ошибка импорта графа: {e}", exc_info=True)
            return False
    
    def _import_from_json(self, data: Dict[str, Any]) -> bool:
        """Импортирует граф из JSON."""
        from eva_ai.knowledge.knowledge_graph_types import KnowledgeNode, KnowledgeEdge
        
        nodes = data.get("nodes", [])
        for node_data in nodes:
            node = KnowledgeNode.from_dict(node_data)
            self.nodes[node.id] = node
            self._save_node_to_db(node)
        
        edges = data.get("edges", [])
        for edge_data in edges:
            edge = KnowledgeEdge.from_dict(edge_data)
            self.edges[edge.id] = edge
            self._save_edge_to_db(edge)
        
        self._init_indexes()
        
        logger.info(f"Импортировано {len(nodes)} узлов и {len(edges)} связей")
        return True
    
    def _import_from_graphml(self, data: str) -> bool:
        """Импортирует граф из GraphML."""
        import xml.etree.ElementTree as ET
        
        root = ET.fromstring(data)
        namespace = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
        
        graph = root.find('graphml:graph', namespace)
        if graph is None:
            raise ValueError("Не найден элемент графа в GraphML")
        
        for node in graph.findall('graphml:node', namespace):
            node_id = node.get('id')
            
            description = ""
            domain = "general"
            strength = 0.5
            
            for data_elem in node.findall('graphml:data', namespace):
                key = data_elem.get('key')
                if key == 'd0':
                    description = data_elem.text or ""
                elif key == 'd1':
                    domain = data_elem.text or "general"
                elif key == 'd2':
                    try:
                        strength = float(data_elem.text or "0.5")
                    except ValueError:
                        strength = 0.5
            
            name = node_id
            self.add_node(name, description, domain=domain, strength=strength)
        
        for edge in graph.findall('graphml:edge', namespace):
            source_id = edge.get('source', edge.get('source_id', ''))
            target_id = edge.get('target', edge.get('target_id', ''))
            edge_id = edge.get('id', f"edge_{hash(edge)}")
            
            relation_type = "related_to"
            for data_elem in edge.findall('graphml:data', namespace):
                if data_elem.get('key') == 'd3':
                    relation_type = data_elem.text or "related_to"
            
            self.add_edge(source_id, target_id, relation_type)
        
        logger.info("Граф успешно импортирован из GraphML")
        return True
    
    def _import_from_gexf(self, data: str) -> bool:
        """Импортирует граф из GEXF."""
        import xml.etree.ElementTree as ET
        
        root = ET.fromstring(data)
        namespace = {'gexf': 'http://www.gexf.net/1.2draft'}
        
        graph = root.find('gexf:graph', namespace)
        if graph is None:
            raise ValueError("Не найден элемент графа в GEXF")
        
        nodes = graph.find('gexf:nodes', namespace)
        if nodes is not None:
            for node in nodes.findall('gexf:node', namespace):
                node_id = node.get('id')
                label = node.get('label', node_id)
                
                description = ""
                domain = "general"
                strength = 0.5
                
                attvalues = node.find('gexf:attvalues', namespace)
                if attvalues is not None:
                    for attvalue in attvalues.findall('gexf:attvalue', namespace):
                        for_attr = attvalue.get('for')
                        value = attvalue.get('value', '')
                        
                        if for_attr == 'description':
                            description = value
                        elif for_attr == 'domain':
                            domain = value
                        elif for_attr == 'strength':
                            try:
                                strength = float(value)
                            except ValueError:
                                strength = 0.5
                
                self.add_node(label, description, domain=domain, strength=strength)
        
        edges = graph.find('gexf:edges', namespace)
        if edges is not None:
            for edge in edges.findall('gexf:edge', namespace):
                source_id = edge.get('source')
                target_id = edge.get('target')
                
                relation_type = "related_to"
                attvalues = edge.find('gexf:attvalues', namespace)
                if attvalues is not None:
                    for attvalue in attvalues.findall('gexf:attvalue', namespace):
                        if attvalue.get('for') == 'relation_type':
                            relation_type = attvalue.get('value', 'related_to')
                
                self.add_edge(source_id, target_id, relation_type)
        
        logger.info("Граф успешно импортирован из GEXF")
        return True
    
    def analyze_knowledge_gaps(self, domain: str, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Анализирует пробелы в знаниях в указанной области."""
        try:
            domain_nodes = self.get_nodes_by_domain(domain)
            
            gaps = []
            node_ids = [n.id for n in domain_nodes]
            
            for node in domain_nodes[:num_samples]:
                connected = set()
                for edge in self.get_edges(node.id):
                    connected.add(edge.source_id)
                    connected.add(edge.target_id)
                
                unconnected = [nid for nid in node_ids if nid != node.id and nid not in connected]
                
                if len(unconnected) > 3:
                    gaps.append({
                        "node_id": node.id,
                        "name": node.name,
                        "potential_connections": len(unconnected),
                        "unconnected_nodes": unconnected[:5]
                    })
            
            return gaps
            
        except Exception as e:
            logger.error(f"Ошибка анализа пробелов в знаниях: {e}", exc_info=True)
            return []
    
    def find_central_nodes(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Находит центральные узлы в графе (по количеству связей)."""
        try:
            node_scores = {}
            
            nodes_to_check = self.nodes.values() if domain is None else self.get_nodes_by_domain(domain)
            
            for node in nodes_to_check:
                edges = self.get_edges(node.id)
                node_scores[node.id] = {
                    "id": node.id,
                    "name": node.name,
                    "degree": len(edges),
                    "domain": node.domain
                }
            
            sorted_nodes = sorted(node_scores.values(), key=lambda x: x["degree"], reverse=True)
            return sorted_nodes[:20]
            
        except Exception as e:
            logger.error(f"Ошибка поиска центральных узлов: {e}", exc_info=True)
            return []
    
    def find_islands(self, min_degree: int = 1) -> List[Dict[str, Any]]:
        """Находит изолированные узлы или группы (островные узлы)."""
        try:
            islands = []
            
            for node in self.nodes.values():
                edges = self.get_edges(node.id)
                if len(edges) <= min_degree:
                    islands.append({
                        "node_id": node.id,
                        "name": node.name,
                        "degree": len(edges),
                        "domain": node.domain
                    })
            
            islands.sort(key=lambda x: x["degree"])
            return islands
            
        except Exception as e:
            logger.error(f"Ошибка поиска изолированных узлов: {e}", exc_info=True)
            return []
    
    def calculate_node_importance(self, node_id: str) -> Dict[str, float]:
        """Вычисляет важность узла различными методами."""
        try:
            node = self.get_node(node_id)
            if not node:
                return {}
            
            degree = len(self.get_edges(node_id))
            
            in_degree = len(self.get_edges(node_id, direction="target"))
            out_degree = len(self.get_edges(node_id, direction="source"))
            
            related = self.get_related_nodes(node_id, max_distance=2)
            indirect_connections = len(related)
            
            centrality = {
                "degree_centrality": degree / max(1, len(self.nodes) - 1),
                "in_degree": in_degree,
                "out_degree": out_degree,
                "indirect_connections": indirect_connections,
                "normalized_centrality": degree / max(1, sum(len(self.get_edges(n.id)) for n in self.nodes.values()))
            }
            
            return centrality
            
        except Exception as e:
            logger.error(f"Ошибка вычисления важности узла: {e}", exc_info=True)
            return {}
    
    def find_communities(self) -> List[List[str]]:
        """Находит сообщества в графе (упрощенная реализация)."""
        try:
            visited = set()
            communities = []
            
            for node_id in self.nodes:
                if node_id in visited:
                    continue
                
                community = []
                queue = deque([node_id])
                
                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    
                    visited.add(current)
                    community.append(current)
                    
                    for edge in self.get_edges(current):
                        neighbor = edge.target_id if edge.source_id == current else edge.source_id
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                if community:
                    communities.append(community)
            
            return communities
            
        except Exception as e:
            logger.error(f"Ошибка поиска сообществ: {e}", exc_info=True)
            return []
    
    def get_graph_density(self) -> float:
        """Вычисляет плотность графа."""
        try:
            n = len(self.nodes)
            if n < 2:
                return 0.0
            
            max_edges = n * (n - 1)
            actual_edges = len(self.edges)
            
            return actual_edges / max_edges if max_edges > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Ошибка вычисления плотности графа: {e}", exc_info=True)
            return 0.0
    
    def get_diameter(self) -> int:
        """Вычисляет диаметр графа (упрощенно - максимальное кратчайшее расстояние)."""
        try:
            if not self.nodes:
                return 0
            
            max_distance = 0
            
            sample_nodes = list(self.nodes.keys())[:min(50, len(self.nodes))]
            
            for start in sample_nodes:
                visited = {start: 0}
                queue = deque([start])
                
                while queue:
                    current = queue.popleft()
                    current_dist = visited[current]
                    
                    if current_dist > max_distance:
                        max_distance = current_dist
                    
                    if current_dist >= 5:
                        continue
                    
                    for edge in self.get_edges(current):
                        neighbor = edge.target_id if edge.source_id == current else edge.source_id
                        if neighbor not in visited:
                            visited[neighbor] = current_dist + 1
                            queue.append(neighbor)
            
            return max_distance
            
        except Exception as e:
            logger.error(f"Ошибка вычисления диаметра графа: {e}", exc_info=True)
            return 0
