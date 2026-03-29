"""
Методы поиска для графа знаний CogniFlex
Часть модуля knowledge_graph.py (разделение на логические компоненты)
"""
from typing import Dict, List, Optional, Any, Tuple


class KnowledgeGraphSearch:
    """Методы поиска в графе знаний."""
    
    def search_nodes(self, query: str, limit: int = 10) -> List[Any]:
        """Поиск узлов по запросу."""
        results = []
        query_lower = query.lower()
        
        for node in self.nodes.values():
            score = 0
            if query_lower in node.name.lower():
                score += 10
            if query_lower in node.description.lower():
                score += 5
            if hasattr(node, 'keyword_index') and node.keyword_index:
                for kw in node.keyword_index:
                    if query_lower in str(kw).lower():
                        score += 2
            
            if score > 0:
                results.append((node, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:limit]]
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Алиас для search_nodes."""
        nodes = self.search_nodes(query, limit)
        return [
            {
                "id": node.id,
                "name": node.name,
                "description": node.description,
                "type": node.node_type,
                "domain": node.domain,
                "strength": node.strength
            }
            for node in nodes
        ]
    
    def find_nodes(self, pattern: str, node_type: Optional[str] = None) -> List[Any]:
        """Поиск узлов по шаблону."""
        results = []
        pattern_lower = pattern.lower()
        
        for node in self.nodes.values():
            if node_type and node.node_type != node_type:
                continue
            if pattern_lower in node.name.lower() or pattern_lower in node.description.lower():
                results.append(node)
        
        return results
    
    def query_nodes(self, filters: Dict[str, Any]) -> List[Any]:
        """Запрос узлов с фильтрами."""
        results = []
        
        for node in self.nodes.values():
            match = True
            for key, value in filters.items():
                if not hasattr(node, key):
                    match = False
                    break
                node_value = getattr(node, key)
                if isinstance(value, list):
                    if node_value not in value:
                        match = False
                        break
                elif node_value != value:
                    match = False
                    break
            
            if match:
                results.append(node)
        
        return results
    
    def get_edges(self, node_id: str) -> List[Any]:
        """Получить все связи для узла."""
        edges = []
        for edge in self.edges.values():
            if edge.source_id == node_id or edge.target_id == node_id:
                edges.append(edge)
        return edges
    
    def get_node(self, node_id: str) -> Optional[Any]:
        """Получить узел по ID."""
        return self.nodes.get(node_id)
    
    def get_sources_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        """Возвращает источники для узла."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        
        sources = node.meta.get('sources', [])
        result = []
        for source in sources:
            if isinstance(source, dict):
                result.append({
                    'name': source.get('source', 'unknown'),
                    'reliability': source.get('reliability', 0.5),
                    'timestamp': source.get('timestamp', 0),
                    'user_id': source.get('user_id'),
                    'version': source.get('version', 1)
                })
            elif hasattr(source, 'source'):
                result.append({
                    'name': getattr(source, 'source', 'unknown'),
                    'reliability': getattr(source, 'reliability', 0.5),
                    'timestamp': getattr(source, 'timestamp', 0),
                    'user_id': getattr(source, 'user_id', None),
                    'version': getattr(source, 'version', 1)
                })
        return result
