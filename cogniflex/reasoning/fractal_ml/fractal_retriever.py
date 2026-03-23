"""
Fractal Retriever для CogniFlex Self-Reasoning
Поиск и извлечение данных из FractalStorage
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("cogniflex.reasoning.fractal_retriever")


class FractalRetriever:
    """Ретривер для извлечения данных из FractalStorage."""
    
    def __init__(self, storage, embedder=None):
        self.storage = storage
        self.embedder = embedder
    
    def retrieve_with_depth(self, level: Optional[int] = None, max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Получить узлы на определённом уровне или обход с глубиной.
        
        Args:
            level: Уровень для выборки (None = все уровни)
            max_depth: Максимальная глубина обхода
        
        Returns:
            List[Dict]: Список узлов в виде словарей
        """
        if level is not None:
            nodes = self.storage.get_nodes_by_level(level)
            return [self._node_to_dict(n) for n in nodes]
        
        results = []
        for lvl in range(max_depth + 1):
            nodes = self.storage.get_nodes_by_level(lvl)
            for node in nodes:
                results.append({**self._node_to_dict(node), "retrieval_depth": lvl})
        
        return results
    
    def retrieve_path(self, node_id: str) -> List[Dict[str, Any]]:
        """Получить полный путь от корня до узла."""
        path = self.storage.get_path_to_root(node_id)
        return [self._node_to_dict(n) for n in path]
    
    def retrieve_with_depth_from_node(self, node_id: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Получить узлы начиная с node_id с обходом вниз."""
        results = []
        visited = set()
        self._retrieve_recursive(node_id, 0, max_depth, visited, results)
        return results
    
    def _retrieve_recursive(self, node_id: str, current_depth: int, max_depth: int, visited: set, results: List):
        if node_id in visited or current_depth > max_depth:
            return
        
        visited.add(node_id)
        node = self.storage.get_node(node_id)
        
        if node:
            node_dict = {**self._node_to_dict(node), "retrieval_depth": current_depth}
            results.append(node_dict)
            
            for child_id in node.child_ids:
                self._retrieve_recursive(child_id, current_depth + 1, max_depth, visited, results)
    
    def retrieve_by_confidence(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Получить узлы по уровню доверия."""
        all_nodes = []
        
        for node in self.storage.nodes.values():
            confidence = node.context.get("confidence", 1.0) if node.context else 1.0
            if confidence >= threshold:
                all_nodes.append({**self._node_to_dict(node), "confidence": confidence})
        
        return sorted(all_nodes, key=lambda x: x.get("confidence", 0), reverse=True)
    
    def retrieve_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Получить узлы сессии рассуждений."""
        results = []
        
        for node in self.storage.nodes.values():
            if node.context and node.context.get("session_id") == session_id:
                results.append(self._node_to_dict(node))
        
        return sorted(results, key=lambda x: x.get("created_at", 0))
    
    def retrieve_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получить последние узлы."""
        results = []
        
        for node in self.storage.nodes.values():
            results.append(self._node_to_dict(node))
        
        return sorted(results, key=lambda x: x.get("created_at", 0), reverse=True)[:limit]
    
    def semantic_search(self, query_content: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Текстовый поиск в хранилище."""
        nodes = self.storage.search_by_content(query_content, limit=max_results * 2)
        return [self._node_to_dict(n) for n in nodes[:max_results]]
    
    def search_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Поиск узлов по типу."""
        results = []
        
        for node in self.storage.nodes.values():
            if node.node_type == node_type:
                results.append(self._node_to_dict(node))
        
        return results
    
    def get_subtree(self, node_id: str, include_root: bool = True) -> List[Dict[str, Any]]:
        """Получить все дочерние узлы."""
        results = []
        
        if include_root:
            node = self.storage.get_node(node_id)
            if node:
                results.append(self._node_to_dict(node))
        
        children = self.storage.get_children(node_id)
        for child in children:
            results.extend(self.get_subtree(child.id, include_root=True))
        
        return results
    
    def _node_to_dict(self, node) -> Dict[str, Any]:
        """Конвертировать FractalNode в словарь."""
        if hasattr(node, 'to_dict'):
            return node.to_dict()
        return {
            "id": node.id,
            "content": node.content,
            "node_type": node.node_type,
            "level": node.level,
            "parent_id": node.parent_id,
            "child_ids": node.child_ids,
            "context": node.context,
            "created_at": node.created_at,
            "updated_at": node.updated_at
        }
    
    def retrieve_with_embedding(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Семантический поиск по эмбеддингам."""
        if not self.embedder:
            logger.warning("Embedder не инициализирован, используем текстовый поиск")
            return self.semantic_search(query, max_results=top_k)
        
        query_embedding = self.embedder.embed_text(query)
        results = []
        
        for node in self.storage.nodes.values():
            node_embedding = self.embedder.embed_node(node)
            similarity = self.embedder.compute_similarity(query_embedding, node_embedding)
            
            results.append({
                **self._node_to_dict(node),
                "similarity": similarity
            })
        
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return results[:top_k]
    
    def retrieve_cross_level(self, query: str, levels: List[int] = None) -> Dict[int, List[Dict]]:
        """Извлечение с разных уровней иерархии."""
        if levels is None:
            levels = [0, 1, 2, 3]
        
        result = {}
        for level in levels:
            nodes = self.storage.get_nodes_by_level(level)
            query_lower = query.lower()
            filtered = [
                self._node_to_dict(n) 
                for n in nodes 
                if query_lower in n.content.lower()
            ]
            result[level] = filtered
        
        return result
    
    def retrieve_reasoning_context(
        self, 
        query: str, 
        levels: List[int] = None
    ) -> Dict[int, List[Dict]]:
        """Извлечь контекст рассуждения для каждого уровня абстракции."""
        if levels is None:
            levels = [0, 1, 2, 3]
        
        context = {}
        
        for level in levels:
            level_nodes = self.storage.get_nodes_by_level(level)
            
            relevant = []
            for node in level_nodes:
                ctx = node.context or {}
                if ctx.get("query") == query or query.lower() in node.content.lower():
                    relevant.append(self._node_to_dict(node))
            
            context[level] = relevant
        
        return context
    
    def search_reasoning_graph(
        self, 
        query: str,
        max_depth: int = 5,
        relation_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Поиск в графе рассуждений."""
        if relation_types is None:
            relation_types = ["follows_from", "related_to", "based_on"]
        
        start_nodes = self.storage.search_by_content(query, limit=20)
        
        results = []
        visited = set()
        
        for start_node in start_nodes:
            self._graph_search_recursive(
                start_node.id, 
                query, 
                0, 
                max_depth, 
                relation_types,
                visited,
                results
            )
        
        return results
    
    def _graph_search_recursive(
        self, 
        node_id: str, 
        query: str,
        depth: int, 
        max_depth: int,
        relation_types: List[str],
        visited: set,
        results: List
    ):
        if depth > max_depth or node_id in visited:
            return
        
        visited.add(node_id)
        node = self.storage.get_node(node_id)
        
        if not node:
            return
        
        if query.lower() in node.content.lower():
            results.append({
                **self._node_to_dict(node),
                "search_depth": depth
            })
        
        for rel_type in relation_types:
            if rel_type in node.relations:
                for related_id in node.relations[rel_type]:
                    self._graph_search_recursive(
                        related_id, query, depth + 1, max_depth,
                        relation_types, visited, results
                    )
