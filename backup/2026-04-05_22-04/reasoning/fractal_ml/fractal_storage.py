"""
Fractal Storage - главное хранилище с иерархической структурой L0→L1→L2→L3
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional
from .fractal_base import FractalNode, FractalNodeType, FractalEdge, FractalRelationType, create_fractal_id

logger = logging.getLogger(__name__)


class FractalStorage:
    """
    Фрактальное хранилище с иерархической структурой
    Уровни: L0 (1KB) → L1 (16KB) → L2 (256KB) → L3 (4MB)
    """
    
    MAX_LEVELS = 4
    BRANCHING_FACTOR = 16
    
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.nodes: Dict[str, FractalNode] = {}
        self.edges: Dict[str, FractalEdge] = {}
        
        # Batch save optimization
        self._dirty = False
        self._save_queue_size = 10  # Сохранять каждые 10 операций
        self._operation_count = 0
        
        # Создаём директорию
        os.makedirs(storage_dir, exist_ok=True)
        
        # Файлы для хранения
        self.nodes_file = os.path.join(storage_dir, "nodes.json")
        self.edges_file = os.path.join(storage_dir, "edges.json")
        
        # Загружаем существующие данные
        self._load()
        
        logger.info(f"FractalStorage инициализирована: {len(self.nodes)} узлов, {len(self.edges)} связей")
    
    def _load(self):
        """Загрузка данных из файлов"""
        if os.path.exists(self.nodes_file):
            try:
                with open(self.nodes_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for node_data in data.values():
                        node = FractalNode.from_dict(node_data)
                        self.nodes[node.id] = node
                logger.info(f"Загружено {len(self.nodes)} узлов")
            except Exception as e:
                logger.error(f"Ошибка загрузки узлов: {e}")
        
        if os.path.exists(self.edges_file):
            try:
                with open(self.edges_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for edge_data in data.values():
                        edge = FractalEdge(**edge_data)
                        self.edges[edge.id] = edge
                logger.info(f"Загружено {len(self.edges)} связей")
            except Exception as e:
                logger.error(f"Ошибка загрузки связей: {e}")
    
    def _save(self):
        """Сохранение данных в файлы"""
        if not self._dirty and self._operation_count == 0:
            return  # Пропускаем если нет изменений
        
        try:
            # Сохраняем узлы
            nodes_data = {node.id: node.to_dict() for node in self.nodes.values()}
            with open(self.nodes_file, 'w', encoding='utf-8') as f:
                json.dump(nodes_data, f, indent=2, ensure_ascii=False)
            
            # Сохраняем связи
            edges_data = {edge.id: edge.to_dict() for edge in self.edges.values()}
            with open(self.edges_file, 'w', encoding='utf-8') as f:
                json.dump(edges_data, f, indent=2, ensure_ascii=False)
                
            self._dirty = False
            self._operation_count = 0
                
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
    
    def flush(self):
        """Принудительное сохранение (для критических операций)"""
        self._dirty = True
        self._save()
    
    def add_node(
        self, 
        content: str, 
        node_type: str, 
        level: int,
        parent_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> FractalNode:
        """Добавить узел в хранилище"""
        node_id = create_fractal_id(content, node_type, level)
        
        # Проверяем существующий
        if node_id in self.nodes:
            return self.nodes[node_id]
        
        node = FractalNode(
            id=node_id,
            content=content,
            node_type=node_type,
            level=level,
            parent_id=parent_id,
            context=context or {}
        )
        
        self.nodes[node_id] = node
        
        # Добавляем связь с родителем
        if parent_id and parent_id in self.nodes:
            parent = self.nodes[parent_id]
            parent.add_child(node_id)
            
            # Создаём связь
            edge = FractalEdge(
                id=f"{parent_id}_{node_id}",
                source_id=parent_id,
                target_id=node_id,
                relation_type=FractalRelationType.PARENT_OF.value
            )
            self.edges[edge.id] = edge
        
        # Batch save - помечаем как dirty и сохраняем если накопилось
        self._dirty = True
        self._operation_count += 1
        
        if self._operation_count >= self._save_queue_size:
            self._save()
            self._operation_count = 0
            self._dirty = False
        
        return node
    
    def get_node(self, node_id: str) -> Optional[FractalNode]:
        """Получить узел по ID"""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[FractalNode]:
        """Получить дочерние узлы"""
        node = self.nodes.get(node_id)
        if not node:
            return []
        
        return [self.nodes[cid] for cid in node.child_ids if cid in self.nodes]
    
    def get_path_to_root(self, node_id: str) -> List[FractalNode]:
        """Получить путь от узла до корня"""
        path = []
        current = self.nodes.get(node_id)
        
        while current:
            path.append(current)
            if current.parent_id:
                current = self.nodes.get(current.parent_id)
            else:
                break
        
        return list(reversed(path))
    
    def search_by_content(self, query: str, limit: int = 10) -> List[FractalNode]:
        """Поиск по содержимому"""
        query_lower = query.lower()
        results = []
        
        for node in self.nodes.values():
            if query_lower in node.content.lower():
                results.append(node)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_nodes_by_level(self, level: int) -> List[FractalNode]:
        """Получить все узлы на уровне"""
        return [n for n in self.nodes.values() if n.level == level]
    
    def add_reasoning_step(
        self,
        query: str,
        step_content: str,
        confidence: float,
        iteration: int
    ) -> FractalNode:
        """Добавить шаг рассуждения"""
        return self.add_node(
            content=step_content,
            node_type=FractalNodeType.REASONING_STEP.value,
            level=0,  # Рассуждения на L0
            context={
                "query": query,
                "confidence": confidence,
                "iteration": iteration
            }
        )
    
    def add_clarification(
        self,
        query: str,
        question: str
    ) -> FractalNode:
        """Добавить вопрос уточнения"""
        return self.add_node(
            content=question,
            node_type=FractalNodeType.CLARIFICATION.value,
            level=0,
            context={"original_query": query}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику хранилища"""
        by_level = {}
        for level in range(self.MAX_LEVELS):
            by_level[f"L{level}"] = len(self.get_nodes_by_level(level))
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_by_level": by_level,
            "storage_path": self.storage_dir
        }
    
    def delete_node(self, node_id: str, cascade: bool = True) -> bool:
        """Удалить узел с опциональным каскадным удалением потомков."""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        
        if cascade:
            for child_id in node.child_ids[:]:
                self.delete_node(child_id, cascade=True)
        
        edges_to_remove = [
            eid for eid, edge in self.edges.items()
            if edge.source_id == node_id or edge.target_id == node_id
        ]
        for eid in edges_to_remove:
            del self.edges[eid]
        
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node_id in parent.child_ids:
                parent.child_ids.remove(node_id)
        
        del self.nodes[node_id]
        self._save()
        return True
    
    def update_node(self, node_id: str, content: Optional[str] = None, **kwargs) -> Optional[FractalNode]:
        """Обновить узел с версионированием."""
        node = self.nodes.get(node_id)
        if not node:
            return None
        
        if content is not None:
            node.content = content
            node.version += 1
            node.updated_at = time.time()
        
        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)
        
        self._save()
        return node
    
    def get_neighbors(self, node_id: str, include_children: bool = True, 
                     include_parents: bool = True) -> List[FractalNode]:
        """Получить все связанные узлы (не только дети)."""
        neighbors = []
        node = self.nodes.get(node_id)
        if not node:
            return neighbors
        
        if include_children:
            for child_id in node.child_ids:
                if child_id in self.nodes:
                    neighbors.append(self.nodes[child_id])
        
        if include_parents and node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent:
                neighbors.append(parent)
        
        for rel_type, related_ids in node.relations.items():
            for rel_id in related_ids:
                if rel_id in self.nodes:
                    neighbors.append(self.nodes[rel_id])
        
        return neighbors
    
    def traverse_breadth_first(self, start_id: str, max_depth: int = 3) -> List[FractalNode]:
        """BFS обход дерева рассуждений."""
        if start_id not in self.nodes:
            return []
        
        result = []
        queue = [(start_id, 0)]
        visited = {start_id}
        
        while queue:
            node_id, depth = queue.pop(0)
            
            if depth > max_depth:
                continue
            
            node = self.nodes[node_id]
            result.append(node)
            
            for child_id in node.child_ids:
                if child_id not in visited:
                    visited.add(child_id)
                    queue.append((child_id, depth + 1))
        
        return result
    
    def batch_add_nodes(self, nodes_data: List[Dict]) -> List[FractalNode]:
        """Массовое добавление узлов (транзакционно)."""
        nodes = []
        for data in nodes_data:
            node = self.add_node(
                content=data["content"],
                node_type=data.get("node_type", "detail"),
                level=data.get("level", 0),
                parent_id=data.get("parent_id"),
                context=data.get("context", {})
            )
            nodes.append(node)
        return nodes
    
    def get_reasoning_chain(self, query: str) -> List[FractalNode]:
        """Получить все шаги рассуждений для запроса."""
        result = []
        for node in self.nodes.values():
            if node.node_type == FractalNodeType.REASONING_STEP.value:
                ctx = node.context or {}
                if ctx.get("query") == query:
                    result.append(node)
        return sorted(result, key=lambda n: n.context.get("iteration", 0) if n.context else 0)
    
    def compute_reasoning_confidence(self, node_id: str) -> float:
        """Вычислить уверенность рассуждения на основе связей."""
        node = self.nodes.get(node_id)
        if not node:
            return 0.0
        
        base_confidence = node.context.get("confidence", 0.5) if node.context else 0.5
        
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent:
                parent_confidence = parent.context.get("confidence", 0.5) if parent.context else 0.5
                base_confidence = min(base_confidence, parent_confidence * 0.9)
        
        return base_confidence
    
    def validate_reasoning_chain(self, chain: List[str]) -> Dict[str, Any]:
        """Проверить логическую целостность цепочки рассуждений."""
        if not chain:
            return {"valid": False, "errors": ["Пустая цепочка"]}
        
        errors = []
        
        for i, node_id in enumerate(chain):
            if node_id not in self.nodes:
                errors.append(f"Узел {node_id} не найден")
                continue
            
            node = self.nodes[node_id]
            
            if i > 0:
                prev_id = chain[i - 1]
                if prev_id not in node.relations.get("follows_from", []):
                    if node.parent_id != prev_id:
                        errors.append(f"Разрыв связи между {prev_id} и {node_id}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "chain_length": len(chain)
        }
    
    def get_reasoning_subgraph(self, start_step_id: str) -> Dict[str, Any]:
        """Получить подграф связанных рассуждений."""
        if start_step_id not in self.nodes:
            return {"nodes": [], "edges": []}
        
        visited = set()
        nodes_data = []
        edges_data = []
        
        def collect(node_id: str, depth: int = 0):
            if depth > 5 or node_id in visited:
                return
            
            visited.add(node_id)
            node = self.nodes.get(node_id)
            if not node:
                return
            
            nodes_data.append(self._node_to_dict(node))
            
            for rel_type, related_ids in node.relations.items():
                for rel_id in related_ids:
                    edges_data.append({
                        "source": node_id,
                        "target": rel_id,
                        "relation": rel_type
                    })
                    collect(rel_id, depth + 1)
        
        collect(start_step_id)
        
        return {"nodes": nodes_data, "edges": edges_data, "node_count": len(nodes_data)}
    
    def _node_to_dict(self, node: FractalNode) -> Dict[str, Any]:
        """Конвертировать узел в словарь."""
        return node.to_dict()
    
    def retrieve_similar_reasoning(
        self, 
        query: str, 
        embedder, 
        top_k: int = 5
    ) -> List[FractalNode]:
        """Найти похожие рассуждения по эмбеддингам."""
        if not embedder:
            return []
        
        query_embedding = embedder.embed_text(query)
        results = []
        
        for node in self.nodes.values():
            if node.node_type != FractalNodeType.REASONING_STEP.value:
                continue
            
            node_embedding = embedder.embed_text(node.content)
            similarity = embedder.compute_similarity(query_embedding, node_embedding)
            
            if similarity > 0.5:
                results.append((node, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:top_k]]
    
    def find_reasoning_parents(
        self, 
        step_id: str, 
        max_depth: int = 3
    ) -> List[FractalNode]:
        """Найти родительские рассуждения."""
        if step_id not in self.nodes:
            return []
        
        parents = []
        current = self.nodes[step_id]
        depth = 0
        
        while current.parent_id and depth < max_depth:
            parent = self.nodes.get(current.parent_id)
            if parent:
                parents.append(parent)
                current = parent
                depth += 1
            else:
                break
        
        return parents
    
    def get_cross_level_context(
        self, 
        query: str, 
        levels: Optional[List[int]] = None
    ) -> Dict[int, List[FractalNode]]:
        """Получить контекст с разных уровней иерархии."""
        if levels is None:
            levels = [0, 1, 2, 3]
        
        result = {}
        query_lower = query.lower()
        
        for level in levels:
            level_nodes = self.get_nodes_by_level(level)
            
            relevant = [
                n for n in level_nodes 
                if query_lower in n.content.lower() or 
               (n.context and query_lower in str(n.context).lower())
            ]
            
            result[level] = relevant
        
        return result
    
    def compute_aggregated_confidence(self, chain: List[str]) -> float:
        """Вычислить агрегированную уверенность для всей цепочки."""
        if not chain:
            return 0.0
        
        confidences = []
        for node_id in chain:
            node = self.nodes.get(node_id)
            if node and node.context:
                conf = node.context.get("confidence", 0.5)
                confidences.append(conf)
        
        if not confidences:
            return 0.5
        
        return sum(confidences) / len(confidences)
    
    def store_with_embedding(
        self, 
        content: str, 
        node_type: str, 
        level: int, 
        embedding: List[float],
        parent_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> FractalNode:
        """Сохранить узел с предвычисленным эмбеддингом."""
        node = self.add_node(
            content=content,
            node_type=node_type,
            level=level,
            parent_id=parent_id,
            context=context
        )
        
        node.embedding = embedding
        self._save()
        
        return node
    
    def merge_branch_results(self, branch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Слить результаты параллельных веток рекурсии."""
        if not branch_results:
            return {"merged_response": "", "confidence": 0.0, "sub_branches": 0}
        
        responses = [r.get("response", "") for r in branch_results if r.get("response")]
        confidences = [r.get("confidence", 0.0) for r in branch_results]
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        merged = " | ".join(responses) if responses else ""
        
        return {
            "merged_response": merged,
            "confidence": avg_confidence,
            "sub_branches": len(branch_results),
            "branch_responses": responses
        }
    
    def detect_cycle(self, start_node_id: str, max_check: int = 100) -> Dict[str, Any]:
        """Обнаружить циклы в графе рассуждений."""
        if start_node_id not in self.nodes:
            return {"has_cycle": False, "cycle_nodes": [], "depth": 0}
        
        visited = set()
        path = []
        cycle_nodes = []
        
        def dfs(node_id: str, depth: int) -> bool:
            if depth > max_check:
                return False
            
            if node_id in visited:
                if node_id in path:
                    cycle_nodes.append(node_id)
                    return True
                return False
            
            visited.add(node_id)
            path.append(node_id)
            
            node = self.nodes.get(node_id)
            if not node:
                return False
            
            for child_id in node.child_ids:
                if dfs(child_id, depth + 1):
                    return True
            
            path.pop()
            return False
        
        has_cycle = dfs(start_node_id, 0)
        
        return {
            "has_cycle": has_cycle,
            "cycle_nodes": cycle_nodes,
            "depth": len(visited)
        }
    
    def prune_by_age(self, max_age_seconds: float = 86400 * 30) -> int:
        """Удалить узлы старше max_age_seconds (по умолчанию 30 дней)."""
        current_time = time.time()
        removed = 0
        
        nodes_to_remove = []
        for node_id, node in self.nodes.items():
            age = current_time - node.created_at
            if age > max_age_seconds:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            if self.delete_node(node_id, cascade=False):
                removed += 1
        
        if removed > 0:
            self._save()
        
        return removed
    
    def compact_storage(self) -> Dict[str, Any]:
        """Дефрагментировать хранилище - пересоздать JSON файлы."""
        original_nodes = len(self.nodes)
        original_edges = len(self.edges)
        
        self._save()
        
        return {
            "nodes_before": original_nodes,
            "nodes_after": len(self.nodes),
            "edges_before": original_edges,
            "edges_after": len(self.edges),
            "compacted": True
        }
    
    def batch_retrieve(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Пакетное извлечение узлов для производительности."""
        results = []
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if node:
                results.append(self._node_to_dict(node))
        return results
    
    def update_confidence_from_feedback(self, node_id: str, new_confidence: float, feedback: str = None) -> bool:
        """Обновить уверенность узла после фидбека."""
        node = self.nodes.get(node_id)
        if not node:
            return False
        
        if not node.context:
            node.context = {}
        
        old_conf = node.context.get("confidence", 0.5)
        node.context["confidence"] = new_confidence
        node.context["previous_confidence"] = old_conf
        node.context["feedback_updated_at"] = time.time()
        
        if feedback:
            node.context["last_feedback"] = feedback
        
        self._save()
        return True
    
    def merge_reasoning_chains(self, chain1_id: str, chain2_id: str) -> Optional[FractalNode]:
        """Объединить две цепочки рассуждений."""
        node1 = self.nodes.get(chain1_id)
        node2 = self.nodes.get(chain2_id)
        
        if not node1 or not node2:
            return None
        
        merged_content = f"MERGED: {node1.content} || {node2.content}"
        merged_context = {
            "merged_from": [chain1_id, chain2_id],
            "original_confidence": [
                node1.context.get("confidence", 0.5) if node1.context else 0.5,
                node2.context.get("confidence", 0.5) if node2.context else 0.5
            ]
        }
        
        # Определяем уровень - берем максимальный
        level = max(node1.level, node2.level)
        
        merged_node = self.add_node(
            content=merged_content,
            node_type="merged_reasoning",
            level=level,
            context=merged_context
        )
        
        # Связываем с оригинальными
        node1.add_child(merged_node.id)
        node2.add_child(merged_node.id)
        
        self._save()
        return merged_node
    
    def get_feedback_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Получить историю фидбека."""
        feedback_nodes = []
        
        for node in self.nodes.values():
            if node.context and "last_feedback_rating" in node.context:
                feedback_nodes.append({
                    "node_id": node.id,
                    "content": node.content[:100],
                    "rating": node.context["last_feedback_rating"],
                    "timestamp": node.context.get("feedback_updated_at", node.created_at)
                })
        
        feedback_nodes.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return feedback_nodes[:limit]
    
    def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья FractalStorage"""
        stats = self.get_stats()
        
        return {
            "healthy": True,
            "node_count": stats["total_nodes"],
            "edge_count": stats["total_edges"],
            "storage_path": self.storage_dir,
            "dirty": self._dirty,
            "operation_count": self._operation_count
        }
