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
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.nodes: Dict[str, FractalNode] = {}
        self.edges: Dict[str, FractalEdge] = {}
        
        # Создаём директорию
        os.makedirs(storage_path, exist_ok=True)
        
        # Файлы для хранения
        self.nodes_file = os.path.join(storage_path, "nodes.json")
        self.edges_file = os.path.join(storage_path, "edges.json")
        
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
        try:
            # Сохраняем узлы
            nodes_data = {node.id: node.to_dict() for node in self.nodes.values()}
            with open(self.nodes_file, 'w', encoding='utf-8') as f:
                json.dump(nodes_data, f, indent=2, ensure_ascii=False)
            
            # Сохраняем связи
            edges_data = {edge.id: edge.to_dict() for edge in self.edges.values()}
            with open(self.edges_file, 'w', encoding='utf-8') as f:
                json.dump(edges_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
    
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
        
        self._save()
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
            "storage_path": self.storage_path
        }
