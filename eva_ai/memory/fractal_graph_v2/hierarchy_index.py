"""
Hierarchical Index - Иерархическая навигация по фрактальным уровням

Обеспечивает:
- Быстрый подъём по уровням (L0 → L1 → L2 → L3)
- Загрузку только релевантных данных
- O(log n) навигацию вместо O(n)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict

logger = logging.getLogger("eva_ai.hierarchy_index")


class HierarchicalIndex:
    """
    Иерархический индекс для фрактальной навигации.
    
    Уровни:
    - L0 (ROOT): 1-10 мета-узлов - ссылки на L1
    - L1 (CONCEPTS): ~500 концептов - ссылки на L2
    - L2 (FACTS): ~10k фактов - ссылки на L3
    - L3 (DETAILS): полные данные (~440k)
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        
        # Кэш мета-данных по уровням (всегда в памяти)
        self._level_cache: Dict[int, Dict[str, Dict]] = {}
        
        # Указатели: level -> node_id -> set of child node_ids
        self._pointers: Dict[int, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        # Индексы для быстрого поиска
        self._level_centroids: Dict[int, np.ndarray] = {}  # Средний вектор каждого уровня
        self._node_to_level: Dict[str, int] = {}  # node_id -> level mapping
        
        # Статистика
        self._stats = {
            "nav_requests": 0,
            "cache_hits": 0,
            "levels_traversed": 0
        }
    
    def build_from_graph(self, nodes: Dict, groups: Dict, nodes_by_level: Dict[int, List[str]]):
        """
        Построить иерархический индекс из графа (LAZY - без загрузки всех данных).
        
        Args:
            nodes: Словарь узлов {node_id: FractalNode} (может быть пустым в lazy mode)
            groups: Словарь групп {group_id: SemanticGroup} (может быть пустым)
            nodes_by_level: Индекс узлов по уровням {level: [node_ids]}
        """
        logger.info("Building hierarchical index (lazy mode)...")
        
        # Очищаем старые данные
        self._pointers.clear()
        self._node_to_level.clear()
        
        # Шаг 1: Только ID и level - НЕ загружаем embeddings!
        # Только мета-индексы, без контента
        for level, node_ids in nodes_by_level.items():
            # Ограничиваем количество - не более 1000 на уровень
            limited_ids = node_ids[:1000] if len(node_ids) > 1000 else node_ids
            
            for node_id in limited_ids:
                self._node_to_level[node_id] = level
        
        # Шаг 2: Только центроиды групп (без узлов)
        # Получаем центроиды из групп напрямую - они уже там есть
        for group_id, group in groups.items():
            if group.embedding and group.level in [1, 2]:
                self._level_centroids[group.level] = group.embedding[:self.embedding_dim]
        
        logger.info(f"Hierarchical index built: {len(self._node_to_level)} nodes indexed (limited)")
        
        # Шаг 2: Вычисляем центроиды каждого уровня
        for level, node_ids in nodes_by_level.items():
            if not node_ids:
                continue
                
            vectors = []
            for node_id in node_ids:
                if node_id in nodes and nodes[node_id].embedding:
                    vectors.append(np.array(nodes[node_id].embedding))
            
            if vectors:
                centroid = np.mean(vectors, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                self._level_centroids[level] = centroid
        
        # Шаг 3: Строим кэш L0-L1 (всегда в памяти)
        self._build_level_cache(0, nodes, groups)
        self._build_level_cache(1, nodes, groups)
        
        logger.info(f"Hierarchical index built: {len(self._node_to_level)} nodes indexed")
        logger.info(f"Level centroids: {list(self._level_centroids.keys())}")
    
    def _build_level_cache(self, level: int, nodes: Dict, groups: Dict):
        """Построить кэш для конкретного уровня."""
        cache = {}
        
        node_ids = self._pointers.get(level, {})
        for node_id in node_ids:
            if node_id in nodes:
                node = nodes[node_id]
                cache[node_id] = {
                    "content": node.content[:100],  #-preview
                    "node_type": node.node_type,
                    "level": node.level,
                    "embedding": node.embedding[:20] if node.embedding else None,  # first 20 dims
                    "children": list(self._pointers[level][node_id])[:50]  # first 50 children
                }
        
        # Также кэшируем группы уровня
        for group_id, group in groups.items():
            if group.level == level:
                cache[group_id] = {
                    "name": group.name,
                    "node_type": "semantic_group",
                    "level": group.level,
                    "embedding": group.embedding[:20] if group.embedding else None,
                    "member_count": group.member_count
                }
        
        self._level_cache[level] = cache
        logger.info(f"Cached level {level}: {len(cache)} items")
    
    def navigate_to_level(
        self,
        query_embedding: List[float],
        target_level: int,
        nodes: Dict,
        groups: Dict,
        min_similarity: float = 0.5
    ) -> Tuple[List[str], List[str]]:
        """
        Навигация от L0 к целевому уровню.
        
        Args:
            query_embedding: Вектор запроса
            target_level: Целевой уровень (0-3)
            nodes: Узлы графа
            groups: Группы графа
            min_similarity: Минимальная схожесть
            
        Returns:
            (relevant_node_ids, relevant_group_ids)
        """
        self._stats["nav_requests"] += 1
        
        query_vec = np.array(query_embedding)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        results_nodes = []
        results_groups = []
        
        # Начинаем с L0 (всегда в кэше)
        current_level = 0
        
        while current_level < target_level:
            self._stats["levels_traversed"] += 1
            
            # Ищем релевантные узлы текущего уровня
            candidates = self._level_cache.get(current_level, {})
            
            scored_candidates = []
            for cand_id, cand_data in candidates.items():
                if cand_data.get("embedding"):
                    cand_vec = np.array(cand_data["embedding"])
                    # Используем только первые dims для экономии
                    if len(cand_vec) < self.embedding_dim:
                        # Паддим нулями
                        cand_vec = np.pad(cand_vec, (0, self.embedding_dim - len(cand_vec)))
                    cand_vec = cand_vec / (np.linalg.norm(cand_vec) + 1e-8)
                    
                    sim = float(np.dot(query_vec[:len(cand_vec)], cand_vec[:len(cand_vec)]))
                    if sim >= min_similarity:
                        scored_candidates.append((cand_id, sim, cand_data.get("children", [])))
            
            # Сортируем и берём top-k
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = scored_candidates[:10]
            
            # Собираем детей для следующего уровня
            next_level_nodes = set()
            for cand_id, sim, children in top_candidates:
                if current_level == 0:
                    results_nodes.append(cand_id)
                else:
                    results_groups.append(cand_id)
                
                # Добавляем детей для навигации глубже
                for child_id in children:
                    if child_id in nodes:
                        child_node = nodes[child_id]
                        if child_node.level == current_level + 1:
                            next_level_nodes.add(child_id)
            
            # Если нет детей - используем всех соседей того же уровня
            if not next_level_nodes and top_candidates:
                for cand_id, sim, children in top_candidates[:3]:
                    for child_id in children:
                        if child_id in nodes and nodes[child_id].level == current_level + 1:
                            next_level_nodes.add(child_id)
            
            # Загружаем следующий уровень если нужно
            if current_level + 1 not in self._level_cache and next_level_nodes:
                self._load_level_cache(current_level + 1, list(next_level_nodes), nodes, groups)
            
            current_level += 1
        
        return list(results_nodes), list(results_groups)
    
    def _load_level_cache(self, level: int, node_ids: List[str], nodes: Dict, groups: Dict):
        """Ленивая загрузка кэша уровня."""
        cache = {}
        
        for node_id in node_ids:
            if node_id in nodes:
                node = nodes[node_id]
                cache[node_id] = {
                    "content": node.content[:100],
                    "node_type": node.node_type,
                    "level": node.level,
                    "embedding": node.embedding[:20] if node.embedding else None,
                    "children": list(self._pointers.get(level, {}).get(node_id, set()))[:50]
                }
        
        # Добавляем группы этого уровня
        for group_id, group in groups.items():
            if group.level == level and group_id not in cache:
                cache[group_id] = {
                    "name": group.name,
                    "node_type": "semantic_group",
                    "level": group.level,
                    "embedding": group.embedding[:20] if group.embedding else None,
                    "member_count": group.member_count
                }
        
        self._level_cache[level] = cache
        logger.debug(f"Lazy loaded level {level}: {len(cache)} items")
    
    def get_subtree(
        self,
        root_ids: List[str],
        max_depth: int,
        nodes: Dict,
        groups: Dict
    ) -> Tuple[Set[str], Set[str]]:
        """
        Получить поддерево от корневых узлов.
        
        Args:
            root_ids: Корневые ID
            max_depth: Максимальная глубина
            nodes: Узлы
            groups: Группы
            
        Returns:
            (node_ids, group_ids)
        """
        result_nodes = set()
        result_groups = set()
        
        to_visit = [(rid, 0) for rid in root_ids]
        visited = set()
        
        while to_visit:
            node_id, depth = to_visit.pop(0)
            
            if node_id in visited or depth > max_depth:
                continue
            visited.add(node_id)
            
            if node_id in nodes:
                result_nodes.add(node_id)
                
                # Добавляем детей
                level = nodes[node_id].level
                children = self._pointers.get(level, {}).get(node_id, set())
                for child_id in children:
                    if child_id in nodes:
                        to_visit.append((child_id, depth + 1))
            
            elif node_id in groups:
                result_groups.add(node_id)
        
        return result_nodes, result_groups
    
    def get_level_stats(self) -> Dict[str, Any]:
        """Получить статистику по уровням."""
        return {
            "cached_levels": list(self._level_cache.keys()),
            "total_indexed_nodes": len(self._node_to_level),
            "level_centroids": {k: v.shape[0] for k, v in self._level_centroids.items()},
            "stats": self._stats
        }


def create_hierarchical_index(embedding_dim: int = 768) -> HierarchicalIndex:
    """Фабричная функция для создания иерархического индекса."""
    return HierarchicalIndex(embedding_dim=embedding_dim)