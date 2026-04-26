"""
FractalGraphV2 - Векторный граф знаний для FCP.

Реализация на основе Update.txt:
- Vector DB с нормализованными векторами
- Поиск противоречий через косинусное сходство
- Интеграция с EmbeddingsManager
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from eva_ai.fcp_core.config import FCPConfig

logger = logging.getLogger("eva_ai.fcp_core.fractal_graph")


class FractalGraphV2:
    """
    Полная реализация векторного графа знаний.
    Хранит узлы как векторы и вычисляет отношения "на лету".
    """
    
    def __init__(self, nodes: np.ndarray = None, config: FCPConfig = None):
        """
        Args:
            nodes: [N, D] - матрица эмбеддингов узлов
            config: FCPConfig конфигурация
        """
        self.config = config or FCPConfig()
        
        if nodes is not None:
            self.nodes = nodes
            self.node_count = nodes.shape[0]
            # Нормализуем узлы один раз для быстрого поиска
            norms = np.linalg.norm(self.nodes, axis=1, keepdims=True)
            self.nodes_norm = self.nodes / (norms + 1e-8)
        else:
            self.nodes = None
            self.nodes_norm = None
            self.node_count = 0
    
    def set_nodes(self, nodes: np.ndarray):
        """Установить узлы графа"""
        self.nodes = nodes
        self.node_count = nodes.shape[0]
        norms = np.linalg.norm(self.nodes, axis=1, keepdims=True)
        self.nodes_norm = self.nodes / (norms + 1e-8)
        logger.info(f"[FractalGraphV2] Loaded {self.node_count} nodes")
    
    def retrieve_subgraph(self, query_vec: np.ndarray, top_k: int = None) -> dict:
        """
        Поиск топ-K релевантных узлов (Cosine Search)
        
        Args:
            query_vec: [D] - вектор запроса
            top_k: количество ближайших узлов
            
        Returns:
            dict: {
                "indices": np.ndarray индексы топ-K узлов,
                "embeddings": np.ndarray [K, D] векторы узлов,
                "scores": np.ndarray [K] косинусные сходства
            }
        """
        if top_k is None:
            top_k = self.config.graph_top_k
        
        if self.nodes is None or self.node_count == 0:
            return {"indices": np.array([]), "embeddings": np.array([]), "scores": np.array([])}
        
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        similarities = self.nodes_norm @ q_norm
        
        # Находим индексы топ-K
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return {
            "indices": top_indices,
            "embeddings": self.nodes[top_indices],
            "scores": similarities[top_indices]
        }
    
    def detect_contradictions(self, subgraph_indices: np.ndarray) -> int:
        """
        Вычисляет количество противоречащих пар в подграфе.
        Противоречие = косинусное сходство ниже порога.
        
        Args:
            subgraph_indices: индексы узлов подграфа
            
        Returns:
            int: количество противоречий
        """
        count = 0
        if len(subgraph_indices) < 2:
            return 0
        
        # Векторы узлов подграфа
        subset_vecs = self.nodes_norm[subgraph_indices]
        
        # Матрица сходства
        sim_matrix = subset_vecs @ subset_vecs.T
        
        # Верхний треугольник без диагонали
        mask = np.triu(np.ones_like(sim_matrix), k=1).astype(bool)
        conflicts = sim_matrix[mask] < self.config.contradiction_sim_threshold
        
        return int(np.sum(conflicts))
    
    def add_node(self, embedding: np.ndarray, metadata: dict = None) -> int:
        """
        Добавить узел в граф.
        
        Args:
            embedding: [D] вектор узла
            metadata: опциональные метаданные
            
        Returns:
            int: индекс добавленного узла
        """
        if self.nodes is None:
            self.nodes = embedding.reshape(1, -1)
            self.nodes_norm = self.nodes / (np.linalg.norm(self.nodes, axis=1, keepdims=True) + 1e-8)
            self.node_count = 1
            self._metadata = [metadata] if metadata else [None]
        else:
            self.nodes = np.vstack([self.nodes, embedding.reshape(1, -1)])
            new_norm = embedding.reshape(1, -1) / (np.linalg.norm(embedding) + 1e-8)
            self.nodes_norm = np.vstack([self.nodes_norm, new_norm])
            self.node_count += 1
            self._metadata.append(metadata)
        
        return self.node_count - 1
    
    def get_node(self, index: int) -> Optional[np.ndarray]:
        """Получить вектор узла по индексу"""
        if 0 <= index < self.node_count:
            return self.nodes[index]
        return None
    
    def save(self, path: str):
        """Сохранение графа в файл"""
        np.save(path, self.nodes)
        logger.info(f"[FractalGraphV2] Saved {self.node_count} nodes to {path}")
    
    @classmethod
    def load(cls, path: str, config: FCPConfig = None) -> 'FractalGraphV2':
        """Загрузка графа из файла"""
        nodes = np.load(path)
        graph = cls(nodes=nodes, config=config)
        logger.info(f"[FractalGraphV2] Loaded {graph.node_count} nodes from {path}")
        return graph


def create_fractal_graph_from_texts(
    texts: List[str],
    embeddings_manager,
    config: FCPConfig = None
) -> FractalGraphV2:
    """
    Создание графа из списка текстов.
    
    Args:
        texts: список текстов
        embeddings_manager: EmbeddingsManager для векторизации
        config: FCPConfig
        
    Returns:
        FractalGraphV2: созданный граф
    """
    config = config or FCPConfig()
    graph = FractalGraphV2(config=config)
    
    if not texts:
        return graph
    
    # Векторизуем тексты
    embeddings = embeddings_manager.encode(texts, normalize=True)
    
    if embeddings is None:
        logger.warning("[FractalGraphV2] Failed to encode texts, returning empty graph")
        return graph
    
    graph.set_nodes(embeddings)
    return graph