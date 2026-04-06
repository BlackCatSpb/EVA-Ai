"""
Graph ML Inference - Inference, prediction, and similarity for MemoryGraphML.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List

logger = logging.getLogger("eva.memory_graph_ml")


def predict_relation(self, source_node: str, target_node: str) -> Dict[str, Any]:
    """Предсказывает вероятность связи между узлами."""
    try:
        if source_node not in self.embeddings or target_node not in self.embeddings:
            return {'probability': 0.0, 'confidence': 0.0, 'reason': 'Node not found'}
        
        source_emb = self.embeddings[source_node].vector
        target_emb = self.embeddings[target_node].vector
        
        similarity = self._cosine_similarity(source_emb, target_emb)
        
        related = self._get_related_concepts(source_node, depth=2)
        is_related = target_node in related
        
        probability = similarity * 0.6 + (0.4 if is_related else 0.0)
        confidence = min(0.95, similarity * 0.8 + (0.2 if is_related else 0.0))
        
        return {
            'probability': float(probability),
            'confidence': float(confidence),
            'similarity': float(similarity),
            'is_directly_related': is_related,
            'reason': 'computed'
        }
        
    except Exception as e:
        logger.debug(f"Ошибка предсказания связи: {e}")
        return {'probability': 0.0, 'confidence': 0.0, 'reason': str(e)}


def find_similar_nodes(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
    """Находит наиболее похожие узлы по эмбеддингу."""
    try:
        similarities = []
        for node_id, embedding in self.embeddings.items():
            sim = self._cosine_similarity(query_embedding, embedding.vector)
            similarities.append({
                'node_id': node_id,
                'similarity': float(sim),
                'node_type': embedding.node_type,
                'metadata': embedding.metadata
            })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
        
    except Exception as e:
        logger.debug(f"Ошибка поиска похожих узлов: {e}")
        return []


def classify_node(self, node_id: str) -> Dict[str, Any]:
    """Классифицирует узел на основе его окружения."""
    try:
        if node_id not in self.embeddings:
            return {'class': 'unknown', 'confidence': 0.0}
        
        embedding = self.embeddings[node_id]
        related = self._get_related_concepts(node_id, depth=1)
        
        type_counts = {}
        for rel_node in related:
            if rel_node in self.embeddings:
                rel_type = self.embeddings[rel_node].node_type
                type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
        
        if type_counts:
            predicted_class = max(type_counts, key=type_counts.get)
            confidence = type_counts[predicted_class] / len(related)
        else:
            predicted_class = embedding.node_type
            confidence = 0.5
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'neighbor_types': type_counts,
            'node_type': embedding.node_type
        }
        
    except Exception as e:
        logger.debug(f"Ошибка классификации узла: {e}")
        return {'class': 'unknown', 'confidence': 0.0}


def _setup_inference(self):
    """Setup inference methods on the MemoryGraphML class."""
    pass
