"""
Graph ML Patterns - Pattern detection, clustering, and analysis for MemoryGraphML.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

logger = logging.getLogger("eva.memory_graph_ml")


def detect_clusters(self, n_clusters: int = 5) -> Dict[str, Any]:
    """Обнаруживает кластеры в графе на основе эмбеддингов."""
    try:
        if len(self.embeddings) < n_clusters:
            return {'clusters': [], 'n_clusters': 0, 'reason': 'Not enough nodes'}
        
        node_ids = list(self.embeddings.keys())
        vectors = np.array([self.embeddings[nid].vector for nid in node_ids])
        
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)
            
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                clusters[int(label)].append({
                    'node_id': node_ids[i],
                    'node_type': self.embeddings[node_ids[i]].node_type,
                    'distance': float(np.linalg.norm(vectors[i] - kmeans.cluster_centers_[label]))
                })
            
            result = {
                'clusters': [
                    {
                        'cluster_id': cid,
                        'size': len(members),
                        'members': members[:20],
                        'centroid': kmeans.cluster_centers_[cid].tolist()
                    }
                    for cid, members in clusters.items()
                ],
                'n_clusters': n_clusters,
                'inertia': float(kmeans.inertia_)
            }
            
            return result
            
        except ImportError:
            return {'clusters': [], 'n_clusters': 0, 'reason': 'sklearn not available'}
        
    except Exception as e:
        logger.debug(f"Ошибка обнаружения кластеров: {e}")
        return {'clusters': [], 'n_clusters': 0, 'reason': str(e)}


def find_frequent_patterns(self, min_support: int = 3) -> List[Dict[str, Any]]:
    """Находит частые паттерны связей в графе."""
    try:
        if not hasattr(self, 'brain') or not self.brain:
            return []
        kg = getattr(self.brain, 'knowledge_graph', None)
        if not kg:
            return []
        
        if not hasattr(kg, 'get_all_relations'):
            return []
        
        relations = kg.get_all_relations()
        
        node_relations = defaultdict(list)
        for rel in relations:
            from_node = rel.get('from') or rel.get('source')
            to_node = rel.get('to') or rel.get('target')
            rel_type = rel.get('type') or rel.get('relation_type')
            
            if from_node and to_node and rel_type:
                node_relations[from_node].append((to_node, rel_type))
        
        pattern_counts = defaultdict(int)
        pattern_examples = defaultdict(list)
        
        for start_node, first_level in node_relations.items():
            for mid_node, rel1 in first_level:
                if mid_node in node_relations:
                    for end_node, rel2 in node_relations[mid_node]:
                        if end_node != start_node:
                            pattern_key = f"{rel1}->{rel2}"
                            pattern_counts[pattern_key] += 1
                            if len(pattern_examples[pattern_key]) < 3:
                                pattern_examples[pattern_key].append({
                                    'start': start_node,
                                    'mid': mid_node,
                                    'end': end_node
                                })
        
        frequent = []
        for pattern_key, count in pattern_counts.items():
            if count >= min_support:
                parts = pattern_key.split("->")
                frequent.append({
                    'pattern': pattern_key,
                    'relations': parts if len(parts) == 2 else [],
                    'support': count,
                    'confidence': min(0.95, 0.5 + count * 0.05),
                    'examples': pattern_examples[pattern_key]
                })
        
        frequent.sort(key=lambda x: x['support'], reverse=True)
        return frequent
        
    except Exception as e:
        logger.debug(f"Ошибка поиска частых паттернов: {e}")
        return []


def analyze_graph_structure(self) -> Dict[str, Any]:
    """Анализирует структуру графа и возвращает статистику."""
    try:
        if not hasattr(self, 'brain') or not self.brain:
            return {'error': 'Brain not available'}
        kg = getattr(self.brain, 'knowledge_graph', None)
        if not kg:
            return {'error': 'KnowledgeGraph not available'}
        
        node_types = defaultdict(int)
        for node_id, embedding in self.embeddings.items():
            node_types[embedding.node_type] += 1
        
        relation_types = defaultdict(int)
        if hasattr(kg, 'get_all_relations'):
            for rel in kg.get_all_relations():
                rel_type = rel.get('type') or rel.get('relation_type', 'unknown')
                relation_types[rel_type] += 1
        
        embedding_stats = {}
        if self.embeddings:
            vectors = np.array([e.vector for e in self.embeddings.values()])
            embedding_stats = {
                'mean_norm': float(np.mean(np.linalg.norm(vectors, axis=1))),
                'std_norm': float(np.std(np.linalg.norm(vectors, axis=1))),
                'dimension': vectors.shape[1] if len(vectors.shape) > 1 else 0
            }
        
        return {
            'total_nodes': len(self.embeddings),
            'total_patterns': len(self.patterns),
            'node_type_distribution': dict(node_types),
            'relation_type_distribution': dict(relation_types),
            'embedding_stats': embedding_stats,
            'fractal_levels': self.fractal_levels,
            'training_samples': len(self.training_data)
        }
        
    except Exception as e:
        logger.debug(f"Ошибка анализа структуры графа: {e}")
        return {'error': str(e)}


def get_pattern_insights(self, query: str) -> List[Dict[str, Any]]:
    """Получает релевантные паттерны для запроса."""
    try:
        query_embedding = self._compute_query_embedding(query)
        
        relevant_patterns = []
        for pattern in self.patterns:
            if pattern.embedding is not None:
                sim = self._cosine_similarity(query_embedding, pattern.embedding)
                if sim > self.similarity_threshold:
                    relevant_patterns.append({
                        'pattern_id': pattern.pattern_id,
                        'relations': pattern.relations,
                        'frequency': pattern.frequency,
                        'confidence': pattern.confidence,
                        'similarity': float(sim)
                    })
        
        relevant_patterns.sort(key=lambda x: x['similarity'], reverse=True)
        return relevant_patterns[:10]
        
    except Exception as e:
        logger.debug(f"Ошибка получения паттернов: {e}")
        return []


def _setup_patterns(self):
    """Setup pattern methods on the MemoryGraphML class."""
    pass
