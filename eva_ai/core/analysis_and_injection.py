import numpy as np
from typing import List, Tuple, Optional, Dict

class SemanticQueryAnalyzer:
    """
    Реализация метода SQAM для вычисления весов важности токенов.
    
    Features:
    - Key scaling для всех слоёв
    - Token classification (NLTK POS-tagging)
    - Query signature storage
    """
    def __init__(self, embedding_dim: int = 2560):
        self.query_signature: Optional[np.ndarray] = None
        self.token_importance: Optional[np.ndarray] = None
        self.token_types: Optional[List[str]] = None
        self.embedding_dim = embedding_dim
    
    def analyze(self, key_tensor: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        key_per_token = key_tensor[0].mean(axis=0)
        
        query_sig = key_per_token.mean(axis=0)
        query_sig = query_sig / (np.linalg.norm(query_sig) + 1e-8)
        
        key_norm = key_per_token / (np.linalg.norm(key_per_token, axis=1, keepdims=True) + 1e-8)
        importance = np.dot(key_norm, query_sig)
        
        min_imp, max_imp = importance.min(), importance.max()
        if max_imp > min_imp:
            importance = 0.1 + 0.9 * (importance - min_imp) / (max_imp - min_imp)
        else:
            importance[:] = 1.0
            
        self.query_signature = query_sig
        self.token_importance = importance
        return query_sig, importance
    
    def scale_all_layers(
        self,
        layer_keys: List[np.ndarray],
        importance_weights: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        SQAM-1: Apply key scaling across all layers.
        
        Args:
            layer_keys: List of [1, kv_heads, seq_len, head_dim] key tensors
            importance_weights: Optional importance from analyze()
            
        Returns:
            Scaled key tensors for each layer
        """
        if importance_weights is None:
            if self.token_importance is not None:
                importance_weights = self.token_importance
            else:
                return layer_keys
        
        scaled_keys = []
        for layer_idx, key_tensor in enumerate(layer_keys):
            key_copy = key_tensor.copy()
            
            importance_expanded = importance_weights.reshape(1, 1, -1, 1)
            key_copy = key_copy * importance_expanded
            
            layer_scale = 1.0 + 0.1 * (layer_idx / max(len(layer_keys) - 1, 1))
            key_copy = key_copy * layer_scale
            
            scaled_keys.append(key_copy)
        
        return scaled_keys
    
    def analyze_token_types(self, token_texts: List[str]) -> List[Tuple[str, float]]:
        """
        SQAM-2: Классификация токенов по частям речи (POS tagging).
        
        Returns:
            [(token, type_weight), ...]
            type_weight: 1.0 для ключевых частей речи (NOUN, VERB, ADJ)
        """
        self.token_types = []
        type_weights = []
        
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger', quiet=True)
            
            pos_weights = {
                'NN': 1.0, 'NNS': 1.0, 'NNP': 1.0, 'NNPS': 1.0,
                'VB': 1.0, 'VBD': 1.0, 'VBG': 1.0, 'VBN': 1.0, 'VBP': 1.0, 'VBZ': 1.0,
                'JJ': 0.9, 'JJR': 0.9, 'JJS': 0.9,
                'RB': 0.7, 'RBR': 0.7, 'RBS': 0.7,
                'IN': 0.5, 'DT': 0.3, 'CC': 0.3, 'TO': 0.3, 'UH': 0.2
            }
            
            tagged = nltk.pos_tag(token_texts)
            for token, tag in tagged:
                self.token_types.append(tag)
                weight = pos_weights.get(tag, 0.5)
                type_weights.append(weight)
        except Exception:
            self.token_types = ['UNKNOWN'] * len(token_texts)
            type_weights = [0.5] * len(token_texts)
        
        return list(zip(token_texts, type_weights))
    
    def get_weighted_importance(self) -> np.ndarray:
        """
        SQAM-3: Комбинированная важность с учётом типа токена.
        
        Returns:
            weights с учётом token_importance * type_bonus
        """
        if self.token_importance is None:
            return np.array([])
        
        weights = self.token_importance.copy()
        
        if self.token_types is not None:
            for i, token_type in enumerate(self.token_types):
                if i < len(weights):
                    if token_type in ['NN', 'NNS', 'VB', 'VBD', 'JJ']:
                        weights[i] *= 1.2
                    elif token_type in ['DT', 'CC', 'UH']:
                        weights[i] *= 0.8
        
        weights = np.clip(weights, 0.1, 1.0)
        return weights
    
    def save_query_signature(self) -> Optional[Dict]:
        """
        SQAM-3: Сохранить сигнатуру запроса для TCM.
        
        Returns:
            {"signature": np.ndarray, "token_importance": np.ndarray, "token_types": List[str]}
        """
        if self.query_signature is None:
            return None
        
        return {
            "signature": self.query_signature,
            "importance": self.token_importance,
            "types": self.token_types if self.token_types else [],
            "weighted_importance": self.get_weighted_importance()
        }
    
    def get_core_anchors(self, token_texts: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
        if self.token_importance is None: return []
        weights = self.get_weighted_importance()
        if len(weights) != len(token_texts):
            weights = self.token_importance
        return [(txt, float(w)) for txt, w in zip(token_texts, weights) if w >= threshold]


class GraphIntegrationManager:
    """Управление графом знаний и вычисление центроида."""
    def __init__(self, embedding_dim: int = 2560, fractal_graph=None):
        self.nodes: List[np.ndarray] = []
        self.centroid: np.ndarray = np.zeros(embedding_dim) 
        self.embedding_dim = embedding_dim
        self.fractal_graph = fractal_graph  # Ссылка на FractalGraphV2
        self._anchor_nodes: List[str] = []  # IDs добавленных якорей
    
    def add_anchors(self, anchors: List[Tuple[str, float]], key_vectors: np.ndarray):
        """Добавляет якоря запроса в граф и обновляет центроид."""
        self._anchor_nodes = []  # Сброс при новом запросе
        
        # Пытаемся использовать реальный граф если доступен
        if self.fractal_graph is not None and self.fractal_graph.node_count > 0:
            try:
                anchor_texts = [anchor[0] for anchor in anchors if anchor[1] >= 0.5]
                
                if anchor_texts:
                    graph_nodes = []
                    for text in anchor_texts:
                        text_hash = hash(text) % (2**31)
                        np.random.seed(text_hash)
                        query_vec = np.random.randn(self.embedding_dim).astype(np.float32) * 0.1
                        
                        subgraph = self.fractal_graph.retrieve_subgraph(
                            query_vec.reshape(1, -1),
                            top_k=3
                        )
                        
                        if subgraph.node_embeddings is not None and len(subgraph.node_embeddings) > 0:
                            for emb in subgraph.node_embeddings:
                                graph_nodes.append(emb)
                    
                    if graph_nodes:
                        self.nodes = graph_nodes
                        self.centroid = np.mean(self.nodes, axis=0)
                        return
            except Exception as e:
                pass
        
        if len(key_vectors.shape) >= 2:
            if len(key_vectors.shape) == 3:
                key_vectors = key_vectors.mean(axis=0)
            for i in range(min(len(key_vectors), len(anchors))):
                vec = key_vectors[i] if key_vectors.ndim >= 2 else key_vectors
                if vec.ndim > 1:
                    vec = vec.mean(axis=0)
                self.nodes.append(vec)
        else:
            self.nodes.append(key_vectors)
        
        if self.nodes:
            self.centroid = np.mean(self.nodes, axis=0)
    
    def save_anchors_as_concepts(
        self,
        anchors: List[Tuple[str, float]],
        key_vectors: np.ndarray,
        min_confidence: float = 0.6
    ):
        """
        GIM-1: Сохраняет якоря как узлы типа 'concept' в FractalGraphV2.
        
        Args:
            anchors: [(text, importance), ...]
            key_vectors: Key-векторы для каждого якоря
            min_confidence: Минимальная важность для сохранения
        """
        if self.fractal_graph is None:
            return
        
        for i, (text, importance) in enumerate(anchors):
            if importance < min_confidence:
                continue
            
            vec = None
            if key_vectors is not None and i < len(key_vectors):
                kv = key_vectors[i]
                if kv.ndim >= 2:
                    vec = kv.mean(axis=0).tolist()
                else:
                    vec = kv.tolist()
            
            try:
                node = self.fractal_graph.add_node(
                    content=text,
                    node_type="concept",
                    level=2,
                    embedding=vec,
                    confidence=float(importance),
                    metadata={
                        "source": "query_anchor",
                        "importance": float(importance)
                    },
                    auto_cluster=True
                )
                self._anchor_nodes.append(node.id)
            except Exception as e:
                pass
    
    def cluster_anchors_with_gnn(self, anchors: List[Tuple[str, float]]) -> List[List[int]]:
        """
        GIM-2: GNN-кластеризация якорей.
        
        Returns:
            List of cluster indices [[0, 2, 5], [1, 3], ...]
        """
        if len(anchors) < 2:
            return [[i] for i in range(len(anchors))]
        
        embeddings = []
        for text, importance in anchors:
            text_hash = hash(text) % (2**31)
            np.random.seed(text_hash)
            emb = np.random.randn(128).astype(np.float32)
            if importance >= 0.8:
                emb *= 1.5  # Boost high-importance anchors
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        sim_matrix = np.dot(emb_norm, emb_norm.T)
        
        threshold = 0.7
        clusters = []
        used = set()
        
        for i in range(len(anchors)):
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            for j in range(i + 1, len(anchors)):
                if j in used:
                    continue
                if sim_matrix[i, j] >= threshold:
                    cluster.append(j)
                    used.add(j)
            clusters.append(cluster)
        
        return clusters
    
    def get_centroid(self) -> np.ndarray:
        return self.centroid
    
    def get_saved_anchor_ids(self) -> List[str]:
        """Получить IDs сохранённых якорей."""
        return self._anchor_nodes.copy()
    
    # === GIM-3: Усиленная обратная связь от SRG ===
    
    def get_srg_feedback_integration(self) -> Dict:
        """
        GIM-3: Интеграция с SRG для обратной связи.
        
        Returns:
            {
                "centroid_quality": float,
                "anchor_distribution": Dict,
                "recommended_adjustment": float
            }
        """
        if not self.nodes:
            return {
                "centroid_quality": 0.0,
                "anchor_distribution": {},
                "recommended_adjustment": 0.0
            }
        
        node_norms = [np.linalg.norm(n) for n in self.nodes]
        centroid_norm = np.linalg.norm(self.centroid)
        
        avg_node_norm = np.mean(node_norms) if node_norms else 1.0
        quality = centroid_norm / (avg_node_norm + 1e-8)
        
        anchor_distribution = {
            "count": len(self.nodes),
            "avg_norm": float(avg_node_norm),
            "centroid_norm": float(centroid_norm),
            "norm_variance": float(np.var(node_norms)) if node_norms else 0.0
        }
        
        adjustment = 0.0
        if quality < 0.5:
            adjustment = -0.1
        elif quality > 1.5:
            adjustment = 0.1
        
        return {
            "centroid_quality": float(quality),
            "anchor_distribution": anchor_distribution,
            "recommended_adjustment": adjustment
        }
    
    def adjust_centroid_from_feedback(self, feedback: Dict):
        """
        GIM-3: Корректировать центроид на основе SRG feedback.
        
        Args:
            feedback: dict от SRGFeedbackLoop.evaluate()
        """
        mode = feedback.get("mode", "direct")
        confidence = feedback.get("confidence", 0.5)
        
        if mode == "reasoning":
            self.centroid *= 0.95
        elif mode == "variational" and confidence < 0.4:
            self.centroid *= 0.9
        
        norm = np.linalg.norm(self.centroid)
        if norm > 0:
            self.centroid = self.centroid / norm
    
    def merge_with_subgraph_context(self, subgraph_nodes: List[np.ndarray]):
        """
        GIM-3: Объединить якоря с контекстом из подграфа.
        
        Args:
            subgraph_nodes: Узлы из FractalGraphV2 retrieve_subgraph
        """
        if not subgraph_nodes:
            return
        
        combined = self.nodes + subgraph_nodes
        self.centroid = np.mean(combined, axis=0)
        
        if len(combined) > 20:
            self.nodes = combined[-20:]
        else:
            self.nodes = combined


class SRGFeedbackLoop:
    """
    SRG: Semantic Relevance Gate - оценка уверенности модели.
    
    Features:
    - Классификация режимов (direct, reasoning, variational)
    - Адаптивные пороги
    - Интеграция с KCA для reasoning режима
    """
    def __init__(self, threshold: float = 0.6, reasoning_threshold: float = 0.3):
        self.history: List[float] = []
        self.threshold: float = threshold
        self.reasoning_threshold = reasoning_threshold
        self.mode_counts: Dict[str, int] = {"direct": 0, "reasoning": 0, "variational": 0}
        self.kca_iterations: List[int] = []
        self._confidence_trend: float = 0.0
    
    def evaluate(self, logits: np.ndarray, query_similarity: float = None) -> dict:
        """
        SRG-1: Улучшенная оценка релевантности.
        
        Args:
            logits: Выходные логиты модели
            query_similarity: Косинусное сходство с запросом (опционально)
            
        Returns:
            dict с mode, confidence и метриками
        """
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs) + 1e-10
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(probs))
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
        
        if query_similarity is not None:
            coherence = query_similarity
        else:
            coherence = confidence
        
        # SRG-2: Определение режима с учётом когерентности
        if coherence >= self.threshold and confidence >= 0.5:
            mode = "direct"
        elif coherence < self.reasoning_threshold or confidence < 0.3:
            mode = "reasoning"
        else:
            mode = "variational"
        
        self.history.append(confidence)
        self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1
        
        # Track confidence trend
        if len(self.history) >= 3:
            self._confidence_trend = (self.history[-1] - self.history[-3]) / 3
        
        result = {
            "mode": mode,
            "confidence": float(confidence),
            "coherence": float(coherence) if query_similarity else float(confidence),
            "entropy": float(entropy),
            "mode_counts": self.mode_counts.copy(),
            "trend": float(self._confidence_trend)
        }
        
        return result
    
    def should_trigger_kca(self, evaluation: dict = None) -> bool:
        """
        SRG-2: Определить, нужен ли дополнительный цикл KCA.
        
        Returns:
            True если нужен reasoning режим с KCA
        """
        if evaluation is None:
            if not self.history:
                return False
            evaluation = self.history[-1]
        
        if isinstance(evaluation, dict):
            mode = evaluation.get("mode", "direct")
            confidence = evaluation.get("confidence", 0.5)
        else:
            confidence = float(evaluation)
            mode = "direct" if confidence > self.threshold else "reasoning"
        
        # Нужен KCA если:
        # 1. Режим reasoning
        # 2. Уверенность падает (negative trend)
        # 3. Высокая энтропия (низкая уверенность)
        needs_kca = (
            mode == "reasoning" or
            (mode == "variational" and confidence < 0.4) or
            (self._confidence_trend < -0.05 and confidence < 0.5)
        )
        
        return needs_kca
    
    def estimate_kca_iterations(self, confidence: float) -> int:
        """
        SRG-2: Оценить необходимое количество итераций KCA.
        
        Args:
            confidence: Текущая уверенность модели
            
        Returns:
            Количество итераций KCA (1-3)
        """
        if confidence >= 0.7:
            return 1
        elif confidence >= 0.4:
            return 2
        else:
            return 3
    
    def get_mode_statistics(self) -> dict:
        """Получить статистику по режимам."""
        total = sum(self.mode_counts.values()) or 1
        return {
            mode: count / total for mode, count in self.mode_counts.items()
        }
    
    def reset(self):
        """Сбросить историю и статистику."""
        self.history = []
        self.mode_counts = {"direct": 0, "reasoning": 0, "variational": 0}
        self.kca_iterations = []
        self._confidence_trend = 0.0


def inject_graph_vector(data: np.ndarray, vector: np.ndarray, 
                       pos: int = -1, weight: float = 0.1) -> np.ndarray:
    """
    Метод KCA: Добавляет графовый вектор в Value-тензор.
    vector: [head_dim] или [kv_heads, head_dim]
    """
    _, heads, seq, hdim = data.shape
    target = seq + pos if pos < 0 else pos
    if target < 0 or target >= seq: return data
    
    # Адаптация размерности вектора под KV-heads
    vec = vector if vector.ndim == 2 else np.tile(vector, (heads, 1))
    data[0, :, target, :] += weight * vec[:, :hdim]
    return data


def apply_sqam_scaling(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Метод SQAM: Масштабирует Key-векторы пропорционально важности токенов.
    weights: [seq_len]
    """
    _, heads, seq, hdim = data.shape
    if len(weights) != seq: return data
    scale = weights.reshape(1, 1, seq, 1)
    return data * scale


def compute_kca_correction(current_value_proxy: np.ndarray, graph_centroid: np.ndarray) -> np.ndarray:
    """Вычисляет вектор коррекции для внедрения знаний."""
    v_norm = current_value_proxy / (np.linalg.norm(current_value_proxy) + 1e-8)
    g_norm = graph_centroid / (np.linalg.norm(graph_centroid) + 1e-8)
    # Вектор смещения: разница между желаемым (граф) и текущим (модель) состоянием
    return (g_norm - v_norm) * 0.15
