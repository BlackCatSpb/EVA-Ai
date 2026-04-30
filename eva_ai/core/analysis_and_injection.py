import numpy as np
from typing import List, Tuple, Optional

class SemanticQueryAnalyzer:
    """Реализация метода SQAM для вычисления весов важности токенов."""
    def __init__(self):
        self.query_signature: Optional[np.ndarray] = None
        self.token_importance: Optional[np.ndarray] = None

    def analyze(self, key_tensor: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        # key_tensor shape: [1, kv_heads, seq_len, head_dim]
        # 1. Pooling по головам
        key_per_token = key_tensor[0].mean(axis=0)  # [seq_len, head_dim]
        
        # 2. Вычисление сигнатуры запроса
        query_sig = key_per_token.mean(axis=0)
        query_sig = query_sig / (np.linalg.norm(query_sig) + 1e-8)
        
        # 3. Вычисление важности (Cosine Similarity)
        key_norm = key_per_token / (np.linalg.norm(key_per_token, axis=1, keepdims=True) + 1e-8)
        importance = np.dot(key_norm, query_sig)  # [seq_len]
        
        # 4. Нормализация в [0.1, 1.0]
        min_imp, max_imp = importance.min(), importance.max()
        if max_imp > min_imp:
            importance = 0.1 + 0.9 * (importance - min_imp) / (max_imp - min_imp)
        else:
            importance[:] = 1.0
            
        self.query_signature = query_sig
        self.token_importance = importance
        return query_sig, importance

    def get_core_anchors(self, token_texts: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
        if self.token_importance is None: return []
        return [(txt, float(imp)) for txt, imp in zip(token_texts, self.token_importance) if imp >= threshold]


class GraphIntegrationManager:
    """Управление графом знаний и вычисление центроида."""
    def __init__(self, embedding_dim: int = 2560):
        self.nodes: List[np.ndarray] = []
        self.centroid: np.ndarray = np.zeros(embedding_dim) 
        self.embedding_dim = embedding_dim

    def add_anchors(self, anchors: List[Tuple[str, float]], key_vectors: np.ndarray):
        """Добавляет якоря запроса в граф и обновляет центроид."""
        # В упрощённой версии просто используем ключевые векторы как узлы графа
        # В реальной реализации здесь будет поиск по индексу в графе знаний
        if len(key_vectors.shape) >= 2:
            # Если у нас есть матрица векторов [seq_len, dim] или [heads, seq_len, dim]
            if len(key_vectors.shape) == 3:
                # [heads, seq_len, dim] -> усредняем по головам
                key_vectors = key_vectors.mean(axis=0)  # [seq_len, dim]
            # Берем первые несколько векторов как представление якорных токенов
            for i in range(min(len(key_vectors), len(anchors))):
                vec = key_vectors[i] if key_vectors.ndim >= 2 else key_vectors
                if vec.ndim > 1:
                    vec = vec.mean(axis=0)  # Усредняем если нужно
                self.nodes.append(vec)
        else:
            # Если у нас уже есть готовый вектор
            self.nodes.append(key_vectors)
        
        if self.nodes:
            self.centroid = np.mean(self.nodes, axis=0)

    def get_centroid(self) -> np.ndarray:
        return self.centroid


class SRGFeedbackLoop:
    """Оценка уверенности модели (Semantic Relevance Gate)."""
    def __init__(self, threshold: float = 0.6):
        self.history: List[float] = []
        self.threshold: float = threshold

    def evaluate(self, logits: np.ndarray) -> dict:
        # Применяем softmax для получения вероятностей
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs) + 1e-10
        # Вычисляем энтропию
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        # Нормализованная уверенность: 1 - (энтропия / макс_энтропия)
        max_entropy = np.log2(len(probs))
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
        
        mode = "direct" if confidence > self.threshold else "reasoning"
        self.history.append(confidence)
        return {"mode": mode, "confidence": float(confidence)}


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