import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger("FCP.CrossAttention")


class CrossAttentionFusion:
    """
    Cross-Attention слияние (EVA.txt раздел 2.1).
    Комбинирует информацию от модели (query) и графа (key/value).
    Полная реализация без упрощений.
    """
    
    def __init__(self, hidden_dim: int = 2560, graph_dim: int = 384, num_heads: int = 8):
        """
        Args:
            hidden_dim: размерность скрытых состояний модели
            graph_dim: размерность эмбеддингов графа
            num_heads: количество голов внимания
        """
        self.hidden_dim = hidden_dim
        self.graph_dim = graph_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Обучаемые матрицы весов (если нужно - инициализируем)
        self.W_q = self._init_weight(hidden_dim, hidden_dim)  # Query проекция
        self.W_k = self._init_weight(graph_dim, hidden_dim)    # Key проекция (для графа)
        self.W_v = self._init_weight(graph_dim, hidden_dim)    # Value проекция (для графа)
        self.W_o = self._init_weight(hidden_dim, hidden_dim)   # Output проекция
        
        logger.info(f"CrossAttentionFusion initialized: hidden_dim={hidden_dim}, graph_dim={graph_dim}, heads={num_heads}")
    
    def _init_weight(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Инициализация весов (для полной реализации нужно обучение)."""
        # Xavier инициализация
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        return np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
    
    def compute_cross_attention(self, 
                               model_hidden: np.ndarray, 
                               graph_embeddings: np.ndarray,
                               mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисление Cross-Attention между моделью и графом.
        
        Args:
            model_hidden: скрытые состояния модели [batch, seq_len, hidden_dim]
            graph_embeddings: эмбеддинги узлов графа [num_nodes, graph_dim]
            mask: маска для padding (опционально)
            
        Returns:
            (output, attention_weights)
            output: результат внимания [batch, seq_len, hidden_dim]
            attention_weights: веса внимания [batch, num_heads, seq_len, num_nodes]
        """
        batch_size = 1
        seq_len = 1  # Для инференса обычно одни токен
        
        if len(model_hidden.shape) == 1:
            model_hidden = model_hidden.reshape(1, 1, -1)
            batch_size = 1
            seq_len = 1
        elif len(model_hidden.shape) == 2:
            model_hidden = model_hidden.reshape(1, -1, self.hidden_dim)
            batch_size = 1
            seq_len = model_hidden.shape[1]
        
        # 1. Линейные проекции
        # Query от модели
        Q = np.dot(model_hidden, self.W_q)  # [batch, seq_len, hidden_dim]
        
        # Key и Value от графа
        K = np.dot(graph_embeddings, self.W_k)  # [num_nodes, hidden_dim]
        V = np.dot(graph_embeddings, self.W_v)  # [num_nodes, hidden_dim]
        
        # 2. Решейпинг для multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(1, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(1, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # 3. Вычисление scores (скалярное произведение)
        # Q: [batch, heads, seq_len, head_dim]
        # K: [1, heads, num_nodes, head_dim] -> броадкастится до [batch, heads, num_nodes, head_dim]
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))  # [batch, heads, seq_len, num_nodes]
        scores = scores / np.sqrt(self.head_dim)
        
        # 4. Маска (если есть)
        if mask is not None:
            scores = scores + mask.reshape(1, 1, 1, -1)
        
        # 5. Softmax для получения весов внимания
        # Для стабильности вычитаем максимум
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 6. Применение весов к Value
        output = np.matmul(attn_weights, V.transpose(0, 1, 2, 3))  # [batch, heads, seq_len, head_dim]
        
        # 7. Конкатенация голов и финальная проекция
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_dim)
        output = np.dot(output, self.W_o)
        
        return output.squeeze(), attn_weights.squeeze()
    
    def update_weights(self, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, W_o: np.ndarray):
        """Обновление весов (для обучения)."""
        self.W_q = W_q.astype(np.float32)
        self.W_k = W_k.astype(np.float32)
        self.W_v = W_v.astype(np.float32)
        self.W_o = W_o.astype(np.float32)
        logger.info("CrossAttention weights updated")
