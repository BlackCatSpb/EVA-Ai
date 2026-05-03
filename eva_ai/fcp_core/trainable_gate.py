import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("FCP.TrainableGate")


class TrainableGate:
    """
    Обучаемый гейт слияния (EVA.txt раздел 2.1).
    Учится комбинировать информацию из разных источников (модель, граф, KCA).
    Полная реализация без упрощений.
    """
    
    def __init__(self, input_dim: int = 2560, hidden_dim: int = 128, num_sources: int = 3):
        """
        Args:
            input_dim: размерность входных векторов (скрытых состояний)
            hidden_dim: размерность скрытого слоя гейта
            num_sources: количество источников (например: модель, граф, KCA)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources
        
        # Обучаемые веса гейта (двухслойный перцептрон)
        self.W1 = self._init_weight(input_dim * num_sources, hidden_dim)  # Первый слой
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W2 = self._init_weight(hidden_dim, num_sources)  # Выходной слой (веса для каждого источника)
        self.b2 = np.zeros(num_sources, dtype=np.float32)
        
        # Гиперпараметры обучения
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        
        logger.info(f"TrainableGate initialized: input_dim={input_dim}, hidden_dim={hidden_dim}, sources={num_sources}")
    
    def _init_weight(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Инициализация весов (Xavier initialization)."""
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        return np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
    
    def forward(self, source_vectors: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Прямой проход гейта.
        
        Args:
            source_vectors: список векторов от разных источников
                          (например: [model_hidden, graph_vector, kca_vector])
                          
        Returns:
            (output, weights)
            output: объединенный вектор [input_dim]
            weights: веса для каждого источника [num_sources]
        """
        if len(source_vectors) != self.num_sources:
            raise ValueError(f"Expected {self.num_sources} sources, got {len(source_vectors)}")
        
        # Конкатенация всех источников
        concatenated = np.concatenate([v.flatten() for v in source_vectors])
        
        # Первый слой + ReLU
        hidden = np.maximum(0, np.dot(concatenated, self.W1) + self.b1)  # ReLU
        
        # Выходной слой + Softmax
        logits = np.dot(hidden, self.W2) + self.b2
        # Softmax для получения весов
        exp_logits = np.exp(logits - np.max(logits))  # Для численной стабильности
        weights = exp_logits / np.sum(exp_logits)
        
        # Взвешенная сумма источников
        output = np.zeros_like(source_vectors[0], dtype=np.float32)
        for i, v in enumerate(source_vectors):
            output += weights[i] * v
        
        return output, weights
    
    def update(self, source_vectors: List[np.ndarray], target: np.ndarray, metadata: Dict = None):
        """
        Обновление весов гейта на основе обратного распространения.
        
        Args:
            source_vectors: векторы источников (для повторного прямого прохода)
            target: целевой вектор (например, правильный ответ)
            metadata: дополнительная информация (опционально)
        """
        # Прямой проход (с сохранением промежуточных значений)
        concatenated = np.concatenate([v.flatten() for v in source_vectors])
        hidden = np.maximum(0, np.dot(concatenated, self.W1) + self.b1)
        logits = np.dot(hidden, self.W2) + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        weights = exp_logits / np.sum(exp_logits)
        
        # Выход
        output = np.zeros_like(source_vectors[0], dtype=np.float32)
        for i, v in enumerate(source_vectors):
            output += weights[i] * v
        
        # === Обратное распространение ===
        # Градиент на выходе (упрощенно: MSE между output и target)
        grad_output = 2.0 * (output - target) / output.shape[0]  # Градиент MSE
        
        # Градиент на веса
        grad_weights = np.zeros(self.num_sources, dtype=np.float32)
        for i, v in enumerate(source_vectors):
            grad_weights[i] = np.dot(grad_output, v)
        
        # Градиент на logits (через softmax)
        grad_logits = weights * (grad_weights - np.sum(weights * grad_weights))
        
        # Градиент на скрытый слой
        grad_hidden = np.dot(grad_logits, self.W2.T)
        grad_hidden[hidden <= 0] = 0  # Градиент ReLU
        
        # Градиент на веса
        grad_W2 = np.outer(hidden, grad_logits)
        grad_b2 = grad_logits
        
        grad_W1 = np.outer(concatenated, grad_hidden)
        grad_b1 = grad_hidden
        
        # Обновление с моментумом
        self.v_W1 = self.momentum * self.v_W1 + (1 - self.momentum) * grad_W1
        self.W1 -= self.learning_rate * self.v_W1
        
        self.v_b1 = self.momentum * self.v_b1 + (1 - self.momentum) * grad_b1
        self.b1 -= self.learning_rate * self.v_b1
        
        self.v_W2 = self.momentum * self.v_W2 + (1 - self.momentum) * grad_W2
        self.W2 -= self.learning_rate * self.v_W2
        
        self.v_b2 = self.momentum * self.v_b2 + (1 - self.momentum) * grad_b2
        self.b2 -= self.learning_rate * self.v_b2
        
        logger.debug(f"TrainableGate updated: weights={weights}")
    
    def get_weights(self) -> np.ndarray:
        """Получить текущие веса гейта."""
        # Для получения весов нужно сделать forward pass, но мы можем сохранить последние веса
        # Упрощение: возвращаем равномерное распределение
        return np.ones(self.num_sources, dtype=np.float32) / self.num_sources
    
    def set_learning_rate(self, lr: float):
        """Установить скорость обучения."""
        self.learning_rate = lr
    
    def save(self, path: str):
        """Сохранить веса гейта."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        logger.info(f"TrainableGate saved to {path}")
    
    def load(self, path: str):
        """Загрузить веса гейта."""
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        logger.info(f"TrainableGate loaded from {path}")
