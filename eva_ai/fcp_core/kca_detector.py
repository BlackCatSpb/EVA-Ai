"""
Knowledge Cognitive Analyzer (KCA) - Обнаружение лакун и противоречий

EVA.txt раздел 9.3:
"На каждом слое работает KCA: обнаруживает лакуны и противоречия, 
формирует корректирующий эмбеддинг и добавляет его к скрытому состоянию через гейт."

KCA анализирует graph_vector и состояния модели для:
1. Обнаружения knowledge gaps (лакун) - областей с низким вниманием
2. Обнаружения противоречий - конфликтующих концептов
3. Генерации корректирующего эмбеддинга для исправления
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Optional
from dataclasses import dataclass

logger = logging.getLogger("FCP.KCA")


@dataclass
class KCAResult:
    """Результат работы KCA."""
    lacuna_detected: bool
    contradiction_detected: bool
    lacuna_layers: List[int]      # Слои с лакунами
    contradiction_layers: List[int]  # Слои с противоречиями
    correction_embedding: np.ndarray  # Корректирующий эмбеддинг [256]
    confidence: float            # Уверенность в коррекции [0, 1]


class KCADetector:
    """
    Knowledge Cognitive Analyzer для обнаружения лакун и противоречий.
    
    Согласно EVA.txt:
    - KCA работает на каждом слое
    - Обнаруживает лакуны: "If attention to most relevant node is low → knowledge gap"
    - Обнаруживает противоречия: конфликтующие концепты в подграфе
    - Формирует корректирующий эмбеддинг через механизм attention
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        lacuna_threshold: float = 0.3,
        contradiction_threshold: float = 0.4,
        correction_strength: float = 0.1
    ):
        """
        Args:
            embedding_dim: размерность эмбеддингов
            lacuna_threshold: порог обнаружения лакуны
            contradiction_threshold: порог обнаружения противоречия
            correction_strength: сила коррекции
        """
        self.embedding_dim = embedding_dim
        self.lacuna_threshold = lacuna_threshold
        self.contradiction_threshold = contradiction_threshold
        self.correction_strength = correction_strength
        
        # Attention механизм для взвешивания концептов
        self._attention_weights = None
        
        # Статистика
        self._detection_count = 0
        self._lacuna_count = 0
        self._contradiction_count = 0
        
        logger.info(f"[KCADetector] Initialized: dim={embedding_dim}, "
                   f"lacuna_th={lacuna_threshold}, contr_th={contradiction_threshold}")
    
    def detect(
        self,
        graph_vector: np.ndarray,
        layer_indices: List[int],
        gate_weights: np.ndarray,
        subgraph_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Обнаружить лакуны и противоречия, сформировать коррекции.
        
        Args:
            graph_vector: [256] графовый вектор от GNN
            layer_indices: список индексов слоёв (обычно 36)
            gate_weights: [36] веса гейтов для каждого слоя
            subgraph_embeddings: опционально, эмбеддинги узлов для анализа
            
        Returns:
            Dict с результатами:
            {
                "lacuna_detected": bool,
                "contradiction_detected": bool,
                "lacuna_layers": [layer_indices],
                "contradiction_layers": [layer_indices],
                "corrections": {layer_idx: {"embedding": np.ndarray, "strength": float}},
                "confidence": float
            }
        """
        self._detection_count += 1
        
        result = {
            "lacuna_detected": False,
            "contradiction_detected": False,
            "lacuna_layers": [],
            "contradiction_layers": [],
            "corrections": {},
            "confidence": 0.0,
            "gate_weights": gate_weights.tolist() if isinstance(gate_weights, np.ndarray) else gate_weights
        }
        
        # 1. Анализ лакун (Knowledge Gaps)
        # Согласно EVA: "If attention to most relevant node is low → knowledge gap"
        lacuna_layers = self._detect_lacunas(graph_vector, layer_indices, gate_weights)
        result["lacuna_layers"] = lacuna_layers
        result["lacuna_detected"] = len(lacuna_layers) > 0
        
        if result["lacuna_detected"]:
            self._lacuna_count += 1
            logger.debug(f"[KCA] Detected {len(lacuna_layers)} lacunas at layers {lacuna_layers}")
        
        # 2. Анализ противоречий (Contradictions)
        contradiction_layers = self._detect_contradictions(
            graph_vector, layer_indices, gate_weights, subgraph_embeddings
        )
        result["contradiction_layers"] = contradiction_layers
        result["contradiction_detected"] = len(contradiction_layers) > 0
        
        if result["contradiction_detected"]:
            self._contradiction_count += 1
            logger.debug(f"[KCA] Detected {len(contradiction_layers)} contradictions at layers {contradiction_layers}")
        
        # 3. Формирование корректирующего эмбеддинга
        corrections = self._generate_corrections(
            graph_vector, lacuna_layers, contradiction_layers, gate_weights
        )
        result["corrections"] = corrections
        
        # 4. Вычисление уверенности
        result["confidence"] = self._compute_confidence(
            lacuna_layers, contradiction_layers, gate_weights
        )
        
        return result
    
    def _detect_lacunas(
        self,
        graph_vector: np.ndarray,
        layer_indices: List[int],
        gate_weights: np.ndarray
    ) -> List[int]:
        """
        Обнаружить слои с knowledge gaps (лакунами).
        
        Лакуна = область с низким вниманием к релевантным концептам.
        
        Согласно EVA: gate_weight < threshold означает область невнимания.
        """
        lacuna_layers = []
        
        for i, layer_idx in enumerate(layer_indices):
            gate = gate_weights[i] if i < len(gate_weights) else 0.5
            
            # Слой с лакуной если:
            # 1. Gate weight низкий (мало внимания)
            # 2. Graph vector имеет низкую активацию в соответствующих областях
            if gate < (1.0 - self.lacuna_threshold):
                lacuna_layers.append(layer_idx)
            elif i < len(graph_vector):
                # Дополнительная проверка: низкая активация в graph_vector
                activation = graph_vector[i % len(graph_vector)]
                if activation < self.lacuna_threshold:
                    lacuna_layers.append(layer_idx)
        
        return lacuna_layers
    
    def _detect_contradictions(
        self,
        graph_vector: np.ndarray,
        layer_indices: List[int],
        gate_weights: np.ndarray,
        subgraph_embeddings: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Обнаружить слои с противоречиями.
        
        Противоречие = конфликт между концептами в подграфе.
        
        Использует:
        1. Variance в graph_vector - высокий variance = конфликтующие концепты
        2. Анализ subgraph_embeddings если доступен
        """
        contradiction_layers = []
        
        # Вычисляем variance graph_vector как индикатор противоречий
        vector_variance = np.var(graph_vector) if len(graph_vector) > 0 else 0.0
        
        # Высокий variance указывает на конфликтующие концепты
        if vector_variance > self.contradiction_threshold:
            # Найти слои с противоречиями по gate_weights
            # Слои с gate_weight около 0.5 - зона неопределённости
            for i, layer_idx in enumerate(layer_indices):
                gate = gate_weights[i] if i < len(gate_weights) else 0.5
                
                # Слой с противоречием если gate около 0.5 (неопределённость)
                # или если есть аномалии в activation
                if 0.4 < gate < 0.6 or (i < len(graph_vector) and abs(graph_vector[i]) > 0.5):
                    contradiction_layers.append(layer_idx)
        
        # Дополнительно анализируем subgraph_embeddings если доступны
        if subgraph_embeddings is not None and len(subgraph_embeddings) >= 2:
            contr_from_emb = self._analyze_embedding_contradictions(subgraph_embeddings)
            contradiction_layers.extend(contr_from_emb)
        
        return list(set(contradiction_layers))  # Убрать дубликаты
    
    def _analyze_embedding_contradictions(self, embeddings: np.ndarray) -> List[int]:
        """
        Анализировать противоречия между embeddingами узлов.
        
        Использует cosine similarity для обнаружения противоположных векторов.
        """
        contradiction_layers = []
        
        if len(embeddings) < 2:
            return contradiction_layers
        
        # Вычислить pairwise cosine similarities
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Для каждой пары смежных концептов
        for i in range(len(normalized) - 1):
            cos_sim = np.dot(normalized[i], normalized[i + 1])
            
            # Противоречие если similarity очень низкая или отрицательная
            if cos_sim < -self.contradiction_threshold:
                # Этот концепт создаёт противоречие
                layer_idx = i % 36  # Маппинг на слои
                if layer_idx not in contradiction_layers:
                    contradiction_layers.append(layer_idx)
        
        return contradiction_layers
    
    def _generate_corrections(
        self,
        graph_vector: np.ndarray,
        lacuna_layers: List[int],
        contradiction_layers: List[int],
        gate_weights: np.ndarray
    ) -> Dict[int, Dict[str, Any]]:
        """
        Сгенерировать корректирующие эмбеддинги для слоёв.
        
        Формула: correction = direction * strength * gate_weight
        
        где direction указывает на:
        - Для лакун: направление к недостающим концептам
        - Для противоречий: нейтрализующий вектор
        """
        corrections = {}
        
        for i, layer_idx in enumerate(lacuna_layers + contradiction_layers):
            if layer_idx in corrections:
                continue
            
            gate = gate_weights[i] if i < len(gate_weights) else 0.5
            
            # Определить направление коррекции
            if layer_idx in lacuna_layers:
                # Лакуна: усилить активацию в направлении graph_vector
                direction = self._get_lacuna_direction(graph_vector, layer_idx)
                strength = self.correction_strength * (1.0 - gate)  # Больше для низких gate
            else:
                # Противоречие: нейтрализовать конфликт
                direction = self._get_contradiction_direction(graph_vector, layer_idx)
                strength = self.correction_strength * gate
            
            # Создать корректирующий эмбеддинг
            correction_embedding = direction * strength
            
            corrections[layer_idx] = {
                "embedding": correction_embedding.astype(np.float32),
                "strength": strength,
                "is_lacuna": layer_idx in lacuna_layers,
                "is_contradiction": layer_idx in contradiction_layers,
                "gate_weight": gate
            }
        
        return corrections
    
    def _get_lacuna_direction(self, graph_vector: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        Получить направление для заполнения лакуны.
        
        Использует градиент внимания: усилить области с низкой активацией.
        """
        direction = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Смещение в сторону областей с низким attention
        low_activation_mask = graph_vector < self.lacuna_threshold
        
        if len(graph_vector) >= self.embedding_dim:
            direction[:self.embedding_dim] = graph_vector[:self.embedding_dim] * (1.0 - low_activation_mask[:self.embedding_dim])
        else:
            direction[:len(graph_vector)] = graph_vector * (1.0 - low_activation_mask)
        
        # Добавить случайный шум для исследования (exploration)
        noise_scale = 0.05 * self.correction_strength
        direction += np.random.randn(self.embedding_dim).astype(np.float32) * noise_scale
        
        return direction
    
    def _get_contradiction_direction(self, graph_vector: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        Получить направление для нейтрализации противоречия.
        
        Использует negative direction: движение противоположное graph_vector.
        """
        direction = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Движение в противоположном направлении от graph_vector
        if len(graph_vector) >= self.embedding_dim:
            direction[:self.embedding_dim] = -graph_vector[:self.embedding_dim] * 0.5
        else:
            direction[:len(graph_vector)] = -graph_vector * 0.5
        
        return direction
    
    def _compute_confidence(
        self,
        lacuna_layers: List[int],
        contradiction_layers: List[int],
        gate_weights: np.ndarray
    ) -> float:
        """
        Вычислить уверенность в коррекции.
        
        Higher confidence если:
        1. Много слоёв с проблемами
        2. Gate weights согласуются (низкая дисперсия)
        """
        total_issues = len(lacuna_layers) + len(contradiction_layers)
        
        if total_issues == 0:
            return 0.0
        
        # Base confidence = proportion of layers with issues
        base_conf = min(1.0, total_issues / 36.0)
        
        # Adjust by gate weight consistency
        if len(gate_weights) > 0:
            gate_variance = np.var(gate_weights)
            consistency = 1.0 - min(1.0, gate_variance * 2)  # Lower variance = higher consistency
            confidence = base_conf * (0.5 + 0.5 * consistency)
        else:
            confidence = base_conf
        
        return float(confidence)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику работы KCA."""
        return {
            "detection_count": self._detection_count,
            "lacuna_detections": self._lacuna_count,
            "contradiction_detections": self._contradiction_count,
            "config": {
                "lacuna_threshold": self.lacuna_threshold,
                "contradiction_threshold": self.contradiction_threshold,
                "correction_strength": self.correction_strength
            }
        }