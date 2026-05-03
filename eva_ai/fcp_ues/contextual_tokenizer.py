import os
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger("FCP.ContextualTokenizer")


class ContextualTokenizer:
    """
    Контекстуальный токенизатор (EVA.txt раздел 2.1).
    Адаптирует токенизацию под контекст запроса и состояние графа знаний.
    """
    
    def __init__(self, base_tokenizer, fractal_graph=None, embedding_model=None):
        """
        Args:
            base_tokenizer: базовый токенизатор (например, GPT2Tokenizer)
            fractal_graph: FractalGraphV2 для уточнения токенов
            embedding_model: модель для получения эмбеддингов токенов
        """
        self.base_tokenizer = base_tokenizer
        self.fractal_graph = fractal_graph
        self.embedding_model = embedding_model
        self.special_tokens = self._load_special_tokens()
        logger.info("ContextualTokenizer initialized")
    
    def _load_special_tokens(self) -> Dict[str, int]:
        """Загрузка специальных токенов из базового токенизатора."""
        special = {}
        if hasattr(self.base_tokenizer, 'special_tokens_map'):
            special = self.base_tokenizer.special_tokens_map
        return special
    
    def tokenize_with_context(self, text: str, context_nodes: List[str] = None) -> Dict:
        """
        Токенизация с учётом контекста (узлов графа).
        
        Args:
            text: исходный текст
            context_nodes: список ID узлов графа для контекста
            
        Returns:
            Dict с токенами, масками и контекстными весами
        """
        # Базовая токенизация
        base_tokens = self.base_tokenizer.encode(text)
        
        result = {
            "input_ids": base_tokens,
            "attention_mask": [1] * len(base_tokens),
            "context_weights": [1.0] * len(base_tokens),
            "graph_anchors": []
        }
        
        # Если есть граф, обогащаем токены контекстом
        if self.fractal_graph and context_nodes:
            try:
                graph_weights = self._compute_graph_weights(base_tokens, context_nodes)
                result["context_weights"] = graph_weights
                result["graph_anchors"] = self._find_graph_anchors(base_tokens, context_nodes)
            except Exception as e:
                logger.debug(f"Graph context enrichment failed: {e}")
        
        return result
    
    def _compute_graph_weights(self, tokens: List[int], context_nodes: List[str]) -> List[float]:
        """Вычисление весов токенов на основе графа."""
        weights = [1.0] * len(tokens)
        
        if not self.embedding_model:
            return weights
        
        try:
            # Получаем эмбеддинги токенов
            token_texts = self.base_tokenizer.convert_ids_to_tokens(tokens)
            token_embeddings = self.embedding_model.encode(token_texts)
            
            # Получаем эмбеддинги узлов контекста
            context_embeddings = []
            for node_id in context_nodes:
                node = self.fractal_graph.get_node(node_id)
                if node and 'emb' in node:
                    context_embeddings.append(node['emb'])
            
            if not context_embeddings:
                return weights
            
            # Вычисляем сходство каждого токена с контекстом
            for i, tok_emb in enumerate(token_embeddings):
                max_sim = 0.0
                for ctx_emb in context_embeddings:
                    sim = self._cosine_similarity(tok_emb, ctx_emb)
                    max_sim = max(max_sim, sim)
                weights[i] = 0.5 + 0.5 * max_sim  # Нормализация к [0.5, 1.0]
        
        except Exception as e:
            logger.debug(f"Graph weights computation failed: {e}")
        
        return weights
    
    def _find_graph_anchors(self, tokens: List[int], context_nodes: List[str]) -> List[Dict]:
        """Поиск токенов-якорей, соответствующих узлам графа."""
        anchors = []
        
        try:
            token_texts = self.base_tokenizer.convert_ids_to_tokens(tokens)
            
            for node_id in context_nodes:
                node = self.fractal_graph.get_node(node_id)
                if not node:
                    continue
                
                node_label = node.get('label', '').lower()
                
                for i, tok_text in enumerate(token_texts):
                    if node_label in tok_text.lower() or tok_text.lower() in node_label:
                        anchors.append({
                            "token_idx": i,
                            "node_id": node_id,
                            "label": node_label,
                            "score": 1.0
                        })
        
        except Exception as e:
            logger.debug(f"Graph anchor search failed: {e}")
        
        return anchors
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Косинусное сходство между векторами."""
        if len(a) == 0 or len(b) == 0:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def decode_with_context(self, token_ids: List[int], context_weights: List[float] = None) -> str:
        """Декодирование с учётом контекстных весов."""
        return self.base_tokenizer.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """Размер словаря."""
        return len(self.base_tokenizer)
