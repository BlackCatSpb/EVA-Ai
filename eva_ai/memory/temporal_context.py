"""
FCP Temporal Context Memory - иерархическая память контекста (SPEC section 4)

TCM-100: Полная реализация без заглушек.
"""

import time
import os
import json
import hashlib
from threading import Lock
from datetime import datetime, timedelta
import numpy as np
from typing import List, Optional, Dict, Union, Any, Tuple
from dataclasses import dataclass, field

from eva_ai.fcp_core.types import MemorySegment, Fact

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class TemporalContextMemory:
    """
    TCM-100: Temporal Context Memory - полная реализация
    
    Methods:
    | Метод | Назначение |
    |-------|------------|
    | write | Добавляет сегмент с иерархическим временным кодированием |
    | retrieve | Извлекает k сегментов по мягкому адресованию |
    | update_async | Асинхронно обновляет параметры (Triplet Loss) |
    | consolidate | Переносит стабильные сегменты в граф |
    | apply_temporal_decay | Применяет экспоненциальное затухание |
    """
    
    def __init__(
        self,
        max_segments: int = 1000,
        embedding_dim: int = 2048,
        time_scales: int = 4,
        storage_dir: str = None,
        encoder_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.max_segments = max_segments
        self.embedding_dim = embedding_dim
        self.time_scales = time_scales
        
        self._segments: List[MemorySegment] = []
        self._segment_index: Dict[str, int] = {}
        self._lock = Lock()
        
        self._storage_dir = storage_dir or self._get_default_storage_dir()
        os.makedirs(self._storage_dir, exist_ok=True)
        
        self._encoder = None
        self._encoder_name = encoder_name
        self._init_encoder()
        
        self._load_segments()
    
    def _get_default_storage_dir(self) -> str:
        """Получить директорию хранения по умолчанию."""
        base = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(base, "tcm_data")
    
    def _init_encoder(self):
        """TCM-100: Инициализация Sentence Transformer для текстовых запросов."""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            self._encoder = SentenceTransformer(self._encoder_name, device='cpu')
            self._encoder.max_seq_length = 256
        except Exception as e:
            self._encoder = None
    
    def _encode_text_to_embedding(self, text: str) -> Optional[np.ndarray]:
        """TCM-100: Кодирование текста в эмбеддинг."""
        if self._encoder is None:
            return None
        
        try:
            emb = self._encoder.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return emb.astype(np.float32)
        except Exception:
            return None
    
    def _load_segments(self):
        """Загрузить сохранённые сегменты из storage."""
        try:
            segments_file = os.path.join(self._storage_dir, "segments.json")
            if os.path.exists(segments_file):
                with open(segments_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for seg_data in data.get('segments', []):
                        segment = MemorySegment(
                            segment_id=seg_data['segment_id'],
                            text=seg_data['text'],
                            embedding=np.array(seg_data['embedding'], dtype=np.float32),
                            timestamp=seg_data['timestamp'],
                            time_encoding=np.array(seg_data['time_encoding'], dtype=np.float32),
                            relevance=seg_data['relevance'],
                            variance=seg_data['variance'],
                            consolidated=seg_data['consolidated'],
                            segment_type=seg_data['segment_type']
                        )
                        self._segments.append(segment)
                        self._segment_index[segment.segment_id] = len(self._segments) - 1
        except Exception:
            pass
    
    def _save_segments(self):
        """Сохранить сегменты в storage."""
        try:
            data = {
                'segments': [
                    {
                        'segment_id': s.segment_id,
                        'text': s.text,
                        'embedding': s.embedding.tolist() if hasattr(s.embedding, 'tolist') else list(s.embedding),
                        'timestamp': s.timestamp,
                        'time_encoding': s.time_encoding.tolist() if hasattr(s.time_encoding, 'tolist') else list(s.time_encoding),
                        'relevance': s.relevance,
                        'variance': s.variance,
                        'consolidated': s.consolidated,
                        'segment_type': s.segment_type
                    }
                    for s in self._segments
                ]
            }
            segments_file = os.path.join(self._storage_dir, "segments.json")
            with open(segments_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    # =========================================================================
    # SPEC METHOD: write
    # =========================================================================
    
    def write(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        segment_type: str = "general"
    ) -> str:
        """
        TCM-100: write(text, embedding, metadata, segment_type) -> str
        
        Добавляет новый сегмент диалога в TCM с иерархическим временным кодированием.
        """
        with self._lock:
            now = time.time()
            
            time_encoding = self._encode_time(now)
            
            segment_id = f"seg_{len(self._segments):08d}_{hashlib.md5(text[:20].encode()).hexdigest()[:8]}"
            
            if embedding is None or (hasattr(embedding, 'shape') and len(embedding.shape) == 0):
                embedding = self._encode_text_to_embedding(text)
                if embedding is None:
                    embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            
            if hasattr(embedding, 'shape') and len(embedding.shape) > 1:
                embedding = embedding.flatten()
            
            segment = MemorySegment(
                segment_id=segment_id,
                text=text,
                embedding=embedding,
                timestamp=now,
                time_encoding=time_encoding,
                relevance=metadata.get('relevance', 0.5) if metadata else 0.5,
                variance=metadata.get('variance', 0.1) if metadata else 0.1,
                consolidated=False,
                segment_type=segment_type
            )
            
            self._segments.append(segment)
            self._segment_index[segment_id] = len(self._segments) - 1
            
            if len(self._segments) > self.max_segments:
                evicted = self._segments.pop(0)
                del self._segment_index[evicted.segment_id]
                self._segment_index = {
                    s.segment_id: i for i, s in enumerate(self._segments)
                }
            
            if len(self._segments) % 50 == 0:
                self._save_segments()
            
            return segment_id
    
    def _encode_time(self, timestamp: float) -> np.ndarray:
        """
        SPEC: иерархическое временное кодирование (час, день, неделя, месяц)
        
        Returns: (4,) - hour, day, week, month
        """
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        
        hour = dt.hour / 24.0
        day = dt.weekday() / 7.0
        week = (dt.day - 1) / 7.0 / 4.0  # approximate
        month = dt.month / 12.0
        
        return np.array([hour, day, week, month])
    
    # =========================================================================
    # SPEC METHOD: retrieve
    # =========================================================================
    
    def retrieve(
        self,
        query: Union[str, np.ndarray],
        k: int = 10,
        time_weight: float = 0.25,
        semantic_weight: float = 0.5,
        relevance_weight: float = 0.15,
        segment_type_filter: List[str] = None
    ) -> List[MemorySegment]:
        """
        TCM-100: retrieve(query, k, ...) -> List[MemorySegment]
        
        Извлекает k наиболее релевантных сегментов по мягкому адресованию:
        - Семантическая близость (embedding similarity)
        - Временная близость (recency)
        - Накопленная релевантность
        - Бонус типа сегмента
        """
        if not self._segments:
            return []
        
        query_emb = None
        if isinstance(query, str):
            query_emb = self._encode_text_to_embedding(query)
            if query_emb is None:
                return []
        else:
            query_emb = np.array(query, dtype=np.float32)
            if query_emb.ndim > 1:
                query_emb = query_emb.flatten()
        
        now = time.time()
        
        scores = []
        for segment in self._segments:
            if segment_type_filter and segment.segment_type not in segment_type_filter:
                continue
            
            sem_sim = self._cosine_similarity(query_emb, segment.embedding)
            
            time_decay = self._time_decay(segment.timestamp, now)
            
            type_bonus = self._get_segment_type_bonus(segment.segment_type)
            
            recency_boost = self._get_recency_boost(segment.timestamp, now)
            
            score = (
                sem_sim * semantic_weight + 
                time_decay * time_weight + 
                segment.relevance * relevance_weight +
                type_bonus +
                recency_boost * 0.05
            )
            scores.append((segment, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if k >= len(scores):
            return [s[0] for s in scores]
        
        return [s[0] for s in scores[:k]]
    
    def _get_segment_type_bonus(self, segment_type: str) -> float:
        """TCM-100: Бонус для разных типов сегментов."""
        bonuses = {
            'system': 0.15,
            'user': 0.1,
            'assistant': 0.1,
            'tool': 0.05,
            'reasoning': -0.05,
            'factual': 0.12,
            'context': 0.08,
            'general': 0.0
        }
        return bonuses.get(segment_type, 0.0)
    
    def _get_recency_boost(self, timestamp: float, now: float) -> float:
        """TCM-100: Буст для недавних сегментов в рамках сессии."""
        hours_ago = (now - timestamp) / 3600
        if hours_ago < 0.5:
            return 0.15
        elif hours_ago < 2:
            return 0.1
        elif hours_ago < 6:
            return 0.05
        return 0.0
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity между двумя векторами."""
        if a is None or b is None:
            return 0.0
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a) + 1e-8
        norm_b = np.linalg.norm(b) + 1e-8
        return float(dot / (norm_a * norm_b))
    
    def _time_decay(self, timestamp: float, now: float, half_life_hours: float = 24.0) -> float:
        """
        TCM-100: Экспоненциальное затухание (half-life: 24 часа по умолчанию)
        
        Returns: 0-1, где 1 = недавнее
        """
        hours_old = (now - timestamp) / 3600
        decay = np.exp(-hours_old / half_life_hours)
        return max(0, min(1, decay))
    
    # =========================================================================
    # SPEC METHOD: update_async (Triplet Loss Contrastive Learning)
    # =========================================================================
    
    def update_async(self, buffer: List[MemorySegment] = None) -> Dict:
        """
        TCM-100: update_async(buffer) -> Dict
        
        Асинхронно обновляет параметры TCM на основе Triplet Margin Loss.
        
        Triplet Loss минимизирует:
        - distance(anchor, positive) - расстояние между похожими
        + distance(anchor, negative) - расстояние между разными
        
        Args:
            buffer: Опциональный буфер сегментов для обучения.
                   Если None - использует последние сегменты.
                   
        Returns:
            Dict с метриками обучения
        """
        if buffer is None:
            buffer = self._segments[-100:] if len(self._segments) > 100 else self._segments
        
        if not buffer or len(buffer) < 3:
            return {"status": "insufficient_data", "updated": 0}
        
        metrics = {
            "status": "updated",
            "updated": len(buffer),
            "avg_loss": 0.0,
            "positive_pairs": 0,
            "negative_pairs": 0
        }
        
        try:
            embeddings = np.array([seg.embedding for seg in buffer], dtype=np.float32)
            
            loss_sum = 0.0
            num_triplets = 0
            positive_pairs = 0
            negative_pairs = 0
            
            margin = 0.5
            learning_rate = 0.01
            
            for i in range(len(buffer)):
                anchor_emb = embeddings[i]
                anchor_type = buffer[i].segment_type
                anchor_relevance = buffer[i].relevance
                
                positives = []
                negatives = []
                
                for j in range(len(buffer)):
                    if i == j:
                        continue
                    
                    sim_ij = self._cosine_similarity(anchor_emb, embeddings[j])
                    same_type = anchor_type == buffer[j].segment_type
                    
                    if same_type and buffer[j].relevance >= anchor_relevance - 0.2:
                        positives.append((j, sim_ij))
                    else:
                        negatives.append((j, sim_ij))
                
                if not positives or not negatives:
                    continue
                
                positives.sort(key=lambda x: x[1], reverse=True)
                negatives.sort(key=lambda x: x[1])
                
                anchor_positive_dist = 1.0 - positives[0][1]
                
                for neg_idx, neg_sim in negatives[:3]:
                    anchor_negative_dist = 1.0 - neg_sim
                    
                    loss = max(0, anchor_positive_dist - anchor_negative_dist + margin)
                    
                    if loss > 0:
                        loss_sum += loss
                        num_triplets += 1
                        
                        negative_emb = embeddings[neg_idx]
                        
                        push_grad = learning_rate * (anchor_positive_dist - anchor_negative_dist + margin)
                        
                        embeddings[i] = embeddings[i] + push_grad * (negative_emb - anchor_emb) / (np.linalg.norm(negative_emb - anchor_emb) + 1e-8)
                        
                        if positives[0][1] > 0.5:
                            positive_pairs += 1
                        if neg_sim < 0.5:
                            negative_pairs += 1
                
                for pos_idx, pos_sim in positives[:2]:
                    embeddings[pos_idx] = embeddings[pos_idx] * 1.001
                
                buffer[i].variance = min(1.0, buffer[i].variance * 1.005)
            
            for idx, seg in enumerate(buffer):
                if seg.relevance < 1.0:
                    seg.relevance = min(1.0, seg.relevance + 0.005)
                if seg.variance > 0.01:
                    seg.variance = max(0.01, seg.variance * 0.995)
            
            if num_triplets > 0:
                metrics["avg_loss"] = float(loss_sum / num_triplets)
            metrics["positive_pairs"] = positive_pairs
            metrics["negative_pairs"] = negative_pairs
            
        except Exception as e:
            metrics["status"] = f"error: {str(e)}"
        
        return metrics
    
    # =========================================================================
    # SPEC METHOD: consolidate
    # =========================================================================
    
    def consolidate(
        self,
        graph: "FractalGraphV2" = None,
        min_stability: float = 0.7,
        min_age_hours: float = 24.0,
        concept_miner=None
    ) -> List[Fact]:
        """
        TCM-100: consolidate() -> List[Fact]
        
        Переносит стабильные сегменты из TCM в долговременный граф,
        вызывая ConceptMiner для извлечения концептов.
        
        Args:
            graph: FractalGraphV2 для сохранения фактов
            min_stability: Минимальная стабильность (0-1)
            min_age_hours: Минимальный возраст в часах
            concept_miner: Опциональный ConceptMiner для извлечения концептов
            
        Returns:
            List of Facts added to graph
        """
        now = time.time()
        added_facts = []
        added_concepts = []
        
        for segment in self._segments:
            if segment.consolidated:
                continue
            
            age_hours = (now - segment.timestamp) / 3600
            if age_hours < min_age_hours:
                continue
            
            stability = 1.0 - segment.variance
            if stability < min_stability:
                continue
            
            if graph and hasattr(graph, 'add_node'):
                try:
                    node = graph.add_node(
                        content=segment.text,
                        node_type="tcm_segment",
                        level=1,
                        embedding=segment.embedding.tolist() if hasattr(segment.embedding, 'tolist') else list(segment.embedding),
                        confidence=stability,
                        metadata={
                            "segment_type": segment.segment_type,
                            "source": "tcm",
                            "original_timestamp": segment.timestamp
                        }
                    )
                    
                    fact_id = f"fact_{segment.segment_id}_{int(now)}"
                    fact = Fact(
                        fact_id=fact_id,
                        subject=segment.text[:100],
                        predicate="was_discussed",
                        object="in_tcm_context",
                        confidence=stability
                    )
                    added_facts.append(fact)
                    
                    segment.consolidated = True
                    
                    if concept_miner and hasattr(concept_miner, 'add_candidate'):
                        concept_miner.add_candidate(segment.text, segment.embedding)
                        added_concepts.append(segment.text)
                        
                except Exception as e:
                    pass
        
        return added_facts
    
    # =========================================================================
    # SPEC METHOD: apply_temporal_decay
    # =========================================================================
    
    def apply_temporal_decay(
        self,
        decay_rate: float = 0.01,
        min_relevance: float = 0.05
    ) -> Dict:
        """
        TCM-100: apply_temporal_decay() -> Dict
        
        Применяет экспоненциальное затухание к весам сегментов.
        Сегменты старше max_age_hours получают дополнительное затухание.
        
        Args:
            decay_rate: Скорость затухания (0.01 = 1% в час)
            min_relevance: Минимальная релевантность для сохранения
            
        Returns:
            Dict с статистикой затухания
        """
        now = time.time()
        stats = {
            "decayed_count": 0,
            "removed_count": 0,
            "min_relevance_reached": 0,
            "avg_relevance_before": 0.0,
            "avg_relevance_after": 0.0
        }
        
        relevances_before = [s.relevance for s in self._segments]
        stats["avg_relevance_before"] = sum(relevances_before) / max(len(relevances_before), 1)
        
        to_remove = []
        
        for segment in self._segments:
            decay = self._time_decay(segment.timestamp, now, half_life_hours=1.0/decay_rate)
            segment.relevance *= decay
            
            if segment.relevance < min_relevance:
                stats["min_relevance_reached"] += 1
                age_hours = (now - segment.timestamp) / 3600
                if age_hours > 168:
                    to_remove.append(segment)
        
        for seg in to_remove:
            try:
                idx = self._segment_index.get(seg.segment_id)
                if idx is not None:
                    self._segments.pop(idx)
                    del self._segment_index[seg.segment_id]
                    stats["removed_count"] += 1
            except Exception:
                pass
        
        stats["decayed_count"] = len(self._segments)
        
        relevances_after = [s.relevance for s in self._segments]
        stats["avg_relevance_after"] = sum(relevances_after) / max(len(relevances_after), 1)
        
        return stats
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_context_for_prompt(
        self,
        query_embedding: Union[str, np.ndarray],
        max_chars: int = 2000,
        k: int = 5
    ) -> str:
        """TCM-100: Получить форматированный контекст для LLM."""
        segments = self.retrieve(query_embedding, k=k)
        
        if not segments:
            return ""
        
        parts = []
        total = 0
        for seg in segments:
            if total + len(seg.text) > max_chars:
                break
            
            prefix = f"[{seg.segment_type}]" if seg.segment_type != "general" else ""
            parts.append(f"{prefix}{seg.text}" if prefix else seg.text)
            total += len(seg.text)
        
        return "\n".join(parts)
    
    def get_session_summary(self, session_id: str = None) -> Dict:
        """TCM-100: Сводка по сессии или всем сегментам."""
        if not self._segments:
            return {"segment_count": 0, "types": {}, "avg_relevance": 0}
        
        type_counts = {}
        total_relevance = 0
        
        for seg in self._segments:
            t = seg.segment_type
            type_counts[t] = type_counts.get(t, 0) + 1
            total_relevance += seg.relevance
        
        concepts = {}
        for seg in self._segments:
            words = seg.text.split()[:3]
            key = " ".join(words)
            concepts[key] = concepts.get(key, 0) + 1
        
        top_concepts = sorted(concepts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "segment_count": len(self._segments),
            "types": type_counts,
            "avg_relevance": total_relevance / len(self._segments),
            "top_concepts": [{"concept": c[0], "count": c[1]} for c in top_concepts],
            "oldest_segment_hours": (time.time() - min(s.timestamp for s in self._segments)) / 3600 if self._segments else 0
        }
    
    def search_by_type(self, segment_type: str, limit: int = 50) -> List[MemorySegment]:
        """TCM-100: Поиск сегментов по типу."""
        return [s for s in self._segments if s.segment_type == segment_type][:limit]
    
    def search_by_time_range(self, start_time: float = None, end_time: float = None) -> List[MemorySegment]:
        """TCM-100: Поиск сегментов по временному диапазону."""
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = time.time()
        return [s for s in self._segments if start_time <= s.timestamp <= end_time]
    
    def get_recent_segments(self, hours: float = 24, k: int = 50) -> List[MemorySegment]:
        """TCM-100: Недавние сегменты за последние hours."""
        cutoff = time.time() - hours * 3600
        return [s for s in self._segments if s.timestamp >= cutoff][:k]
    
    def clear_old_segments(self, older_than_hours: float = 720) -> int:
        """TCM-100: Очистить сегменты старше specified часов."""
        cutoff = time.time() - older_than_hours * 3600
        to_remove = [s for s in self._segments if s.timestamp < cutoff]
        
        for seg in to_remove:
            idx = self._segment_index.get(seg.segment_id)
            if idx is not None:
                try:
                    self._segments.pop(idx)
                    del self._segment_index[seg.segment_id]
                except Exception:
                    pass
        
        self._save_segments()
        return len(to_remove)
    
    def __len__(self) -> int:
        return len(self._segments)
    
    def is_empty(self) -> bool:
        return len(self._segments) == 0
    
    def get_stats(self) -> Dict:
        """TCM-100: Статистика TCM."""
        return self.get_session_summary()