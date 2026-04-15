"""
L1: Activation Profiler - Профили активаций для доменов

Хранит статистику генераций по доменам:
- Центроиды (fingerprint'ы запросов)
- Метрики качества (entropy, latency, quality)
- Количество образцов

Узел типа: activation_profile
Связи: derived_from → model_root, similar_to → activation_profile
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from .fractal_graph_l1_l2 import FractalGraphL1L2, ActivationProfileData

logger = logging.getLogger("eumi.profiler")


@dataclass
class ProfileStats:
    """Статистика профиля."""
    avg_entropy: float = 0.0
    avg_latency_ms: float = 0.0
    avg_quality: float = 0.0
    sample_count: int = 0
    last_updated: float = 0.0


@dataclass
class SimilarProfile:
    """Похожий профиль с оценкой сходства."""
    profile: ActivationProfileData
    similarity: float


class ActivationProfiler:
    """
    L1: Structural Shadow - профили активаций.
    
    Каждый профиль хранит центроид (768 float) в embedding поле узла
    и метаданные в JSON поле.
    """
    
    def __init__(
        self,
        graph: FractalGraphL1L2,
        embedder_model: Optional[str] = None
    ):
        """
        Args:
            graph: FractalGraphL1L2 instance
            embedder_model: Модель для sentence embeddings (default: 'all-MiniLM-L6-v2')
        """
        self.graph = graph
        self.embedding_dim = 768
        self._embedder = None
        self._embedder_model = embedder_model
        
    def _get_embedder(self) -> Optional[Any]:
        """Ленивая инициализация embedder'а через singleton кеш."""
        if self._embedder is None:
            try:
                from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer
                self._embedder = get_sentence_transformer(device='auto')
                if self._embedder:
                    self.embedding_dim = getattr(self._embedder, 'get_sentence_embedding_dimension', lambda: 768)()
                    logger.info(f"Loaded embedder via singleton: multilingual-e5-base ({self.embedding_dim}d)")
            except Exception as e:
                logger.warning(f"Failed to load embedder: {e}")
        return self._embedder
    
    def create_profile(
        self,
        domain: str,
        model_id: str = "model_a",
        quant_profile: str = "Q4_K_M",
        initial_fingerprint: Optional[np.ndarray] = None
    ) -> str:
        """
        Создать новый профиль активации.
        
        Args:
            domain: Домен (например, "astrophysics")
            model_id: Какая модель GGUF (model_a или model_b)
            quant_profile: Профиль квантизации
            initial_fingerprint: Начальный fingerprint (768 float)
            
        Returns:
            profile_id: ID созданного узла
        """
        profile_id = self.graph.create_activation_profile(
            domain=domain,
            model_id=model_id,
            quant_profile=quant_profile,
            initial_embedding=initial_fingerprint
        )
        
        logger.info(f"Created activation profile: {profile_id} for domain={domain}")
        return profile_id
    
    def get_or_create_profile(
        self,
        domain: str,
        model_id: str = "model_a",
        quant_profile: str = "Q4_K_M"
    ) -> str:
        """
        Получить существующий профиль или создать новый.
        
        Args:
            domain: Домен
            model_id: ID модели
            quant_profile: Профиль квантизации
            
        Returns:
            profile_id: ID профиля
        """
        profile = self.graph.get_activation_profile(domain, model_id)
        
        if profile is not None:
            return profile.node_id
        
        # Создаём новый
        return self.create_profile(domain, model_id, quant_profile)
    
    def update_profile(
        self,
        profile_id: str,
        fingerprint: np.ndarray,
        stats: Dict[str, float]
    ) -> bool:
        """
        Инкрементальное обновление профиля.
        
        new_centroid = (old_centroid * n + fingerprint) / (n + 1)
        
        Args:
            profile_id: ID профиля
            fingerprint: Вектор запроса (768 float)
            stats: {'entropy': float, 'latency_ms': float, 'quality': float}
            
        Returns:
            True если успешно
        """
        entropy = stats.get('entropy', 0.0)
        latency_ms = stats.get('latency_ms', 0.0)
        quality = stats.get('quality', 0.0)
        
        success = self.graph.update_activation_profile(
            profile_id=profile_id,
            new_fingerprint=fingerprint,
            entropy=entropy,
            latency_ms=latency_ms,
            quality=quality
        )
        
        if success:
            logger.debug(f"Updated profile {profile_id}: entropy={entropy:.3f}, latency={latency_ms:.1f}ms")
        else:
            logger.warning(f"Failed to update profile: {profile_id}")
        
        return success
    
    def record_generation(
        self,
        domain: str,
        model_id: str,
        query: str,
        response: str,
        entropy: float = 0.0,
        latency_ms: float = 0.0,
        quality: float = 0.0
    ) -> bool:
        """
        Записать генерацию и обновить профиль.
        
        Args:
            domain: Домен запроса
            model_id: Какая модель использовалась
            query: Текст запроса
            response: Текст ответа
            entropy: Энтропия генерации
            latency_ms: Время генерации в мс
            quality: Оценка качества (0-1)
            
        Returns:
            True если успешно
        """
        # Получаем или создаём профиль
        profile_id = self.get_or_create_profile(domain, model_id)
        
        # Вычисляем fingerprint
        fingerprint = self.compute_fingerprint(query, response)
        
        # Обновляем профиль
        stats = {
            'entropy': entropy,
            'latency_ms': latency_ms,
            'quality': quality
        }
        
        return self.update_profile(profile_id, fingerprint, stats)
    
    def get_profile(
        self,
        domain: str,
        model_id: str = "model_a"
    ) -> Optional[ActivationProfileData]:
        """
        Получить профиль по домену и модели.
        
        Args:
            domain: Домен
            model_id: ID модели
            
        Returns:
            Профиль или None
        """
        return self.graph.get_activation_profile(domain, model_id)
    
    def find_similar_profiles(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SimilarProfile]:
        """
        Найти похожие профили по тексту запроса.
        
        Args:
            query: Текст запроса
            top_k: Сколько вернуть
            
        Returns:
            Список похожих профилей
        """
        # Вычисляем embedding запроса
        query_embedding = self._compute_embedding(query)
        
        if query_embedding is None:
            return []
        
        # Ищем похожие профили
        similar = self.graph.find_similar_profiles(query_embedding, top_k=top_k)
        
        results = []
        for profile_id, similarity in similar:
            # Получаем полные данные профиля
            # Note: This is inefficient - should optimize in real implementation
            all_profiles = self.graph.list_activation_profiles()
            for profile in all_profiles:
                if profile.node_id == profile_id:
                    results.append(SimilarProfile(profile=profile, similarity=similarity))
                    break
        
        return results
    
    def compute_fingerprint(
        self,
        query: str,
        response: str
    ) -> np.ndarray:
        """
        Вычислить fingerprint для запроса и ответа.
        
        Комбинация embeddings запроса + ответа, нормализованная до 768 float.
        
        Args:
            query: Текст запроса
            response: Текст ответа
            
        Returns:
            Вектор fingerprint (768 float)
        """
        query_emb = self._compute_embedding(query)
        response_emb = self._compute_embedding(response)
        
        if query_emb is None or response_emb is None:
            # Fallback: случайный вектор
            return np.random.randn(self.embedding_dim).astype(np.float32) * 0.01
        
        # Среднее embeddings
        combined = (query_emb + response_emb) / 2.0
        
        # Нормализуем до 768
        if len(combined) < self.embedding_dim:
            # Pad with zeros
            combined = np.pad(combined, (0, self.embedding_dim - len(combined)))
        elif len(combined) > self.embedding_dim:
            # Truncate
            combined = combined[:self.embedding_dim]
        
        return combined.astype(np.float32)
    
    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Вычислить embedding текста.
        
        Args:
            text: Текст
            
        Returns:
            Вектор embedding или None
        """
        embedder = self._get_embedder()
        
        if embedder is None:
            # Fallback: простая хеш-функция
            return self._simple_hash_embedding(text)
        
        try:
            embedding = embedder.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to compute embedding: {e}")
            return self._simple_hash_embedding(text)
    
    def _simple_hash_embedding(self, text: str) -> np.ndarray:
        """
        Простая хеш-функция для embedding (fallback).
        
        Args:
            text: Текст
            
        Returns:
            Вектор (768 float)
        """
        # Простой хеш на основе символов
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for i, char in enumerate(text[:1000]):  # Limit to first 1000 chars
            idx = (i + ord(char)) % self.embedding_dim
            vec[idx] += 1.0
        
        # Нормализуем
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def get_profile_stats(self, profile_id: str) -> Optional[ProfileStats]:
        """
        Получить статистику профиля.
        
        Args:
            profile_id: ID профиля
            
        Returns:
            Статистика или None
        """
        # Получаем все профили и ищем нужный
        all_profiles = self.graph.list_activation_profiles()
        
        for profile in all_profiles:
            if profile.node_id == profile_id:
                return ProfileStats(
                    avg_entropy=profile.avg_entropy,
                    avg_latency_ms=profile.avg_latency_ms,
                    avg_quality=profile.avg_quality,
                    sample_count=profile.sample_count,
                    last_updated=profile.last_accessed
                )
        
        return None
    
    def list_all_profiles(self) -> List[ActivationProfileData]:
        """Получить все профили."""
        return self.graph.list_activation_profiles()
    
    def get_domain_from_query(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Определить домен запроса на основе похожих профилей.
        
        Args:
            query: Текст запроса
            top_k: Сколько доменов вернуть
            
        Returns:
            Список (domain, confidence)
        """
        similar = self.find_similar_profiles(query, top_k=top_k)
        
        # Группируем по доменам
        domain_scores: Dict[str, List[float]] = {}
        for sim in similar:
            domain = sim.profile.domain
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(sim.similarity)
        
        # Усредняем скоры
        results = []
        for domain, scores in domain_scores.items():
            avg_score = sum(scores) / len(scores)
            results.append((domain, avg_score))
        
        # Сортируем по confidence
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def create_default_profiler(
    graph: FractalGraphL1L2,
    embedder_model: Optional[str] = None
) -> ActivationProfiler:
    """Фабричный метод создания профайлера."""
    return ActivationProfiler(graph, embedder_model)
