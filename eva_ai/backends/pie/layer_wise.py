"""
LayerWiseEmbedder - Параллельная многоуровневая система эмбеддингов

Архитектура:
- Каждый слой обрабатывается независимо
- Параллельное выполнение через ThreadPool
- Кэширование на уровне слоёв
- Адаптивная квантизация по слоям

Оптимизации:
1. Layer-level caching - кэш каждого слоя отдельно
2. Parallel layer processing - параллельная обработка
3. Quantization per layer - разная квантизация для разных слоёв
4. Dynamic layer selection - активация только нужных слоёв
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

logger = logging.getLogger("eumi.embeddings.layer_wise")


@dataclass
class LayerConfig:
    """Конфигурация слоя эмбеддинга."""
    name: str
    dimension: int
    quantization_bits: int = 32  # fp32 по умолчанию
    enabled: bool = True
    cache_enabled: bool = True
    parallel_workers: int = 4
    
    # Оптимизации
    use_half_precision: bool = False
    normalize: bool = True
    pooling_strategy: str = "mean"  # mean, cls, max


class LayerCache:
    """Кэш для отдельного слоя."""
    
    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.lock = threading.RLock()
        self.access_count: Dict[str, int] = {}
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Получить из кэша."""
        with self.lock:
            if key in self.cache:
                self.access_count[key] += 1
                return self.cache[key].copy()
            return None
    
    def put(self, key: str, value: np.ndarray) -> None:
        """Сохранить в кэш."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # LRU eviction
                least_used = min(self.access_count, key=self.access_count.get)
                del self.cache[least_used]
                del self.access_count[least_used]
            
            self.cache[key] = value.copy()
            self.access_count[key] = 1
    
    def clear(self) -> None:
        """Очистить кэш."""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()


class EmbeddingLayer:
    """
    Один слой эмбеддинга с оптимизациями.
    
    Может быть:
    - Semantic layer (e5, MiniLM) - для текста
    - Graph layer (Node2Vec, GNN) - для узлов графа
    - Projection layer - для выравнивания пространств
    """
    
    def __init__(self, config: LayerConfig, model_path: Optional[str] = None):
        self.config = config
        self.model_path = model_path
        self.cache = LayerCache()
        self.model = None
        self._lock = threading.RLock()
        
        # Статистика
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_calls": 0,
            "avg_time": 0.0
        }
        
        if model_path:
            self._load_model()
    
    def _load_model(self) -> None:
        """Загрузить модель слоя."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_path)
            logger.info(f"Loaded embedding layer: {self.config.name}")
        except Exception as e:
            logger.error(f"Failed to load layer {self.config.name}: {e}")
            self.model = None
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Получить эмбеддинги для текстов.
        
        Args:
            texts: Список текстов
            
        Returns:
            Массив эмбеддингов (N, dimension)
        """
        import time
        start_time = time.time()
        
        with self._lock:
            self.stats["total_calls"] += 1
        
        # Проверка кэша
        if self.config.cache_enabled:
            cached_results = []
            texts_to_process = []
            indices_to_process = []
            
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    cached_results.append((i, cached))
                    with self._lock:
                        self.stats["cache_hits"] += 1
                else:
                    texts_to_process.append(text)
                    indices_to_process.append(i)
                    with self._lock:
                        self.stats["cache_misses"] += 1
            
            if not texts_to_process:
                # Все в кэше
                results = [None] * len(texts)
                for idx, emb in cached_results:
                    results[idx] = emb
                return np.array(results)
        else:
            texts_to_process = texts
            indices_to_process = list(range(len(texts)))
            cached_results = []
        
        # Обработка
        if self.model is None:
            # Fallback: случайные эмбеддинги
            embeddings = np.random.randn(len(texts_to_process), self.config.dimension)
        else:
            embeddings = self.model.encode(
                texts_to_process,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize
            )
            
            # Квантизация если нужно
            if self.config.quantization_bits < 32:
                embeddings = self._quantize(embeddings, self.config.quantization_bits)
        
        # Сохранение в кэш
        if self.config.cache_enabled:
            for idx, text, emb in zip(indices_to_process, texts_to_process, embeddings):
                self.cache.put(text, emb)
        
        # Сборка результата
        results = [None] * len(texts)
        for (idx, emb) in cached_results:
            results[idx] = emb
        for idx, emb in zip(indices_to_process, embeddings):
            results[idx] = emb
        
        elapsed = time.time() - start_time
        with self._lock:
            self.stats["avg_time"] = (
                self.stats["avg_time"] * (self.stats["total_calls"] - 1) + elapsed
            ) / self.stats["total_calls"]
        
        return np.array(results)
    
    def _quantize(self, embeddings: np.ndarray, bits: int) -> np.ndarray:
        """Квантизировать эмбеддинги."""
        if bits == 16:
            return embeddings.astype(np.float16).astype(np.float32)
        elif bits == 8:
            # Min-max квантизация
            min_val = embeddings.min()
            max_val = embeddings.max()
            scale = (max_val - min_val) / 255.0
            quantized = ((embeddings - min_val) / scale).astype(np.uint8)
            return (quantized.astype(np.float32) * scale + min_val)
        return embeddings
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику слоя."""
        with self._lock:
            cache_hit_rate = 0.0
            if self.stats["total_calls"] > 0:
                cache_hit_rate = self.stats["cache_hits"] / self.stats["total_calls"]
            
            return {
                "name": self.config.name,
                "dimension": self.config.dimension,
                "cache_size": len(self.cache.cache),
                "cache_hit_rate": cache_hit_rate,
                "avg_time_ms": self.stats["avg_time"] * 1000,
                "total_calls": self.stats["total_calls"],
                **self.stats
            }


class LayerWiseEmbedder:
    """
    Многоуровневая система эмбеддингов с параллельной обработкой.
    
    Позволяет:
    - Обрабатывать разные слои параллельно
    - Кэшировать каждый слой отдельно
    - Применять разную квантизацию
    - Активировать только нужные слои
    
    Example:
        >>> embedder = LayerWiseEmbedder()
        >>> embedder.add_layer(LayerConfig("semantic", 384), "e5-model")
        >>> embedder.add_layer(LayerConfig("graph", 128), "node2vec")
        >>> 
        >>> # Параллельная обработка
        >>> results = embedder.embed_parallel("текст", ["semantic", "graph"])
        >>> semantic_emb = results["semantic"]
        >>> graph_emb = results["graph"]
    """
    
    def __init__(self, max_workers: int = 4):
        self.layers: Dict[str, EmbeddingLayer] = {}
        self.max_workers = max_workers
        self._lock = threading.RLock()
        
        logger.info(f"LayerWiseEmbedder initialized (max_workers={max_workers})")
    
    def add_layer(self, config: LayerConfig, model_path: Optional[str] = None) -> None:
        """
        Добавить слой эмбеддинга.
        
        Args:
            config: Конфигурация слоя
            model_path: Путь к модели (если нужна)
        """
        with self._lock:
            self.layers[config.name] = EmbeddingLayer(config, model_path)
            logger.info(f"Added layer: {config.name} (dim={config.dimension})")
    
    def remove_layer(self, name: str) -> None:
        """Удалить слой."""
        with self._lock:
            if name in self.layers:
                del self.layers[name]
                logger.info(f"Removed layer: {name}")
    
    def embed_sequential(
        self,
        texts: List[str],
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Последовательная обработка по слоям.
        
        Args:
            texts: Список текстов
            layer_names: Какие слои использовать (None = все)
            
        Returns:
            Словарь {layer_name: embeddings}
        """
        if layer_names is None:
            layer_names = list(self.layers.keys())
        
        results = {}
        for name in layer_names:
            if name in self.layers and self.layers[name].config.enabled:
                results[name] = self.layers[name].embed(texts)
        
        return results
    
    def embed_parallel(
        self,
        texts: List[str],
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Параллельная обработка слоёв через ThreadPool.
        
        Args:
            texts: Список текстов
            layer_names: Какие слои использовать (None = все)
            
        Returns:
            Словарь {layer_name: embeddings}
        """
        if layer_names is None:
            layer_names = list(self.layers.keys())
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Создаём задачи
            future_to_layer = {}
            for name in layer_names:
                if name in self.layers and self.layers[name].config.enabled:
                    future = executor.submit(self.layers[name].embed, texts)
                    future_to_layer[future] = name
            
            # Собираем результаты
            for future in as_completed(future_to_layer):
                name = future_to_layer[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.error(f"Layer {name} failed: {e}")
                    results[name] = None
        
        return results
    
    def embed_fused(
        self,
        texts: List[str],
        layer_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Получить объединённый эмбеддинг из нескольких слоёв.
        
        Args:
            texts: Список текстов
            layer_weights: Веса слоёв {name: weight}
            
        Returns:
            Объединённые эмбеддинги
        """
        results = self.embed_parallel(texts)
        
        if not results:
            raise RuntimeError("No layers available")
        
        if layer_weights is None:
            # Равные веса
            layer_weights = {name: 1.0 for name in results.keys()}
        
        # Нормализация весов
        total_weight = sum(layer_weights.values())
        
        # Объединение
        fused = None
        for name, embeddings in results.items():
            if name in layer_weights and embeddings is not None:
                weight = layer_weights[name] / total_weight
                if fused is None:
                    fused = embeddings * weight
                else:
                    fused += embeddings * weight
        
        return fused
    
    def project(
        self,
        embeddings: np.ndarray,
        source_layer: str,
        target_layer: str
    ) -> np.ndarray:
        """
        Проецировать эмбеддинги из одного слоя в другой.
        
        Args:
            embeddings: Эмбеддинги для проекции
            source_layer: Исходный слой
            target_layer: Целевой слой
            
        Returns:
            Спроецированные эмбеддинги
        """
        # Простая линейная проекция (можно улучшить до обучаемой)
        source_dim = self.layers[source_layer].config.dimension
        target_dim = self.layers[target_layer].config.dimension
        
        if source_dim == target_dim:
            return embeddings
        
        # Создаём матрицу проекции
        projection_matrix = np.random.randn(source_dim, target_dim) / np.sqrt(source_dim)
        
        return embeddings @ projection_matrix
    
    def get_layer_stats(self) -> Dict[str, Dict[str, Any]]:
        """Получить статистику по всем слоям."""
        with self._lock:
            return {name: layer.get_stats() for name, layer in self.layers.items()}
    
    def clear_all_caches(self) -> None:
        """Очистить кэш всех слоёв."""
        with self._lock:
            for layer in self.layers.values():
                layer.cache.clear()
        logger.info("All layer caches cleared")
    
    def __repr__(self) -> str:
        """Строковое представление."""
        with self._lock:
            layers_info = ", ".join([
                f"{name}({layer.config.dimension}d)"
                for name, layer in self.layers.items()
            ])
            return f"LayerWiseEmbedder(layers=[{layers_info}], max_workers={self.max_workers})"
