"""
EvaModel - Unified API для EUMI моделей

Единый интерфейс для работы с моделями любого формата.
Поддерживает GGUF, Transformers, ONNX с интеграцией
LayerWiseEmbedder для параллельной обработки и
Pie Architecture (L1/L2 слои через ActivationProfiler и RoutingEngine).

Example:
    >>> from eumi import EvaModel
    >>> 
    >>> # Загрузка из .eva
    >>> model = EvaModel.from_eva("path/to/model.eva")
    >>> 
    >>> # Генерация с автоматической маршрутизацией
    >>> result = model.generate(
    ...     "Что такое тёмная материя?",
    ...     mode="auto",
    ...     return_reasoning=True
    ... )
    >>> print(result["text"])
    >>> 
    >>> # Эмбеддинги
    >>> emb = model.embed("текст", layers=["semantic", "graph"])
    >>> 
    >>> # Информация
    >>> info = model.info()
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Iterator
import time
import numpy as np

from ..core.container import EvaContainer
from ..core.config import EvaConfig
from ..backends import BaseBackend, create_backend, GenerationConfig, GenerationResult
from ..embeddings.layer_wise import LayerWiseEmbedder, LayerConfig

# L1/L2 Pie Architecture (these are at src/ level, not inside eumi)
try:
    from memory.fractal_graph_l1_l2 import FractalGraphL1L2, create_l1l2_graph
    from profiles.activation_profiler import ActivationProfiler, create_default_profiler
    from routing.routing_engine import RoutingEngine, create_default_engine, RoutingParams
except ImportError:
    # Fallback for when running as part of eumi package
    import sys
    from pathlib import Path
    src_path = str(Path(__file__).parent.parent.parent)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from memory.fractal_graph_l1_l2 import FractalGraphL1L2, create_l1l2_graph
    from profiles.activation_profiler import ActivationProfiler, create_default_profiler
    from routing.routing_engine import RoutingEngine, create_default_engine, RoutingParams

logger = logging.getLogger("eumi.model")


class ModelInfo:
    """Информация о модели."""
    
    def __init__(self, data: Dict[str, Any]):
        self.name = data.get("name", "unknown")
        self.version = data.get("version", "1.0.0")
        self.backend = data.get("backend", "unknown")
        self.type = data.get("type", "unknown")
        self.vocab_size = data.get("vocab_size", 0)
        self.context_size = data.get("context_size", 0)
        self.embedding_size = data.get("embedding_size", 0)
        self.num_layers = data.get("num_layers", 0)
        self.num_heads = data.get("num_heads", 0)
        self.raw = data
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в словарь."""
        return {
            "name": self.name,
            "version": self.version,
            "backend": self.backend,
            "type": self.type,
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "embedding_size": self.embedding_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            **self.raw
        }


class EvaModel:
    """
    Unified API для моделей EUMI с Pie Architecture.
    
    Объединяет:
    - EvaContainer (загрузка .eva)
    - BaseBackend (генерация текста)
    - LayerWiseEmbedder (эмбеддинги)
    - EvaConfig (конфигурация)
    - ActivationProfiler (L1: профили активаций)
    - RoutingEngine (L2: маршрутизация)
    
    Attributes:
        container: EvaContainer для доступа к файлам модели
        backend: Бэкенд генерации (GGUF/Transformers/ONNX)
        embedder: LayerWiseEmbedder для эмбеддингов
        config: Конфигурация модели
        graph: FractalGraphL1L2 для L1/L2 слоёв
        profiler: ActivationProfiler для профилей
        router: RoutingEngine для маршрутизации
        is_loaded: Флаг загрузки
    """
    
    def __init__(self):
        """Инициализация (пустая, используйте from_* методы)."""
        self.container: Optional[EvaContainer] = None
        self.backend: Optional[BaseBackend] = None
        self.embedder: Optional[LayerWiseEmbedder] = None
        self.config: Optional[EvaConfig] = None
        self.graph: Optional[FractalGraphL1L2] = None
        self.profiler: Optional[ActivationProfiler] = None
        self.router: Optional[RoutingEngine] = None
        self.is_loaded = False
        self._model_info: Optional[ModelInfo] = None
        self._current_domain: str = "general"
    
    @classmethod
    def from_eva(
        cls,
        path: Union[str, Path],
        backend_type: str = "gguf",
        load_embedder: bool = True,
        init_pie_architecture: bool = True
    ) -> "EvaModel":
        """
        Загрузить модель из .eva файла или директории.
        
        Args:
            path: Путь к .eva файлу или директории
            backend_type: Тип бэкенда ('gguf', 'transformers', 'onnx')
            load_embedder: Загрузить ли эмбеддер
            init_pie_architecture: Инициализировать ли L1/L2 слои
            
        Returns:
            Загруженная EvaModel
            
        Example:
            >>> model = EvaModel.from_eva("models/qwen.eva")
            >>> print(model.info().name)
        """
        instance = cls()
        
        # Загрузка контейнера
        logger.info(f"Loading model from: {path}")
        instance.container = EvaContainer(path).mount()
        
        # Загрузка конфига
        config_path = instance.container.temp_dir / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                config_data = json.load(f)
            instance.config = EvaConfig.from_dict(config_data)
        else:
            instance.config = EvaConfig()
        
        # Создание бэкенда
        backend_config = instance.container.get_backend_config(backend_type)
        instance.backend = create_backend(backend_type, backend_config)
        
        # Загрузка модели в бэкенд
        model_path = instance.container.get_model_path("condensed")
        instance.backend.load_model(str(model_path))
        
        # Загрузка эмбеддера
        if load_embedder:
            instance._init_embedder()
        
        # Инициализация Pie Architecture (L1/L2)
        if init_pie_architecture:
            instance._init_pie_architecture()
        
        instance.is_loaded = True
        instance._model_info = ModelInfo(instance.backend.get_model_info())
        
        logger.info(f"Model loaded: {instance._model_info.name}")
        logger.info(f"  Backend: {backend_type}")
        logger.info(f"  Context: {instance._model_info.context_size}")
        if instance.router:
            logger.info(f"  Pie Architecture: L1/L2 initialized")
        
        return instance
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        backend_type: str = "gguf",
        **backend_kwargs
    ) -> "EvaModel":
        """
        Загрузить модель напрямую (без .eva контейнера).
        
        Для обратной совместимости со старым кодом.
        
        Args:
            model_path: Путь к файлу модели
            backend_type: Тип бэкенда
            **backend_kwargs: Дополнительные параметры бэкенда
            
        Returns:
            Загруженная EvaModel
            
        Example:
            >>> model = EvaModel.from_pretrained("model.gguf", backend_type="gguf")
        """
        instance = cls()
        
        instance.config = EvaConfig()
        instance.backend = create_backend(backend_type, backend_kwargs)
        instance.backend.load_model(str(model_path))
        instance.embedder = None  # Без эмбеддера в legacy mode
        instance.graph = None  # Без графа в legacy mode
        instance.profiler = None
        instance.router = None
        instance.is_loaded = True
        instance._model_info = ModelInfo(instance.backend.get_model_info())
        
        return instance
    
    def _init_embedder(self) -> None:
        """Инициализация LayerWiseEmbedder."""
        self.embedder = LayerWiseEmbedder(max_workers=4)
        
        # Добавляем семантический слой (e5/MiniLM)
        embeddings_path = self.container.temp_dir / "embeddings"
        if embeddings_path.exists():
            # Ищем доступные модели
            semantic_models = list(embeddings_path.rglob("*all-MiniLM*"))
            if semantic_models:
                semantic_config = LayerConfig(
                    name="semantic",
                    dimension=384,
                    quantization_bits=32,
                    cache_enabled=True,
                    normalize=True
                )
                self.embedder.add_layer(semantic_config, str(semantic_models[0]))
                logger.info(f"Added semantic layer: {semantic_models[0]}")
        
        # Можно добавить и другие слои из конфига
        if self.config and hasattr(self.config, 'knowledge'):
            if self.config.knowledge.enabled:
                # Добавляем графовый слой если есть граф
                graph_path = self.container.get_graph_path()
                if graph_path:
                    graph_config = LayerConfig(
                        name="graph",
                        dimension=128,  # Размерность Node2Vec
                        cache_enabled=True
                    )
                    # Заглушка - в реальности загрузить Node2Vec
                    self.embedder.add_layer(graph_config, None)
                    logger.info("Added graph layer")
    
    def _init_pie_architecture(self) -> None:
        """Инициализация Pie Architecture (L1/L2 слои)."""
        graph_path = self.container.get_graph_path()
        
        if graph_path is None:
            logger.warning("No graph found, Pie Architecture disabled")
            return
        
        # Создаём L1/L2 граф
        self.graph = create_l1l2_graph(str(graph_path))
        
        # Создаём profiler (L1)
        self.profiler = create_default_profiler(self.graph)
        
        # Создаём router (L2)
        self.router = create_default_engine(self.graph, self.profiler)
        
        # Создаём дефолтные правила если их нет
        self.router.create_default_rules()
        
        logger.info("Pie Architecture initialized (L1/L2)")
    
    def generate(
        self,
        prompt: str,
        mode: str = "auto",  # auto, condensed, extended
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        repetition_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        return_reasoning: bool = False,
        use_routing: bool = True,
        domain_hint: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Сгенерировать текст с поддержкой Pie Architecture.
        
        Args:
            prompt: Промпт для генерации
            mode: Режим ('auto', 'condensed', 'extended')
            max_tokens: Максимум токенов (если None - из routing)
            temperature: Температура (если None - из routing)
            top_p: Top-p сэмплирование
            repetition_penalty: Штраф за повторы (если None - из routing)
            stop_sequences: Стоп-последовательности
            return_reasoning: Вернуть reasoning steps
            use_routing: Использовать ли L2 routing
            domain_hint: Подсказка домена для routing
            **kwargs: Дополнительные параметры
            
        Returns:
            Словарь с результатом:
            {
                "text": str,
                "tokens": List[int],
                "num_tokens": int,
                "generation_time": float,
                "finish_reason": str,
                "routing_params": Dict,  # если use_routing=True
                "reasoning": Optional[List[Dict]]
            }
        """
        if not self.is_loaded or self.backend is None:
            raise RuntimeError("Model not loaded. Call from_eva() or from_pretrained() first.")
        
        start_time = time.time()
        
        # Получаем параметры из RoutingEngine (L2)
        routing_params_dict = None
        if use_routing and self.router:
            routing_params_dict = self.router.get_rule_for_generation(
                query=prompt,
                mode=mode
            )
            self._current_domain = routing_params_dict.get("fallback_chain", ["general"])[0]
        
        # Применяем параметры (explicit params override routing)
        final_max_tokens = max_tokens if max_tokens is not None else (
            routing_params_dict["max_tokens"] if routing_params_dict else 1024
        )
        final_temperature = temperature if temperature is not None else (
            routing_params_dict["temperature"] if routing_params_dict else 0.7
        )
        final_repeat_penalty = repetition_penalty if repetition_penalty is not None else (
            routing_params_dict["repeat_penalty"] if routing_params_dict else 1.1
        )
        
        # Если нужен extended mode и есть модель extended
        if mode == "extended":
            try:
                extended_path = self.container.get_model_path("extended")
                logger.info(f"Using extended model: {extended_path}")
                final_max_tokens = min(final_max_tokens * 2, 2048)
            except ValueError:
                pass  # Используем текущую модель
        
        # Создаём конфиг генерации
        config = GenerationConfig(
            max_tokens=final_max_tokens,
            temperature=final_temperature,
            top_p=top_p,
            repetition_penalty=final_repeat_penalty,
            stop_sequences=stop_sequences or []
        )
        
        # Генерация
        result = self.backend.generate(prompt, config)
        
        generation_time = time.time() - start_time
        
        # Формируем ответ
        response = {
            "text": result.text,
            "tokens": result.tokens,
            "num_tokens": result.num_tokens,
            "generation_time": generation_time,
            "finish_reason": result.finish_reason
        }
        
        # Добавляем routing params если использовались
        if routing_params_dict:
            response["routing_params"] = {
                "temperature": final_temperature,
                "repeat_penalty": final_repeat_penalty,
                "max_tokens": final_max_tokens,
                "quant_profile": routing_params_dict.get("quant_profile", "Q4_K_M"),
                "rule_id": routing_params_dict.get("rule_id")
            }
        
        # Добавляем reasoning если запрошено
        if return_reasoning:
            response["reasoning"] = self._generate_reasoning(prompt, result)
        
        # Estimate entropy from response repetition (heuristic)
        words = response.split()
        if len(words) > 1:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            entropy = 1.0 - repetition_ratio if repetition_ratio < 1.0 else 0.0
        else:
            entropy = 0.5

        # Estimate quality from response characteristics
        quality = 0.8
        if not response or len(response) < 10:
            quality = 0.3
        elif len(response) < 50:
            quality = 0.6
        elif len(unique_words) / len(words) < 0.5:
            quality = 0.5

        # Обновляем L1 профиль (асинхронно бы могли)
        if self.profiler:
            try:
                self.profiler.record_generation(
                    domain=self._current_domain,
                    model_id="model_a",
                    query=prompt,
                    response=result.text,
                    entropy=entropy,
                    latency_ms=generation_time * 1000,
                    quality=quality
                )
            except Exception as e:
                logger.debug(f"Failed to record generation stats: {e}")
        
        return response
    
    def generate_stream(
        self,
        prompt: str,
        mode: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> Iterator[str]:
        """
        Генерировать текст потоком.
        
        Yields:
            Части сгенерированного текста
        """
        if not self.is_loaded or self.backend is None:
            raise RuntimeError("Model not loaded")
        
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
        yield from self.backend.generate_stream(prompt, config)
    
    def embed(
        self,
        texts: Union[str, List[str]],
        layers: Optional[List[str]] = None,
        parallel: bool = True,
        return_fused: bool = False,
        layer_weights: Optional[Dict[str, float]] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Получить эмбеддинги через LayerWiseEmbedder.
        
        Args:
            texts: Текст или список текстов
            layers: Какие слои использовать (None = все)
            parallel: Использовать ли параллельную обработку
            return_fused: Вернуть объединённый эмбеддинг
            layer_weights: Веса слоёв для объединения
            
        Returns:
            Эмбеддинги (numpy array или dict)
            
        Example:
            >>> emb = model.embed("текст", layers=["semantic"])
            >>> print(emb.shape)  # (384,)
            >>> 
            >>> embs = model.embed(["текст1", "текст2"], parallel=True)
            >>> print(embs["semantic"].shape)  # (2, 384)
        """
        if self.embedder is None:
            raise RuntimeError("Embedder not initialized")
        
        if isinstance(texts, str):
            texts = [texts]
        
        if return_fused:
            return self.embedder.embed_fused(texts, layer_weights)
        
        if parallel:
            return self.embedder.embed_parallel(texts, layers)
        else:
            return self.embedder.embed_sequential(texts, layers)
    
    def embed_layers_parallel(
        self,
        texts: List[str],
        layer_configs: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """
        Продвинутая параллельная обработка с конфигурацией слоёв.
        
        Args:
            texts: Тексты
            layer_configs: Конфигурация каждого слоя:
                [{"name": "semantic", "dimension": 384, "quantize": 16}, ...]
                
        Returns:
            Результаты по слоям
        """
        results = {}
        
        # Создаём временные слои по конфигу
        temp_layers = {}
        for cfg in layer_configs:
            layer_config = LayerConfig(
                name=cfg["name"],
                dimension=cfg["dimension"],
                quantization_bits=cfg.get("quantize", 32),
                parallel_workers=cfg.get("workers", 4)
            )
            temp_layers[cfg["name"]] = layer_config
        
        # Параллельная обработка
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=len(layer_configs)) as executor:
            # Note: This is a simplified version - in real implementation
            # we would need actual layer implementations
            futures = {}
            for name, config in temp_layers.items():
                # Placeholder for actual embedding computation
                future = executor.submit(self._dummy_embed, texts, config)
                futures[future] = name
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.error(f"Layer {name} failed: {e}")
        
        return results
    
    def _dummy_embed(self, texts: List[str], config: LayerConfig) -> np.ndarray:
        """Заглушка для embedding (в реальности используется embedder)."""
        # Return random embeddings for demonstration
        return np.random.randn(len(texts), config.dimension).astype(np.float32)
    
    def info(self) -> ModelInfo:
        """Получить информацию о модели."""
        if self._model_info is None:
            raise RuntimeError("Model not loaded")
        return self._model_info
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Получить использование памяти."""
        backend_mem = {"ram_mb": 0, "vram_mb": 0}
        if self.backend:
            backend_mem = self.backend.get_memory_usage()
        
        # Оценка для эмбеддера
        embedder_mem = 0
        if self.embedder:
            for layer in self.embedder.layers.values():
                # Очень приблизительная оценка
                embedder_mem += len(layer.cache.cache) * layer.config.dimension * 4 / 1024 / 1024
        
        return {
            "backend_ram_mb": backend_mem.get("ram_mb", 0),
            "backend_vram_mb": backend_mem.get("vram_mb", 0),
            "embedder_cache_mb": embedder_mem,
            "total_ram_mb": backend_mem.get("ram_mb", 0) + embedder_mem
        }
    
    def get_embedder_stats(self) -> Dict[str, Any]:
        """Получить статистику эмбеддера."""
        if self.embedder is None:
            return {}
        return self.embedder.get_layer_stats()
    
    def get_routing_stats(self) -> Optional[Dict[str, Any]]:
        """Получить статистику routing (L2)."""
        if self.router is None:
            return None
        
        rules = self.router.list_rules()
        return {
            "num_rules": len(rules),
            "domains": [r.fallback_chain[0] if r.fallback_chain else "unknown" for r in rules],
            "current_domain": self._current_domain
        }
    
    def get_profiler_stats(self) -> Optional[Dict[str, Any]]:
        """Получить статистику profiler (L1)."""
        if self.profiler is None:
            return None
        
        profiles = self.profiler.list_all_profiles()
        return {
            "num_profiles": len(profiles),
            "domains": list(set(p.domain for p in profiles)),
            "total_samples": sum(p.sample_count for p in profiles)
        }
    
    def feedback(self, positive: bool, rule_id: Optional[str] = None) -> bool:
        """
        Отправить обратную связь о последней генерации.
        
        Args:
            positive: Положительная ли обратная связь
            rule_id: ID правила (если None - используется последнее)
            
        Returns:
            True если успешно
        """
        if self.router is None:
            return False
        
        return self.router.record_feedback(rule_id, positive)
    
    def _generate_reasoning(
        self,
        prompt: str,
        result: GenerationResult
    ) -> List[Dict[str, Any]]:
        """Сгенерировать reasoning steps."""
        reasoning = []
        
        # Шаг 1: Routing (L2)
        if self.router:
            reasoning.append({
                "step": 1,
                "phase": "routing",
                "action": f"L2 Routing: домен={self._current_domain}",
                "icon": "🎯"
            })
        
        # Шаг 2: Токенизация
        reasoning.append({
            "step": 2,
            "phase": "tokenization",
            "action": "Токенизация промпта",
            "tokens": len(self.backend.tokenize(prompt)),
            "icon": "🔤"
        })
        
        # Шаг 3: Контекст (L3)
        if self.embedder:
            reasoning.append({
                "step": 3,
                "phase": "context_retrieval",
                "action": "L3 Memory: поиск релевантного контекста",
                "icon": "🔍"
            })
        
        # Шаг 4: Генерация (L0)
        reasoning.append({
            "step": 4,
            "phase": "generation",
            "action": f"L0 Generation: {result.num_tokens} токенов",
            "time_ms": result.generation_time * 1000,
            "icon": "✨"
        })
        
        # Шаг 5: Профилирование (L1)
        if self.profiler:
            reasoning.append({
                "step": 5,
                "phase": "profiling",
                "action": "L1 Profiler: обновление статистики",
                "icon": "📊"
            })
        
        return reasoning
    
    def unload(self) -> None:
        """Выгрузить модель и освободить память."""
        if self.backend:
            self.backend.unload()
        
        if self.embedder:
            self.embedder.clear_all_caches()
        
        if self.container:
            self.container.unmount()
        
        self.graph = None
        self.profiler = None
        self.router = None
        self.is_loaded = False
        logger.info("Model unloaded")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
    
    def __repr__(self) -> str:
        """Строковое представление."""
        if not self.is_loaded:
            return "EvaModel(not_loaded)"
        
        info = self._model_info
        embedder_info = f", embedder={len(self.embedder.layers)} layers" if self.embedder else ""
        pie_info = ", pie_arch=L1/L2" if self.router else ""
        return f"EvaModel(name={info.name}, backend={info.backend}{embedder_info}{pie_info})"
