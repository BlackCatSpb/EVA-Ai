"""
OpenVINO Generator - генерация на базе OpenVINO GenAI.

Преимущества над llama.cpp:
- Continuous Batching для параллельных запросов
- PagedAttention для эффективного KV-cache
- INT8 сжатие KV-cache по умолчанию
- Нативная поддержка GGUF моделей

Асинхронная архитектура для разных типов данных:
- QUERY: основные запросы пользователя
- CONTEXT: контекст из истории/кэша
- CONCEPT: извлечение и анализ концептов
- CONTRADICTION: анализ противоречий
- SELF_DIALOG: самодиалог для обучения
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Generator, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

logger = logging.getLogger("eva_ai.openvino_generator")


class DataType(Enum):
    """Типы данных для асинхронной обработки."""
    QUERY = "query"              # Основной запрос
    CONTEXT = "context"          # Контекст из истории
    CONCEPT = "concept"          # Концепты
    CONTRADICTION = "contradiction"  # Противоречия
    SELF_DIALOG = "self_dialog"  # Самодиалог
    CODE = "code"                # Генерация кода


@dataclass
class DataTypeConfig:
    """Конфигурация для типа данных."""
    max_tokens: int = 1024
    temperature: float = 0.7
    device: str = "CPU"  # CPU или GPU
    priority: int = 0  # Приоритет (выше = важнее)


# Конфигурации по умолчанию для каждого типа данных
DEFAULT_DATA_TYPE_CONFIGS: Dict[DataType, DataTypeConfig] = {
    DataType.QUERY: DataTypeConfig(
        max_tokens=1024,
        temperature=0.7,
        device="CPU",
        priority=10
    ),
    DataType.CONTEXT: DataTypeConfig(
        max_tokens=512,
        temperature=0.5,
        device="CPU",
        priority=5
    ),
    DataType.CONCEPT: DataTypeConfig(
        max_tokens=512,
        temperature=0.6,
        device="GPU.0",
        priority=7
    ),
    DataType.CONTRADICTION: DataTypeConfig(
        max_tokens=768,
        temperature=0.7,
        device="GPU.0",
        priority=8
    ),
    DataType.SELF_DIALOG: DataTypeConfig(
        max_tokens=2048,
        temperature=0.8,
        device="GPU.0",
        priority=9
    ),
    DataType.CODE: DataTypeConfig(
        max_tokens=1024,
        temperature=0.3,
        device="GPU.0",
        priority=8
    ),
}


@dataclass
class OpenVINOGenerationResult:
    """Результат генерации OpenVINO."""
    text: str
    model_used: str
    generation_time: float
    tokens_generated: int
    device: str
    data_type: DataType = DataType.QUERY


class OpenVINOGenerator:
    """
    Генератор на базе OpenVINO GenAI.
    
    Использует LLMPipeline с поддержкой:
    - GGUF моделей напрямую
    - Streaming через callback
    - CPU и GPU устройства
    - SchedulerConfig для параллельной обработки
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "CPU",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        n_ctx: int = 4096,
        scheduler_config: Optional[Dict] = None,
        performance_hint: str = "LATENCY",
        num_streams: Optional[int] = None
    ):
        """
        Инициализация OpenVINO Generator.
        
        Args:
            model_path: Путь к GGUF модели
            device: Устройство (CPU, GPU.0, GPU.1)
            max_tokens: Максимум токенов по умолчанию
            temperature: Температура по умолчанию
            n_ctx: Размер контекста
            scheduler_config: Конфиг планировщика {
                cache_size: int = 2,           # GB под KV-кеш
                max_num_seqs: int = 4,         # Параллельные слоты
                max_num_batched_tokens: int = 2048,
                enable_prefix_caching: bool = True
            }
            performance_hint: "LATENCY" или "THROUGHPUT"
            num_streams: Количество потоков (None=AUTO, для CPU авто=物理ческие ядра)
        """
        self.model_path = model_path
        self.device = device
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.n_ctx = n_ctx
        self.scheduler_config = scheduler_config or {}
        self.performance_hint = performance_hint
        self.num_streams = num_streams
        
        self._pipeline = None
        self._model_path_str = None
        
        if model_path:
            self._load_model()
    
    def _load_model(self) -> bool:
        """Загрузить модель в OpenVINO."""
        try:
            import openvino_genai as ov_genai
            
            if self._pipeline is not None:
                return True
            
            if not self.model_path:
                logger.error("No model path specified")
                return False
            
            path_str = str(self.model_path)
            logger.info(f"Loading OpenVINO model from {path_str} on {self.device}")
            
            start = time.time()
            
            config = {}
            
            if self.performance_hint:
                config["PERFORMANCE_HINT"] = self.performance_hint
            
            if self.num_streams is not None:
                config["NUM_STREAMS"] = str(self.num_streams) if self.num_streams != "AUTO" else "AUTO"
            elif self.device == "CPU":
                config["NUM_STREAMS"] = "AUTO"
            
            if self.scheduler_config:
                scheduler = ov_genai.SchedulerConfig()
                if 'cache_size' in self.scheduler_config:
                    scheduler.cache_size = self.scheduler_config['cache_size']
                if 'max_num_seqs' in self.scheduler_config:
                    scheduler.max_num_seqs = self.scheduler_config['max_num_seqs']
                if 'max_num_batched_tokens' in self.scheduler_config:
                    scheduler.max_num_batched_tokens = self.scheduler_config['max_num_batched_tokens']
                if 'enable_prefix_caching' in self.scheduler_config:
                    scheduler.enable_prefix_caching = self.scheduler_config['enable_prefix_caching']
                config["scheduler_config"] = scheduler
            
            self._pipeline = ov_genai.LLMPipeline(path_str, self.device, config=config)
            self._model_path_str = path_str
            
            load_time = time.time() - start
            logger.info(f"OpenVINO model loaded in {load_time:.1f}s on {self.device}")
            if self.scheduler_config:
                logger.info(f"  Scheduler: max_seqs={self.scheduler_config.get('max_num_seqs', 'default')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load OpenVINO model: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_tokens: Optional[List[str]] = None
    ) -> OpenVINOGenerationResult:
        """
        Генерация текста.
        
        Args:
            prompt: Промпт в формате Qwen chatml
            max_tokens: Максимум новых токенов
            temperature: Температура
            stop_tokens: Список стоп-токенов
            
        Returns:
            OpenVINOGenerationResult
        """
        if not self._load_model():
            return OpenVINOGenerationResult(
                text="Ошибка: модель не загружена",
                model_used="none",
                generation_time=0.0,
                tokens_generated=0,
                device=self.device
            )
        
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature
        
        config = self._create_config(max_tokens, temperature, stop_tokens)
        
        start_time = time.time()
        
        try:
            result = self._pipeline.generate(prompt, config)
            
            # Очищаем от служебных токенов
            text = self._clean_output(result)
            
            tokens = len(text.split())
            gen_time = time.time() - start_time
            
            return OpenVINOGenerationResult(
                text=text,
                model_used="openvino_gguf",
                generation_time=gen_time,
                tokens_generated=tokens,
                device=self.device
            )
            
        except Exception as e:
            logger.error(f"OpenVINO generation error: {e}")
            return OpenVINOGenerationResult(
                text=f"Ошибка генерации: {e}",
                model_used="openvino_gguf",
                generation_time=time.time() - start_time,
                tokens_generated=0,
                device=self.device
            )
    
    def generate_streaming(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_tokens: Optional[List[str]] = None,
        chunk_size: int = 25
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Генерация со стримингом.
        
        Args:
            prompt: Промпт
            max_tokens: Максимум токенов
            temperature: Температура
            stop_tokens: Список стоп-токенов
            chunk_size: Размер чанка для буферизации
            
        Yields:
            Dict с данными чанка
        """
        import queue
        import threading
        
        if not self._load_model():
            yield {
                'type': 'error',
                'text': 'Модель не загружена',
                'is_final': True
            }
            return
        
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature
        
        config = self._create_config(max_tokens, temperature, stop_tokens)
        
        chunk_queue = queue.Queue()
        full_text = []
        start_time = time.time()
        
        def streamer(s: str) -> bool:
            full_text.append(s)
            chunk_queue.put(s)
            return False
        
        def generate_in_thread():
            try:
                self._pipeline.generate(prompt, config, streamer=streamer)
            except Exception as e:
                chunk_queue.put(('error', str(e)))
            finally:
                chunk_queue.put(None)
        
        thread = threading.Thread(target=generate_in_thread, daemon=True)
        thread.start()
        
        buffer = ""
        chunk_count = 0
        
        while True:
            try:
                item = chunk_queue.get(timeout=30)
            except queue.Empty:
                continue
            
            if item is None:
                break
            
            if isinstance(item, tuple) and item[0] == 'error':
                yield {
                    'type': 'error',
                    'text': item[1],
                    'is_final': True,
                    'tokens_count': len(full_text),
                    'elapsed_ms': int((time.time() - start_time) * 1000)
                }
                return
            
            buffer += item
            
            if len(buffer) >= chunk_size:
                elapsed = int((time.time() - start_time) * 1000)
                yield {
                    'type': 'chunk',
                    'text': buffer,
                    'is_final': False,
                    'tokens_count': len(full_text),
                    'elapsed_ms': elapsed
                }
                buffer = ""
                chunk_count += 1
        
        if buffer:
            yield {
                'type': 'chunk',
                'text': buffer,
                'is_final': False,
                'tokens_count': len(full_text),
                'elapsed_ms': int((time.time() - start_time) * 1000)
            }
        
        final_text = self._clean_output(''.join(full_text))
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        yield {
            'type': 'complete',
            'text': final_text,
            'is_final': True,
            'tokens_count': len(final_text.split()),
            'elapsed_ms': elapsed_ms,
            'chunk_count': chunk_count
        }
    
    def _create_config(self, max_tokens: int, temperature: float, stop_tokens: Optional[List[str]]):
        """Создать GenerationConfig для OpenVINO."""
        import openvino_genai as ov_genai
        
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = max_tokens
        config.temperature = temperature
        
        if stop_tokens:
            config.stop_strings = set(stop_tokens)
        
        return config
    
    def _clean_output(self, text: str) -> str:
        """Очистить вывод от служебных токенов."""
        text = text.replace("<|im_end|>", "")
        text = text.replace("<|im_start|>", "")
        text = text.replace("<|endoftext|>", "")
        return text.strip()
    
    def is_loaded(self) -> bool:
        """Проверить загружена ли модель."""
        return self._pipeline is not None
    
    def get_device(self) -> str:
        """Получить устройство."""
        return self.device


class OpenVINORouter:
    """
    Роутер для выбора между llama.cpp и OpenVINO генераторами.
    
    Может использовать OpenVINO для:
    - Параллельных запросов (continuous batching)
    - Длинных контекстов (PagedAttention)
    - CPU-эффективности (INT8 KV-cache)
    """
    
    def __init__(self):
        self.llama_generator = None
        self.openvino_generator = None
        self._use_openvino = True
    
    def set_llama_generator(self, generator):
        """Установить llama.cpp генератор."""
        self.llama_generator = generator
    
    def set_openvino_generator(self, generator: OpenVINOGenerator):
        """Установить OpenVINO генератор."""
        self.openvino_generator = generator
    
    def should_use_openvino(self, query: str, parallel: bool = False) -> bool:
        """
        Определить какой генератор использовать.
        
        Args:
            query: Запрос
            parallel: Нужна ли параллельная обработка
            
        Returns:
            True если использовать OpenVINO
        """
        if not self._use_openvino:
            return False
        
        if not self.openvino_generator or not self.openvino_generator.is_loaded():
            return False
        
        # OpenVINO для параллельных запросов эффективнее
        if parallel:
            return True
        
        return True  # По умолчанию OpenVINO
    
    def generate(self, prompt: str, **kwargs):
        """Генерировать с выбранным генератором."""
        if self.should_use_openvino(""):
            return self.openvino_generator.generate(prompt, **kwargs)
        elif self.llama_generator:
            return self.llama_generator.generate(prompt, **kwargs)
        else:
            raise RuntimeError("No generator available")


def create_openvino_generator(
    model_path: Optional[Path] = None,
    device: str = "CPU"
) -> Optional[OpenVINOGenerator]:
    """
    Создать OpenVINO Generator.
    
    Args:
        model_path: Путь к модели
        device: Устройство
        
    Returns:
        OpenVINOGenerator или None
    """
    try:
        if model_path is None:
            from eva_ai.core.pie_model_paths import get_pie_model_path
            model_path = get_pie_model_path('ruadapt_qwen3_4b', 'condensed')
        
        return OpenVINOGenerator(
            model_path=model_path,
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to create OpenVINO generator: {e}")
        return None


class OpenVINOCacheAdapter:
    """
    Адаптер для интеграции OpenVINO с HybridTokenCache.
    
    Обеспечивает:
    - Семантический кеш запросов
    - Предиктивный префетч контекста
    - Синхронизацию с KV-кешем OpenVINO
    """
    
    def __init__(self, openvino_generator: OpenVINOGenerator, hybrid_cache=None):
        self.generator = openvino_generator
        self.hybrid_cache = hybrid_cache
        self.cache_hits = 0
        self.cache_misses = 0
    
    def check_cache(self, query: str, similarity_threshold: float = 0.85) -> Optional[str]:
        """
        Проверить кеш на похожий запрос.
        
        Args:
            query: Запрос пользователя
            similarity_threshold: Порог схожести (0-1)
            
        Returns:
            cached_response если найден, иначе None
        """
        if not self.hybrid_cache:
            return None
        
        try:
            results = self.hybrid_cache.search(query, top_k=5)
            for item in results:
                if isinstance(item, dict):
                    text = item.get('text', '') or item.get('content', '')
                    similarity = item.get('similarity', 0)
                    if similarity >= similarity_threshold and text:
                        self.cache_hits += 1
                        logger.info(f"[CACHE HIT] similarity={similarity:.2f}")
                        return text
            self.cache_misses += 1
        except Exception as e:
            logger.debug(f"Cache check error: {e}")
        
        return None
    
    def store_in_cache(self, query: str, response: str, metadata: Optional[Dict] = None):
        """
        Сохранить результат в кеш.
        
        Args:
            query: Запрос
            response: Сгенерированный ответ
            metadata: Дополнительные метаданные
        """
        if not self.hybrid_cache:
            return
        
        try:
            entry = {
                'text': response,
                'query': query,
                'timestamp': time.time(),
                'model': 'openvino',
                'device': self.generator.device
            }
            if metadata:
                entry.update(metadata)
            
            self.hybrid_cache.add_token(f"openvino_{hash(query)}", entry)
            logger.debug(f"[CACHE STORE] query_hash={hash(query)}")
        except Exception as e:
            logger.debug(f"Cache store error: {e}")
    
    def get_context_for_generation(self, query: str, max_context_tokens: int = 2000) -> str:
        """
        Получить релевантный контекст из кеша для генерации.
        
        Args:
            query: Запрос
            max_context_tokens: Максимум токенов контекста
            
        Returns:
            Строка с контекстом
        """
        if not self.hybrid_cache:
            return ""
        
        try:
            contexts = []
            results = self.hybrid_cache.search(query, top_k=10)
            
            for item in results[:5]:
                if isinstance(item, dict):
                    text = item.get('text', '') or item.get('content', '')
                    if text and len(text) < max_context_tokens:
                        contexts.append(text)
            
            return '\n\n'.join(contexts) if contexts else ""
        except Exception as e:
            logger.debug(f"Context retrieval error: {e}")
            return ""
    
    def prefetch_related(self, query: str):
        """
        Предиктивный префетч связанных данных.
        
        Args:
            query: Запрос
        """
        if not self.hybrid_cache:
            return
        
        try:
            if hasattr(self.hybrid_cache, 'prefetch'):
                self.hybrid_cache.prefetch(query)
            elif hasattr(self.hybrid_cache, 'warm_up'):
                self.hybrid_cache.warm_up(query)
        except Exception as e:
            logger.debug(f"Prefetch error: {e}")
    
    def get_stats(self) -> Dict:
        """Получить статистику кеша."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total
        }


async def async_generate(
    generator: OpenVINOGenerator,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
) -> OpenVINOGenerationResult:
    """
    Асинхронная обёртка для синхронного вызова генерации.
    
    Использует asyncio.to_thread для неблокирующего выполнения.
    
    Args:
        generator: OpenVINOGenerator
        prompt: Промпт
        max_tokens: Максимум токенов
        temperature: Температура
        
    Returns:
        OpenVINOGenerationResult
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: generator.generate(prompt, max_tokens, temperature)
    )


async def async_generate_batch(
    generator: OpenVINOGenerator,
    prompts: List[str],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
) -> List[OpenVINOGenerationResult]:
    """
    Параллельная генерация для нескольких промптов.
    
    Использует asyncio.gather для одновременного выполнения.
    
    Args:
        generator: OpenVINOGenerator
        prompts: Список промптов
        max_tokens: Максимум токенов
        temperature: Температура
        
    Returns:
        Список OpenVINOGenerationResult
    """
    tasks = [
        async_generate(generator, prompt, max_tokens, temperature)
        for prompt in prompts
    ]
    return await asyncio.gather(*tasks)


async def async_generate_query(
    generator: OpenVINOGenerator,
    prompt: str,
    config: Optional[DataTypeConfig] = None
) -> OpenVINOGenerationResult:
    """
    Асинхронная генерация основного запроса.
    
    Args:
        generator: OpenVINOGenerator
        prompt: Промпт
        config: Конфигурация (или default для QUERY)
        
    Returns:
        OpenVINOGenerationResult
    """
    cfg = config or DEFAULT_DATA_TYPE_CONFIGS[DataType.QUERY]
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: generator.generate(prompt, cfg.max_tokens, cfg.temperature)
    )
    result.data_type = DataType.QUERY
    return result


async def async_generate_context(
    generator: OpenVINOGenerator,
    prompt: str,
    config: Optional[DataTypeConfig] = None
) -> OpenVINOGenerationResult:
    """
    Асинхронная генерация контекста.
    
    Args:
        generator: OpenVINOGenerator
        prompt: Промпт с контекстом
        config: Конфигурация (или default для CONTEXT)
        
    Returns:
        OpenVINOGenerationResult
    """
    cfg = config or DEFAULT_DATA_TYPE_CONFIGS[DataType.CONTEXT]
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: generator.generate(prompt, cfg.max_tokens, cfg.temperature)
    )
    result.data_type = DataType.CONTEXT
    return result


async def async_generate_concept(
    generator: OpenVINOGenerator,
    prompt: str,
    config: Optional[DataTypeConfig] = None
) -> OpenVINOGenerationResult:
    """
    Асинхронная генерация для извлечения концептов.
    
    Args:
        generator: OpenVINOGenerator
        prompt: Промпт для анализа концептов
        config: Конфигурация (или default для CONCEPT)
        
    Returns:
        OpenVINOGenerationResult
    """
    cfg = config or DEFAULT_DATA_TYPE_CONFIGS[DataType.CONCEPT]
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: generator.generate(prompt, cfg.max_tokens, cfg.temperature)
    )
    result.data_type = DataType.CONCEPT
    return result


async def async_generate_contradiction(
    generator: OpenVINOGenerator,
    prompt: str,
    config: Optional[DataTypeConfig] = None
) -> OpenVINOGenerationResult:
    """
    Асинхронная генерация для анализа противоречий.
    
    Args:
        generator: OpenVINOGenerator
        prompt: Промпт для анализа противоречий
        config: Конфигурация (или default для CONTRADICTION)
        
    Returns:
        OpenVINOGenerationResult
    """
    cfg = config or DEFAULT_DATA_TYPE_CONFIGS[DataType.CONTRADICTION]
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: generator.generate(prompt, cfg.max_tokens, cfg.temperature)
    )
    result.data_type = DataType.CONTRADICTION
    return result


async def async_generate_self_dialog(
    generator: OpenVINOGenerator,
    prompt: str,
    config: Optional[DataTypeConfig] = None
) -> OpenVINOGenerationResult:
    """
    Асинхронная генерация для самодиалога.
    
    Args:
        generator: OpenVINOGenerator
        prompt: Промпт для самодиалога
        config: Конфигурация (или default для SELF_DIALOG)
        
    Returns:
        OpenVINOGenerationResult
    """
    cfg = config or DEFAULT_DATA_TYPE_CONFIGS[DataType.SELF_DIALOG]
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: generator.generate(prompt, cfg.max_tokens, cfg.temperature)
    )
    result.data_type = DataType.SELF_DIALOG
    return result


async def async_generate_code(
    generator: OpenVINOGenerator,
    prompt: str,
    config: Optional[DataTypeConfig] = None
) -> OpenVINOGenerationResult:
    """
    Асинхронная генерация кода.
    
    Args:
        generator: OpenVINOGenerator
        prompt: Промпт для генерации кода
        config: Конфигурация (или default для CODE)
        
    Returns:
        OpenVINOGenerationResult
    """
    cfg = config or DEFAULT_DATA_TYPE_CONFIGS[DataType.CODE]
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: generator.generate(prompt, cfg.max_tokens, cfg.temperature)
    )
    result.data_type = DataType.CODE
    return result


class AsyncDataProcessor:
    """
    Процессор для асинхронной обработки разных типов данных.
    
    Обеспечивает:
    - Параллельную обработку разных типов данных
    - Приоритизацию запросов
    - Роутинг на правильное устройство
    """
    
    def __init__(
        self,
        cpu_generator: OpenVINOGenerator,
        gpu_generator: OpenVINOGenerator
    ):
        self.cpu_generator = cpu_generator
        self.gpu_generator = gpu_generator
        
        self.stats = {
            dt.value: {'count': 0, 'total_time': 0.0}
            for dt in DataType
        }
    
    def _get_generator_for_data_type(self, data_type: DataType) -> OpenVINOGenerator:
        """Получить генератор для типа данных."""
        config = DEFAULT_DATA_TYPE_CONFIGS[data_type]
        if config.device == "GPU.0" and self.gpu_generator:
            return self.gpu_generator
        return self.cpu_generator
    
    async def process_query(
        self,
        prompt: str,
        data_type: DataType = DataType.QUERY
    ) -> OpenVINOGenerationResult:
        """Обработать запрос указанного типа."""
        generator = self._get_generator_for_data_type(data_type)
        config = DEFAULT_DATA_TYPE_CONFIGS[data_type]
        
        start = time.time()
        
        if data_type == DataType.QUERY:
            result = await async_generate_query(generator, prompt, config)
        elif data_type == DataType.CONTEXT:
            result = await async_generate_context(generator, prompt, config)
        elif data_type == DataType.CONCEPT:
            result = await async_generate_concept(generator, prompt, config)
        elif data_type == DataType.CONTRADICTION:
            result = await async_generate_contradiction(generator, prompt, config)
        elif data_type == DataType.SELF_DIALOG:
            result = await async_generate_self_dialog(generator, prompt, config)
        elif data_type == DataType.CODE:
            result = await async_generate_code(generator, prompt, config)
        else:
            result = await async_generate_query(generator, prompt, config)
        
        elapsed = time.time() - start
        self.stats[data_type.value]['count'] += 1
        self.stats[data_type.value]['total_time'] += elapsed
        
        return result
    
    async def process_batch(
        self,
        items: List[Tuple[str, DataType]]
    ) -> List[OpenVINOGenerationResult]:
        """
        Параллельная обработка списка запросов.
        
        Args:
            items: List[(prompt, data_type), ...]
            
        Returns:
            List[OpenVINOGenerationResult]
        """
        tasks = [
            self.process_query(prompt, data_type)
            for prompt, data_type in items
        ]
        return await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict:
        """Получить статистику обработки."""
        return self.stats
