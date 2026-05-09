"""
GeneratorQueueManager - Потокобезопасный менеджер очереди для генерации.

Обходит ошибку OpenVINO: 'Generate cannot be called while ContinuousBatchingPipeline is already in running state'.

Паттерн: Producer-Consumer
- Async поток добавляет запросы в очередь
- Sync поток (worker) последовательно исполняет их
"""

import asyncio
import threading
import queue
from typing import Optional, Any, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Запрос на генерацию."""
    prompt: str
    request_id: str
    config: Dict[str, Any]
    future: Optional[asyncio.Future] = None
    is_segment: bool = False


class GeneratorQueueManager:
    """
    Менеджер очереди генерации.
    
    Обеспечивает:
    - Потокобезопасность
    - Последовательное выполнение (один generate() в любой момент)
    - Async интерфейс для внешнего кода
    """
    
    def __init__(self, model_instance: Any, max_queue_size: int = 32, worker_threads: int = 1):
        """
        Args:
            model_instance: Модель с generate() методом
            max_queue_size: Макс. размер очереди
            worker_threads: Кол-во worker потоков
        """
        self.model = model_instance
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        
        self._workers = []
        self._executor = ThreadPoolExecutor(max_workers=worker_threads)
        self.is_running = True
        
        for i in range(worker_threads):
            t = threading.Thread(
                target=self._worker_loop, 
                name=f"GenWorker-{i}",
                daemon=True
            )
            t.start()
            self._workers.append(t)
        
        logger.info(f"GeneratorQueueManager: {worker_threads} workers started")
    
    def _worker_loop(self):
        """Worker поток - последовательная обработка запросов."""
        while self.is_running:
            try:
                req = self.request_queue.get(timeout=0.5)
                
                if req is None:
                    self.request_queue.task_done()
                    break
                
                try:
                    response_text = self._execute_generation(req)
                    
                    if req.future and not req.future.cancelled():
                        req.future.set_result(response_text)
                        
                except Exception as e:
                    logger.error(f"Generation failed for {req.request_id}: {e}")
                    if req.future and not req.future.done():
                        req.future.set_exception(e)
                
                finally:
                    self.request_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.critical(f"Worker critical error: {e}")
    
    def _execute_generation(self, req: GenerationRequest) -> str:
        """Вызов модели."""
        prompt = req.prompt
        config = req.config
        
        if hasattr(self.model, 'create_chat_completion'):
            response = self.model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **config
            )
            if isinstance(response, dict):
                return response.get('choices', [{}])[0].get('text', '')
            return str(response)
        
        elif hasattr(self.model, 'generate'):
            result = self.model.generate(prompt, **config)
            if isinstance(result, dict):
                return result.get('text', '')
            return str(result)
        
        # Model does not have required generation method
        logger.error(f"Model {type(self.model).__name__} does not have generate() or create_chat_completion() method")
        return f"[Error: Model {type(self.model).__name__} does not support generation]"
    
    async def submit(self, prompt: str, config: Dict, request_id: str) -> str:
        """Async отправка запроса."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        req = GenerationRequest(
            prompt=prompt,
            request_id=request_id,
            config=config,
            future=future
        )
        
        await loop.run_in_executor(None, self.request_queue.put, req)
        
        return await future
    
    async def submit_batch(self, prompts: list, config: Dict) -> list:
        """Parallel submission - очередь сама распараллелит."""
        tasks = []
        for i, prompt in enumerate(prompts):
            tasks.append(self.submit(prompt, config, f"batch_{i}"))
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def shutdown(self):
        """Остановка workers."""
        self.is_running = False
        
        for _ in self._workers:
            try:
                self.request_queue.put(None)
            except:
                pass
        
        for t in self._workers:
            t.join(timeout=2)
        
        self._executor.shutdown(wait=False)
        logger.info("GeneratorQueueManager stopped")


def create_queue_manager(model_instance: Any, workers: int = 1) -> GeneratorQueueManager:
    """Factory."""
    return GeneratorQueueManager(model_instance, worker_threads=workers)