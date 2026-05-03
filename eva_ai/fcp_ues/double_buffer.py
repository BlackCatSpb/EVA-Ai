import threading
import openvino_genai as ov_genai
from typing import Optional, Dict, Any

import logging
logger = logging.getLogger("FCP.UES")


class DoubleBufferPipeline:
    """
    Обеспечивает атомарную замену активного LLMPipeline без остановки генерации.
    Пока новые запросы обслуживаются обновлённой моделью, старые завершаются с предыдущей.
    """
    
    def __init__(self, model_path: str, device: str = "CPU",
                 scheduler_config: Optional[Dict[str, Any]] = None):
        self.model_path = model_path
        self.device = device
        self.scheduler_config = scheduler_config or {}
        self._active: ov_genai.LLMPipeline = self._create_pipeline()
        self._standby: Optional[ov_genai.LLMPipeline] = None
        self._lock = threading.Lock()
        self._swap_pending = False
    
    def _create_pipeline(self) -> ov_genai.LLMPipeline:
        return ov_genai.LLMPipeline(
            self.model_path, self.device,
            config={"scheduler_config": self.scheduler_config}
        )
    
    @property
    def active(self) -> ov_genai.LLMPipeline:
        with self._lock:
            return self._active
    
    def prepare_swap(self, adapter_path: Optional[str] = None, alpha: float = 0.8):
        """Создаёт новый пайплайн в фоне (возможно, с LoRA-адаптером)."""
        new_pipeline = self._create_pipeline()
        if adapter_path:
            adapter = ov_genai.Adapter(adapter_path)
            config = ov_genai.AdapterConfig()
            config.add(adapter, alpha=alpha)
            new_pipeline.set_adapters(config)
        
        with self._lock:
            self._standby = new_pipeline
            self._swap_pending = True
        logger.info("New pipeline prepared for swap")
    
    def commit_swap(self):
        """Атомарно активирует подготовленный пайплайн."""
        with self._lock:
            if self._standby is not None:
                old = self._active
                self._active = self._standby
                self._standby = None
                self._swap_pending = False
                logger.info("Pipeline swap committed")
                # Старый пайплайн будет удалён сборщиком мусора после завершения всех запросов
    
    def rollback(self):
        """Отменяет ожидающий свап."""
        with self._lock:
            self._standby = None
            self._swap_pending = False
            logger.info("Swap rolled back")
