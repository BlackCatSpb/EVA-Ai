import os
from .topology import TopologyDiscoverer, ComputeTopology, ComputeUnit
from .auto_tune import PGOAutoTuner
from .resource_pin import ResourcePinner
from .qat_trainer import QATTrainer
from .double_buffer import DoubleBufferPipeline

import logging
logger = logging.getLogger("FCP.UES")


class UES:
    """Универсальная подсистема исполнения — точка входа для всех оптимизаций."""
    
    def __init__(self, model_path: str, device: str = "CPU"):
        self.topology = TopologyDiscoverer.discover()
        self.buffer = None
        self.auto_tuner: Optional[PGOAutoTuner] = None
        
        # Проверяем OpenVINO и модель перед инициализацией буфера
        try:
            if model_path and os.path.exists(model_path):
                import openvino_genai as ov_genai
                self.buffer = DoubleBufferPipeline(model_path, device)
                logger.info(f"UES buffer initialized (model: {model_path})")
            else:
                logger.warning(f"UES buffer skipped: model_path not found ({model_path})")
        except Exception as e:
            logger.warning(f"UES buffer skipped: {e}")
        
        logger.info(f"UES initialized: {len(self.topology.units)} compute units, {self.topology.total_memory_gb:.1f}GB RAM")
    
    def optimize_pipeline(self, benchmark_fn) -> dict:
        """Запускает полный цикл оптимизации."""
        self.auto_tuner = PGOAutoTuner(benchmark_fn)
        params = self.auto_tuner.tune()
        return self.auto_tuner.get_optimal_config()
    
    def pin_gnn_to_e_cores(self) -> dict:
        return ResourcePinner.pin_gnn_to_e_cores()
    
    def pin_llm_to_p_cores(self) -> dict:
        return ResourcePinner.pin_llm_to_p_cores()
    
    @staticmethod
    def qat_train(model, train_loader, epochs=2):
        quantized = QATTrainer.quantize_model(model)
        return QATTrainer.fine_tune(quantized, train_loader, epochs)
