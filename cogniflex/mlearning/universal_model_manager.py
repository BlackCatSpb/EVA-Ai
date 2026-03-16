"""
Универсальный ModelManager для CogniFlex
Поддержка: RuGPT, Qwen3.5, BitNet
"""
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Доступные модели
MODELS = {
    # === BitNet (1-bit) - САМАЯ БЫСТРАЯ на CPU ===
    "bitnet-2b": {
        "provider": "bitnet",
        "params": "2B",
        "ram": "0.4GB",
        "speed": "40-48 tok/s",
        "description": "Microsoft BitNet - 1-bit, самая быстрая на CPU",
        "requires_gpu": False
    },
    
    # === Qwen3.5 (РЕКОМЕНДУЕТСЯ) ===
    "qwen3.5-0.8b": {
        "provider": "qwen",
        "params": "0.8B",
        "ram": "1.6GB",
        "speed": "20-30 tok/s",
        "description": "Минимальная мультимодальная",
        "requires_gpu": False
    },
    "qwen3.5-2b": {
        "provider": "qwen",
        "params": "2B", 
        "ram": "4GB",
        "speed": "15-25 tok/s",
        "description": "Оптимально для интегрированной графики (РЕКОМЕНДУЕТСЯ)",
        "requires_gpu": False
    },
    "qwen3.5-3b": {
        "provider": "qwen",
        "params": "3B",
        "ram": "6GB",
        "speed": "10-20 tok/s",
        "description": "Баланс качества и скорости",
        "requires_gpu": False
    },
    "qwen3.5-4b": {
        "provider": "qwen",
        "params": "4B",
        "ram": "8GB", 
        "speed": "8-15 tok/s",
        "description": "Для дискретной графики",
        "requires_gpu": True
    },
    "qwen3.5-9b": {
        "provider": "qwen",
        "params": "9B",
        "ram": "18GB",
        "speed": "5-10 tok/s",
        "description": "Лучшее качество, требует GPU",
        "requires_gpu": True
    },
    
    # === RuGPT (текущая) ===
    "rugpt3large": {
        "provider": "fractal",
        "params": "760M",
        "ram": "2GB",
        "speed": "5-10 tok/s",
        "description": "Русская модель (текущая)",
        "requires_gpu": False
    }
}


class UniversalModelManager:
    """
    Универсальный менеджер моделей с автовыбором
    """
    
    def __init__(
        self, 
        model_name: str = "auto",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            model_name: Имя модели или "auto" для автовыбора
            device: auto/cpu/cuda
            cache_dir: Директория кэша
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.current_model = None
        self.provider = None
        
        # Автовыбор модели
        if model_name == "auto":
            model_name = self._auto_select()
        
        self.model_name = model_name
        
        if model_name in MODELS:
            model_info = MODELS[model_name]
            self.provider = model_info["provider"]
            logger.info(f"Выбрана модель: {model_name} ({model_info['description']})")
        else:
            logger.warning(f"Модель {model_name} не найдена, используем qwen3.5-2b")
            self.model_name = "qwen3.5-2b"
            self.provider = "qwen"
        
        self._load_model(**kwargs)
    
    def _auto_select(self) -> str:
        """Автоматический выбор лучшей модели"""
        import torch
        import psutil
        
        # Определяем доступные ресурсы
        has_cuda = torch.cuda.is_available()
        available_ram = psutil.virtual_memory().available / (1024**3)
        
        logger.info(f"Автовыбор: CUDA={has_cuda}, RAM={available_ram:.1f}GB")
        
        # Логика выбора
        if has_cuda:
            if available_ram > 20:
                return "qwen3.5-9b"
            elif available_ram > 10:
                return "qwen3.5-4b"
            else:
                return "qwen3.5-3b"
        else:
            # CPU режим
            if available_ram > 8:
                return "qwen3.5-2b"
            elif available_ram > 4:
                return "qwen3.5-0.8b"
            else:
                return "bitnet-2b"
    
    def _load_model(self, **kwargs):
        """Загружает выбранную модель"""
        try:
            if self.provider == "bitnet":
                from .bitnet_model_manager import BitNetModelManager
                self.current_model = BitNetModelManager(
                    model_size="bitnet-b1.58-2b",
                    cache_dir=self.cache_dir,
                    **kwargs
                )
                
            elif self.provider == "qwen":
                from .qwen_model_manager import QwenModelManager
                self.current_model = QwenModelManager(
                    model_size=self.model_name,
                    cache_dir=self.cache_dir,
                    **kwargs
                )
                
            elif self.provider == "fractal":
                from .fractal_model_manager import FractalModelManager
                self.current_model = FractalModelManager(
                    cache_dir=self.cache_dir,
                    **kwargs
                )
            else:
                logger.error(f"Неизвестный провайдер: {self.provider}")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Генерирует текст"""
        if self.current_model:
            return self.current_model.generate(prompt, **kwargs)
        return "Ошибка: модель не загружена"
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус модели"""
        info = {
            "model": self.model_name,
            "provider": self.provider,
            "available": self.current_model is not None,
            "initialized": False
        }
        
        if self.current_model and hasattr(self.current_model, 'initialized'):
            info["initialized"] = self.current_model.initialized
            
        if self.model_name in MODELS:
            info.update(MODELS[self.model_name])
            
        return info
    
    @staticmethod
    def list_models() -> Dict[str, Dict]:
        """Список всех доступных моделей"""
        return MODELS.copy()
    
    @staticmethod
    def get_recommendation() -> str:
        """Рекомендуемая модель для текущей системы"""
        import torch
        import psutil
        
        has_cuda = torch.cuda.is_available()
        available_ram = psutil.virtual_memory().available / (1024**3)
        
        if has_cuda and available_ram > 20:
            return "qwen3.5-9b"
        elif available_ram > 8:
            return "qwen3.5-2b" 
        elif available_ram > 4:
            return "qwen3.5-0.8b"
        else:
            return "bitnet-2b"
