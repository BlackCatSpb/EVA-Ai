"""
Модуль выбора модели для CogniFlex
Позволяет переключаться между RuGPT, Qwen3.5 и BitNet
"""
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Конфигурация моделей
# Qwen3.5-0.8b - активная модель (из brain_config.json)
MODEL_CONFIGS = {
    "qwen3.5-0.8b": {
        "name": "Qwen3.5-0.8B",
        "type": "qwen",
        "params": "0.8B",
        "ram": "~1.6GB",
        "speed": "быстро",
        "quality": "высоко",
        "description": "Основная модель (активная)",
        "status": "ready",
        "enabled": True
    },
    "qwen3.5-2b": {
        "name": "Qwen3.5-2B",
        "type": "qwen",
        "params": "2B",
        "ram": "~4GB",
        "speed": "средне",
        "quality": "хорошо",
        "description": "Увеличенная версия",
        "status": "disabled",
        "enabled": False
    },
    "bitnet-2b": {
        "name": "BitNet 2B",
        "type": "bitnet",
        "params": "2B",
        "ram": "~0.4GB",
        "speed": "очень быстро",
        "quality": "ниже среднего",
        "description": "Microsoft 1-bit (самая быстрая)",
        "status": "disabled",
        "requires": "llama-cpp-python",
        "enabled": False
    }
}


class ModelSelector:
    """Класс для управления выбором модели"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cogniflex_cache", "models")
        self.current_model = "qwen3.5-0.8b"
        self.loaded_model = None
        
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"ModelSelector инициализирован, модель: {self.current_model}")
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о текущей модели"""
        return MODEL_CONFIGS.get(self.current_model, {})
    
    def list_models(self) -> Dict[str, Dict]:
        """Список всех доступных моделей"""
        return MODEL_CONFIGS.copy()
    
    def switch_model(self, model_name: str) -> bool:
        """
        Переключается на указанную модель
        
        Args:
            model_name: Имя модели (qwen3.5-0.8b, qwen3.5-2b, bitnet-2b)
            
        Returns:
            True если успешно
        """
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Модель {model_name} не найдена")
            return False
        
        logger.info(f"Переключение на модель: {model_name}")
        self.current_model = model_name
        
        # Обновляем статус
        MODEL_CONFIGS[model_name]["status"] = "switching"
        
        return True
    
    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Загружает модель в память
        
        Args:
            model_name: Имя модели или None для текущей
            
        Returns:
            True если успешно
        """
        model_name = model_name or self.current_model
        
        # Check if model is enabled
        if model_name in MODEL_CONFIGS and not MODEL_CONFIGS[model_name].get('enabled', True):
            logger.warning(f"Модель {model_name} отключена конфигурацией")
            return False
        
        # Загружать другие модели запрещено
        logger.warning(f"Модель {model_name} недоступна")
        return False
    
    def _load_qwen(self, model_name: str) -> bool:
        """Загружает Qwen модель"""
        try:
            from cogniflex.mlearning.qwen_model_manager import QwenModelManager
            
            logger.info(f"Загрузка Qwen: {model_name}")
            
            # Отображаем имя
            qwen_size = model_name.replace("qwen3.5-", "qwen3.5-")
            
            self.loaded_model = QwenModelManager(
                model_size=model_name,
                quantize=True,
                cache_dir=self.cache_dir
            )
            
            if self.loaded_model.initialized:
                MODEL_CONFIGS[model_name]["status"] = "ready"
                logger.info(f"Модель {model_name} загружена")
                return True
            else:
                MODEL_CONFIGS[model_name]["status"] = "error"
                return False
                
        except Exception as e:
            logger.error(f"Ошибка загрузки Qwen: {e}")
            return False
    
    def _load_bitnet(self, model_name: str) -> bool:
        """Загружает BitNet модель"""
        try:
            from cogniflex.mlearning.bitnet_model_manager import BitNetModelManager
            
            logger.info(f"Загрузка BitNet: {model_name}")
            
            self.loaded_model = BitNetModelManager(
                model_size="bitnet-b1.58-2b",
                cache_dir=self.cache_dir
            )
            
            if self.loaded_model.initialized:
                MODEL_CONFIGS[model_name]["status"] = "ready"
                logger.info(f"Модель {model_name} загружена")
                return True
            else:
                MODEL_CONFIGS[model_name]["status"] = "error"
                return False
                
        except Exception as e:
            logger.error(f"Ошибка загрузки BitNet: {e}")
            return False
    
    def unload_model(self):
        """Выгружает текущую модель из памяти"""
        if self.loaded_model:
            try:
                self.loaded_model.unload()
            except Exception as e:
                logger.warning(f"Не удалось выгрузить модель: {e}")
            self.loaded_model = None
        
        # Возвращаем статус
        for name in MODEL_CONFIGS:
            if name != "qwen3.5-0.8b":
                MODEL_CONFIGS[name]["status"] = "not_loaded"
    
    def get_recommendation(self) -> str:
        """
        Возвращает рекомендуемую модель для текущей системы
        
        Returns:
            Имя рекомендуемой модели
        """
        import psutil
        
        available_ram = psutil.virtual_memory().available / (1024**3)
        
        if available_ram > 6:
            return "qwen3.5-2b"
        elif available_ram > 2:
            return "qwen3.5-0.8b"
        else:
            return "bitnet-2b"


def print_model_comparison():
    """Выводит сравнение моделей в консоль"""
    print("\n" + "="*60)
    print("Dostupnye modeli CogniFlex")
    print("="*60)
    
    for name, info in MODEL_CONFIGS.items():
        status_icon = {
            "ready": "[OK]",
            "not_loaded": "[  ]",
            "switching": "[..]",
            "error": "[!!]"
        }.get(info.get("status", "not_loaded"), "[??]")
        
        print(f"\n{status_icon} {info['name']} ({name})")
        print(f"   Parametry: {info['params']}")
        print(f"   RAM: {info['ram']}")
        print(f"   Skorost: {info['speed']}")
        print(f"   Opisanie: {info['description']}")
    
    print("\n" + "="*60)
    
    selector = ModelSelector()
    rec = selector.get_recommendation()
    print(f"Rekomendetsya: {MODEL_CONFIGS[rec]['name']}")
    print("="*60 + "\n")
