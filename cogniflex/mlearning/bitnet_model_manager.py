"""
BitNet Model Manager для CogniFlex
Microsoft BitNet b1.58 - 1-bit модели для CPU
"""
import logging
import os
from typing import Optional, Dict, List, Any
import torch

logger = logging.getLogger(__name__)

BITNET_MODELS = {
    "bitnet-b1.58-2b": {
        "name": "microsoft/bitnet-b1.58-2B-4T",
        "gguf_name": "microsoft/bitnet-b1.58-2B-4T-gguf",
        "params": "2B",
        "ram": "0.4GB",
        "description": "Microsoft BitNet - 1-bit, самая быстрая на CPU",
        "tokens_per_sec": "40-48"
    },
    "bitnet-b1.58-3b": {
        "name": "microsoft/BitNet-b1.58-3B",
        "gguf_name": None,
        "params": "3B",
        "ram": "0.6GB",
        "description": "BitNet 3B - больше качество",
        "tokens_per_sec": "30-40"
    }
}


class BitNetModelManager:
    """
    Менеджер для BitNet моделей
    Оптимизирован для CPU - работает без GPU!
    """
    
    def __init__(
        self, 
        model_size: str = "bitnet-b1.58-2b",
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ):
        self.model_size = model_size.lower()
        self.device = "cpu"  # BitNet только для CPU!
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cogniflex_cache", "bitnet_models")
        
        self.model = None
        self.tokenizer = None
        self.initialized = False
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"BitNetModelManager инициализирован: {model_size}")
        self._initialize_model()
    
    def _initialize_model(self):
        """Загружает BitNet модель через bitnet.cpp или transformers"""
        try:
            if self.model_size not in BITNET_MODELS:
                logger.warning(f"Модель {self.model_size} не найдена, используем bitnet-b1.58-2b")
                self.model_size = "bitnet-b1.58-2b"
            
            model_info = BITNET_MODELS[self.model_size]
            model_name = model_info["name"]
            
            logger.info(f"Загрузка BitNet: {model_name}")
            
            # Пробуем загрузить через transformers (требует специальной версии)
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                logger.info("Загрузка через transformers...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=self.cache_dir
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",
                    cache_dir=self.cache_dir
                )
                
                self.initialized = True
                logger.info(f"✓ BitNet загружена!")
                
            except Exception as e:
                logger.warning(f"Не удалось загрузить через transformers: {e}")
                logger.info("Попробуйте установить bitnet.cpp отдельно")
                self._fallback_load()
            
        except Exception as e:
            logger.error(f"Ошибка загрузки BitNet: {e}")
            self.initialized = False
    
    def _fallback_load(self):
        """Fallback - пробуем загрузить через llama.cpp (GGUF формат)"""
        try:
            # Пробуем llama.cpp
            try:
                from llama_cpp import Llama
                
                model_path = os.path.join(self.cache_dir, f"{self.model_size}.gguf")
                
                if os.path.exists(model_path):
                    logger.info(f"Загрузка GGUF модели: {model_path}")
                    self.model = Llama(model_path=model_path)
                    self.initialized = True
                    logger.info("✓ Загружено через llama.cpp")
                else:
                    logger.warning(f"GGUF файл не найден: {model_path}")
                    logger.info("Скачайте модель в формате GGUF с HuggingFace")
            except ImportError:
                logger.warning("llama_cpp не установлен")
                logger.info("Установите: pip install llama-cpp-python")
                
        except Exception as e:
            logger.error(f"Fallback ошибка: {e}")
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Генерирует текст"""
        if not self.initialized:
            return "Ошибка: BitNet модель не инициализирована"
        
        try:
            if hasattr(self, 'model') and hasattr(self.model, '__call__'):
                # Transformers формат
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # Llama.cpp формат
                output = self.model(
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                return output['choices'][0]['text']
                
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return f"Ошибка: {str(e)}"
    
    def get_info(self) -> Dict[str, Any]:
        if self.model_size in BITNET_MODELS:
            info = BITNET_MODELS[self.model_size].copy()
        else:
            info = {"name": "Unknown"}
        
        info.update({
            "initialized": self.initialized,
            "device": self.device,
            "cache_dir": self.cache_dir
        })
        return info
    
    @staticmethod
    def list_available_models() -> Dict[str, Dict]:
        return BITNET_MODELS.copy()
    
    def unload(self):
        if self.model:
            del self.model
            self.model = None
        self.initialized = False
        logger.info("BitNet модель выгружена")


def install_bitnet_support():
    """Установка зависимостей для BitNet"""
    os.system("pip install llama-cpp-python")
    logger.info("Рекомендуется также установить bitnet.cpp отдельно")
