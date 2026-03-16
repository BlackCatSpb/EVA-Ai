"""
Qwen3.5 Model Manager для CogniFlex
Поддержка Qwen3.5 Small Series - 0.8B, 1.5B, 3B, 4B, 9B
"""
import logging
import os
from typing import Optional, Dict, List, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logger = logging.getLogger(__name__)

QWEN_MODELS = {
    # Qwen3.5 Small Series (Март 2026) - РЕКОМЕНДУЕТСЯ
    "qwen3.5-0.8b": {
        "name": "Qwen/Qwen3.5-0.8B",
        "params": "0.8B",
        "ram_fp16": "1.6GB",
        "ram_q4": "0.5GB",
        "description": "Минимальная - для телефонов/IoT (ИНТЕГРИРОВАННАЯ ГРАФИКА)"
    },
    "qwen3.5-2b": {
        "name": "Qwen/Qwen3.5-2B",
        "params": "2B",
        "ram_fp16": "4GB",
        "ram_q4": "1GB",
        "description": "Оптимально для интегрированной графики (РЕКОМЕНДУЕТСЯ)"
    },
    "qwen3.5-3b": {
        "name": "Qwen/Qwen3.5-3B",
        "params": "3B",
        "ram_fp16": "6GB", 
        "ram_q4": "2GB",
        "description": "Баланс качества и скорости"
    },
    "qwen3.5-4b": {
        "name": "Qwen/Qwen3.5-4B",
        "params": "4B",
        "ram_fp16": "8GB",
        "ram_q4": "2.5GB",
        "description": "Требует больше RAM"
    },
    "qwen3.5-9b": {
        "name": "Qwen/Qwen3.5-9B",
        "params": "9B",
        "ram_fp16": "18GB",
        "ram_q4": "5GB",
        "description": "Для дискретной графики"
    },
    # Qwen3 (базовые модели)
    "qwen3-0.6b": {
        "name": "Qwen/Qwen3-0.6B",
        "params": "0.6B",
        "ram_fp16": "1.2GB",
        "ram_q4": "0.4GB",
        "description": "Минимальная модель"
    },
    "qwen3-1.7b": {
        "name": "Qwen/Qwen3-1.7B", 
        "params": "1.7B",
        "ram_fp16": "3.4GB",
        "ram_q4": "1GB",
        "description": "Компактная модель"
    },
    "qwen3-4b": {
        "name": "Qwen/Qwen3-4B",
        "params": "4B",
        "ram_fp16": "8GB", 
        "ram_q4": "2.5GB",
        "description": "Средняя модель"
    },
    "qwen3-8b": {
        "name": "Qwen/Qwen3-8B",
        "params": "8B",
        "ram_fp16": "16GB",
        "ram_q4": "5GB",
        "description": "Большая модель"
    },
    # MoE модели
    "qwen3-30b-a3b": {
        "name": "Qwen/Qwen3-30B-A3B",
        "params": "30B total / 3B active",
        "ram_fp16": "60GB",
        "ram_q4": "15GB",
        "description": "MoE - требует мощное железо"
    }
}


class QwenModelManager:
    """
    Менеджер для работы с моделями Qwen3.5
    Оптимизирован для CPU с поддержкой квантизации
    """
    
    def __init__(
        self, 
        model_size: str = "qwen3.5-3b",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        quantize: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Инициализация Qwen Model Manager
        
        Args:
            model_size: Размер модели (qwen3.5-0.8b, qwen3.5-1.5b, qwen3.5-3b, qwen3.5-4b, qwen3.5-9b)
            device: Устройство (auto, cpu, cuda)
            cache_dir: Директория для кэша моделей
            quantize: Использовать квантизацию
            load_in_8bit: Загрузить в 8-bit
            load_in_4bit: Загрузить в 4-bit (требует bitsandbytes)
        """
        self.model_size = model_size.lower()
        self.device = self._get_device(device)
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cogniflex_cache", "qwen_models")
        self.quantize = quantize
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.initialized = False
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"QwenModelManager инициализирован: {model_size}, device={self.device}")
        
        self._initialize_model()
    
    def _get_device(self, device: str) -> str:
        """Определяет устройство для загрузки"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _initialize_model(self):
        """Загружает модель и токенизатор"""
        try:
            if self.model_size not in QWEN_MODELS:
                logger.warning(f"Модель {self.model_size} не найдена, используем qwen3.5-3b")
                self.model_size = "qwen3.5-3b"
            
            model_info = QWEN_MODELS[self.model_size]
            model_name = model_info["name"]
            
            logger.info(f"Загрузка модели: {model_name}")
            logger.info(f"Параметры: {model_info['params']}, RAM: {model_info['ram_fp16']}")
            
            # Настройки загрузки
            load_kwargs = {
                "device_map": "auto" if self.device == "cpu" else self.device,
                "trust_remote_code": True,
                "cache_dir": self.cache_dir,
                "torch_dtype": torch.float16 if self.device == "cpu" else torch.float32,
            }
            
            # Квантизация
            if self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    logger.info("Используется 4-bit квантизация")
                except ImportError:
                    logger.warning("bitsandbytes не установлен, используем 8-bit")
                    self.load_in_8bit = True
            
            if self.load_in_8bit and not self.load_in_4bit:
                load_kwargs["load_in_8bit"] = True
                logger.info("Используется 8-bit квантизация")
            
            # Загрузка токенизатора
            logger.info("Загрузка токенизатора...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Загрузка модели
            logger.info("Загрузка модели (может занять несколько минут)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )
            
            if self.device == "cpu" and not (self.load_in_8bit or self.load_in_4bit):
                self.model = self.model.to(torch.float16)
            
            # Конфигурация генерации
            self.generation_config = GenerationConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            
            self.initialized = True
            logger.info(f"✓ Модель {model_name} успешно загружена!")
            logger.info(f"  Устройство: {self.device}")
            logger.info(f"  Параметров: {model_info['params']}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.initialized = False
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> str:
        """
        Генерирует текст на основе промпта
        
        Args:
            prompt: Входной текст
            max_new_tokens: Максимум новых токенов
            temperature: Температура (0-2)
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repetition_penalty: Штраф за повторения
            
        Returns:
            Сгенерированный текст
        """
        if not self.initialized:
            return "Ошибка: модель не инициализирована"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Переносим на устройство
            if self.device == "cpu" or torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )
            
            # Декодируем только новые токены
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return f"Ошибка генерации: {str(e)}"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Генерирует ответ в формате чата
        
        Args:
            messages: Список сообщений [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            Ответ модели
        """
        if not self.initialized:
            return "Ошибка: модель не инициализирована"
        
        try:
            # Форматируем в текст
            chat_text = self._format_chat(messages)
            return self.generate(chat_text, **kwargs)
        except Exception as e:
            logger.error(f"Ошибка в чате: {e}")
            return f"Ошибка: {str(e)}"
    
    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Форматирует сообщения для модели"""
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                text += f"System: {content}\n\n"
            elif role == "user":
                text += f"User: {content}\n\n"
            elif role == "assistant":
                text += f"Assistant: {content}\n\n"
        return text
    
    def get_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели"""
        if self.model_size in QWEN_MODELS:
            info = QWEN_MODELS[self.model_size].copy()
        else:
            info = {"name": "Unknown", "params": "N/A"}
        
        info.update({
            "initialized": self.initialized,
            "device": self.device,
            "quantize": self.quantize,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "cache_dir": self.cache_dir,
        })
        
        return info
    
    @staticmethod
    def list_available_models() -> Dict[str, Dict]:
        """Возвращает список доступных моделей"""
        return QWEN_MODELS.copy()
    
    def unload(self):
        """Выгружает модель из памяти"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.initialized = False
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Модель выгружена из памяти")
