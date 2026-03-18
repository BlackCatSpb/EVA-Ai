"""
Интеграция ruGPT-3 с фрактальным хранилищем CogniFlex
"""

import os
import logging
import json
import hashlib
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel

from cogniflex.mlearning.storage.fractal_store import FractalWeightStore
from cogniflex.mlearning.storage.fractal_model_loader import FractalModelLoader
from cogniflex.mlearning.storage.model_storage_config import ModelStorageConfig

logger = logging.getLogger(__name__)

# Русскоязычные модели для локального использования
RUSSIAN_MODELS = {
    # Модели от Сбера (ruGPT-3)
    "rugpt3medium": {
        "name": "sberbank-ai/rugpt3large_based_on_gpt2",
        "size_mb": 1500,
        "description": "ruGPT-3 Medium (355M параметров)",
        "requires_download": True,
        "supports_russian": True,
        "quality": 9
    },
    "rugpt3large": {
        "name": "sberbank-ai/rugpt3large_based_on_gpt2",
        "size_mb": 3000,
        "description": "ruGPT-3 Large (760M параметров)",
        "requires_download": True,
        "supports_russian": True,
        "quality": 10
    },
    
    # Модели от DeepPavlov
    "rubert_base_cased": {
        "name": "DeepPavlov/rubert-base-cased",
        "size_mb": 400,
        "description": "RuBERT (110M параметров)",
        "requires_download": True,
        "supports_russian": True,
        "quality": 7,
        "type": "encoder"
    },
    
    # GPT-2 модели, обученные на русском
    "gpt2_russian": {
        "name": "sberbank-ai/rugpt2small",
        "size_mb": 500,
        "description": "ruGPT-2 Small (124M параметров)",
        "requires_download": True,
        "supports_russian": True,
        "quality": 6
    },
    
    # T5 модели для русского
    "rut5_base": {
        "name": "sberbank-ai/ruT5-base",
        "size_mb": 800,
        "description": "RuT5 Base (220M параметров)",
        "requires_download": True,
        "supports_russian": True,
        "quality": 7,
        "type": "seq2seq"
    },
    
    # Локальные альтернативы (не требуют интернета)
    "fractal_russian": {
        "name": "fractal_russian_v1",
        "size_mb": 300,
        "description": "Фрактальная русская модель (локальная)",
        "requires_download": False,
        "supports_russian": True,
        "quality": 5,
        "type": "fractal"
    }
}

class FractalRuGPT3Manager:
    """
    Менеджер ruGPT-3 с фрактальным хранилищем
    """
    
    def __init__(self, brain=None, model_name: str = "rugpt3large", 
                 storage_path: str = "./cogniflex_cache/ml_unit/fractal_storage/rugpt3large"):
        self.brain = brain
        self.model_name = model_name
        self.storage_path = storage_path
        
        # Инициализация фрактального хранилища
        self.config = ModelStorageConfig(
            base_path=storage_path,
            device="cpu",
            block_size=1024,
            fractal_levels=4
        )
        
        self.model_loader = FractalModelLoader(self.config)
        self.fractal_store = None
        
        # Модель и токенизатор
        self.model = None
        self.tokenizer = None
        self.initialized = False
        
        # Метрики
        self.generation_count = 0
        self.cache_hits = 0
        
        logger.info(f"FractalRuGPT3Manager создан для модели: {model_name}")
    
    def initialize(self) -> bool:
        """Инициализация менеджера"""
        try:
            logger.info(f"Инициализация FractalRuGPT3Manager: {self.model_name}")
            
            # Проверяем наличие фрактального хранилища
            if os.path.exists(os.path.join(self.storage_path, self.model_name)):
                # Загружаем из фрактального хранилища
                return self._load_from_fractal_storage()
            else:
                # Создаем модель локально
                return self._create_local_model()
                
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}")
            return False
    
    def _create_local_model(self):
        """Создание модели локально"""
        try:
            from .local_rugpt3_loader import load_rugpt3_medium_local
            
            logger.info(f"Создание модели {self.model_name} локально...")
            
            # Определяем устройство
            device = "auto"
            if hasattr(self, 'device') and self.device:
                device = self.device
            
            # Загружаем модель и токенизатор
            self.model, self.tokenizer = load_rugpt3_medium_local(
                storage_path=self.storage_path,
                device=device
            )
            
            if self.model and self.tokenizer:
                self.initialized = True
                logger.info(f"✅ Модель {self.model_name} создана локально")
                return True
            else:
                logger.error(f"❌ Не удалось создать модель {self.model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка создания локальной модели: {e}")
            return False
    
    def _load_from_fractal_storage(self):
        """Загрузка модели из фрактального хранилища"""
        try:
            from .local_rugpt3_loader import load_rugpt3_medium_local
            
            logger.info(f"Загрузка модели {self.model_name} из фрактального хранилища...")
            
            device = "auto"
            if hasattr(self, 'device') and self.device:
                device = self.device
            
            self.model, self.tokenizer = load_rugpt3_medium_local(
                storage_path=self.storage_path,
                device=device
            )
            
            if self.model and self.tokenizer:
                self.initialized = True
                logger.info(f"✅ Модель {self.model_name} загружена из фрактального хранилища")
                return True
            else:
                logger.error(f"❌ Не удалось загрузить модель {self.model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка загрузки из фрактального хранилища: {e}")
            return False

    def _load_and_save_model(self):
        """Загрузка модели и сохранение во фрактальное хранилище"""
        try:
            model_info = RUSSIAN_MODELS.get(self.model_name, RUSSIAN_MODELS["rugpt3large"])
            
            if model_info.get("type") == "fractal":
                self._load_fractal_model()
            else:
                logger.info(f"Загружаем модель {model_info['name']}...")
                
                os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_info["name"],
                    trust_remote_code=True,
                    use_fast=False
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_info["name"],
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                logger.info("Сохраняем модель во фрактальное хранилище...")
                self.model_loader.save_model(self.model_name, self.model)
                
                tokenizer_path = os.path.join(self.storage_path, "tokenizers", self.model_name)
                os.makedirs(tokenizer_path, exist_ok=True)
                self.tokenizer.save_pretrained(tokenizer_path)
            
            logger.info(f"Модель сохранена во фрактальном хранилище")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки и сохранения: {e}")
            raise

    
    def _load_fractal_model(self):
        """Загрузка локальной фрактальной модели"""
        try:
            # Проверяем наличие фрактального токенизатора
            fractal_tokenizer_path = os.path.join(
                "cogniflex/mlearning/tokenizers/fractal_unified_tokenizer"
            )
            
            if os.path.exists(fractal_tokenizer_path):
                logger.info("Используем фрактальный токенизатор")
                self.tokenizer = AutoTokenizer.from_pretrained(fractal_tokenizer_path)
                
                # Создаем простую модель на основе GPT-2 с фрактальными весами
                self.model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
                
                # Применяем фрактальные веса если есть
                fractal_weights_path = os.path.join(self.storage_path, "fractal_weights")
                if os.path.exists(fractal_weights_path):
                    self._apply_fractal_weights(fractal_weights_path)
                
            else:
                # Fallback на GPT-2
                logger.warning("Фрактальный токенизатор не найден, используем GPT-2")
                self.tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
                self.model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Ошибка загрузки фрактальной модели: {e}")
            raise
    
    def _apply_fractal_weights(self, weights_path: str):
        """Применение фрактальных весов к модели"""
        try:
            # Здесь можно применить фрактальные веса
            # Это упрощенная версия
            logger.info("Применяем фрактальные веса...")
            # TODO: Реализовать применение фрактальных весов
        except Exception as e:
            logger.warning(f"Не удалось применить фрактальные веса: {e}")
    
    def generate_response(self, query: str, max_tokens: int = 500, 
                         temperature: float = 0.7, **kwargs) -> str:
        """Генерация ответа с использованием фрактальной модели"""
        if not self.initialized:
            return "Модель не инициализирована"
        
        try:
            self.generation_count += 1
            
            # Подготовка входа
            inputs = self.tokenizer.encode(query, return_tensors="pt")
            # Переносим на устройство модели
            inputs = inputs.to(self.model.device)
            
            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Декодирование
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Очистка от запроса
            if response.startswith(query):
                response = response[len(query):].strip()
            
            return response if response else "Понимаю ваш вопрос."
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return f"Ошибка генерации: {str(e)}"
    
    def export_model(self, export_path: str) -> bool:
        """Экспорт модели из фрактального хранилища"""
        try:
            logger.info(f"Экспортируем модель {self.model_name} в {export_path}")
            
            # Создаем директорию
            os.makedirs(export_path, exist_ok=True)
            
            # Экспортируем модель
            if self.model:
                model_export_path = os.path.join(export_path, "model")
                self.model.save_pretrained(model_export_path)
            
            # Экспортируем токенизатор
            if self.tokenizer:
                tokenizer_export_path = os.path.join(export_path, "tokenizer")
                self.tokenizer.save_pretrained(tokenizer_export_path)
            
            # Экспортируем метаданные
            metadata = {
                "model_name": self.model_name,
                "model_info": RUSSIAN_MODELS.get(self.model_name, {}),
                "generation_count": self.generation_count,
                "cache_hits": self.cache_hits,
                "fractal_storage": True
            }
            
            with open(os.path.join(export_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Модель успешно экспортирована в {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта: {e}")
            return False
    
    def get_available_models(self) -> Dict[str, Any]:
        """Возвращает доступные модели"""
        return RUSSIAN_MODELS
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о текущей модели"""
        model_info = RUSSIAN_MODELS.get(self.model_name, {})
        return {
            "model_name": self.model_name,
            "initialized": self.initialized,
            "generation_count": self.generation_count,
            "cache_hits": self.cache_hits,
            "fractal_storage": True,
            "model_info": model_info
        }
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Токенизация текста (для совместимости с тестами)"""
        if not self.initialized or not self.tokenizer:
            raise ValueError("Модель или токенизатор не инициализированы")
        
        try:
            # Используем токенизатор для получения токенов
            tokens = self.tokenizer.tokenize(text, **kwargs)
            return tokens
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 100, 
                temperature: float = 0.7, **kwargs) -> str:
        """Генерация текста (для совместимости с тестами)"""
        if not self.initialized:
            raise ValueError("Модель не инициализирована")
        
        # Вызываем существующий метод generate_response
        return self.generate_response(
            prompt, 
            max_tokens=max_length, 
            temperature=temperature, 
            **kwargs
        )

def create_fractal_rugpt3_manager(brain=None, model_name: str = "fractal_russian") -> FractalRuGPT3Manager:
    """Создает менеджер фрактальной ruGPT-3 модели"""
    return FractalRuGPT3Manager(brain=brain, model_name=model_name)
