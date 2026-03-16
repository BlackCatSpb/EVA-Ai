#!/usr/bin/env python3
"""
Простой менеджер моделей для тестирования
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class SimpleModelManager:
    """Простой менеджер моделей для тестирования"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Инициализация"""
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.device = torch.device(self.config.get("device", "cpu"))
        
        # Инициализируем модель
        self._initialize_model()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Загружает конфигурацию"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Конфигурация по умолчанию
        return {
            "model_name": "gpt2",
            "device": "cpu",
            "max_length": 256
        }
    
    def _initialize_model(self):
        """Инициализирует модель"""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch не доступен")
            return
        
        try:
            model_name = self.config.get("model_name", "gpt2")
            logger.info(f"Загрузка модели {model_name}...")
            
            # Загружаем модель и токенизатор
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            
            # Устанавливаем pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Переносим на устройство
            self.model.to(self.device)
            self.model.eval()
            
            self.initialized = True
            logger.info(f"Модель {model_name} успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.initialized = False
    
    def generate_response(self, query: str, max_tokens: int = 100) -> str:
        """Генерирует ответ"""
        if not self.initialized:
            return "Модель не инициализирована"
        
        try:
            # Токенизируем
            inputs = self.tokenizer.encode(query, return_tensors='pt').to(self.device)
            
            # Генерируем
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Декодируем
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Убираем исходный запрос
            if response.startswith(query):
                response = response[len(query):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return f"Ошибка генерации: {str(e)}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности"""
        return {
            "initialized": self.initialized,
            "model_name": self.config.get("model_name"),
            "device": str(self.device)
        }
