#!/usr/bin/env python3
"""
Интеграция ruGPT-3 Large с системой CogniFlex
Создает единый интерфейс для работы с моделью через фрактальное хранилище
"""
import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Добавляем путь к CogniFlex
sys.path.append('.')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rugpt3large_integration")

class Rugpt3LargeCogniFlex:
    """
    Интегрированный интерфейс ruGPT-3 Large с CogniFlex
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        
        # Пути к фрактальному хранилищу
        self.base_storage_path = Path("cogniflex_cache/ml_unit/fractal_storage")
        self.model_name = "rugpt3_large_fractal"
        self.model_path = self.base_storage_path / "models" / self.model_name
        self.tokenizer_path = self.base_storage_path / "tokenizers" / self.model_name
        
        # Конфигурация
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 2048
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        
        # Модель и токенизатор
        self.model = None
        self.tokenizer = None
        self.initialized = False
        
        # Метрики
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.cache_hits = 0
        
        logger.info(f"Rugpt3LargeCogniFlex инициализирован")
        logger.info(f"  Устройство: {self.device}")
        logger.info(f"  Модель: {self.model_name}")
    
    def initialize(self) -> bool:
        """Инициализация модели и токенизатора"""
        try:
            logger.info("🚀 Инициализация ruGPT-3 Large...")
            
            # 1. Загрузка токенизатора
            if not self._load_tokenizer():
                return False
            
            # 2. Загрузка модели
            if not self._load_model():
                return False
            
            # 3. Настройка параметров
            self._configure_parameters()
            
            # 4. Интеграция с CogniFlex
            if self.brain:
                self._integrate_with_brain()
            
            self.initialized = True
            logger.info("✅ ruGPT-3 Large успешно инициализирована")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            return False
    
    def _load_tokenizer(self) -> bool:
        """Загрузка токенизатора из фрактального хранилища"""
        try:
            from transformers import AutoTokenizer
            
            logger.info("📦 Загрузка токенизатора...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            vocab_size = len(self.tokenizer.get_vocab())
            logger.info(f"✅ Токенизатор загружен: {vocab_size:,} токенов")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки токенизатора: {e}")
            return False
    
    def _load_model(self) -> bool:
        """Загрузка модели ruGPT-3 Large"""
        try:
            from transformers import AutoModelForCausalLM
            
            logger.info("📦 Загрузка модели...")
            
            # Загружаем модель
            model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            self.model = self.model.to(self.device)
            
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"✅ Модель загружена: {param_count:,} параметров")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def _configure_parameters(self):
        """Настройка параметров генерации"""
        try:
            # Загружаем метаданные модели
            metadata_path = self.model_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Применяем параметры из метаданных
                architecture = metadata.get('architecture', {})
                self.max_length = architecture.get('max_position_embeddings', 2048)
                
                logger.info(f"📋 Параметры настроены: max_length={self.max_length}")
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка настройки параметров: {e}")
    
    def _integrate_with_brain(self):
        """Интеграция с мозгом CogniFlex"""
        try:
            logger.info("🧠 Интеграция с CogniFlex...")
            
            # Регистрируем модель в мозге
            if hasattr(self.brain, 'register_model'):
                self.brain.register_model(self.model_name, self)
            
            # Добавляем в кэш токенов если доступно
            if hasattr(self.brain, 'token_cache'):
                self.token_cache = self.brain.token_cache
            
            logger.info("✅ Интеграция с CogniFlex завершена")
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка интеграции с мозгом: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Генерация текста с использованием ruGPT-3 Large
        
        Args:
            prompt: Входной текст
            **kwargs: Дополнительные параметры генерации
            
        Returns:
            Сгенерированный текст
        """
        if not self.initialized:
            return "Модель не инициализирована"
        
        try:
            self.generation_count += 1
            
            # Параметры генерации
            max_length = kwargs.get('max_length', 100)
            temperature = kwargs.get('temperature', self.temperature)
            top_p = kwargs.get('top_p', self.top_p)
            top_k = kwargs.get('top_k', self.top_k)
            do_sample = kwargs.get('do_sample', True)
            
            # Кодируем промпт
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            inputs = inputs.to(self.device)
            
            # Ограничиваем максимальную длину
            max_new_tokens = min(max_length, self.max_length - inputs.shape[1])
            
            # Генерируем ответ
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Декодируем результат
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Очищаем от промпта
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Обновляем метрики
            self.total_tokens_generated += len(self.tokenizer.encode(response))
            
            logger.debug(f"Сгенерировано: {len(response)} символов")
            
            return response if response else "Понимаю ваш вопрос."
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации: {e}")
            return f"Ошибка генерации: {str(e)}"
    
    def generate_response(self, query: str, context: Optional[str] = None, **kwargs) -> str:
        """
        Генерация ответа на запрос с опциональным контекстом
        
        Args:
            query: Запрос пользователя
            context: Дополнительный контекст
            **kwargs: Параметры генерации
            
        Returns:
            Сгенерированный ответ
        """
        # Формируем полный промпт
        if context:
            prompt = f"Контекст: {context}\n\nВопрос: {query}\n\nОтвет:"
        else:
            prompt = f"Вопрос: {query}\n\nОтвет:"
        
        return self.generate(prompt, **kwargs)
    
    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста"""
        if not self.initialized:
            raise ValueError("Модель не инициализирована")
        
        return self.tokenizer.tokenize(text)
    
    def encode(self, text: str) -> List[int]:
        """Кодирование текста в токены"""
        if not self.initialized:
            raise ValueError("Модель не инициализирована")
        
        return self.tokenizer.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Декодирование токенов в текст"""
        if not self.initialized:
            raise ValueError("Модель не инициализирована")
        
        return self.tokenizer.decode(tokens)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        info = {
            "model_name": self.model_name,
            "initialized": self.initialized,
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "generation_count": self.generation_count,
            "total_tokens_generated": self.total_tokens_generated,
            "cache_hits": self.cache_hits
        }
        
        if self.initialized:
            param_count = sum(p.numel() for p in self.model.parameters())
            vocab_size = len(self.tokenizer.get_vocab())
            
            info.update({
                "parameters": param_count,
                "vocab_size": vocab_size,
                "model_memory_gb": param_count * 4 / 1024**3,
                "tokenizer_path": str(self.tokenizer_path),
                "model_path": str(self.model_path)
            })
        
        return info
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Получение информации об использовании памяти"""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
                "gpu_reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
                "gpu_max_allocated_gb": torch.cuda.max_memory_allocated(0) / 1024**3
            })
        
        if self.initialized:
            param_count = sum(p.numel() for p in self.model.parameters())
            memory_info.update({
                "model_parameters": param_count,
                "model_memory_gb": param_count * 4 / 1024**3,
                "vocab_size": len(self.tokenizer.get_vocab())
            })
        
        return memory_info
    
    def cleanup(self):
        """Очистка ресурсов"""
        try:
            logger.info("🧹 Очистка ресурсов...")
            
            # Сохраняем метаданные
            if self.initialized:
                self._save_metrics()
            
            # Очищаем модель
            if self.model:
                del self.model
                self.model = None
            
            # Очищаем токенизатор
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            # Очищаем GPU память
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.initialized = False
            logger.info("✅ Ресурсы очищены")
            
        except Exception as e:
            logger.error(f"❌ Ошибка очистки: {e}")
    
    def _save_metrics(self):
        """Сохранение метрик использования"""
        try:
            metrics = {
                "generation_count": self.generation_count,
                "total_tokens_generated": self.total_tokens_generated,
                "cache_hits": self.cache_hits,
                "avg_tokens_per_generation": self.total_tokens_generated / max(1, self.generation_count)
            }
            
            metrics_path = self.model_path / "usage_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            logger.info("📊 Метрики сохранены")
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка сохранения метрик: {e}")

def create_rugpt3large_cogniflex(brain=None) -> Rugpt3LargeCogniFlex:
    """Создание интегрированного экземпляра ruGPT-3 Large"""
    return Rugpt3LargeCogniFlex(brain=brain)

def main():
    """Демонстрация работы интеграции"""
    logger.info("🚀 ДЕМОНСТРАЦИЯ ИНТЕГРАЦИИ RU-GPT-3 LARGE С COGNIFLEX")
    logger.info("=" * 70)
    
    try:
        # Создаем экземпляр
        rugpt3 = Rugpt3LargeCogniFlex()
        
        # Инициализируем
        if not rugpt3.initialize():
            logger.error("❌ Не удалось инициализировать модель")
            return 1
        
        # Показываем информацию о модели
        info = rugpt3.get_model_info()
        logger.info("📋 Информация о модели:")
        for key, value in info.items():
            logger.info(f"   {key}: {value}")
        
        # Тестируем генерацию
        test_queries = [
            "Привет, как дела?",
            "Что такое искусственный интеллект?",
            "Расскажи о России кратко",
            "Объясни квантовые вычисления простыми словами"
        ]
        
        logger.info("\n🧪 Тестирование генерации:")
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n{i}. 📝 '{query}'")
            response = rugpt3.generate(query, max_length=80)
            logger.info(f"   💬 '{response}'")
        
        # Показываем использование памяти
        memory = rugpt3.get_memory_usage()
        logger.info(f"\n📊 Использование памяти:")
        for key, value in memory.items():
            logger.info(f"   {key}: {value}")
        
        # Очищаем ресурсы
        rugpt3.cleanup()
        
        logger.info("\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
