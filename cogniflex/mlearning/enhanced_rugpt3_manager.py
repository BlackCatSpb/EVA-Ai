"""
Улучшенный RuGPT3ModelManager с фрактальным хранилищем
"""

import os
import logging
import json
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from cogniflex.mlearning.fractal_rugpt3_manager import FractalRuGPT3Manager, RUSSIAN_MODELS
from cogniflex.memory.hybrid_token_cache import HybridTokenCache

logger = logging.getLogger(__name__)

class EnhancedRuGPT3ModelManager:
    """
    Улучшенный менеджер ruGPT-3 с фрактальным хранилищем и гибридным кэшем
    """
    
    def __init__(self, brain=None, model_name: str = "fractal_russian", 
                 cache_dir: str = "./cache", device: str = "auto",
                 max_memory_gb: float = 1.5, enable_gpu_tokenization: bool = True,
                 cache_tokens: bool = True):
        
        self.brain = brain
        self.model_name = model_name
        self.cache_dir = self._get_cache_dir(cache_dir)
        self.device = self._determine_device(device)
        self.max_memory_gb = max_memory_gb
        self.enable_gpu_tokenization = enable_gpu_tokenization
        self.cache_tokens = cache_tokens
        
        # GPU доступность
        self.gpu_available = torch.cuda.is_available() and device != "cpu"
        
        # Оптимизации производительности
        self.cache_tokenization = True
        self.parallel_tokenization = True
        self.tokenization_workers = 4
        self.memory_optimization = True
        self.tensor_pool_size = 1000
        
        # Инициализация компонентов
        self.fractal_manager = None
        self.hybrid_cache = None
        self.model = None
        self.tokenizer = None
        self.initialized = False
        
        # Пулы и executors для оптимизации
        self.tokenization_cache = {}
        self.tensor_pool = []
        self.tokenization_executor = None
        self.background_executor = None
        
        # Статистика производительности
        self.performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "tokenization_time": 0.0,
            "generation_time": 0.0,
            "memory_saved_mb": 0.0
        }
        
        # Статистика
        self.stats_lock = threading.Lock()
        self.total_tokens = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.gpu_tokenizations = 0
        self.cpu_tokenizations = 0
        self.total_tokenization_time = 0.0
        self.generation_count = 0
        
        # Инициализация
        self._initialize_components()
    
    def _get_cache_dir(self, cache_dir: str) -> str:
        """Определяет директорию кэша."""
        try:
            if hasattr(self.brain, 'cache_dir') and self.brain.cache_dir:
                return self.brain.cache_dir
            return os.environ.get('COGNIFLEX_CACHE_DIR', cache_dir)
        except Exception:
            return cache_dir
    
    def _determine_device(self, device: str) -> str:
        """Безопасно определяет устройство с fallback на CPU."""
        try:
            # Проверяем доступность CUDA
            if torch.cuda.is_available():
                device = torch.device("cuda")
                # Тестируем создание тензора
                test_tensor = torch.tensor([1.0], device=device)
                logger.info(f"CUDA доступен, используем устройство: {device}")
                return str(device)
            else:
                logger.info("CUDA недоступен, используем CPU")
                return "cpu"
        except Exception as e:
            logger.warning(f"Ошибка при определении устройства: {e}, используем CPU")
            return "cpu"
    
    def _initialize_components(self):
        """Инициализирует компоненты с оптимизациями производительности"""
        try:
            # Инициализация переменных и пулов
            self.tokenization_cache = {}
            self.tensor_pool = []
            
            # Исполнители для параллельной обработки
            self.tokenization_executor = ThreadPoolExecutor(max_workers=self.tokenization_workers)
            self.background_executor = ThreadPoolExecutor(max_workers=2)
            
            # Инициализация фрактального менеджера
            storage_path = "./cogniflex_cache/ml_unit/fractal_storage/rugpt3large"
            self.fractal_manager = FractalRuGPT3Manager(
                brain=self.brain,
                model_name=self.model_name,
                storage_path=storage_path
            )
            
            # Инициализация гибридного кэша
            if self.cache_tokens:
                self.hybrid_cache = HybridTokenCache(
                    brain=self.brain,
                    max_memory_gb=self.max_memory_gb,
                    device=self.device
                )
                logger.info("Гибридный кэш токенов инициализирован")
            
            # Инициализация модели
            if self.fractal_manager.initialize():
                self.model = self.fractal_manager.model
                self.tokenizer = self.fractal_manager.tokenizer
                
                # Обеспечиваем что модель остается на GPU как горячее окно
                if self.gpu_available:
                    try:
                        # Явно переносим модель на GPU и фиксируем там
                        self.model = self.model.to(self.device)
                        # Создаем горячий буфер в GPU памяти для модели
                        with torch.no_grad():
                            dummy_input = torch.tensor([[1]], device=self.device)
                            _ = self.model(dummy_input)
                            torch.cuda.empty_cache()  # Очищаем кэш после прогрева
                        logger.info(f"Модель {self.model_name} закреплена в GPU памяти как горячее окно")
                    except Exception as e:
                        logger.warning(f"Не удалось закрепить модель в GPU: {e}")
                
                self.initialized = True
            else:
                logger.error("Не удалось инициализировать фрактальный менеджер")
                self._fallback_to_standard()
                
        except Exception as e:
            logger.error(f"Ошибка инициализации компонентов: {e}")
            self.initialized = False
    
    def _fallback_to_standard(self):
        """Fallback на стандартную модель (только локальная)"""
        try:
            logger.info("Используем fallback на стандартную модель")
            
            # Флаг отключения HuggingFace
            DISABLE_HUGGINGFACE_FALLBACK = True
            
            # Ищем локальную модель ruGPT-3 Large
            # ВАЖНО: модель находится в models/rugpt3_small_fractal/model/
            local_paths = [
                "cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_small_fractal/model",
                "cogniflex_cache/ml_unit/fractal_storage/models/rugpt3_small_fractal",
                "cogniflex_cache/ml_unit/fractal_storage/tokenizers/rugpt3_small_fractal",
            ]
            
            tokenizer = None
            model = None
            
            for path in local_paths:
                if os.path.exists(path):
                    try:
                        # Пробуем загрузить токенизатор локально
                        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                        # Пробуем загрузить модель локально
                        model_path = path.replace("tokenizers", "models")
                        if os.path.exists(model_path):
                            model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
                            logger.info(f"✅ Локальная модель загружена из: {path}")
                            break
                    except Exception as e:
                        logger.warning(f"Не удалось загрузить из {path}: {e}")
                        continue
            
            if tokenizer is None or model is None:
                if DISABLE_HUGGINGFACE_FALLBACK:
                    logger.error("❌ HuggingFace fallback отключен. Локальная модель обязательна.")
                    self.initialized = False
                    return
                else:
                    # Fallback на HuggingFace (если разрешено)
                    logger.warning("Используем fallback на HuggingFace")
                    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
                    model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = model.to(self.device)
            self.tokenizer = tokenizer
            self.model = model
            self.initialized = True
            logger.info("✅ Fallback модель успешно загружена")
                
        except Exception as e:
            logger.error(f"Ошибка fallback: {e}")
            self.initialized = False
    
    def tokenize_with_cache(self, text: str, return_tensors: str = "pt") -> torch.Tensor:
        """Токенизация с кэшированием"""
        if not self.tokenizer:
            raise ValueError("Токенизатор не инициализирован")
        
        start_time = time.time()
        
        # Проверяем кэш
        cache_key = None
        if self.hybrid_cache:
            cache_key = self._generate_cache_key(text)
            cached_tokens = self.hybrid_cache.get(cache_key)
            if cached_tokens is not None:
                with self.stats_lock:
                    self.cache_hits += 1
                return cached_tokens
        
        # Токенизация
        with self.stats_lock:
            self.cache_misses += 1
        
        if False and self.enable_gpu_tokenization and self.gpu_available:
            tokens = self._gpu_tokenize(text, return_tensors)
            with self.stats_lock:
                self.gpu_tokenizations += 1
        else:
            tokens = self.tokenizer.encode(text, return_tensors=return_tensors)
            with self.stats_lock:
                self.cpu_tokenizations += 1
        
        # Сохраняем в кэш
        if self.hybrid_cache and cache_key:
            self.hybrid_cache.put(cache_key, tokens)
        
        # Обновляем статистику
        tokenization_time = time.time() - start_time
        with self.stats_lock:
            self.total_tokenization_time += tokenization_time
            if tokens.numel() > 0:
                self.total_tokens += tokens.numel()
        
        return tokens
    
    def _gpu_tokenize(self, text: str, return_tensors: str = "pt") -> torch.Tensor:
        """GPU токенизация"""
        try:
            # Перемещаем токенизатор на GPU если возможно
            if hasattr(self.tokenizer, 'model') and self.tokenizer.model:
                # Для быстрой токенизации на GPU
                tokens = self.tokenizer.encode(text, return_tensors=return_tensors)
                return tokens.to(self.device)
            else:
                return self.tokenizer.encode(text, return_tensors=return_tensors)
        except Exception as e:
            logger.warning(f"GPU токенизация не удалась: {e}, используем CPU")
            return self.tokenizer.encode(text, return_tensors=return_tensors)
    
    def _generate_cache_key(self, text: str) -> str:
        """Генерирует ключ кэша"""
        return hashlib.md5(f"{self.model_name}_{text}".encode()).hexdigest()
    
    def generate_response(self, query: str, max_tokens: int = 200,
                         temperature: float = 0.3, top_p: float = 0.9,
                         top_k: int = 50, do_sample: bool = True,
                         no_repeat_ngram_size: int = 3, **kwargs) -> str:
        """Генерация ответа"""
        if not self.initialized or not self.model or not self.tokenizer:
            return self._get_fallback_response(query)
        
        try:
            with self.stats_lock:
                self.generation_count += 1
            
            # Улучшенный промпт для разных моделей
            if self.model_name.startswith("fractal") or "rubert" in self.model_name.lower():
                # Для русских моделей используем русский промпт
                system_prompt = "Ответь на вопрос по теме, кратко и по существу: "
                full_query = system_prompt + query
                need_translate_back = False
            else:
                # Для других моделей пробуем перевод
                system_prompt = "Answer the question clearly and concisely. "
                translated_query = self._translate_to_english(query)
                full_query = system_prompt + translated_query
                need_translate_back = True
            
            # Ограничение max_tokens
            max_tokens = min(max_tokens, 200)
            
            # Токенизация с кэшированием
            inputs = self.tokenize_with_cache(full_query, return_tensors="pt")
            # Переносим на устройство модели
            inputs = inputs.to(self.model.device)
            
            # Генерация
            with torch.no_grad():
                generation_params = {
                    'input_ids': inputs,
                    'max_length': inputs.shape[1] + max_tokens,
                    'num_return_sequences': 1,
                    'do_sample': do_sample,
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k,
                    'pad_token_id': self.tokenizer.eos_token_id,
                    'attention_mask': torch.ones_like(inputs),
                    'use_cache': True,
                    'repetition_penalty': 1.2,  # Предотвращаем повторения
                    'length_penalty': 1.0,
                }
                
                if no_repeat_ngram_size > 0:
                    generation_params['no_repeat_ngram_size'] = no_repeat_ngram_size
                
                outputs = self.model.generate(**generation_params)
            
            # Декодирование
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Очистка от промпта
            if response.startswith(system_prompt):
                response = response[len(system_prompt):].strip()
            
            # Обратный перевод если нужно
            if need_translate_back:
                response = self._translate_to_russian(response)
            
            # Очистка от исходного запроса
            if response.startswith(query):
                response = response[len(query):].strip()
            
            response = response.strip()
            
            # Проверка качества - валидация coherent ответа
            if not self._is_response_coherent(response, query):
                logger.warning("Сгенерирован incoherent ответ, пробуем fallback генерацию")
                return self._generate_fallback_response(query)
            
            # Проверка качества
            if len(response) < 3:
                return self._get_fallback_response(query)
            
            # Проверка качества для не-русских моделей
            if not (self.model_name.startswith("fractal") or "rugpt" in self.model_name.lower() or "rubert" in self.model_name.lower()):
                russian_chars = sum(1 for c in response if 'а' <= c.lower() <= 'я')
                total_chars = len(response)
                russian_ratio = russian_chars / total_chars if total_chars > 0 else 0
                
                if russian_ratio < 0.3 or len(response) < 20:
                    return self._get_fallback_response(query)
            
            # Ограничение длины
            if len(response) > 500:
                response = response[:500] + "..."
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return self._get_fallback_response(query)
    
    def _is_response_coherent(self, response: str, query: str) -> bool:
        """Проверяет coherent ли сгенерированный ответ"""
        if len(response) < 5:
            return False
            
        # Проверяем на бессвязный текст (много повторяющихся слов)
        words = response.lower().split()
        if len(words) < 3:
            return False
            
        # Считаем повторения слов
        from collections import Counter
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1] if word_counts else 0
        
        # Если самое частое слово повторяется более 30% от общего количества слов
        if most_common_count > len(words) * 0.3:
            logger.warning(f"Обнаружено слишком много повторений в ответе: {response[:100]}...")
            return False
            
        # Проверяем на наличие coherent структуры (предложения)
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) == 0:
            return False
            
        # Минимум одно предложение должно быть coherent
        coherent_sentences = 0
        for sentence in sentences[:3]:  # Проверяем первые 3 предложения
            words_in_sentence = sentence.split()
            if len(words_in_sentence) >= 3:  # Минимум 3 слова в предложении
                coherent_sentences += 1
                
        return coherent_sentences >= 1
    
    def _generate_fallback_response(self, query: str) -> str:
        """Генерация fallback ответа с более консервативными параметрами"""
        try:
            logger.info("Пробуем fallback генерацию с консервативными параметрами")
            
            # Создаем простой промпт
            system_prompt = "Ответь кратко: "
            full_query = system_prompt + query
            
            # Токенизация
            inputs = self.tokenize_with_cache(full_query, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            # Более консервативные параметры генерации
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs,
                    max_length=inputs.shape[1] + 50,  # Короче
                    num_return_sequences=1,
                    do_sample=False,  # Детерминированная генерация
                    temperature=0.1,  # Очень низкая температура
                    top_k=10,  # Ограниченный top_k
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs),
                    use_cache=True,
                    repetition_penalty=1.5,  # Сильное наказание за повторения
                )
            
            # Декодирование
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Очистка
            if response.startswith(system_prompt):
                response = response[len(system_prompt):].strip()
            if response.startswith(query):
                response = response[len(query):].strip()
                
            response = response.strip()
            
            # Финальная проверка
            if len(response) >= 3 and self._is_response_coherent(response, query):
                logger.info("Fallback генерация успешна")
                return response
            else:
                logger.warning("Fallback генерация тоже не удалась")
                return self._get_fallback_response(query)
                
        except Exception as e:
            logger.error(f"Ошибка fallback генерации: {e}")
            return self._get_fallback_response(query)
    
    def _translate_to_english(self, text: str) -> str:
        """Простой перевод русского на английский"""
        translations = {
            "что такое": "what is", "как работает": "how does", "расскажи о": "tell about",
            "искусственный интеллект": "artificial intelligence", "машинное обучение": "machine learning",
            "нейронные сети": "neural networks", "языки программирования": "programming languages",
            "привет": "hello", "спасибо": "thank you", "пока": "bye", "тест": "test"
        }
        
        result = text.lower()
        for ru, en in translations.items():
            result = result.replace(ru, en)
        
        return result.capitalize() if result != text.lower() else text
    
    def _translate_to_russian(self, text: str) -> str:
        """Простой перевод английского на русский"""
        translations = {
            "artificial intelligence": "искусственный интеллект", "machine learning": "машинное обучение",
            "neural networks": "нейронные сети", "programming languages": "языки программирования",
            "hello": "привет", "thank you": "спасибо", "bye": "пока", "test": "тест"
        }
        
        result = text.lower()
        for en, ru in translations.items():
            result = result.replace(en, ru)
        
        return result.capitalize()
    
    def _get_fallback_response(self, query: str) -> str:
        """Fallback ответы"""
        query_lower = query.lower()
        
        if "привет" in query_lower or "hello" in query_lower:
            return "Привет! Я когнитивная система CogniFlex с фрактальной архитектурой. Чем могу помочь?"
        
        if "искусственный интеллект" in query_lower or "artificial intelligence" in query_lower:
            return "Искусственный интеллект - это область, создающая системы с человеческим интеллектом. Моя фрактальная архитектура позволяет эффективно обрабатывать знания."
        
        if "машинное обучение" in query_lower or "machine learning" in query_lower:
            return "Машинное обучение позволяет системам улучшаться на основе опыта. Фрактальное хранилище оптимизирует этот процесс."
        
        if "нейронные сети" in query_lower or "neural networks" in query_lower:
            return "Нейронные сети вдохновлены структурой мозга. Моя архитектура использует фрактальные принципы для эффективности."
        
        return "Это интересный вопрос. Моя фрактальная модель анализирует информацию для оптимального ответа. Хотите узнать подробнее?"
    
    def export_model(self, export_path: str) -> bool:
        """Экспорт модели"""
        if self.fractal_manager:
            return self.fractal_manager.export_model(export_path)
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику"""
        with self.stats_lock:
            cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses)) if (self.cache_hits + self.cache_misses) > 0 else 0.0
            avg_tokenization_time = self.total_tokenization_time / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
            
            return {
                'total_tokens': self.total_tokens,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'gpu_tokenizations': self.gpu_tokenizations,
                'cpu_tokenizations': self.cpu_tokenizations,
                'avg_tokenization_time': avg_tokenization_time,
                'cache_hit_rate': cache_hit_rate,
                'generation_count': self.generation_count,
                'model_name': self.model_name,
                'device': self.device,
                'gpu_available': self.gpu_available,
                'initialized': self.initialized,
                'fractal_storage': self.fractal_manager is not None
            }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Использование памяти"""
        usage = {"memory_usage_mb": 0.0}
        
        try:
            if self.gpu_available and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.device)
                usage["gpu_memory_mb"] = memory_allocated / (1024**2)
            
            # RAM память
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            usage["ram_memory_mb"] = memory_info.rss / (1024**2)
            
            # Гибридный кэш
            if self.hybrid_cache:
                cache_stats = self.hybrid_cache.get_cache_stats()
                usage.update(cache_stats)
                
        except Exception as e:
            logger.warning(f"Ошибка получения информации о памяти: {e}")
        
        return usage
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о текущей модели"""
        model_info = RUSSIAN_MODELS.get(self.model_name, {})
        return {
            'model_name': self.model_name,
            'initialized': self.initialized,
            'generation_count': self.generation_count,
            'cache_hits': self.cache_hits,
            'fractal_storage': self.fractal_manager is not None,
            'model_info': model_info,
            'device': self.device,
            'gpu_available': self.gpu_available,
            'enable_gpu_tokenization': self.enable_gpu_tokenization
        }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Возвращает доступные модели (только текущая)"""
        if not self.initialized:
            return {}
        
        # Возвращаем только текущую модель
        current_model = {
            self.model_name: RUSSIAN_MODELS.get(self.model_name, {
                "name": self.model_name,
                "description": "Текущая модель",
                "initialized": self.initialized,
                "quality": 8
            })
        }
        
        return current_model
    
    def switch_model(self, new_model_name: str) -> bool:
        """Переключение модели"""
        if new_model_name == self.model_name:
            return True
        
        try:
            logger.info(f"Переключение на модель: {new_model_name}")
            
            # Сохраняем текущую модель
            if self.fractal_manager and self.initialized:
                current_export = os.path.join(self.cache_dir, "current_model_backup")
                self.export_model(current_export)
            
            # Создаем новый менеджер
            storage_path = os.path.join(self.cache_dir, "fractal_storage")
            new_manager = FractalRuGPT3Manager(
                brain=self.brain,
                model_name=new_model_name,
                storage_path=storage_path
            )
            
            if new_manager.initialize():
                self.fractal_manager = new_manager
                self.model = new_manager.model
                self.tokenizer = new_manager.tokenizer
                self.model_name = new_model_name
                self.initialized = True
                
                logger.info(f"Успешно переключено на модель: {new_model_name}")
                return True
            else:
                logger.error(f"Не удалось инициализировать модель: {new_model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка переключения модели: {e}")
            return False
    
    def generate(self, text: str, **kwargs) -> str:
        """
        Метод generate для совместимости с существующим кодом.
        Делегирует вызов к generate_response.
        """
        return self.generate_response(text, **kwargs)
    
    def cleanup(self):
        """Очистка ресурсов"""
        try:
            if self.hybrid_cache:
                self.hybrid_cache.cleanup()
            
            if self.model:
                del self.model
                self.model = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Ресурсы менеджера очищены")
            
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")
    
    def get_model_for_task(self, task_type: str, model_name: Optional[str] = None, **kwargs) -> Tuple[Any, Any, str]:
        """
        Получает модель и токенизатор для указанной задачи.
        
        Args:
            task_type: Тип задачи (например, 'text-generation', 'fractal-text-generation')
            model_name: Имя модели (опционально)
            **kwargs: Дополнительные параметры для инициализации модели
            
        Returns:
            Кортеж (модель, токенизатор, имя_модели)
        """
        if not self.initialized or not self.model or not self.tokenizer:
            logger.error(f"Модель не инициализирована для задачи '{task_type}'")
            return None, None, None
        
        # Используем переданное имя модели или имя задачи
        actual_model_name = model_name or task_type
        
        # Возвращаем текущую модель и токенизатор
        logger.debug(f"Возвращаем модель для задачи '{task_type}' с именем '{actual_model_name}'")
        return self.model, self.tokenizer, actual_model_name
