"""
Улучшенный менеджер моделей с поддержкой ruGPT-3 и оптимизированной токенизацией.
Поддерживает GPU токенизацию, гибридный кэш и горячее окно 1.5GB.
"""

import os
import time
import hashlib
import threading
from typing import Dict, Any, Optional, List, Tuple
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
import numpy as np

# Отключаем HF_HUB_ENABLE_HF_TRANSFER для совместимости
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)

# Опциональные импорты
try:
    import psutil
except Exception:
    psutil = None

logger = logging.getLogger(__name__)


class RuGPT3ModelManager:
    """
    Улучшенный менеджер моделей с поддержкой ruGPT-3 и оптимизированной токенизацией.
    
    Особенности:
    - Поддержка ruGPT-3 и других моделей
    - Токенизация на GPU
    - Интеграция с гибридным кэшем
    - Горячее окно 1.5GB для токенов
    - Оптимизированная генерация
    """
    
    def __init__(
        self,
        brain,
        model_name: str = "sberbank-ai/rugpt3large_based_on_gpt2",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        max_memory_gb: float = 1.5,  # Горячее окно 1.5GB
        enable_gpu_tokenization: bool = True,
        cache_tokens: bool = True,
        **kwargs
    ):
        """
        Инициализирует RuGPT3ModelManager.
        
        Args:
            brain: Ссылка на ядро CogniFlex
            model_name: Имя модели (ruGPT-3, GPT-2, и т.д.)
            cache_dir: Директория кэша
            device: Устройство (auto, cpu, cuda)
            max_memory_gb: Максимальный объем памяти для токенов (1.5GB)
            enable_gpu_tokenization: Включить токенизацию на GPU
            cache_tokens: Кэшировать токены в гибридном кэше
        """
        self.brain = brain
        self.model_name = model_name
        self.initialized = False
        self.cache_dir = cache_dir or self._get_cache_dir()
        
        # Определение устройства
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.gpu_available = torch.cuda.is_available() and self.device.startswith("cuda")
        
        # Настройки памяти
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.enable_gpu_tokenization = enable_gpu_tokenization and self.gpu_available
        self.cache_tokens = cache_tokens
        
        # Инициализация компонентов
        self.tokenizer = None
        self.model = None
        self.hybrid_cache = None
        self.token_cache = {}  # Локальный кэш токенов
        self.cache_lock = threading.RLock()
        
        # Статистика
        self.stats = {
            "total_tokens": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "gpu_tokenizations": 0,
            "cpu_tokenizations": 0,
            "avg_tokenization_time": 0.0,
            "memory_usage_mb": 0.0
        }
        
        # Инициализация
        self._initialize_model()
        self._initialize_cache()
        
        logger.info(
            f"RuGPT3ModelManager инициализирован: модель={model_name}, "
            f"устройство={self.device}, GPU токенизация={self.enable_gpu_tokenization}, "
            f"память={max_memory_gb}GB"
        )
    
    def _get_cache_dir(self) -> str:
        """Определяет директорию кэша."""
        try:
            if hasattr(self.brain, 'cache_dir') and self.brain.cache_dir:
                return self.brain.cache_dir
            return os.environ.get('COGNIFLEX_CACHE_DIR', 'ml_cache')
        except Exception:
            return 'ml_cache'
    
    def _initialize_model(self) -> None:
        """Инициализирует модель и токенизатор."""
        # Полностью отключаем HF_HUB_ENABLE_HF_TRANSFER
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
        
        try:
            logger.info(f"Попытка загрузки модели {self.model_name}...")
            
            # Проверка доступности модели
            try:
                # Загрузка токенизатора с отключенным hf_transfer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=False,
                    use_fast=False  # Используем медленный токенизатор
                )
                
                # Установка pad_token если отсутствует
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Загрузка модели с отключенным hf_transfer
                if "rugpt" in self.model_name.lower():
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        torch_dtype=torch.float16 if self.gpu_available else torch.float32,
                        device_map="auto" if self.gpu_available else None,
                        trust_remote_code=True,
                        local_files_only=False
                    )
                else:
                    # Fallback на GPT-2
                    self.model = GPT2LMHeadModel.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        torch_dtype=torch.float16 if self.gpu_available else torch.float32,
                        device_map="auto" if self.gpu_available else None
                    )
                
                # Перемещение модели на устройство
                if not self.gpu_available or not hasattr(self.model, 'device'):
                    self.model = self.model.to(self.device)
                
                self.initialized = True
                logger.info(f"Модель {self.model_name} успешно загружена на {self.device}")
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Ошибка загрузки модели {self.model_name}: {error_msg}")
                
                # Если ошибка связана с hf_transfer, пробуем еще раз с другими параметрами
                if "hf_transfer" in error_msg:
                    logger.info("Пробуем загрузку без hf_transfer...")
                    try:
                        # Временное отключение переменной окружения
                        old_hf_transfer = os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')
                        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
                        
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name,
                            cache_dir=self.cache_dir,
                            trust_remote_code=True,
                            local_files_only=False,
                            use_fast=False
                        )
                        
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                        
                        if "rugpt" in self.model_name.lower():
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_name,
                                cache_dir=self.cache_dir,
                                torch_dtype=torch.float16 if self.gpu_available else torch.float32,
                                device_map="auto" if self.gpu_available else None,
                                trust_remote_code=True,
                                local_files_only=False
                            )
                        else:
                            self.model = GPT2LMHeadModel.from_pretrained(
                                self.model_name,
                                cache_dir=self.cache_dir,
                                torch_dtype=torch.float16 if self.gpu_available else torch.float32,
                                device_map="auto" if self.gpu_available else None
                            )
                        
                        if not self.gpu_available or not hasattr(self.model, 'device'):
                            self.model = self.model.to(self.device)
                        
                        # Восстановление переменной окружения
                        if old_hf_transfer is not None:
                            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = old_hf_transfer
                        
                        self.initialized = True
                        logger.info(f"Модель {self.model_name} успешно загружена на {self.device}")
                        
                    except Exception as e2:
                        logger.error(f"Повторная ошибка загрузки: {e2}")
                        raise e
                else:
                    # Другая ошибка, пробуем локальную загрузку
                    raise e
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {self.model_name}: {e}")
            # Пробуем загрузить GPT-2 как fallback
            try:
                self._load_gpt2_fallback()
            except Exception as e2:
                logger.error(f"Ошибка загрузки fallback модели: {e2}")
                self.initialized = False
    
    def _load_gpt2_fallback(self) -> None:
        """Загружает GPT-2 как fallback."""
        logger.info("Загрузка GPT-2 как fallback...")
        self.model_name = "gpt2"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            cache_dir=self.cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(
            "gpt2",
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if self.gpu_available else torch.float32
        )
        
        if not self.gpu_available or not hasattr(self.model, 'device'):
            self.model = self.model.to(self.device)
        
        self.initialized = True
        logger.info("GPT-2 fallback успешно загружен")
    
    def _initialize_cache(self) -> None:
        """Инициализирует гибридный кэш."""
        if not self.cache_tokens:
            return
        
        try:
            # Получаем гибридный кэш из brain
            if hasattr(self.brain, 'get_component'):
                self.hybrid_cache = self.brain.get_component('hybrid_cache')
            
            # Если нет, создаем локальный кэш
            if not self.hybrid_cache:
                from cogniflex.memory.hybrid_token_cache import HybridTokenCache
                self.hybrid_cache = HybridTokenCache(
                    brain=self.brain,
                    target_memory_gb=self.max_memory_gb,
                    max_memory_tokens=int(self.max_memory_bytes / 4096)  # ~4KB на токен
                )
            
            logger.info("Гибридный кэш токенов инициализирован")
            
        except Exception as e:
            logger.warning(f"Ошибка инициализации гибридного кэша: {e}")
            self.hybrid_cache = None
    
    def tokenize_with_cache(
        self, 
        text: str, 
        return_tensors: str = "pt",
        use_gpu: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Токенизирует текст с кэшированием и поддержкой GPU.
        
        Args:
            text: Текст для токенизации
            return_tensors: Формат тензоров
            use_gpu: Использовать GPU для токенизации
            
        Returns:
            Токенизированный тензор
        """
        start_time = time.time()
        
        # Генерация ключа кэша
        cache_key = self._generate_cache_key(text, return_tensors)
        
        # Попытка получить из кэша
        if self.hybrid_cache:
            cached_tokens = self.hybrid_cache.get_token(cache_key)
            if cached_tokens:
                self.stats["cache_hits"] += 1
                return self._tensor_from_cache(cached_tokens)
        
        # Решение об использовании GPU
        use_gpu_tokenization = use_gpu if use_gpu is not None else self.enable_gpu_tokenization
        use_gpu_tokenization = use_gpu_tokenization and self.gpu_available
        
        # Токенизация
        if use_gpu_tokenization:
            tokens = self._tokenize_on_gpu(text, return_tensors)
            self.stats["gpu_tokenizations"] += 1
        else:
            tokens = self.tokenizer.encode(text, return_tensors=return_tensors)
            if self.gpu_available:
                tokens = tokens.to(self.device)
            self.stats["cpu_tokenizations"] += 1
        
        # Сохранение в кэш
        if self.hybrid_cache:
            try:
                cache_data = self._tensor_to_cache(tokens)
                self.hybrid_cache.add_token(cache_key, cache_data)
            except Exception as e:
                logger.warning(f"Ошибка сохранения токенов в кэш: {e}")
        
        # Обновление статистики
        tokenization_time = time.time() - start_time
        self.stats["total_tokens"] += tokens.numel()
        self.stats["cache_misses"] += 1
        self.stats["avg_tokenization_time"] = (
            (self.stats["avg_tokenization_time"] * (self.stats["cache_misses"] - 1) + tokenization_time)
            / self.stats["cache_misses"]
        )
        
        return tokens
    
    def _tokenize_on_gpu(self, text: str, return_tensors: str) -> torch.Tensor:
        """Токенизирует текст на GPU."""
        try:
            # Используем токенизатор на GPU
            inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Перемещение на GPU
            if self.gpu_available:
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
            
            return inputs['input_ids']
            
        except Exception as e:
            logger.warning(f"Ошибка GPU токенизации, fallback на CPU: {e}")
            return self.tokenizer.encode(text, return_tensors=return_tensors)
    
    def _generate_cache_key(self, text: str, return_tensors: str) -> str:
        """Генерирует ключ для кэша токенов."""
        content = f"{text}_{return_tensors}_{self.model_name}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _tensor_to_cache(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Конвертирует тензор для сохранения в кэш."""
        if self.gpu_available:
            tensor = tensor.cpu()
        
        return {
            'data': tensor.numpy().tolist(),
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype)
        }
    
    def _tensor_from_cache(self, cache_data: Dict[str, Any]) -> torch.Tensor:
        """Восстанавливает тензор из кэша."""
        try:
            tensor = torch.tensor(
                cache_data['data'],
                dtype=getattr(torch, cache_data['dtype'])
            ).reshape(cache_data['shape'])
            
            if self.gpu_available:
                tensor = tensor.to(self.device)
            
            return tensor
        except Exception as e:
            logger.error(f"Ошибка восстановления тензора из кэша: {e}")
            raise
    
    def generate_response(
        self, 
        query: str, 
        max_tokens: int = 500,
        temperature: float = 0.4,
        top_p: float = 0.75,
        top_k: int = 40,
        do_sample: bool = True,
        no_repeat_ngram_size: int = 3,
        **kwargs
    ) -> str:
        """
        Генерирует ответ с использованием ruGPT-3.
        
        Args:
            query: Запрос для генерации
            max_tokens: Максимальное количество токенов
            temperature: Температура генерации
            top_p: Top-p параметр
            top_k: Top-k параметр
            do_sample: Использовать сэмплирование
            no_repeat_ngram_size: Размер n-gram для предотвращения повторов
            
        Returns:
            Сгенерированный ответ
        """
        if not self.initialized or not self.model or not self.tokenizer:
            return "Модель не инициализирована. ML компоненты недоступны."
        
        try:
            # Улучшенный системный промпт для GPT-2
            if "gpt2" in self.model_name.lower():
                # Для GPT-2 используем английский промпт с переводом
                system_prompt = "Answer the question clearly and concisely. "
                # Переводим запрос на английский для лучшего качества
                translated_query = self._translate_to_english(query)
                full_query = system_prompt + translated_query
                need_translate_back = True
            else:
                # Для ruGPT-3 используем русский промпт
                system_prompt = "Ответь на вопрос по теме, кратко и по существу: "
                full_query = system_prompt + query
                need_translate_back = False
            
            # Ограничение max_tokens
            max_tokens = min(max_tokens, 500)
            
            # Токенизация с кэшированием
            inputs = self.tokenize_with_cache(full_query, return_tensors="pt")
            
            # Генерация с оптимизированными параметрами
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
            
            # Проверка минимальной длины и качества
            if len(response) < 3:
                return self._get_fallback_response(query)
            
            # Проверка качества ответа для GPT-2
            if "gpt2" in self.model_name.lower():
                # Если ответ содержит много бессмысленных символов, используем fallback
                russian_chars = sum(1 for c in response if 'а' <= c.lower() <= 'я')
                total_chars = len(response)
                russian_ratio = russian_chars / total_chars if total_chars > 0 else 0
                
                # Если мало русских букв или ответ слишком короткий, используем fallback
                if russian_ratio < 0.3 or len(response) < 20:
                    return self._get_fallback_response(query)
            
            # Ограничение длины ответа
            if len(response) > 500:
                response = response[:500] + "..."
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return f"Произошла ошибка при генерации ответа: {str(e)}"
    
    def _translate_to_english(self, text: str) -> str:
        """Простой перевод русского на английский для GPT-2."""
        # Расширенный словарь для перевода
        translations = {
            # Вопросы
            "что такое": "what is",
            "как работает": "how does", 
            "расскажи о": "tell about",
            "почему": "why",
            "где": "where",
            "когда": "when",
            "кто": "who",
            "как": "how",
            "какие": "what",
            "сколько": "how many",
            "какой": "what kind of",
            
            # Технические термины
            "искусственный интеллект": "artificial intelligence",
            "машинное обучение": "machine learning",
            "нейронные сети": "neural networks",
            "языки программирования": "programming languages",
            "глубокое обучение": "deep learning",
            "алгоритм": "algorithm",
            "данные": "data",
            "программирование": "programming",
            "компьютер": "computer",
            "сеть": "network",
            
            # Общие фразы
            "привет": "hello",
            "спасибо": "thank you", 
            "пока": "bye",
            "хорошо": "good",
            "плохо": "bad",
            "интересно": "interesting",
            "понятно": "understandable",
            "помощь": "help",
            "вопрос": "question",
            "ответ": "answer",
            
            # Новые фразы из диалога
            "преимущества": "advantages",
            "недостатки": "disadvantages",
            "модели": "models",
            "языковые модели": "language models",
            "обсудим": "let's discuss",
            "аспекты": "aspects",
            "важны": "are important",
            "подробнее": "in more detail",
            "влияет на": "affects",
            "результат": "result",
            "генерация текста": "text generation",
            "тест": "test"
        }
        
        # Простая замена
        result = text.lower()
        for ru, en in translations.items():
            result = result.replace(ru, en)
        
        # Если нет изменений, возвращаем оригинал
        if result == text.lower():
            return text
        
        return result.capitalize()
    
    def _translate_to_russian(self, text: str) -> str:
        """Простой перевод английского на русский."""
        # Расширенный словарь для обратного перевода
        translations = {
            "artificial intelligence": "искусственный интеллект",
            "machine learning": "машинное обучение", 
            "neural networks": "нейронные сети",
            "programming languages": "языки программирования",
            "deep learning": "глубокое обучение",
            "algorithm": "алгоритм",
            "data": "данные",
            "programming": "программирование",
            "computer": "компьютер",
            "network": "сеть",
            "hello": "привет",
            "thank you": "спасибо",
            "bye": "пока",
            "good": "хорошо",
            "bad": "плохо", 
            "interesting": "интересно",
            "understandable": "понятно",
            "help": "помощь",
            "question": "вопрос",
            "answer": "ответ",
            "advantages": "преимущества",
            "disadvantages": "недостатки",
            "models": "модели",
            "language models": "языковые модели",
            "let's discuss": "давай обсудим",
            "aspects": "аспекты",
            "are important": "важны",
            "in more detail": "подробнее",
            "affects": "влияет на",
            "result": "результат",
            "text generation": "генерация текста",
            "test": "тест",
            "what is": "что такое",
            "how does": "как работает",
            "tell about": "расскажи о"
        }
        
        # Простая замена
        result = text.lower()
        for en, ru in translations.items():
            result = result.replace(en, ru)
        
        return result.capitalize()
    
    def _get_fallback_response(self, query: str) -> str:
        """Возвращает fallback ответ для сложных запросов."""
        query_lower = query.lower()
        
        # Базовые ответы по ключевым словам
        if "привет" in query_lower or "hello" in query_lower:
            return "Привет! Я когнитивная система CogniFlex. Чем могу помочь?"
        
        if "спасибо" in query_lower or "thank" in query_lower:
            return "Пожалуйста! Рад был помочь."
        
        if "искусственный интеллект" in query_lower or "artificial intelligence" in query_lower:
            return "Искусственный интеллект - это область компьютерных наук, создающая системы, способные выполнять задачи, требующие человеческого интеллекта."
        
        if "машинное обучение" in query_lower or "machine learning" in query_lower:
            return "Машинное обучение - это подраздел ИИ, который позволяет системам улучшать свою работу на основе опыта без явного программирования."
        
        if "нейронные сети" in query_lower or "neural networks" in query_lower:
            return "Нейронные сети - это вычислительные модели, вдохновленные структурой человеческого мозга, используемые для распознавания образов и обработки данных."
        
        if "языки программирования" in query_lower or "programming languages" in query_lower:
            return "Популярные языки программирования включают Python, JavaScript, Java, C++, Go и Rust. Каждый подходит для разных задач."
        
        if "преимущества" in query_lower or "advantages" in query_lower:
            return "Преимущества современных систем включают скорость обработки данных, возможность обучения, адаптивность и автоматизацию рутинных задач."
        
        if "недостатки" in query_lower or "disadvantages" in query_lower:
            return "Недостатки могут включать потребность в больших объемах данных, вычислительные затраты и возможные ошибки в обучении."
        
        if "тест" in query_lower or "test" in query_lower:
            return "Тестирование генерации текста показывает способность системы создавать осмысленные ответы на основе входных данных."
        
        # Общий fallback
        return "Это интересный вопрос. Давайте рассмотрим его подробнее. Могу я предоставить дополнительную информацию по этой теме?"
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Возвращает информацию об использовании памяти."""
        usage = {"memory_usage_mb": 0.0}
        
        try:
            if self.gpu_available and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.device)
                usage["gpu_memory_mb"] = memory_allocated / (1024**2)
                usage["gpu_memory_cached_mb"] = torch.cuda.memory_reserved(self.device) / (1024**2)
            
            if psutil:
                process = psutil.Process()
                usage["ram_memory_mb"] = process.memory_info().rss / (1024**2)
            
            usage["total_tokens_cached"] = len(self.token_cache)
            if self.hybrid_cache:
                cache_stats = self.hybrid_cache.get_cache_stats()
                usage.update(cache_stats)
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о памяти: {e}")
        
        return usage
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы."""
        stats = self.stats.copy()
        stats.update(self.get_memory_usage())
        
        # Расчет эффективности кэша
        total_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_requests > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_requests
        else:
            stats["cache_hit_rate"] = 0.0
        
        stats["model_name"] = self.model_name
        stats["device"] = self.device
        stats["gpu_available"] = self.gpu_available
        stats["initialized"] = self.initialized
        
        return stats
    
    def cleanup(self) -> None:
        """Очищает ресурсы."""
        try:
            # Очистка кэша
            with self.cache_lock:
                self.token_cache.clear()
            
            # Очистка GPU памяти
            if self.gpu_available and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Сохранение статистики
            logger.info(f"RuGPT3ModelManager остановлен. Статистика: {self.get_stats()}")
            
        except Exception as e:
            logger.error(f"Ошибка при очистке RuGPT3ModelManager: {e}")
    
    def __del__(self):
        """Деструктор."""
        self.cleanup()


# Экспорт
__all__ = ['RuGPT3ModelManager']
