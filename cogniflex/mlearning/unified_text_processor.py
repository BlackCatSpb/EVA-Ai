"""Объединенный текстовый процессор с асинхронной токенизацией для CogniFlex
Обновленная версия с интеграцией гибридного кэша и правильной архитектурой
"""

import os
import time
import logging
import multiprocessing
import numpy as np
import nltk
import re
import hashlib
import json
from collections import defaultdict, Counter, OrderedDict
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import psutil
import spacy
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from transformers import pipeline

logger = logging.getLogger("cogniflex.unified_text_processor")

# Попытка загрузить необходимые ресурсы NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    # Пытаемся загрузить русскую модель spaCy
    nlp = spacy.load("ru_core_news_sm")
except OSError:
    try:
        # Если не получается, загружаем английскую модель
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Если и это не получается, загружаем базовую модель
        nlp = spacy.blank("ru")
        logger.warning("Не удалось загрузить языковые модели spaCy, используем базовую модель")

class UnifiedTextProcessor:
    """Объединенный текстовый процессор с асинхронной токенизацией для CogniFlex
    Обновленная версия с интеграцией гибридного кэша и ResponseGenerator"""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None, 
                 use_gpu: bool = False, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 max_workers: Optional[int] = None, use_async: bool = True,
                 hybrid_cache: Optional[Any] = None):
        """Инициализирует объединенный текстовый процессор.
        
        Args:
            brain: Ссылка на ядро CogniFlex
            cache_dir: Путь к директории кэша
            use_gpu: Использовать GPU если доступен
            model_name: Название модели эмбеддингов
            max_workers: Максимальное количество рабочих потоков
            use_async: Использовать асинхронную обработку
            hybrid_cache: Экземпляр гибридного кэша (должен быть предоставлен извне)
        """
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cogniflex_cache", "text_processing")
        os.makedirs(self.cache_dir, exist_ok=True)
        

        
        # Определяем количество ядер
        self.physical_cores = psutil.cpu_count(logical=False)
        self.logical_cores = psutil.cpu_count(logical=True)
        
        # Устанавливаем количество рабочих потоков
        self.max_workers = max_workers or min(4, max(1, self.logical_cores - 1))
        
        # Параметры
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model_name = model_name
        self.use_async = use_async
        
        # Компоненты
        self.hybrid_cache = hybrid_cache
        
        # Инициализация NLP компонентов
        self._init_nlp_models()
        
        # Инициализация компонентов интеграции
        self._init_integration_components()
        
        # Устанавливаем обратную связь с brain через событийную систему
        if brain:
            self.brain = brain
            # Уведомляем brain через событийную систему, что текстовый процессор готов
            if hasattr(brain, 'events'):
                brain.events.trigger('text_processor_ready', self)
                logger.info("Уведомление о готовности text_processor отправлено через событийную систему")
            # Поддерживаем старый механизм для обратной совместимости
            elif hasattr(brain, 'on_text_processor_ready'):
                callbacks = brain.on_text_processor_ready
                brain.on_text_processor_ready = []
                for callback in callbacks:
                    try:
                        callback(self)
                    except Exception as e:
                        logger.error(f"Ошибка в обратном вызове текстового процессора: {e}")
        
        logger.info(f"UnifiedTextProcessor инициализирован с {self.max_workers} воркерами")
        logger.info(f"Физических ядер: {self.physical_cores}, логических: {self.logical_cores}")
    
    def _init_integration_components(self):
        """Инициализирует компоненты для интеграции с другими модулями."""
        # Проверяем наличие brain
        if not self.brain:
            logger.warning("Brain не инициализирован. Некоторые функции могут быть недоступны.")
        
        # Гибридный кэш
        self.hybrid_cache = None
        if self.brain and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager:
            self.hybrid_cache = self.brain.memory_manager.get_hybrid_cache()
            if self.hybrid_cache:
                logger.debug("Используем гибридный кэш из MemoryManager")
        
        # Если гибридный кэш не получен из MemoryManager, но HybridTokenCache доступен
        if self.hybrid_cache is None:
            try:
                from cogniflex.memory.hybrid_token_cache import HybridTokenCache
                logger.debug("HybridTokenCache импортирован успешно")
                
                # Убеждаемся, что у brain есть cache_dir перед созданием кэша
                if not self.brain:
                    # Создаем минимальный brain объект
                    class MinimalBrain:
                        def __init__(self, cache_dir):
                            self.cache_dir = cache_dir
                    self.brain = MinimalBrain(self.cache_dir)
                elif not hasattr(self.brain, 'cache_dir'):
                    self.brain.cache_dir = self.cache_dir
                
                # Инициализируем гибридный кэш
                self.hybrid_cache = HybridTokenCache(
                    brain=self.brain,
                    max_memory_tokens=10000,
                    disk_cache_dir="hybrid_cache"
                )
                logger.debug("Создан внутренний гибридный кэш")
                
                # Сохраняем кэш в memory_manager, если он существует
                if self.brain and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager:
                    self.brain.memory_manager.hybrid_cache = self.hybrid_cache
                    logger.debug("Гибридный кэш сохранен в MemoryManager")
            except Exception as e:
                logger.warning(f"Не удалось создать гибридный кэш: {e}")
    
    def _init_nlp_models(self):
        """Инициализирует NLP модели и компоненты."""
        try:
            # Загрузка основных моделей
            try:
                self.nlp = spacy.load("ru_core_news_sm", disable=["parser", "ner", "textcat"])
                logger.debug("Русская SpaCy модель загружена")
            except Exception as e:
                try:
                    self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
                    logger.debug("Английская SpaCy модель загружена как fallback")
                except Exception as e2:
                    logger.warning(f"Не удалось загрузить SpaCy модель: {e}")
                    self.nlp = None
            
            # Загрузка модели эмбеддингов
            try:
                device = "cuda" if self.use_gpu else "cpu"
                self.embedder = SentenceTransformer(self.model_name, device=device)
                logger.info(f"Модель эмбеддингов '{self.model_name}' загружена на {device}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель эмбеддингов: {e}")
                self.embedder = None
            
            # Инициализация анализаторов
            self.sentiment_analyzer = None
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
                logger.debug("Sentiment analyzer инициализирован")
            except Exception as e:
                logger.warning(f"Sentiment analyzer недоступен: {e}")
            
            self.stemmer = SnowballStemmer('russian')
            self.lemmatizer = WordNetLemmatizer()
            
            # Инициализация пула процессов для асинхронной обработки
            if self.use_async:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            else:
                self.executor = None
            
            logger.info("NLP-модели успешно инициализированы")
        except Exception as e:
            logger.error(f"Ошибка инициализации NLP моделей: {e}", exc_info=True)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Основной метод обработки текста.
        
        Args:
            text: Текст для обработки
            
        Returns:
            Dict[str, Any]: Результат обработки
        """
        if not text or not isinstance(text, str):
            return {
                "text": "",
                "tokens": [],
                "lemmas": [],
                "keywords": [],
                "sentiment": {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0},
                "embeddings": None,
                "metadata": {
                    "processor": "unified_text_processor",
                    "version": "2.1",
                    "status": "empty_input"
                }
            }
        
        start_time = time.time()
        try:
            # Токенизация
            tokens = self.tokenize(text)
            
            # Лемматизация
            lemmas = self.lemmatize(tokens)
            
            # Извлечение ключевых слов
            keywords = self.extract_keywords(text, tokens)
            
            # Анализ тональности
            sentiment = self.analyze_sentiment(text)
            
            # Векторизация
            embeddings = self.get_embeddings(text) if self.embedder else None
            
            # Формируем результат
            result = {
                "text": text,
                "tokens": tokens,
                "lemmas": lemmas,
                "keywords": keywords,
                "sentiment": sentiment,
                "embeddings": embeddings.tolist() if embeddings is not None else None,
                "processed_at": time.time(),
                "metadata": {
                    "processor": "unified_text_processor",
                    "version": "2.1",
                    "token_count": len(tokens)
                }
            }
            
            # Эмиссия нормализованных метрик обработки текста
            duration = max(0.0, time.time() - start_time)
            try:
                self._emit_metrics([
                    {
                        "name": "text_processor.requests_total",
                        "component": "text_processor",
                        "type": "counter",
                        "value": 1.0,
                        "labels": {"result": "success"}
                    },
                    {
                        "name": "text_processor.process_time_seconds",
                        "component": "text_processor",
                        "type": "summary",
                        "value": float(duration),
                    },
                    {
                        "name": "text_processor.tokens_per_request",
                        "component": "text_processor",
                        "type": "summary",
                        "value": float(len(tokens)),
                    },
                ])
            except Exception:
                pass

            return result
            
        except Exception as e:
            logger.error(f"Ошибка обработки текста: {e}", exc_info=True)
            # Эмиссия метрики ошибки
            try:
                duration = max(0.0, time.time() - start_time)
                self._emit_metrics([
                    {
                        "name": "text_processor.requests_total",
                        "component": "text_processor",
                        "type": "counter",
                        "value": 1.0,
                        "labels": {"result": "error"}
                    },
                    {
                        "name": "text_processor.process_time_seconds",
                        "component": "text_processor",
                        "type": "summary",
                        "value": float(duration),
                    },
                ])
            except Exception:
                pass
            return {
                "text": text,
                "tokens": text.split(),
                "lemmas": text.split(),
                "keywords": [("ошибка", 1.0)],
                "sentiment": {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0},
                "embeddings": None,
                "error": str(e),
                "metadata": {
                    "processor": "unified_text_processor",
                    "version": "2.1",
                    "status": "error",
                    "token_count": len(text.split())
                }
            }
    
    def tokenize(self, text: str) -> List[str]:
        """Токенизирует текст.
        
        Args:
            text: Текст для токенизации
            
        Returns:
            List[str]: Список токенов
        """
        start_time = time.time()
        # Удаляем пунктуацию и приводим к нижнему регистру
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Токенизируем
        tokens = word_tokenize(text)
        
        # Удаляем стоп-слова
        stop_words = set(stopwords.words('english') + stopwords.words('russian'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Метрики токенизации (синхронный путь)
        try:
            duration = max(0.0, time.time() - start_time)
            self._emit_metrics([
                {
                    "name": "text_processor.tokenize_latency_seconds",
                    "component": "text_processor",
                    "type": "summary",
                    "value": float(duration),
                },
                {
                    "name": "text_processor.tokens_per_tokenize",
                    "component": "text_processor",
                    "type": "summary",
                    "value": float(len(tokens)),
                },
            ])
        except Exception:
            pass
        
        return tokens
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Лемматизирует токены.
        
        Args:
            tokens: Список токенов
            
        Returns:
            List[str]: Список лемм
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def extract_keywords(self, text: str, tokens: List[str]) -> List[Tuple[str, float]]:
        """Извлекает ключевые слова из текста.
        
        Args:
            text: Исходный текст
            tokens: Токены текста
            
        Returns:
            List[Tuple[str, float]]: Список ключевых слов с весами
        """
        # Простая реализация - используем частоту
        word_freq = Counter(tokens)
        total = sum(word_freq.values())
        if total == 0:
            return []
        
        keywords = [(word, freq/total) for word, freq in word_freq.most_common(10)]
        
        return keywords
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Анализирует тональность текста.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Dict[str, float]: Результат анализа тональности
        """
        if self.sentiment_analyzer:
            try:
                # Адаптация под вывод pipeline
                result = self.sentiment_analyzer(text)[0]
                score = result['score']
                label = result['label']

                # Конвертируем '5 stars' в sentiment dict
                sentiment = {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
                if '1 star' in label or '2 stars' in label:
                    sentiment['neg'] = score
                    sentiment['compound'] = -score
                elif '3 stars' in label:
                    sentiment['neu'] = score
                elif '4 stars' in label or '5 stars' in label:
                    sentiment['pos'] = score
                    sentiment['compound'] = score
                return sentiment
            except Exception as e:
                logger.warning(f"Ошибка анализа тональности: {e}")
        
        # Резервная реализация
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Получает эмбеддинги для текста.
        
        Args:
            text: Текст для получения эмбеддингов
            
        Returns:
            np.ndarray: Вектор эмбеддингов
        """
        if self.embedder:
            try:
                return self.embedder.encode([text])[0]
            except Exception as e:
                logger.warning(f"Ошибка получения эмбеддингов: {e}")
        
        # Возвращаем нулевой вектор как заглушку
        return np.zeros(384)  # Стандартный размер для all-MiniLM-L6-v2
    
    def tokenize_async(self, text: str) -> Dict[str, Any]:
        """Асинхронная токенизация текста.
        
        Args:
            text: Текст для токенизации
            
        Returns:
            Dict[str, Any]: Результат токенизации
        """
        if not text or not isinstance(text, str):
            return {
                "tokens": [],
                "token_count": 0,
                "original_text": "",
                "error": "empty_input",
                "metadata": {
                    "processor": "unified_text_processor",
                    "version": "2.1",
                    "status": "empty_input"
                }
            }
        
        if self.use_async and self.executor:
            # Выполняем токенизацию в фоновом потоке
            future = self.executor.submit(self._tokenize_text, text)
            try:
                return future.result(timeout=5.0)
            except Exception as e:
                logger.warning(f"Таймаут асинхронной токенизации: {e}")
                # Падаем обратно на синхронную обработку
                return self._tokenize_text(text)
        else:
            # Синхронная обработка
            return self._tokenize_text(text)
    
    def _tokenize_text(self, text: str) -> Dict[str, Any]:
        """Внутренний метод токенизации текста.
        
        Args:
            text: Текст для токенизации
            
        Returns:
            Dict[str, Any]: Результат токенизации
        """
        start_time = time.time()
        try:
            tokens = self.tokenize(text)
            result = {
                "tokens": tokens,
                "token_count": len(tokens),
                "original_text": text[:100] + "..." if len(text) > 100 else text,
                "processed_at": time.time(),
                "metadata": {
                    "processor": "unified_text_processor",
                    "version": "2.1",
                    "token_count": len(tokens)
                }
            }
            # Метрики токенизации (асинхронный путь)
            try:
                duration = max(0.0, time.time() - start_time)
                self._emit_metrics([
                    {
                        "name": "text_processor.tokenize_requests_total",
                        "component": "text_processor",
                        "type": "counter",
                        "value": 1.0,
                        "labels": {"result": "success"}
                    },
                    {
                        "name": "text_processor.tokenize_latency_seconds",
                        "component": "text_processor",
                        "type": "summary",
                        "value": float(duration),
                    },
                    {
                        "name": "text_processor.tokens_per_tokenize",
                        "component": "text_processor",
                        "type": "summary",
                        "value": float(len(tokens)),
                    },
                ])
            except Exception:
                pass
            return result
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            try:
                duration = max(0.0, time.time() - start_time)
                self._emit_metrics([
                    {
                        "name": "text_processor.tokenize_requests_total",
                        "component": "text_processor",
                        "type": "counter",
                        "value": 1.0,
                        "labels": {"result": "error"}
                    },
                    {
                        "name": "text_processor.tokenize_latency_seconds",
                        "component": "text_processor",
                        "type": "summary",
                        "value": float(duration),
                    },
                ])
            except Exception:
                pass
            return {
                "tokens": [],
                "token_count": 0,
                "original_text": text[:100] + "..." if len(text) > 100 else text,
                "processed_at": time.time(),
                "error": str(e),
                "metadata": {
                    "processor": "unified_text_processor",
                    "version": "2.1",
                    "status": "error",
                    "token_count": 0
                }
            }
    
    def is_ready(self) -> bool:
        """Проверяет готовность текстового процессора к работе."""
        return (self.nlp is not None or 
                self.embedder is not None or 
                self.sentiment_analyzer is not None)
    
    def shutdown(self):
        """Завершение работы процессора с сохранением кэша"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # Сохраняем кэш
        if self.hybrid_cache and hasattr(self.hybrid_cache, 'save'):
            self.hybrid_cache.save()
        
        logger.info("UnifiedTextProcessor завершил работу")
    
    def _emit_metrics(self, metrics: List[Dict[str, Any]]):
        """Безопасно отправляет метрики: через событийную систему ('metrics') и напрямую emit_metrics."""
        try:
            if getattr(self, "brain", None):
                # Сначала пробуем событийную шину для унифицированного транспорта
                try:
                    if hasattr(self.brain, "events") and self.brain.events:
                        self.brain.events.trigger('metrics', metrics)
                except Exception:
                    pass
                # Обратная совместимость: прямой вызов emit_metrics
                try:
                    if hasattr(self.brain, "emit_metrics"):
                        self.brain.emit_metrics(metrics)
                except Exception:
                    pass
        except Exception:
            pass
    
    def __del__(self):
        """Деструктор для освобождения ресурсов"""
        self.shutdown()

# Алиас для обратной совместимости
TextProcessor = UnifiedTextProcessor