"""Объединенный текстовый процессор с асинхронной токенизацией для CogniFlex
Обновленная версия с интеграцией гибридного кэша и правильной архитектурой
"""

import os
import time
import logging
import multiprocessing
import threading
import numpy as np
import re
import hashlib
import json
import nltk
from collections import defaultdict, Counter, OrderedDict
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Type, TypeVar
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import psutil
import spacy
import torch
from nltk.stem import WordNetLemmatizer

# Import config
try:
    from cogniflex.config import is_embedding_loading_disabled, DISABLE_ALL_MODELS
except ImportError:
    # Fallback if config doesn't exist
    def is_embedding_loading_disabled():
        return False
    DISABLE_ALL_MODELS = False

# Импортируем базовый класс компонента
try:
    from cogniflex.core.base_component import BaseComponent
except ImportError:
    # Для обратной совместимости
    BaseComponent = object
    logger = logging.getLogger("cogniflex.unified_text_processor")
else:
    logger = logging.getLogger("cogniflex.unified_text_processor")

# Check if embeddings are disabled
if not is_embedding_loading_disabled():
    try:
        # Optional dependency; allow absence in test environments
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:  # pragma: no cover - optional dependency may be absent
        SentenceTransformer = None  # type: ignore
else:
    SentenceTransformer = None
    logger.info("SentenceTransformer disabled by configuration")

# Импортируем singleton для кэширования sentence-transformers моделей
try:
    from cogniflex.mlearning.sentence_transformers_cache import get_sentence_transformer
except ImportError:
    get_sentence_transformer = None

from transformers import pipeline

# Попытка загрузить необходимые ресурсы NLTK (offline-safe)
NLTK_AVAILABLE = True
try:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"NLTK resources not fully available, some features may be limited: {e}")
    NLTK_AVAILABLE = False

class UnifiedTextProcessor(BaseComponent):
    """
    Объединенный текстовый процессор с асинхронной токенизацией для CogniFlex
    
    Этот компонент предоставляет единый интерфейс для обработки текста, включая токенизацию,
    лемматизацию, извлечение эмбеддингов и другие операции над текстом.
    """
    
    def __init__(self, brain: Any = None, config: Optional[Dict[str, Any]] = None):
        """Инициализирует объединенный текстовый процессор.

        Args:
            brain: Ссылка на ядро CogniFlex, которое предоставляет зависимости.
            config: Конфигурация процессора. Может включать:
                - use_gpu (bool): Использовать GPU если доступен.
                - model_name (str): Название модели эмбеддингов.
                - use_async (bool): Использовать асинхронную обработку.
                - max_workers (int): Максимальное количество рабочих потоков.
        """
        
        # Вызываем родительский конструктор
        super().__init__(name="unified_text_processor", brain=brain)
        
        # Устанавливаем конфигурацию по умолчанию
        self.config = config or {}
        self.use_gpu = self.config.get('use_gpu', True)
        self.model_name = self.config.get('model_name', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.use_async = self.config.get('use_async', True)
        self.max_workers = self.config.get('max_workers', 4)
        
        # Пути для моделей
        self.models_base_path = self.config.get('models_base_path', './models/embeddings')
        
        # Инициализируем кэш
        self.token_cache = {}
        self.cache_lock = threading.RLock()
        
        # Инициализируем токенизатор по умолчанию
        self.tokenizer = None
        
        # Инициализируем анализаторы
        self.sentiment_analyzer = None
        
        # Инициализируем модели эмбеддингов
        self.embedder = None
        self.embedding_model = None
        
        # Инициализируем модели NLP
        self._init_nlp_models()
        
        logger.debug(f"UnifiedTextProcessor инициализирован с {self.max_workers} воркерами.")
    
        
    def _setup_component(self) -> None:
        """Настраивает компонент после проверки зависимостей."""
        # Получаем зависимости из CoreBrain
        if hasattr(self.brain, 'components') and self.brain.components:
            self.tokenizer = self.brain.components.get('tokenizer')
            self.hybrid_cache = self.brain.components.get('hybrid_cache')
        else:
            # Fallback если components еще не инициализирован
            self.tokenizer = getattr(self.brain, 'tokenizer', None)
            self.hybrid_cache = getattr(self.brain, 'hybrid_cache', None)
        
        # Инициализируем исполнитель для асинхронных операций
        if self.use_async:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
            
        # Отправляем уведомление о готовности
        if hasattr(self.brain, 'events'):
            self.brain.events.trigger('text_processor_ready', self)
            logger.info("Уведомление о готовности text_processor отправлено.")
    
    def _init_nlp_models(self):
        """Инициализирует NLP модели и компоненты."""
        try:
            # Инициализируем лемматизатор NLTK
            self.lemmatizer = WordNetLemmatizer()
            
            # Создаем локальный токенизатор
            try:
                from cogniflex.nlp.text_processor import TextProcessor
                self.tokenizer = TextProcessor(model_name="qwen3.5-2b")
                logger.info("Локальный токенизатор Qwen3.5-2B создан успешно")
            except Exception as e:
                logger.warning(f"Не удалось создать локальный токенизатор: {e}")
                self.tokenizer = None
            
            # Пытаемся загрузить языковую модель spaCy
            try:
                # Сначала пробуем загрузить русскую модель
                self.nlp = spacy.load("ru_core_news_sm")
            except OSError:
                try:
                    # Если не получается, загружаем английскую модель
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # Если и это не получается, загружаем базовую модель
                    self.nlp = spacy.blank("ru")
                    logger.warning("Не удалось загрузить языковые модели spaCy, используем базовую модель")
            
            # Загрузка моделей для эмбеддингов и семантического поиска
            self._load_embedding_models()
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации NLP моделей: {e}", exc_info=True)
            raise
            
    def _load_embedding_models(self):
        """Загружает модели для эмбеддингов и семантического поиска."""
        # Check if embeddings are disabled
        if is_embedding_loading_disabled():
            logger.info("Загрузка эмбеддингов отключена конфигурацией")
            self.embedding_model = None
            self.embedder = None
            return
            
        try:
            device = "cuda" if self.use_gpu else "cpu"
            
            # Используем singleton для загрузки sentence-transformers модели
            if get_sentence_transformer is not None:
                logger.info(f"Загружаем модель эмбеддингов через singleton (устройство: {device})")
                self.embedding_model = get_sentence_transformer(self.model_name, device=device)
                self.embedder = self.embedding_model
                if self.embedding_model is not None:
                    logger.info("Модель эмбеддингов загружена успешно (через singleton)")
                else:
                    logger.warning("Не удалось загрузить модель эмбеддингов через singleton")
            elif SentenceTransformer is not None:
                # Fallback - прямая загрузка без singleton
                logger.info(f"Загружаем модель эмбеддингов напрямую: {self.model_name}")
                try:
                    self.embedding_model = SentenceTransformer(
                        self.model_name,
                        device=device
                    )
                    self.embedder = self.embedding_model
                    logger.info("Модель эмбеддингов загружена успешно")
                except Exception as e:
                    logger.warning(f"Ошибка загрузки модели эмбеддингов: {e}")
                    self.embedding_model = None
                    self.embedder = None
            else:
                logger.warning("SentenceTransformer не доступен. Функции эмбеддингов отключены.")
                self.embedding_model = None
                self.embedder = None
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели эмбеддингов: {e}", exc_info=True)
            self.embedding_model = None
            self.embedder = None
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
            
            # Определяем размерность эмбеддинга
            try:
                if self.embedder and hasattr(self.embedder, 'get_sentence_embedding_dimension'):
                    self.embedding_dim = int(self.embedder.get_sentence_embedding_dimension())
                else:
                    # разумный дефолт для MiniLM/MPNet семейств
                    self.embedding_dim = 384
            except Exception:
                self.embedding_dim = 384
            
            # Инициализация анализаторов
            self.sentiment_analyzer = None
            try:
                sentiment_model_path = os.path.join(self.models_base_path, 'bert-base-multilingual-uncased-sentiment')
                if os.path.exists(sentiment_model_path):
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis", 
                        model=sentiment_model_path,
                        local_files_only=True  # Только локальные файлы
                    )
                    logger.debug("Sentiment analyzer инициализирован локально")
                else:
                    logger.warning(f"Локальная модель sentiment analyzer не найдена: {sentiment_model_path}")
            except Exception as e:
                logger.warning(f"Sentiment analyzer недоступен: {e}")
            
            self.lemmatizer = WordNetLemmatizer()
            
            # Инициализация пула процессов для асинхронной обработки
            if self.use_async:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            else:
                self.executor = None
            
            logger.info("NLP-модели инициализированы")
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
            # Предварительная обработка текста
            processed_text = text.lower()

            # Токенизация
            tokens = self.tokenize(processed_text)
            
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
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Токенизирует текст, используя центральный токенизатор из CoreBrain.

        Args:
            text: Текст для токенизации
            **kwargs: Дополнительные параметры токенизации

        Returns:
            Список токенов
        """
        try:
            if self.tokenizer and hasattr(self.tokenizer, 'tokenize'):
                return self.tokenizer.tokenize(text, **kwargs)
            elif self.tokenizer and hasattr(self.tokenizer, 'encode'):
                # Для transformers токенизаторов
                tokens = self.tokenizer.encode(text, **kwargs)
                if isinstance(tokens, list):
                    return [str(t) for t in tokens]
                return tokens
            else:
                # Fallback на простой split
                return text.split()
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            return text.split()
    
    def encode(self, text: str, **kwargs) -> Dict[str, Any]:
        """Кодирует текст в формат, совместимый с ML моделями.
        
        Args:
            text: Текст для кодирования
            **kwargs: Дополнительные параметры

        Returns:
            Словарь с input_ids и другими полями
        """
        try:
            if self.tokenizer and hasattr(self.tokenizer, 'encode'):
                # Для transformers токенизаторов
                encoded = self.tokenizer.encode(text, **kwargs)
                
                if hasattr(encoded, 'input_ids'):
                    return {
                        'input_ids': encoded.input_ids,
                        'attention_mask': encoded.attention_mask if hasattr(encoded, 'attention_mask') else None
                    }
                elif isinstance(encoded, list):
                    return {
                        'input_ids': encoded,
                        'attention_mask': [1] * len(encoded)
                    }
                else:
                    return {
                        'input_ids': encoded,
                        'attention_mask': None
                    }
            else:
                # Fallback - простая токенизация
                tokens = self.tokenize(text)
                return {
                    'input_ids': tokens,
                    'attention_mask': [1] * len(tokens)
                }
        except Exception as e:
            logger.error(f"Ошибка кодирования текста: {e}")
            return {
                'input_ids': text.split(),
                'attention_mask': [1] * len(text.split())
            }

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

    def classify_domain(self, concept: str, description: str = "") -> str:
        """
        Классифицирует домен концепта по простым эвристикам.
        Возвращает строку домена, либо "general" при отсутствии уверенности.

        Args:
            concept: Название концепта
            description: Описание/контекст (необязательно)
        """
        try:
            text = f"{concept} {description}".lower()
            # Простые эвристики по ключевым словам
            domains = {
                "science": ["физик", "хим", "биолог", "квант", "наук"],
                "technology": ["алгоритм", "программ", "сеть", "данн", "машинн", "ml", "ai"],
                "mathematics": ["теорем", "функц", "матриц", "вектор", "интеграл", "дифференци"],
                "medicine": ["лечен", "болезн", "симптом", "диагноз", "терап"],
                "economics": ["рынок", "капитал", "инфляц", "эконом", "инвест"],
                "metaknowledge": ["определен", "контекст", "мета", "обобщен"],
            }
            best_domain = None
            best_hits = 0
            for dom, kws in domains.items():
                hits = sum(1 for kw in kws if kw in text)
                if hits > best_hits:
                    best_hits = hits
                    best_domain = dom
            if best_domain and best_hits > 0:
                return best_domain

            # Если доступен embedder, можно реализовать продвинутую схему со словарем якорей
            # Здесь оставим фолбэк
            return "general"
        except Exception:
            return "general"
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Получает эмбеддинги для текста(ов).
        
        Args:
            texts: Строка или список строк
            
        Returns:
            np.ndarray: Если на входе строка — массив формы (D,), если список — (N, D)
        """
        # Нормализуем вход
        if texts is None:
            return np.zeros((0, getattr(self, 'embedding_dim', 384)), dtype=np.float32)
        if isinstance(texts, str):
            batch = [texts]
            single_input = True
        elif isinstance(texts, list):
            batch = [t for t in texts if isinstance(t, str) and t.strip()]
            single_input = False
        else:
            batch = [str(texts)]
            single_input = True
        
        if len(batch) == 0:
            return np.zeros((0, getattr(self, 'embedding_dim', 384)), dtype=np.float32) if not single_input else np.zeros(getattr(self, 'embedding_dim', 384), dtype=np.float32)
        
        # Основной путь через sentence-transformers
        if self.embedding_model:  # Исправляем: embedding_model вместо embedder
            try:
                embs = self.embedding_model.encode(batch)
                embs = np.array(embs)
                if single_input:
                    # Возвращаем вектор (D,)
                    return embs[0]
                return embs
            except Exception as e:
                logger.warning(f"Ошибка получения эмбеддингов: {e}")
                # Падение в резервный путь ниже
        
        # Резервный путь: нулевые векторы нужной размерности
        D = getattr(self, 'embedding_dim', 384)
        if single_input:
            return np.zeros(D, dtype=np.float32)
        else:
            return np.zeros((len(batch), D), dtype=np.float32)
    
    
    def is_ready(self) -> bool:
        """Проверяет готовность текстового процессора к работе."""
        return (self.nlp is not None or 
                self.embedder is not None or 
                self.sentiment_analyzer is not None)
    
    def cleanup(self):
        """Очищает ресурсы, используемые процессором."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=True)
                
            # Освобождаем ресурсы моделей
            if hasattr(self, 'embedding_model'):
                del self.embedding_model
                
            if hasattr(self, 'nlp'):
                del self.nlp
                
            # Вызываем cleanup родительского класса
            super().cleanup()
            
            logger.info("Ресурсы UnifiedTextProcessor успешно освобождены")
            
        except Exception as e:
            logger.error(f"Ошибка при очистке ресурсов: {e}", exc_info=True)
            raise
        except Exception:
            pass

        # Сохраняем кэш
        try:
            if self.hybrid_cache and hasattr(self.hybrid_cache, 'save'):
                self.hybrid_cache.save()
        except Exception:
            pass

        # Безопасное логирование (во время завершения интерпретатора globals могут быть None)
        try:
            _lg = logger if logger else logging.getLogger("cogniflex.unified_text_processor")
            if _lg:
                _lg.info("UnifiedTextProcessor завершил работу")
        except Exception:
            pass
    
    def _emit_metrics(self, metrics: List[Dict[str, Any]]):
        """Безопасно отправляет метрики: предпочитает событийную систему; fallback на прямой вызов."""
        try:
            brain = getattr(self, "brain", None)
            if not brain:
                return
            # Сначала пробуем событийную шину
            try:
                if hasattr(brain, "events") and brain.events:
                    brain.events.trigger('metrics', metrics)
                    return  # избегаем двойной эмиссии
            except Exception:
                pass
            # Fallback: прямой вызов
            try:
                if hasattr(brain, "emit_metrics"):
                    brain.emit_metrics(metrics)
            except Exception:
                pass
        except Exception:
            pass
    
    def __del__(self):
        """Деструктор для освобождения ресурсов"""
        try:
            self.shutdown()
        except Exception:
            # Нельзя бросать исключения из деструктора
            pass

# Алиас для обратной совместимости
TextProcessor = UnifiedTextProcessor
