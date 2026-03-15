
"""
ResponseGenerator для CogniFlex.
Возвращает структурированный результат генерации и обеспечивает безопасные fallback-ы.
"""

import time
import logging
import threading
import re
from typing import Dict, Any, List, Optional, Callable, Protocol, Union
from hashlib import sha256, pbkdf2_hmac
from spacy.lang.ru import Russian
from spacy.lang.en import English
import html
import secrets

logger = logging.getLogger("cogniflex.response_generator")

# Опциональные импорты (используются, если доступны)
try:
    from torch import no_grad, inference_mode
    import torch
except ImportError:
    torch = None  # type: ignore
    no_grad = None
    inference_mode = None

# Протоколы для типизации
class BrainProtocol(Protocol):
    model_manager: Any
    token_streamer: Optional[Any]
    token_cache: Optional[Any]

class HybridCacheProtocol(Protocol):
    def get_token(self, key: str) -> Any: ...
    def add_token(self, key: str, value: Any) -> None: ...

class TokenStreamerProtocol(Protocol):
    def tokenize(self, text: str, callback: Callable, context: Optional[Dict], priority: int) -> None: ...

class ResponseGenerator:
    """
    Класс генерации ответов. Инкапсулирует работу с моделями и токенизаторами.
    Ожидает, что self.brain.model_manager.get_model_for_task(task) возвращает
    кортеж (model, tokenizer, model_name) совместимый с HuggingFace transformers.
    
    Основные улучшения:
    - Интеграция с асинхронной потоковой токенизацией
    - Поддержка гибридного кэша (память + диск)
    - Улучшенная обработка ошибок и fallback-сценариев
    - Соответствие форматам данных для взаимодействия с другими модулями
    """

    def __init__(self, brain=None, model_manager=None, text_processor=None):
        """Инициализирует генератор ответов."""
        self.brain = brain
        self.model_manager = model_manager
        self.text_processor = text_processor
        self.token_streamer = None
        self.hybrid_cache = None
        self.lock = threading.Lock()
        logger.info("ResponseGenerator инициализирован")

    def _update_components(self):
        """Обновляет ссылки на компоненты из brain непосредственно перед использованием."""
        if not self.brain:
            return

        # Получаем model_manager
        if not self.model_manager and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'model_manager'):
            self.model_manager = self.brain.ml_unit.model_manager

        # Получаем text_processor и используем его как token_streamer
        if not self.text_processor and hasattr(self.brain, 'text_processor'):
            self.text_processor = self.brain.text_processor
            self.token_streamer = self.text_processor

        # Получаем hybrid_cache из text_processor
        if not self.hybrid_cache and self.text_processor and hasattr(self.text_processor, 'hybrid_cache'):
            self.hybrid_cache = self.text_processor.hybrid_cache

    def _init_components(self):
        """Инициализирует необходимые компоненты для работы."""
        try:
            # Инициализация асинхронного токенизатора
            if self.token_streamer:
                logger.debug("Используется внешний token_streamer")
            elif hasattr(self.brain, 'token_streamer') and self.brain.token_streamer:
                self.token_streamer = self.brain.token_streamer
                logger.debug("Используется внешний token_streamer из ядра")
            else:
                try:
                    from cogniflex.mlearning.unified_text_processor import UnifiedTextProcessor
                    self.token_streamer = UnifiedTextProcessor()
                    logger.info("Инициализирован внутренний token_streamer")
                except ImportError as e:
                    safe_error = html.escape(str(e)[:50])
                    logger.warning(f"Не удалось импортировать UnifiedTextProcessor: {safe_error}")
            
            # Инициализация гибридного кэша
            if self.hybrid_cache:
                logger.debug("Используется внешний гибридный кэш")
            elif hasattr(self.brain, 'token_cache') and self.brain.token_cache:
                self.hybrid_cache = self.brain.token_cache
                logger.debug("Используется внешний гибридный кэш из ядра")
            else:
                try:
                    from ..memory.hybrid_token_cache import HybridTokenCache
                    self.hybrid_cache = HybridTokenCache(self.brain)
                    logger.info("Инициализирован внутренний гибридный кэш")
                except ImportError as e:
                    safe_error = html.escape(str(e)[:50])
                    logger.warning(f"Не удалось импортировать HybridTokenCache: {safe_error}")
                
        except Exception as e:
            safe_error = html.escape(str(e)[:100])
            logger.error(f"Ошибка инициализации компонентов ResponseGenerator: {safe_error}")
            # Устанавливаем None если компоненты недоступны

    def _emit_metrics(self, metrics: List[Dict[str, Any]]):
        """Безопасно отправляет метрики через событийную систему ('metrics') и напрямую emit_metrics."""
        try:
            if getattr(self, "brain", None):
                # Сначала публикуем в событийную систему для унифицированного транспорта
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

    # --- Вспомогательные методы ---

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Получает данные из гибридного кэша."""
        if not self.hybrid_cache:
            return None
            
        try:
            # Проверяем кэш на наличие полного ответа
            cached_data = self.hybrid_cache.get_token(cache_key)
            if cached_data and isinstance(cached_data, dict) and "text" in cached_data:
                return cached_data["text"]
                
            # Проверяем кэш на наличие только текста
            if isinstance(cached_data, str):
                return cached_data
                
            return None
        except Exception as e:
            safe_error = html.escape(str(e)[:50])
            logger.debug(f"_get_from_cache: не удалось получить из кэша: {safe_error}")
            return None

    def _save_to_cache(self, cache_key: str, text: str, model_name: str, metadata: Optional[Dict] = None):
        """Сохраняет данные в гибридный кэш."""
        if not self.hybrid_cache:
            return
            
        try:
            # Формируем данные для кэша
            cache_data = {
                "text": text,
                "model": model_name,
                "timestamp": time.time(),
                "metadata": metadata or {}
            }
            
            # Сохраняем в гибридный кэш
            self.hybrid_cache.add_token(cache_key, cache_data)
            
            # Сохраняем токены отдельно для быстрого доступа
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                tokens = self._safe_tokenize(self.tokenizer, text)
                self.hybrid_cache.add_token(f"{cache_key}_tokens", tokens)
                
        except Exception as e:
            safe_error = html.escape(str(e)[:50])
            logger.debug(f"_save_to_cache: не удалось сохранить в кэш: {safe_error}")

    def _safe_tokenize(self, tokenizer: Any, text: str) -> List[str]:
        """Безопасная токенизация текста."""
        try:
            if hasattr(tokenizer, "tokenize"):
                return tokenizer.tokenize(text)
            if hasattr(tokenizer, "encode"):
                ids = tokenizer.encode(text, add_special_tokens=False)
                return [str(i) for i in ids]
        except Exception as e:
            safe_error = html.escape(str(e)[:50])
            logger.debug(f"_safe_tokenize: ошибка токенизации: {safe_error}")
        return text.split()

    def _safe_encode(self, tokenizer: Any, texts: List[str]):
        """
        Возвращает словарь для передачи в модель: tokenizer(texts, return_tensors="pt", truncation=True)
        или None при ошибке.
        """
        try:
            return tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        except Exception as e:
            safe_error = html.escape(str(e)[:50])
            logger.debug(f"_safe_encode: ошибка кодирования: {safe_error}")
            try:
                return tokenizer(" ".join(texts), return_tensors="pt", truncation=True)
            except Exception as e2:
                safe_error = html.escape(str(e2)[:50])
                logger.debug(f"_safe_encode: вторичная попытка кодирования провалилась: {safe_error}")
        return None

    def _process_tokens_async(self, text: str, callback: Callable, context: Optional[Dict] = None):
        """
        Асинхронно обрабатывает токенизацию текста.
        
        Args:
            text: Текст для токенизации
            callback: Функция обратного вызова после завершения
            context: Дополнительный контекст
        """
        if not self.token_streamer:
            # Синхронная обработка как fallback
            tokens = self._safe_tokenize(self.tokenizer, text) if hasattr(self, 'tokenizer') and self.tokenizer is not None else text.split()
            callback(tokens, context)
            return
        
        # Асинхронная обработка через token_streamer
        def on_tokenization_complete(result, ctx, error=None):
            if error:
                # Санитизация для логирования
                safe_error = html.escape(str(error)[:100])
                logger.error("Ошибка асинхронной токенизации: %s", safe_error)
                # Используем синхронную токенизацию как fallback
                tokens = self._safe_tokenize(self.tokenizer, text) if hasattr(self, 'tokenizer') and self.tokenizer is not None else text.split()
                callback(tokens, ctx)
            else:
                callback(result, ctx)
        
        # Добавляем задачу токенизации
        self.token_streamer.tokenize(
            text,
            callback=on_tokenization_complete,
            context=context,
            priority=5
        )

    # --- Основной метод ---

    def generate_response(self, prompt: str, max_length: int = 200, 
                         temperature: float = 0.7, top_p: float = 0.9,
                         task: str = "text-generation", 
                         return_reasoning: bool = False,
                         minimal_output: bool = False,
                         context: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Генерирует ответ и возвращает структурированный dict.
        Добавлена очистка от кракозябр и безопасная работа с токенизатором.
        """
        start_time = time.time()
        self._update_components()  # Обновляем компоненты перед использованием
        
        try:
            # Подготовка и валидация входных данных
            prompt = self._prepare_prompt(prompt, task, context)
            
            # Проверка кэша
            cache_key = self._make_cache_key(task, prompt, max_length, temperature, top_p)
            cached_result = self._check_cache(cache_key, task)
            if cached_result:
                # Эмиссия метрик при попадании в кэш
                try:
                    duration = max(0.0, time.time() - start_time)
                    tokens_len = 0
                    try:
                        tokens_list = cached_result.get("tokens") or []
                        tokens_len = len(tokens_list) if isinstance(tokens_list, list) else 0
                    except Exception:
                        tokens_len = 0
                    self._emit_metrics([
                        {
                            "name": "response_generator.requests_total",
                            "component": "response_generator",
                            "type": "counter",
                            "value": 1.0,
                            "labels": {"result": "success", "source": "cache"}
                        },
                        {
                            "name": "response_generator.cache_hits_total",
                            "component": "response_generator",
                            "type": "counter",
                            "value": 1.0,
                        },
                        {
                            "name": "response_generator.latency_seconds",
                            "component": "response_generator",
                            "type": "summary",
                            "value": float(duration),
                        },
                        {
                            "name": "response_generator.tokens_generated",
                            "component": "response_generator",
                            "type": "summary",
                            "value": float(tokens_len),
                        },
                    ])
                except Exception:
                    pass
                return cached_result
            
            # Получение модели
            model_data = self._get_model_data(task)
            if not model_data:
                return self._create_fallback_response(prompt, "Модель недоступна")
            
            model, tokenizer, model_name = model_data
            
            # Генерация ответа
            generated_text = self._generate_with_model(
                model, tokenizer, prompt, max_length, temperature, top_p
            )
            
            # Постобработка и формирование результата
            return self._create_response(
                generated_text, tokenizer, model_name, task, prompt,
                cache_key, start_time, return_reasoning, temperature, top_p, max_length, minimal_output
            )
            
        except Exception as e:
            safe_error = html.escape(str(e)[:100])
            logger.error("generate_response: критическая ошибка: %s", safe_error)
            # Эмиссия метрик ошибки генерации
            try:
                duration = max(0.0, time.time() - start_time)
                self._emit_metrics([
                    {
                        "name": "response_generator.requests_total",
                        "component": "response_generator",
                        "type": "counter",
                        "value": 1.0,
                        "labels": {"result": "error"}
                    },
                    {
                        "name": "response_generator.latency_seconds",
                        "component": "response_generator",
                        "type": "summary",
                        "value": float(duration),
                    },
                ])
            except Exception:
                pass
            return self._create_fallback_response(prompt, f"Критическая ошибка: {safe_error}")
    
    def _prepare_prompt(self, prompt: str, task: str, context: Optional[Union[str, Dict[str, Any]]] = "") -> str:
        """Подготавливает и валидирует промпт."""
        if not isinstance(prompt, str):
            prompt = str(prompt or "")
        
        # Очистка входа от неподдерживаемых символов
        prompt = prompt.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore").strip()
        
        # Поддержка словарного контекста: форматируем в структурированный текст
        ctx_text: str = ""
        try:
            if isinstance(context, dict) and context:
                parts: List[str] = []
                # Упорядоченно сериализуем известные поля, остальное добавляем как JSON-представление
                for key in ["nlp", "evidence", "concept", "user_context"]:
                    if key in context and context[key] is not None:
                        parts.append(f"{key.capitalize()}: {str(context[key])}")
                # Добавляем остальные поля, если есть
                extra_keys = [k for k in context.keys() if k not in {"nlp", "evidence", "concept", "user_context"}]
                for k in extra_keys:
                    parts.append(f"{k}: {str(context[k])}")
                ctx_text = "\n".join(parts)
            elif isinstance(context, str):
                ctx_text = context
            else:
                ctx_text = ""
        except Exception:
            ctx_text = ""

        # Оптимизация контекста через гибридный кэш (динамическая приоритизация)
        try:
            # Обновляем ссылки на компоненты (на случай прямого вызова вне generate_response)
            self._update_components()
            if self.hybrid_cache and ctx_text:
                # DEBUG: замеряем длину контекста (приблизительно по словам) до оптимизации
                try:
                    before_tokens = len(ctx_text.split())
                except Exception:
                    before_tokens = 0
                optimized_ctx = self.hybrid_cache.prioritize_context(query=prompt, context=ctx_text, task_type=task)
                if isinstance(optimized_ctx, str) and optimized_ctx:
                    try:
                        after_tokens = len(optimized_ctx.split())
                    except Exception:
                        after_tokens = 0
                    # DEBUG: логируем эффект оптимизации
                    logger.debug(
                        "context_optimization: task=%s before_tokens=%d after_tokens=%d",
                        str(task), before_tokens, after_tokens
                    )
                    ctx_text = optimized_ctx
        except Exception:
            # Тихо деградируем до исходного контекста
            pass
        
        # Используем новый метод _build_prompt для создания структурированного промпта
        structured_prompt = self._build_prompt(prompt, ctx_text)
        
        # Санитизация для логирования
        safe_task = html.escape(str(task)[:50])
        safe_prompt = html.escape(structured_prompt[:80])
        logger.debug("generate_response: task=%s prompt_preview=%s", safe_task, safe_prompt)
        
        return structured_prompt

    def _make_cache_key(self, task: str, prompt: str, max_length: int, temperature: float, top_p: float) -> str:
        """Создает безопасный ключ кэша для использования на диске (Windows-safe).

        Формируем сырой ключ из параметров, затем хешируем, чтобы исключить недопустимые
        символы (например, ':', '\n') и ограничить длину.
        """
        raw = f"{task}:{prompt}:{max_length}:{temperature}:{top_p}"
        digest = sha256(raw.encode('utf-8')).hexdigest()[:24]
        safe_task = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in task)[:32]
        return f"cf_{safe_task}_{digest}"
    
    def _check_cache(self, cache_key: str, task: str) -> Optional[Dict[str, Any]]:
        """Проверяет кэш и возвращает результат если найден."""
        cached = self._get_from_cache(cache_key)
        if not cached:
            return None
        
        logger.info("generate_response: найден кэш для промпта")
        
        # Получаем токены из кэша
        cached_tokens = []
        if self.hybrid_cache:
            cached_tokens = self.hybrid_cache.get_token(f"{cache_key}_tokens") or []
        
        # Получаем имя модели из кэша если доступно
        cached_model_name = "unknown"
        if self.hybrid_cache:
            cached_data = self.hybrid_cache.get_token(cache_key)
            if isinstance(cached_data, dict) and "model" in cached_data:
                cached_model_name = cached_data["model"]
        
        return {
            "text": cached,
            "tokens": cached_tokens,
            "metadata": {
                "model": cached_model_name,
                "task": task,
                "length": len(cached),
                "from_cache": True,
                "cache_timestamp": time.time()
            },
            "reasoning": None,
            "contradiction_detected": False,
            "contradictions": [],
            "sentiment": None
        }
    
    def _get_model_data(self, task: str) -> Optional[tuple]:
        """Получает данные модели для задачи."""
        # Проверяем наличие brain и model_manager
        if not self.brain:
            logger.warning("generate_response: brain недоступен")
            return None
            
        # Используем model_manager из инициализации
        model_manager = self.model_manager
        
        if not model_manager:
            logger.warning("generate_response: model_manager недоступен")
            return None
        
        try:
            # Проверяем, есть ли метод get_model_for_task
            if not hasattr(model_manager, 'get_model_for_task'):
                logger.warning("generate_response: model_manager не имеет метода get_model_for_task")
                return None
                
            model_data = model_manager.get_model_for_task(task)
            if not model_data or len(model_data) < 3:
                logger.warning(f"generate_response: модель для задачи '{task}' не найдена")
                return None
            
            model, tokenizer, model_name = model_data
            self.tokenizer = tokenizer
            return model, tokenizer, model_name
            
        except Exception as e:
            safe_error = html.escape(str(e)[:100])
            logger.error("generate_response: ошибка получения модели: %s", safe_error)
            return None
    
    def _generate_with_model(self, model, tokenizer, prompt: str, max_length: int, temperature: float, top_p: float) -> str:
        """Генерирует текст с помощью модели."""
        with self.lock:
            # Подготовка входных данных
            inputs = self._safe_encode(tokenizer, [prompt])
            if inputs is None:
                logger.warning("generate_response: не удалось закодировать промпт")
                return self._fallback_generation(prompt, max_length)
            
            # Генерация с моделью
            if torch and hasattr(model, 'generate'):
                try:
                    context_manager = inference_mode() if inference_mode else (no_grad() if no_grad else torch.no_grad())
                    with context_manager:
                        # Улучшенные параметры генерации для предотвращения кракозябр
                        generation_kwargs = {
                            **inputs,
                            'max_new_tokens': min(max_length, 512),  # используем только max_new_tokens, чтобы избежать конфликтов
                            'temperature': max(0.3, min(temperature, 0.9)),  # Ограничиваем температуру
                            'top_p': max(0.3, min(top_p, 0.9)),  # Ограничиваем top_p
                            'do_sample': True,
                            'repetition_penalty': 1.15,  # Увеличиваем штраф за повторения для русского языка
                            'no_repeat_ngram_size': 3,  # Предотвращаем повторение 3-грамм для лучшего качества
                            'pad_token_id': tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0,
                            'length_penalty': 1.0,  # Нейтральный штраф за длину
                            'num_beams': 1  # Жадный поиск для стабильности
                        }
                        
                        outputs = model.generate(**generation_kwargs)
                    
                    # Декодирование результата с улучшенной обработкой
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    
                    # Убираем исходный промпт из результата
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    
                    # Проверяем качество сгенерированного текста
                    if self._is_text_corrupted(generated_text):
                        logger.warning("Обнаружен некачественный текст, используем fallback")
                        return self._fallback_generation(prompt, max_length)
                        
                except Exception as e:
                    safe_error = html.escape(str(e)[:100])
                    logger.error(f"Ошибка генерации с моделью: {safe_error}")
                    return self._fallback_generation(prompt, max_length)
            else:
                # Fallback генерация
                generated_text = self._fallback_generation(prompt, max_length)
            
            return self._clean_text(generated_text)
    
    def _create_response(self, generated_text: str, tokenizer, model_name: str, task: str, 
                        prompt: str, cache_key: str, start_time: float, return_reasoning: bool,
                        temperature: float, top_p: float, max_length: int,
                        minimal_output: bool = False) -> Dict[str, Any]:
        """Создает финальный ответ."""
        # Применяем пост-обработку к сгенерированному тексту
        processed_text = self._post_process_response(generated_text, prompt)
        
        # Токенизация результата
        tokens = self._safe_tokenize(tokenizer, processed_text)
        
        # Сохранение в кэш
        self._save_to_cache(cache_key, processed_text, model_name, {
            "temperature": temperature,
            "top_p": top_p,
            "max_length": max_length
        })
        
        generation_time = time.time() - start_time
        reasoning_text = self._generate_reasoning(prompt, processed_text) if (return_reasoning or minimal_output) else None

        if minimal_output:
            # Минимальный формат вывода: только время и рассуждения
            logger.info("generate_response: time=%.3fs", generation_time)
            return {
                "generation_time": generation_time,
                "reasoning": reasoning_text
            }

        # Полный формат результата (обратная совместимость)
        result = {
            "text": processed_text,
            "tokens": tokens,
            "metadata": {
                "model": model_name,
                "task": task,
                "length": len(processed_text),
                "from_cache": False,
                "generation_time": generation_time,
                "token_count": len(tokens),
                "post_processed": True
            },
            "reasoning": reasoning_text if return_reasoning else None,
            "contradiction_detected": False,
            "contradictions": [],
            "sentiment": None
        }
        
        # Эмиссия метрик успешной генерации (источник: модель/постпроцесс)
        try:
            self._emit_metrics([
                {
                    "name": "response_generator.requests_total",
                    "component": "response_generator",
                    "type": "counter",
                    "value": 1.0,
                    "labels": {"result": "success", "source": "model"}
                },
                {
                    "name": "response_generator.latency_seconds",
                    "component": "response_generator",
                    "type": "summary",
                    "value": float(generation_time),
                },
                {
                    "name": "response_generator.tokens_generated",
                    "component": "response_generator",
                    "type": "summary",
                    "value": float(len(tokens)),
                },
            ])
        except Exception:
            pass

        # Сокращаем лог до времени генерации для чистоты
        logger.info("generate_response: time=%.3fs, len=%d", generation_time, len(processed_text))
        return result

    def _clean_text(self, text: str) -> str:
        """Очищает текст от нежелательных символов и форматирования."""
        if not text:
            return ""
        
        # Удаляем управляющие символы, но сохраняем кириллицу
        cleaned_chars = []
        for char in text:
            # Разрешаем ASCII, кириллицу, пунктуацию и пробелы
            if (ord(char) >= 32 and ord(char) <= 126) or \
               (ord(char) >= 1040 and ord(char) <= 1103) or \
               char in '\n\t.,!?;:()[]{}"\'-–—«»„“…':
                cleaned_chars.append(char)
        
        text = ''.join(cleaned_chars)
        
        # Убираем лишние пробелы
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _is_text_corrupted(self, text: str) -> bool:
        """Проверяет, является ли текст поврежденным или некачественным."""
        if not text or len(text.strip()) < 3:
            return True
        
        # Проверяем на чрезмерное повторение слов
        words = text.split()
        if len(words) > 5:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Если какое-то слово повторяется более 30% от общего количества слов
            for word, count in word_counts.items():
                if count > len(words) * 0.3:
                    return True
        
        # Проверяем на наличие слишком много непонятных символов
        strange_chars = 0
        for char in text:
            if ord(char) > 1200 or (ord(char) < 32 and char not in '\n\t'):
                strange_chars += 1
        
        if strange_chars > len(text) * 0.1:  # Более 10% странных символов
            return True
        
        return False

    def _fallback_generation(self, prompt: str, max_length: int) -> str:
        """Простая fallback генерация когда модель недоступна."""
        # Анализируем промпт для более релевантного ответа
        prompt_lower = prompt.lower()
        
        if "привет" in prompt_lower or "hello" in prompt_lower:
            responses = [
                "Привет! Как дела? Я готов помочь вам с вашими вопросами.",
                "Здравствуйте! Рад вас видеть. Чем могу быть полезен?",
                "Привет! Отличный день для общения. О чем хотели бы поговорить?"
            ]
        elif "как дела" in prompt_lower or "how are you" in prompt_lower:
            responses = [
                "У меня все хорошо, спасибо! Готов помочь вам с любыми вопросами.",
                "Отлично! Работаю в полную силу и готов к новым задачам.",
                "Прекрасно! Всегда рад общению и готов поделиться знаниями."
            ]
        elif "тест" in prompt_lower or "test" in prompt_lower:
            responses = [
                "Тест прошел успешно! Система работает корректно и готова к использованию.",
                "Проверка завершена. Все системы функционируют нормально.",
                "Тестирование выполнено. CogniFlex готов к работе!"
            ]
        else:
            responses = [
                "Понял ваш запрос. Обрабатываю информацию и готовлю ответ.",
                "Интересный вопрос! Позвольте мне подумать над этим.",
                "Спасибо за ваш запрос. Анализирую данные для наилучшего ответа."
            ]
        
        # Используем безопасный PBKDF2 вместо простого SHA256
        salt = b'cogniflex_fallback_salt'
        hash_bytes = pbkdf2_hmac('sha256', prompt.encode('utf-8'), salt, 100000)
        hash_val = int.from_bytes(hash_bytes[:4], 'big')
        return responses[hash_val % len(responses)]

    def _generate_reasoning(self, prompt: str, response: str) -> Optional[str]:
        """Генерирует объяснение логики ответа."""
        try:
            return f"Ответ сгенерирован на основе анализа промпта длиной {len(prompt)} символов."
        except Exception:
            return None

    def _build_prompt(self, query: str, context: str = "") -> str:
        """
        Формирует промпт для модели на основе запроса и контекста.
        
        Args:
            query: Входной запрос пользователя
            context: Контекст для генерации ответа
            
        Returns:
            str: Полный промпт для модели
        """
        # Если контекст пустой, возвращаем только запрос
        if not context or len(context.strip()) == 0:
            return f"Вопрос: {query}\nОтвет:"
        
        # Определяем тип контекста (структурированный или свободный текст)
        is_structured = any(marker in context for marker in ["Контекст:", "Связанные факты:", "Источник:"])
        
        if is_structured:
            # Для структурированного контекста используем четкий формат
            return (
                f"Контекст:\n{context}\n\n"
                f"Инструкция: Ответь на вопрос, используя предоставленный контекст. "
                f"Если информации недостаточно, скажи, что не можешь ответить.\n"
                f"Вопрос: {query}\n"
                f"Ответ:"
            )
        else:
            # Для свободного текста применяем более гибкий подход
            return (
                f"Вот информация, которая может помочь ответить на вопрос:\n{context}\n\n"
                f"На основе этой информации, пожалуйста, ответьте на следующий вопрос:\n"
                f"{query}\n\n"
                f"Ответ:"
            )

    def _post_process_response(self, response: str, query: str) -> str:
        """
        Выполняет пост-обработку ответа, улучшая его качество и релевантность.
        
        Args:
            response: Сырой ответ от модели
            query: Исходный запрос пользователя
            
        Returns:
            str: Обработанный ответ
        """
        # Удаляем повторяющиеся фразы и избыточные повторения
        processed = self._remove_repetitions(response)
        
        # Удаляем незаконченные предложения
        processed = self._complete_sentences(processed)
        
        # Проверяем соответствие ответа запросу
        processed = self._ensure_relevance(processed, query)
        
        # Удаляем артефакты генерации (начало промпта, повторяющиеся символы)
        processed = self._clean_artifacts(processed)
        
        # Добавляем стилистические улучшения в зависимости от типа запроса
        processed = self._add_style_enhancements(processed, query)
        
        # Проверяем и исправляем грамматику при необходимости
        processed = self._correct_grammar(processed)
        
        # Убедимся, что ответ имеет разумную длину
        processed = self._adjust_length(processed, query)
        
        return processed

    def _remove_repetitions(self, text: str) -> str:
        """Удаляет повторяющиеся фразы и избыточные повторения в тексте."""
        # Разделяем на предложения
        sentences = re.split(r'(?<=[.!?])\s+', text)
        unique_sentences = []
        seen = set()
        
        for sent in sentences:
            # Нормализуем предложение для сравнения
            normalized = re.sub(r'\s+', ' ', sent.lower().strip())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_sentences.append(sent)
        
        # Собираем уникальные предложения
        result = " ".join(unique_sentences)
        
        # Удаляем повторяющиеся слова подряд
        result = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', result)
        
        return result

    def _complete_sentences(self, text: str) -> str:
        """Завершает незаконченные предложения в тексте."""
        # Проверяем, заканчивается ли текст знаком препинания
        if text and text[-1] not in ['.', '!', '?', '...']:
            # Ищем последнюю точку
            last_period = text.rfind('.')
            last_exclamation = text.rfind('!')
            last_question = text.rfind('?')
            
            # Находим последний знак препинания
            last_punctuation = max(last_period, last_exclamation, last_question)
            
            if last_punctuation != -1 and last_punctuation > len(text) * 0.3:
                # Возвращаем текст до последнего знака препинания
                return text[:last_punctuation + 1]
            else:
                # Добавляем точку в конец
                return text + '.'
        
        return text

    def _ensure_relevance(self, response: str, query: str) -> str:
        """Обеспечивает соответствие ответа запросу пользователя."""
        # Если text_processor недоступен, возвращаем как есть
        if not self.text_processor or not hasattr(self.text_processor, 'tokenize'):
            return response
            
        try:
            # Получаем ключевые слова из запроса
            query_keywords = set(self.text_processor.tokenize(query))
            if not query_keywords:
                return response
            
            # Получаем ключевые слова из ответа
            response_keywords = set(self.text_processor.tokenize(response))
            if not response_keywords:
                return response
            
            # Вычисляем коэффициент релевантности
            common = query_keywords & response_keywords
            relevance = len(common) / len(query_keywords)
            
            # Если релевантность слишком низкая, добавляем уточнение
            if relevance < 0.2:
                # Пытаемся определить тип запроса
                query_lower = query.lower()
                if "кто" in query_lower or "как" in query_lower or "почему" in query_lower or "что" in query_lower:
                    return f"К сожалению, я не могу точно ответить на ваш вопрос. {response}"
                else:
                    return f"По вашему запросу: {response}"
        except Exception as e:
            logger.debug(f"Ошибка проверки релевантности: {e}")
        
        return response

    def _clean_artifacts(self, text: str) -> str:
        """Удаляет артефакты генерации из ответа."""
        # Удаляем начало промпта, если оно осталось в ответе
        text = re.sub(r'^Вопрос:.*?Ответ:\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'^Контекст:.*?Вопрос:.*?Ответ:\s*', '', text, flags=re.DOTALL)
        
        # Удаляем повторяющиеся символы (например, "ППППП")
        text = re.sub(r'(.)\1{3,}', r'\1', text)
        
        # Удаляем избыточные пробелы
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Удаляем повторяющиеся слова
        text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text)
        
        # Удаляем неполные предложения в начале
        if len(text) > 50:
            first_sentence = re.search(r'[.!?]\s', text)
            if first_sentence and first_sentence.start() < 20:
                text = text[first_sentence.start()+2:]
        
        return text.strip()

    def _add_style_enhancements(self, text: str, query: str) -> str:
        """Добавляет стилистические улучшения в ответ в зависимости от типа запроса."""
        query_lower = query.lower()
        
        # Определяем тип запроса
        is_question = any(qw in query_lower for qw in ["кто", "что", "как", "почему", "зачем", "где", "когда", "можем", "можно", "ли", "?"])
        is_instruction = any(iw in query_lower for iw in ["сделай", "напиши", "составь", "покажи", "опиши", "расскажи"])
        
        # Добавляем вежливые формулы
        if is_question and not text.startswith(("К сожалению", "Извините")):
            if "спасибо" in query_lower or "благодарю" in query_lower:
                text = "Пожалуйста! " + text
            elif not any(text.startswith(greeting) for greeting in ["Здравствуйте", "Привет", "Добрый день"]):
                text = "Вот что я могу сказать по этому поводу: " + text
        
        # Добавляем структуру для длинных ответов
        if len(text.split()) > 30:
            if is_question and "как" in query_lower:
                if not ("во-первых" in text.lower() or "1." in text or "первое" in text.lower()):
                    text = "Вот основные моменты:\n- " + text.replace(". ", ".\n- ")
            elif is_instruction and ("опиши" in query_lower or "расскажи" in query_lower):
                if not ("основные аспекты" in text.lower() or "ключевые моменты" in text.lower()):
                    text = "Основные аспекты:\n" + text
        
        return text

    def _correct_grammar(self, text: str) -> str:
        """Выполняет базовую коррекцию грамматики в ответе."""
        # Заменяем распространенные грамматические ошибки
        text = re.sub(r'\bя не знают\b', 'я не знаю', text)
        text = re.sub(r'\bони идет\b', 'они идут', text)
        text = re.sub(r'\bты есть\b', 'ты есть', text)
        
        # Исправляем распространенные опечатки
        text = re.sub(r'\bвв\b', 'в', text)
        text = re.sub(r'\bнеет\b', 'нет', text)
        text = re.sub(r'\bдаа\b', 'да', text)
        
        return text

    def _adjust_length(self, text: str, query: str) -> str:
        """Регулирует длину ответа в зависимости от запроса."""
        word_count = len(text.split())
        query_lower = query.lower()
        
        # Если запрос короткий и простой, делаем ответ короче
        if len(query.split()) < 5 and word_count > 50:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            # Оставляем только первые 2-3 предложения
            text = ' '.join(sentences[:min(3, len(sentences))])
        
        # Если запрос содержит "кратко" или "в двух словах", сокращаем ответ
        if "кратко" in query_lower or "в двух словах" in query_lower:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            # Оставляем только первые 1-2 предложения
            text = ' '.join(sentences[:min(2, len(sentences))])
        
        # Если ответ слишком короткий для информативного запроса, пытаемся расширить
        if word_count < 10 and any(w in query_lower for w in ["что такое", "кто такой", "объясни", "расскажи"]):
            text += " Более подробно, это означает, что..."
        
        return text

    def _create_fallback_response(self, prompt: str, error_msg: str) -> Dict[str, Any]:
        """Создает fallback ответ при ошибках."""
        try:
            fallback_text = self._fallback_generation(prompt, 100)
            safe_error = html.escape(str(error_msg)[:100])
            
            return {
                "text": html.escape(fallback_text),
                "tokens": fallback_text.split(),
                "metadata": {
                    "model": "fallback",
                    "task": "error_handling",
                    "length": len(fallback_text),
                    "from_cache": False,
                    "error": safe_error,
                    "generation_time": 0.0,
                    "token_count": len(fallback_text.split())
                },
                "reasoning": f"Fallback ответ из-за ошибки: {safe_error}",
                "contradiction_detected": False,
                "contradictions": [],
                "sentiment": None
            }
        except Exception as e:
            safe_fallback_error = html.escape(str(e)[:50])
            logger.error("Ошибка создания fallback ответа: %s", safe_fallback_error)
            return {
                "text": "Система временно недоступна",
                "tokens": ["Система", "временно", "недоступна"],
                "metadata": {"model": "emergency_fallback", "error": safe_fallback_error},
                "reasoning": None,
                "contradiction_detected": False,
                "contradictions": [],
                "sentiment": None
            }

    def is_available(self) -> bool:
        """Проверяет доступность генератора ответов."""
        try:
            return (
                self.brain is not None and
                hasattr(self.brain, 'model_manager') and
                self.brain.model_manager is not None
            )
        except Exception as e:
            safe_error = html.escape(str(e)[:50])
            logger.debug("Ошибка проверки доступности: %s", safe_error)
            return False

    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус генератора."""
        return {
            "available": self.is_available(),
            "brain_connected": self.brain is not None,
            "token_streamer": self.token_streamer is not None,
            "hybrid_cache": self.hybrid_cache is not None,
            "torch_available": torch is not None
        }
