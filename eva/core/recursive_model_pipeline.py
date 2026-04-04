"""
Recursive Model Pipeline - Three GGUF models working sequentially
для рекурсивной генерации ответов с переводом в натуральный язык и проверкой качества

Использует chat_format="qwen" для автоматической обработки темплейта Qwen

Pipeline:
  Model A (Qwen 2.5 3B) - логика, краткий ответ
  Model B (Qwen 2.5 3B) - развитие мысли, развёрнутый ответ
  Model C (Qwen 2.5 Coder 1.5B) - генерация кода (если запрос содержит код)
"""

import os
import re
import logging
import json
import time
import threading
from typing import Dict, Any, List, Optional
from llama_cpp import Llama

logger = logging.getLogger(__name__)


class AdaptiveParameterController:
    """
    Адаптивный контроллер параметров генерации.
    Анализирует причины провалов + семантическую схожесть ответов через эмбеддер.
    """
    
    DEFAULT_PARAMS = {
        'temperature': 0.3,
        'top_p': 0.9,
        'top_k': 40,
        'repeat_penalty': 1.5,
        'max_tokens': 1024,
    }
    
    # Диапазоны параметров
    PARAM_RANGES = {
        'temperature': (0.05, 1.5),
        'top_p': (0.1, 1.0),
        'top_k': (10, 100),
        'repeat_penalty': (0.5, 3.0),
        'max_tokens': (64, 4096),
    }
    
    SEMANTIC_STUCK_THRESHOLD = 0.85
    
    def __init__(self, base_params: Dict[str, float] = None):
        self.base_params = base_params or dict(self.DEFAULT_PARAMS)
        self.current_params = dict(self.base_params)
        self.failure_history: List[Dict] = []
        self.failed_response_texts: List[str] = []
        self.failed_response_embeddings: list = []
        self.success_count = 0
        self.failure_count = 0
        self._embedder = None
    
    def _get_embedder(self):
        """Ленивая загрузка эмбеддера для семантического анализа."""
        if self._embedder is None:
            try:
                from eva.mlearning.sentence_transformers_cache import get_sentence_transformer
                self._embedder = get_sentence_transformer('intfloat/multilingual-e5-base', device='cpu')
                if self._embedder is not None:
                    logger.debug("AdaptiveController: эмбеддер загружен для семантического анализа")
            except Exception as e:
                logger.debug(f"AdaptiveController: эмбеддер недоступен: {e}")
        return self._embedder
    
    def _compute_embedding(self, text: str) -> Optional[list]:
        """Вычисляет эмбеддинг текста через Model A (или fallback)."""
        embedder = self._get_embedder()
        if embedder is None or not text.strip():
            return None
        try:
            if hasattr(embedder, 'create_embedding'):
                # llama.cpp Llama instance
                result = embedder.create_embedding([text.strip()])
                if result and len(result) > 0:
                    return result[0]
            else:
                # sentence-transformers
                embedding = embedder.encode([text.strip()])[0]
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.debug(f"AdaptiveController: ошибка вычисления эмбеддинга: {e}")
        return None
    
    def _cosine_similarity(self, a: list, b: list) -> float:
        """Вычисляет косинусную схожесть двух векторов."""
        if not a or not b or len(a) != len(b):
            return 0.0
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def _are_embeddings_stuck(self) -> Dict[str, Any]:
        """Проверяет, застряли ли мы в семантически одинаковых ответах."""
        embeddings = self.failed_response_embeddings
        if len(embeddings) < 2:
            return {'is_stuck': False, 'max_similarity': 0.0, 'stuck_count': 0}
        
        # Сравниваем последние эмбеддинги друг с другом
        max_sim = 0.0
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[-1])
            max_sim = max(max_sim, sim)
        
        stuck_count = sum(1 for i in range(len(embeddings) - 1) 
                         if self._cosine_similarity(embeddings[i], embeddings[-1]) > self.SEMANTIC_STUCK_THRESHOLD)
        
        return {
            'is_stuck': max_sim > self.SEMANTIC_STUCK_THRESHOLD,
            'max_similarity': max_sim,
            'stuck_count': stuck_count,
        }
    
    def get_params_for_attempt(self, attempt: int, failure_reasons: List[str] = None) -> Dict[str, float]:
        """
        Возвращает адаптированные параметры для попытки.
        
        Стратегии:
        1. Rule-based: анализ причин провала из check_quality()
        2. Semantic: если прошлые ответы семантически одинаковы → радикальный сдвиг
        """
        # Проверяем семантическую застрялость по прошлым ответам
        semantic_info = self._are_embeddings_stuck()
        if semantic_info['is_stuck']:
            logger.warning(f"Model ЗАСТРЯЛА: семантическая схожесть={semantic_info['max_similarity']:.2f} "
                         f"(порог={self.SEMANTIC_STUCK_THRESHOLD}), застряло попыток: {semantic_info['stuck_count']}")
        
        if not failure_reasons and not semantic_info['is_stuck']:
            # Первая попытка — базовые параметры
            return dict(self.base_params)
        
        params = dict(self.base_params)
        
        # 1. Rule-based адаптация по причинам провала
        if failure_reasons:
            for reason in failure_reasons:
                reason_lower = reason.lower()
                
                if 'зацикл' in reason_lower or 'повтор' in reason_lower or 'loop' in reason_lower:
                    params['temperature'] = min(1.0, params.get('temperature', 0.3) + 0.25)
                    params['repeat_penalty'] = min(2.5, params.get('repeat_penalty', 1.5) + 0.4)
                    params['top_k'] = max(20, params.get('top_k', 40) - 10)
                    params['top_p'] = max(0.7, params.get('top_p', 0.9) - 0.1)
                
                elif 'китайск' in reason_lower or 'chinese' in reason_lower:
                    params['temperature'] = max(0.1, params.get('temperature', 0.3) - 0.15)
                    params['top_p'] = max(0.5, params.get('top_p', 0.9) - 0.2)
                    params['repeat_penalty'] = min(2.5, params.get('repeat_penalty', 1.5) + 0.3)
                
                elif 'английск' in reason_lower or 'english' in reason_lower or 'latin' in reason_lower:
                    params['temperature'] = max(0.1, params.get('temperature', 0.3) - 0.1)
                    params['top_p'] = max(0.5, params.get('top_p', 0.9) - 0.15)
                
                elif 'фраз' in reason_lower or 'паразит' in reason_lower or 'filler' in reason_lower:
                    # Фразы-паразиты — ПОВЫШАЕМ температуру (шаблонность от низкой temp)
                    params['temperature'] = min(1.0, params.get('temperature', 0.3) + 0.2)
                    params['top_k'] = min(80, params.get('top_k', 40) + 20)
                    params['repeat_penalty'] = min(2.5, params.get('repeat_penalty', 1.5) + 0.2)
                
                elif 'пуст' in reason_lower or 'коротк' in reason_lower or 'empty' in reason_lower:
                    params['max_tokens'] = min(2048, params.get('max_tokens', 1024) + 256)
                    params['temperature'] = min(1.0, params.get('temperature', 0.3) + 0.15)
                
                elif 'гласн' in reason_lower or 'vowel' in reason_lower:
                    params['temperature'] = min(1.2, params.get('temperature', 0.3) + 0.35)
                    params['repeat_penalty'] = min(2.5, params.get('repeat_penalty', 1.5) + 0.3)
        
        # 2. Semantic-based адаптация: если застряли — радикальный сдвиг
        if semantic_info['is_stuck']:
            stuck_factor = min(1.0, semantic_info['stuck_count'] / 2.0)
            
            logger.info(f"Semantic adaptation: stuck_factor={stuck_factor:.2f}, "
                       f"drastically changing parameters")
            
            # Радикально меняем все параметры
            params['temperature'] = min(1.5, params.get('temperature', 0.3) + 0.3 + stuck_factor * 0.3)
            params['top_p'] = max(0.3, min(1.0, params.get('top_p', 0.9) + 0.1 * (1 - stuck_factor)))
            params['top_k'] = max(15, min(80, params.get('top_k', 40) + int(20 * stuck_factor)))
            params['repeat_penalty'] = min(3.0, params.get('repeat_penalty', 1.5) + 0.3 + stuck_factor * 0.3)
        
        # 3. Кумулятивная адаптация при множественных провалах
        if len(self.failure_history) >= 2:
            cumulative_factor = min(0.5, len(self.failure_history) * 0.1)
            params['temperature'] = min(1.5, params['temperature'] + cumulative_factor)
            params['repeat_penalty'] = min(3.0, params['repeat_penalty'] + cumulative_factor * 0.5)
        
        # Ограничиваем диапазонами
        for param_name, (min_val, max_val) in self.PARAM_RANGES.items():
            if param_name in params:
                params[param_name] = max(min_val, min(max_val, params[param_name]))
            else:
                params[param_name] = self.base_params.get(param_name, self.DEFAULT_PARAMS.get(param_name, 1.0))
        
        return params
    
    def record_failure(self, attempt: int, reasons: List[str], params_used: Dict, response_text: str = None):
        """Записывает провал + эмбеддинг ответа."""
        self.failure_history.append({
            'attempt': attempt,
            'reasons': reasons,
            'params': params_used,
        })
        self.failure_count += 1
        
        if response_text:
            self.failed_response_texts.append(response_text)
            embedding = self._compute_embedding(response_text)
            if embedding:
                self.failed_response_embeddings.append(embedding)
    
    def record_success(self):
        """Записывает успех."""
        self.success_count += 1
    
    def reset(self):
        """Сбрасывает состояние для нового запроса."""
        self.current_params = dict(self.base_params)
        self.failed_response_texts = []
        self.failed_response_embeddings = []
        self.failure_history = []
    
    def get_stats(self) -> Dict:
        return {
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'total_attempts': self.success_count + self.failure_count,
            'recent_failures': self.failure_history[-5:],
            'semantic_analysis_enabled': self._get_embedder() is not None,
        }


class RecursiveModelPipeline:
    """
    Пайплайн для последовательной работы GGUF моделей:
    1. Model A (Qwen 2.5 3B) - даёт краткий логичный ответ
    2. Model B (Qwen 2.5 3B) - развивает мысль, добавляет детали
    3. Model C (Qwen 2.5 Coder 1.5B) - генерирует код, если нужен
    
    Использует create_chat_completion с автоматическим форматированием Qwen
    """
    
    # Настройки генерации для Model A (основная модель)
    MODEL_A_MAX_TOKENS = 1024
    MODEL_A_TEMPERATURE = 0.3
    MODEL_A_TOP_P = 0.9
    MODEL_A_TOP_K = 40
    MODEL_A_REPEAT_PENALTY = 1.5
    
    # Настройки генерации для Model B (развитие)
    MODEL_B_MAX_TOKENS = 512
    MODEL_B_TEMPERATURE = 0.3
    MODEL_B_TOP_P = 0.9
    MODEL_B_TOP_K = 40
    MODEL_B_REPEAT_PENALTY = 2.0
    
    # Настройки генерации для Model C (кодер)
    MODEL_C_MAX_TOKENS = 512
    MODEL_C_TEMPERATURE = 0.1
    MODEL_C_TOP_P = 0.9
    MODEL_C_TOP_K = 50
    MODEL_C_REPEAT_PENALTY = 1.3
    
    def __init__(
        self,
        model_a_path: str,
        model_b_path: str,
        model_c_path: str = None,
        n_ctx: int = 8192,
        n_threads: int = 8,
        fractal_memory = None
    ):
        self.model_a_path = model_a_path
        self.model_b_path = model_b_path
        self.model_c_path = model_c_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.model_a = None
        self.model_b = None
        self.model_c = None
        self.fractal_memory = fractal_memory
        self.quality_checker = None
        
        self.model_a_params = AdaptiveParameterController({
            'temperature': self.MODEL_A_TEMPERATURE,
            'top_p': self.MODEL_A_TOP_P,
            'top_k': self.MODEL_A_TOP_K,
            'repeat_penalty': self.MODEL_A_REPEAT_PENALTY,
            'max_tokens': self.MODEL_A_MAX_TOKENS,
        })
        self.model_b_params = AdaptiveParameterController({
            'temperature': self.MODEL_B_TEMPERATURE,
            'top_p': self.MODEL_B_TOP_P,
            'top_k': self.MODEL_B_TOP_K,
            'repeat_penalty': self.MODEL_B_REPEAT_PENALTY,
            'max_tokens': self.MODEL_B_MAX_TOKENS,
        })
        
        logger.info(f"RecursiveModelPipeline инициализирован (3-модельный, адаптивные параметры)")
    
    def load_models(self):
        """Загрузка GGUF моделей - Model A и B как отдельные экземпляры"""
        a_ctx = min(self.n_ctx, 2048)
        
        logger.info(f"Загрузка Model A: {self.model_a_path}")
        self.model_a = Llama(
            model_path=self.model_a_path,
            chat_format="qwen",
            n_ctx=a_ctx,
            n_threads=self.n_threads,
            verbose=False,
            cache_type_k='q8_0',
            cache_type_v='q8_0'
        )
        logger.info(f"Model A загружена с контекстом {a_ctx}, KV-кэш q8_0")
        
        if self.fractal_memory:
            self.fractal_memory.register_model_instance("model_a", self.model_a)
        
        # Model B - отдельный экземпляр
        logger.info(f"Загрузка Model B: {self.model_b_path}")
        self.model_b = Llama(
            model_path=self.model_b_path,
            chat_format="qwen",
            n_ctx=a_ctx,
            n_threads=self.n_threads,
            verbose=False,
            cache_type_k='q8_0',
            cache_type_v='q8_0'
        )
        logger.info(f"Model B загружена с контекстом {a_ctx}, KV-кэш q8_0")
        
        if self.fractal_memory:
            self.fractal_memory.register_model_instance("model_b", self.model_b)
        
        # Model C загружается лениво (только при запросе кода)
        self.model_c = None
        if self.model_c_path and os.path.exists(self.model_c_path):
            logger.info(f"Model C будет загружена лениво при запросе кода")
        else:
            logger.info("Model C не указана")
    
    def _is_code_request(self, query: str) -> bool:
        """Определяет, нужен ли код в ответе"""
        code_keywords = [
            'напиши код', 'напиши функцию', 'напиши скрипт', 'код для',
            'функцию для', 'скрипт для', 'программу', 'код на python',
            'код на js', 'код на javascript', 'напиши программу',
            'реализуй', 'реализовать', 'функция которая', 'класс для',
            'def ', 'import ', 'function ', 'class ', 'const ', 'let ',
            '```', 'print(', 'return ', 'async ', 'await '
        ]
        query_lower = query.lower()
        for kw in code_keywords:
            if kw in query_lower:
                return True
        return False
    
    def check_quality(self, text: str) -> Dict[str, Any]:
        """Проверка качества текста с детекцией зацикливания"""
        if not text or len(text.strip()) < 5:
            return {'is_gibberish': True, 'score': 0.1, 'reasons': ['Пустой или слишком короткий текст']}
        
        words = text.split()
        unique_words = set(words) if words else set()
        
        # Проверка на повторения слов
        if len(words) > 10 and len(unique_words) / len(words) < 0.3:
            return {'is_gibberish': True, 'score': 0.2, 'reasons': ['Слишком много повторений слов']}
        
        # Проверка на гласные
        vowels = set('аеёиоуыэюяАЕЁИОУЫЭЮЯaeiouAEIOU')
        if not any(v in text for v in vowels):
            return {'is_gibberish': True, 'score': 0.1, 'reasons': ['Нет гласных букв']}
        
        # Проверка на зацикливание предложений
        lines = text.split('\n')
        repeating_lines = {}
        for line in lines:
            line = line.strip()
            if len(line) > 20:
                repeating_lines[line] = repeating_lines.get(line, 0) + 1
        max_repeats = max(repeating_lines.values()) if repeating_lines else 1
        if max_repeats > 2:
            return {'is_gibberish': True, 'score': 0.3, 'reasons': ['Зацикливание: одинаковые предложения повторяются']}
        
        # Проверка на начальные фразы-паразиты
        filler_starts = ['Вот более', 'Вот что', '---', '***']
        for filler in filler_starts:
            if text.startswith(filler):
                return {'is_gibberish': True, 'score': 0.4, 'reasons': ['Начинается с фразы-паразита']}
        
        # Проверка на смешение языков (китайские/английские вставки)
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        if chinese_chars > 5:
            return {'is_gibberish': True, 'score': 0.1, 'reasons': ['Содержит китайские символы']}
        
        # Проверка на английские блоки (более 30% английских слов)
        words = text.split()
        english_words = sum(1 for w in words if w.isascii() and len(w) > 3)
        if len(words) > 10 and english_words / len(words) > 0.3:
            return {'is_gibberish': True, 'score': 0.3, 'reasons': ['Слишком много английских слов']}
        
        return {'is_gibberish': False, 'score': 0.8, 'reasons': ['OK']}
    
    def _sanitize_response(self, text: str) -> str:
        """Постобработка: удаление артефактов, латиницы в кириллических словах"""
        import re
        
        code_blocks = []
        def save_code_block(match):
            code_blocks.append(match.group(0))
            return f"\x00CODEBLOCK{len(code_blocks) - 1}\x00"
        
        text = re.sub(r'```[\s\S]*?```', save_code_block, text)
        
        # Удаляем артефакты retry
        retry_artifacts = [
            'Прости за ошибку',
            'Извините за предыдущий',
            'ВНИМАНИЕ:',
            'предыдущий ответ был некорректным',
        ]
        for artifact in retry_artifacts:
            if artifact in text:
                lines = text.split('\n')
                lines = [l for l in lines if artifact not in l]
                text = '\n'.join(lines)
        
        # Заменяем смешанные слова с латиницей
        replacements = {
            'ИнтелLECT': 'Интеллект',
            'LECT': 'лект',
            'TEL': 'тел',
            'IA': 'ИИ',
            'AI': 'ИИ',
            'system': 'система',
            'software': 'программа',
            'rek': 'рек',
            'omend': 'оменд',
            'рек': 'рек',
            'омend': 'оменд',
        }
        for eng, rus in replacements.items():
            text = text.replace(eng, rus)
        
        # Находим слова со смешением кириллицы и латиницы и помечаем
        # Паттерн: русское слово с латинскими буквами внутри
        mixed_pattern = re.compile(r'[а-яА-ЯёЁ]+[a-zA-Z]+[а-яА-ЯёЁ]+|[a-zA-Z]+[а-яА-ЯёЁ]+[a-zA-Z]+')
        def fix_mixed(match):
            word = match.group(0)
            # Заменяем латинские буквы на похожие кириллические
            latin_to_cyrillic = {
                'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
                'x': 'х', 'y': 'у', 'A': 'А', 'E': 'Е', 'O': 'О',
                'P': 'Р', 'C': 'С', 'X': 'Х', 'Y': 'У',
                'i': 'і', 'I': 'І', 'j': 'ј', 'J': 'Ј',
                'k': 'к', 'K': 'К', 'm': 'м', 'M': 'М',
                'T': 'Т', 't': 'т', 'B': 'В', 'b': 'ь',
                'H': 'Н', 'h': 'н',
            }
            result = ''
            has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in word)
            for ch in word:
                if has_cyrillic and ch in latin_to_cyrillic:
                    result += latin_to_cyrillic[ch]
                else:
                    result += ch
            return result
        
        text = mixed_pattern.sub(fix_mixed, text)
        
        for i, block in enumerate(code_blocks):
            text = text.replace(f"\x00CODEBLOCK{i}\x00", block)
        
        return text.strip()

    def _generate_with_timeout(self, model, messages: list, params: dict, timeout: int = 45) -> Optional[Dict]:
        """Генерация с таймаутом. Если модель не отвечает — возвращает None."""
        result = [None]
        error = [None]
        
        def _generate():
            try:
                result[0] = model.create_chat_completion(
                    messages=messages,
                    max_tokens=params['max_tokens'],
                    temperature=params['temperature'],
                    top_p=params['top_p'],
                    top_k=params['top_k'],
                    repeat_penalty=params['repeat_penalty'],
                    stop=["</s>"]
                )
            except Exception as e:
                error[0] = e
        
        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            logger.warning(f"Генерация прервана по таймауту ({timeout}с)")
            return None
        
        if error[0]:
            logger.warning(f"Ошибка генерации: {error[0]}")
            return None
        
        return result[0]

    def generate_with_model_a(self, query: str, max_retries: int = 2) -> Dict[str, Any]:
        """Генерация ответа на Model A (логика) с адаптивными параметрами"""
        if not self.model_a:
            raise RuntimeError("Model A не загружена")
        
        logger.info(f"Model A query: {query[:100]}...")
        
        user_content = query
        if len(query.strip()) < 20:
            user_content = 'Пользователь написал: "' + query + '". Ответь вежливо на русском языке.'
        
        self.model_a_params.reset()
        
        for attempt in range(max_retries + 1):
            # Получаем адаптированные параметры для этой попытки
            failure_reasons = None
            if attempt > 0 and self.model_a_params.failure_history:
                last_failure = self.model_a_params.failure_history[-1]
                failure_reasons = last_failure.get('reasons', [])
            
            params = self.model_a_params.get_params_for_attempt(attempt, failure_reasons)
            
            if attempt > 0:
                logger.info(f"Model A attempt {attempt+1} — adapted params: temp={params['temperature']:.2f}, "
                           f"rep={params['repeat_penalty']:.2f}, top_k={params['top_k']}, top_p={params['top_p']:.2f}")
            
            # Модифицируем системный промт при повторных попытках
            system_prompt = (
                "Ты — Модуль Логического Ядра EVA.\n"
                "Задача: Извлекать точные факты из запроса без расширений.\n"
                "Спецификации:\n"
                "1. Отвечай только подтверждёнными фактами.\n"
                "2. Максимум 3 предложения.\n"
                "3. Не используй слова «возможно», «вероятно», «может быть».\n"
                "4. Избегай оценочных суждений.\n"
                "5. Если информации недостаточно — сообщи об этом прямо.\n"
                "Ограничения:\n"
                "- Не добавляй вступления или заключения.\n"
                "- Не повторяй вопрос пользователя.\n"
                "- Отвечай строго на русском языке.\n"
                "Формат вывода: Русский. Ответ начинается сразу с факта.\n"
                "Конец инструкции."
            )
            
            if failure_reasons:
                has_filler = any('фраз' in r.lower() or 'паразит' in r.lower() for r in failure_reasons)
                has_loop = any('зацикл' in r.lower() or 'повтор' in r.lower() for r in failure_reasons)
                
                if has_filler:
                    system_prompt = (
                        "Ты — Модуль Логического Ядра EVA.\n"
                        "Задача: Отвечай строго по существу, без вступлений.\n"
                        "ЗАПРЕЩЕНО начинать с фраз: «Конечно», «Давайте», «Вот», «Это», «Привет», «Здравствуйте».\n"
                        "Спецификации:\n"
                        "1. Начни ответ сразу с факта или прямого ответа.\n"
                        "2. Максимум 3 предложения.\n"
                        "3. Не используй вводные слова и фразы-паразиты.\n"
                        "4. Отвечай строго на русском языке.\n"
                        "5. Избегай общих фраз — только конкретика.\n"
                        "Ограничения:\n"
                        "- Никаких вступлений, приветствий, обращений.\n"
                        "- Не повторяй вопрос пользователя.\n"
                        "Формат вывода: Русский. Сразу с факта.\n"
                        "Конец инструкции."
                    )
                if has_loop:
                    system_prompt = (
                        "Ты — Модуль Логического Ядра EVA.\n"
                        "Задача: Дай уникальный ответ без повторений.\n"
                        "Спецификации:\n"
                        "1. Каждое предложение должно нести новую информацию.\n"
                        "2. Запрещено повторять одинаковые мысли.\n"
                        "3. Максимум 3 предложения.\n"
                        "4. Отвечай строго на русском языке.\n"
                        "5. Используй разнообразные конструкции.\n"
                        "Ограничения:\n"
                        "- Не повторяй предложения или их части.\n"
                        "- Не используй одинаковые начала предложений.\n"
                        "Формат вывода: Русский. Сразу с факта.\n"
                        "Конец инструкции."
                    )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            start_time = time.time()
            output = self._generate_with_timeout(self.model_a, messages, params, timeout=60)
            elapsed = time.time() - start_time
            
            if output is None:
                logger.warning(f"Model A: попытка {attempt+1} прервана по таймауту ({elapsed:.1f}с)")
                self.model_a_params.record_failure(attempt + 1, ['Таймаут генерации'], params, '')
                continue
            
            logger.info(f"Model A generation time: {elapsed:.1f}с")
            
            raw_response = output['choices'][0]['message']['content'].strip()
            raw_response = self._sanitize_response(raw_response)
            quality = self.check_quality(raw_response)
            
            has_chinese = sum(1 for c in raw_response if '一' <= c <= '鿿') > 5
            if has_chinese:
                quality['reasons'].append('Содержит китайские символы')
                self.model_a_params.record_failure(attempt + 1, quality['reasons'], params, raw_response)
                logger.warning(f"Model A: попытка {attempt+1} содержит китайские символы, retry")
                continue
            
            if not quality['is_gibberish']:
                logger.info(f"Model A response (attempt {attempt+1}): '{raw_response[:200]}...'")
                logger.info(f"Model A tokens: {output['usage']['completion_tokens']}")
                self.model_a_params.record_success()
                
                return {
                    'raw_response': raw_response,
                    'natural_response': raw_response,
                    'quality': quality,
                    'model': 'Model A (Logic)',
                    'tokens': output['usage']['completion_tokens']
                }
            
            self.model_a_params.record_failure(attempt + 1, quality['reasons'], params, raw_response)
            logger.warning(f"Model A: попытка {attempt+1} не прошла проверку качества: {quality['reasons']}")
        
        # Fallback — Model B попробует
        logger.warning("Model A: все попытки провалились, использую Model B")
        return {
            'raw_response': '',
            'natural_response': '',
            'quality': {'is_gibberish': True, 'score': 0.0, 'reasons': ['Model A не смогла, передаю Model B']},
            'model': 'Model A (Logic)',
            'tokens': 0
        }
    
    def generate_with_model_b(self, query: str, previous_response: str, max_retries: int = 2) -> Dict[str, Any]:
        """Генерация ответа на Model B с адаптивными параметрами"""
        if not self.model_b:
            raise RuntimeError("Model B не загружена")
        
        self.model_b_params.reset()
        
        for attempt in range(max_retries + 1):
            failure_reasons = None
            if attempt > 0 and self.model_b_params.failure_history:
                last_failure = self.model_b_params.failure_history[-1]
                failure_reasons = last_failure.get('reasons', [])
            
            params = self.model_b_params.get_params_for_attempt(attempt, failure_reasons)
            
            if attempt > 0:
                logger.info(f"Model B attempt {attempt+1} — adapted params: temp={params['temperature']:.2f}, "
                           f"rep={params['repeat_penalty']:.2f}, top_k={params['top_k']}, top_p={params['top_p']:.2f}")
            
            system_prompt = (
                "Ты — Модуль Развития Концепций EVA.\n"
                "Задача: Развивай мысль, добавляй детали и примеры.\n"
                "Спецификации:\n"
                "1. Расширяй факты примерами и пояснениями.\n"
                "2. Используй структурированный формат (списки, абзацы).\n"
                "3. Отвечай строго на русском языке.\n"
                "4. Добавляй контекст и смежные темы.\n"
                "5. Максимум 10 предложений.\n"
                "Ограничения:\n"
                "- Не повторяй факты дословно.\n"
                "- Не используй английские или китайские вставки.\n"
                "Формат вывода: Русский, развёрнутый ответ.\n"
                "Конец инструкции."
            )
            
            if failure_reasons:
                has_filler = any('фраз' in r.lower() or 'паразит' in r.lower() for r in failure_reasons)
                has_chinese = any('китайск' in r.lower() or 'chinese' in r.lower() for r in failure_reasons)
                
                if has_filler:
                    system_prompt = (
                        "Ты — Модуль Развития Концепций EVA.\n"
                        "Задача: Развивай мысль без вступлений.\n"
                        "ЗАПРЕЩЕНО начинать с: «Конечно», «Давайте», «Вот», «Это».\n"
                        "Спецификации:\n"
                        "1. Начни сразу с развития мысли.\n"
                        "2. Используй списки и абзацы.\n"
                        "3. Отвечай строго на русском языке.\n"
                        "4. Максимум 10 предложений.\n"
                        "Ограничения:\n"
                        "- Никаких вступлений и обращений.\n"
                        "Формат вывода: Русский, сразу по делу.\n"
                        "Конец инструкции."
                    )
                if has_chinese:
                    system_prompt = (
                        "Ты — Модуль Развития Концепций EVA.\n"
                        "Задача: Развивай мысль СТРОГО на русском языке.\n"
                        "ЗАПРЕЩЕНО: использовать китайские, английские или иные иностранные символы.\n"
                        "Спецификации:\n"
                        "1. Только русские буквы и знаки препинания.\n"
                        "2. Развивай факты примерами.\n"
                        "3. Максимум 10 предложений.\n"
                        "Ограничения:\n"
                        "- Любой иностранный символ = ошибка.\n"
                        "Формат вывода: Только русский язык.\n"
                        "Конец инструкции."
                    )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            start_time = time.time()
            output = self._generate_with_timeout(self.model_b, messages, params, timeout=45)
            elapsed = time.time() - start_time
            
            if output is None:
                logger.warning(f"Model B: попытка {attempt+1} прервана по таймауту ({elapsed:.1f}с)")
                self.model_b_params.record_failure(attempt + 1, ['Таймаут генерации'], params, '')
                continue
            
            logger.info(f"Model B generation time: {elapsed:.1f}с")
            
            raw_response = output['choices'][0]['message']['content'].strip()
            raw_response = self._sanitize_response(raw_response)
            raw_response = self._clean_filler_start(raw_response)
            raw_response = self._remove_looping_blocks(raw_response)
            quality = self.check_quality(raw_response)
            
            # Никогда не возвращаем ответ с китайскими символами
            has_chinese = sum(1 for c in raw_response if '一' <= c <= '鿿') > 5
            if has_chinese:
                quality['reasons'].append('Содержит китайские символы')
                self.model_b_params.record_failure(attempt + 1, quality['reasons'], params, raw_response)
                logger.warning(f"Model B: попытка {attempt+1} содержит китайские символы, retry")
                continue
            
            if not quality['is_gibberish']:
                logger.info(f"Model B response (attempt {attempt+1}): '{raw_response[:200]}...'")
                logger.info(f"Model B tokens: {output['usage']['completion_tokens']}")
                self.model_b_params.record_success()
                
                return {
                    'raw_response': raw_response,
                    'natural_response': raw_response,
                    'quality': quality,
                    'model': 'Model B (Concept)',
                    'tokens': output['usage']['completion_tokens']
                }
            
            self.model_b_params.record_failure(attempt + 1, quality['reasons'], params, raw_response)
            logger.warning(f"Model B: попытка {attempt+1} не прошла проверку качества: {quality['reasons']}")
        
        # Fallback на Model A
        return {
            'raw_response': previous_response if previous_response else 'Не удалось сформировать ответ.',
            'natural_response': previous_response if previous_response else 'Не удалось сформировать ответ.',
            'quality': {'is_gibberish': True, 'score': 0.0, 'reasons': ['Все попытки не прошли проверку']},
            'model': 'Model B (Concept)',
            'tokens': 0
        }
    
    def _load_model_c(self):
        """Ленивая загрузка Model C (только при необходимости)"""
        if self.model_c is not None:
            return True
        if not self.model_c_path or not os.path.exists(self.model_c_path):
            logger.warning("Model C не найдена")
            return False
        try:
            logger.info(f"Ленивая загрузка Model C: {self.model_c_path}")
            model_c_ctx = min(self.n_ctx, 2048)
            self.model_c = Llama(
                model_path=self.model_c_path,
                chat_format="qwen",
                n_ctx=model_c_ctx,
                n_threads=self.n_threads,
                verbose=False,
                cache_type_k='q8_0',
                cache_type_v='q8_0'
            )
            logger.info(f"Model C загружена с контекстом {model_c_ctx}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки Model C: {e}")
            return False

    def generate_with_model_c(self, query: str, context: str, max_retries: int = 2) -> Dict[str, Any]:
        """Генерация кода на Model C (Coder) с ленивой загрузкой"""
        if not self._load_model_c():
            raise RuntimeError("Model C (Coder) не может быть загружена")
        
        messages = [
            {"role": "system", "content": "Ты - ЕВА, помощница-программист. Пиши чистый, рабочий код. Комментарии на русском языке."},
            {"role": "user", "content": f"[НА РУССКОМ] Контекст: {context}\n\nЗапрос: {query}\n\nНапиши код."}
        ]
        
        logger.info(f"Model C (Coder) query: {query[:100]}...")
        
        for attempt in range(max_retries + 1):
            params = {
                'max_tokens': self.MODEL_C_MAX_TOKENS,
                'temperature': self.MODEL_C_TEMPERATURE + (attempt * 0.05),
                'top_p': self.MODEL_C_TOP_P,
                'top_k': self.MODEL_C_TOP_K,
                'repeat_penalty': self.MODEL_C_REPEAT_PENALTY + (attempt * 0.1),
                'stop': ["</s>"]
            }
            output = self._generate_with_timeout(self.model_c, messages, params, timeout=60)
            if output is None:
                logger.warning(f"Model C: timeout on attempt {attempt+1}")
                continue
            
            raw_response = output['choices'][0]['message']['content'].strip()
            quality = self.check_quality(raw_response)
            
            if not quality['is_gibberish'] or attempt == max_retries:
                logger.info(f"Model C response (attempt {attempt+1}): '{raw_response[:200]}...'")
                logger.info(f"Model C tokens: {output['usage']['completion_tokens']}")
                
                return {
                    'raw_response': raw_response,
                    'natural_response': raw_response,
                    'quality': quality,
                    'model': 'Model C (Coder)',
                    'tokens': output['usage']['completion_tokens']
                }
            
            logger.warning(f"Model C: попытка {attempt+1} не прошла проверку качества: {quality['reasons']}")
    
    def _clean_filler_start(self, text: str) -> str:
        """Удаляет вводные фразы-паразиты из начала ответа"""
        fillers = [
            'Конечно! ', 'Конечно, ', 'Конечно!', 'Конечно',
            'Вот более ', 'Вот что ', 'Вот ',
            'Это всё ', 'Это все ',
            'Ваш вопрос ', 'Ваш запрос ',
            '---', '***',
        ]
        cleaned = text
        for filler in fillers:
            if cleaned.startswith(filler):
                cleaned = cleaned[len(filler):].strip()
                break
        return cleaned
    
    def _remove_looping_blocks(self, text: str, max_repeats: int = 2) -> str:
        """Удаляет зацикливающиеся блоки текста"""
        lines = text.split('\n')
        seen_blocks = {}
        result_lines = []
        
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 30:
                if stripped in seen_blocks:
                    seen_blocks[stripped] += 1
                    if seen_blocks[stripped] > max_repeats:
                        continue
                else:
                    seen_blocks[stripped] = 1
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def process_query(
        self,
        query: str,
        max_iterations: int = 1,
        gen_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Основной метод обработки запроса через 3-модельный пайплайн"""
        if not self.model_a or not self.model_b:
            raise RuntimeError("Модели не загружены. Вызовите load_models()")
        
        # Применяем динамические параметры (локальные переменные)
        a_max_tokens = self.MODEL_A_MAX_TOKENS
        a_temperature = self.MODEL_A_TEMPERATURE
        b_max_tokens = self.MODEL_B_MAX_TOKENS
        b_temperature = self.MODEL_B_TEMPERATURE
        
        if gen_params:
            params_a = gen_params.get('model_a', {})
            params_b = gen_params.get('model_b', {})
            a_max_tokens = params_a.get('max_tokens', self.MODEL_A_MAX_TOKENS)
            a_temperature = params_a.get('temperature', self.MODEL_A_TEMPERATURE)
            b_max_tokens = params_b.get('max_tokens', self.MODEL_B_MAX_TOKENS)
            b_temperature = params_b.get('temperature', self.MODEL_B_TEMPERATURE)
            logger.info(f"Применены динамические параметры: A(temp={self.MODEL_A_TEMPERATURE}), B(temp={self.MODEL_B_TEMPERATURE})")
        
        results = {
            'query': query,
            'model_a_result': None,
            'model_b_result': None,
            'model_c_result': None,
            'reasoning_steps': [],
            'has_code': False,
            'fractal_context': None
        }
        
        # Контекст из графа обучения (только для содержательных запросов)
        enriched_query = query
        if self.fractal_memory and hasattr(self.fractal_memory, 'get_context_for_query'):
            # Не добавляем контекст для коротких приветствий и простых фраз
            skip_context_keywords = ['привет', 'здравствуй', 'добрый', 'хай', 'hello', 'hi', 'hey', 'пока', 'до свидания', 'спасибо', 'благодар']
            query_lower = query.lower().strip()
            is_greeting = any(kw in query_lower for kw in skip_context_keywords) and len(query) < 60
            
            if not is_greeting:
                graph_context = self.fractal_memory.get_context_for_query(query)
                if graph_context:
                    enriched_query = f"{query}\n\nКонтекст из опыта:\n{graph_context}"
                    results['fractal_context'] = graph_context
                    logger.info(f"Контекст из графа обучения: {len(graph_context)} символов")
        
        # Шаг 1: Model A - логический ответ
        logger.info("=== Шаг 1: Генерация ответа на Model A (логическое ядро) ===")
        model_a_result = self.generate_with_model_a(enriched_query)
        logger.info(f"Model A ответ: {model_a_result['natural_response'][:150]}...")
        results['model_a_result'] = model_a_result
        
        results['reasoning_steps'].append({
            'step': 1,
            'phase': 'model_a_generation',
            'thought': model_a_result['natural_response'][:200],
            'confidence': model_a_result['quality'].get('score', 0.8),
            'model': 'Model A (Logic)',
            'action': 'Извлечение фактов',
            'input': query,
            'output': model_a_result['natural_response']
        })
        
        # Шаг 2: Model B - развитие мысли (всегда, даже если Model A провалилась)
        logger.info("=== Шаг 2: Генерация расширенного ответа на Model B ===")
        model_b_result = self.generate_with_model_b(query, model_a_result.get('natural_response', ''))
        logger.info(f"Model B ответ: {model_b_result['natural_response'][:150]}...")
        results['model_b_result'] = model_b_result
        
        results['reasoning_steps'].append({
            'step': 2,
            'phase': 'model_b_generation',
            'thought': model_b_result['natural_response'][:200],
            'confidence': model_b_result['quality'].get('score', 0.8),
            'model': 'Model B (Concept)',
            'action': 'Расширение концепций',
            'input': f"Факты: {model_a_result['natural_response'][:100]}",
            'output': model_b_result['natural_response']
        })
        
        # Финальный ответ - Model B (расширенный)
        results['final_response'] = model_b_result['natural_response']
        
        # Шаг 3: Model C - код (если нужен)
        if self.model_c and self._is_code_request(query):
            logger.info("=== Шаг 3: Генерация кода на Model C (Coder) ===")
            results['has_code'] = True
            model_c_result = self.generate_with_model_c(query, model_b_result['natural_response'])
            logger.info(f"Model C ответ: {model_c_result['natural_response'][:150]}...")
            results['model_c_result'] = model_c_result
            
            results['reasoning_steps'].append({
                'step': 3,
                'phase': 'model_c_generation',
                'thought': model_c_result['natural_response'][:200],
                'confidence': model_c_result['quality'].get('score', 0.8),
                'model': 'Model C (Coder)',
                'action': 'Генерация кода',
                'input': f"Контекст: {model_b_result['natural_response'][:100]}",
                'output': model_c_result['natural_response']
            })
            
            # Финальный ответ = текст Model B + код Model C
            results['final_response'] = model_b_result['natural_response'] + "\n\n" + model_c_result['natural_response']
        else:
            # Финальный ответ - используем Model B (расширенный)
            results['final_response'] = model_b_result['natural_response']
        
        results['final_quality'] = model_b_result['quality']
        
        # Сохраняем опыт в граф обучения (цикл обучения через граф)
        if self.fractal_memory and hasattr(self.fractal_memory, 'save_experience'):
            # Сохраняем оба ответа как опыт
            self.fractal_memory.save_experience(
                query=query,
                response=model_a_result['natural_response'],
                model_used='model_a',
                quality_score=model_a_result['quality'].get('score', 0.5)
            )
            self.fractal_memory.save_experience(
                query=query,
                response=model_b_result['natural_response'],
                model_used='model_b',
                quality_score=model_b_result['quality'].get('score', 0.5)
            )
        
        logger.info("Three-GGUF пайплайн завершён")
        
        return results


def create_recursive_pipeline(
    model_a_path: str = None,
    model_b_path: str = None,
    model_c_path: str = None,
    n_ctx: int = 8192,
    n_threads: int = 8,
    fractal_memory = None
) -> 'RecursiveModelPipeline':
    """Фабричная функция для создания пайплайна"""
    if model_a_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_a_path = os.path.join(project_root, "eva", "memory", "fractal_torch_storage", "gguf_models", "qwen2.5-3b-instruct", "qwen2.5-3b-instruct-q4_k_m.gguf")
    
    if model_b_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_b_path = os.path.join(project_root, "eva", "memory", "fractal_torch_storage", "gguf_models", "qwen2.5-3b-instruct", "qwen2.5-3b-instruct-q4_k_m.gguf")
    
    if model_c_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_c_path = os.path.join(project_root, "eva", "memory", "fractal_torch_storage", "gguf_models", "qwen2.5-coder-1.5b-instruct", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
    
    pipeline = RecursiveModelPipeline(
        model_a_path=model_a_path,
        model_b_path=model_b_path,
        model_c_path=model_c_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        fractal_memory=fractal_memory
    )
    pipeline.load_models()
    return pipeline



