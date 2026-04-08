"""
Generation methods for individual models in the Recursive Model Pipeline.
"""

import os
import re
import logging
import time
from typing import Dict, Any, Optional
from llama_cpp import Llama

from .text_chunker import TextChunker, MAX_INPUT_TOKENS_MODEL_A, MAX_INPUT_TOKENS_MODEL_B
from .pipeline_quality import check_quality, _sanitize_response, _clean_filler_start, _remove_looping_blocks, check_russian_quality

logger = logging.getLogger(__name__)


def _generate_response(
    model: Llama,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.3,
    max_context: int = 4096
) -> str:
    """Helper for Model A review functionality."""
    messages = [
        {"role": "system", "content": "Ты — Модуль Логического Ядра EVA. Отвечай кратко и по существу."},
        {"role": "user", "content": prompt}
    ]
    
    output = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|end|>", "\n\n"],
    )
    
    return output['choices'][0]['message']['content'].strip()


def generate_with_model_a(
    self,
    query: str,
    max_retries: int = 1
) -> Dict[str, Any]:
    """Генерация ответа на Model A (логика) с адаптивными параметрами"""
    if not self.model_a:
        raise RuntimeError("Model A не загружена")
    
    logger.info(f"Model A query: {query[:100]}...")
    
    _chunker = TextChunker()
    estimated_tokens = _chunker.estimate_tokens(query)
    
    if estimated_tokens > MAX_INPUT_TOKENS_MODEL_A:
        logger.warning(f"Model A: запрос слишком большой ({estimated_tokens} токенов), разбиение на чанки")
        return self._generate_with_chunking_a(query)
    
    try:
        output = self.model_a.create_chat_completion(
            messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "test"}],
            max_tokens=1,
            temperature=0.1
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "exceed" in error_msg or "context" in error_msg or "token" in error_msg:
            logger.warning(f"Model A: контекст превышен, используем chunking")
            return self._generate_with_chunking_a(query)
    
    user_content = query
    # Для коротких приветствий - особая обработка
    short_greetings = ['привет', 'здравствуй', 'hi', 'hello', 'hey', 'приветик', 'приветики', 'здорово', 'прив', 'ку', 'приветствую']
    if len(query.strip()) < 25 and any(query.strip().lower().startswith(g) for g in short_greetings):
        user_content = 'Пользователь поздоровался с тобой: "' + query.strip() + '". Поздоровайся в ответ, кратко и дружелюбно.'
    
    self.model_a_params.reset()
    
    for attempt in range(max_retries + 1):
        failure_reasons = None
        if attempt > 0 and self.model_a_params.failure_history:
            last_failure = self.model_a_params.failure_history[-1]
            failure_reasons = last_failure.get('reasons', [])
        
        params = self.model_a_params.get_params_for_attempt(attempt, failure_reasons)
        
        if attempt > 0:
            logger.info(f"Model A attempt {attempt+1} — adapted params: temp={params['temperature']:.2f}, "
                       f"rep={params['repeat_penalty']:.2f}, top_k={params['top_k']}, top_p={params['top_p']:.2f}")
        
        system_prompt = (
            "Ты — Модуль Логического Ядра eva_ai.\n"
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
                    "Ты — Модуль Логического Ядра eva_ai.\n"
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
                    "Ты — Модуль Логического Ядра eva_ai.\n"
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
        
        # Без hard timeout - ждём бесконечно, но с итеративным контролем
        # Контроль осуществляется через CoreBrain через DeferredCommandSystem
        try:
            output = self._generate_with_timeout(self.model_a, messages, params, timeout=None)
        except Exception as e:
            error_msg = str(e).lower()
            if "exceed" in error_msg or "context" in error_msg:
                logger.warning(f"Model A: ошибка контекста ({error_msg}), используем chunking")
                return self._generate_with_chunking_a(query)
            raise
        
        elapsed = time.time() - start_time
        
        if output is None:
            # Таймаут = генерация идёт, но медленно - NOT критическая ошибка
            # Это не "ошибка генерации", а "ещё генерируется"
            logger.warning(f"Model A: генерация продолжается ({elapsed:.1f}с), ожидаем завершения")
            # Возвращаем признак того, что генерация идёт
            return {
                'raw_response': '',
                'natural_response': '',
                'quality': {'is_gibberish': True, 'score': 0.0, 'reasons': ['Генерация в процессе']},
                'tokens': 0,
                'status': 'generating',  # Ключевой признак - идёт генерация!
                'elapsed': elapsed
            }
        
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
    
    logger.warning("Model A: все попытки провалились, использую Model B")
    return {
        'raw_response': '',
        'natural_response': '',
        'quality': {'is_gibberish': True, 'score': 0.0, 'reasons': ['Model A не смогла, передаю Model B']},
        'model': 'Model A (Logic)',
        'tokens': 0
    }


def _generate_with_chunking_a(self, query: str) -> Dict[str, Any]:
    """Генерация с разбиением больших запросов для Model A"""
    _chunker = TextChunker()
    
    chunks = _chunker.chunk_by_sentences(query, max_tokens=MAX_INPUT_TOKENS_MODEL_A - 400)
    logger.info(f"Model A: разбито на {len(chunks)} чанков")
    
    results = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Model A: обработка чанка {i+1}/{len(chunks)}")
        
        user_content = 'Извлеки ключевые факты из текста. Отвечай кратко (1-2 предложения).\n\nТекст: ' + chunk
        
        system_prompt = (
            "Ты — Модуль Логического Ядра eva_ai.\n"
            "Задача: Извлекать точные факты из запроса без расширений.\n"
            "Спецификации:\n"
            "1. Отвечай только подтверждёнными фактами.\n"
            "2. Максимум 2 предложения.\n"
            "3. Не используй слова «возможно», «вероятно».\n"
            "4. Отвечай строго на русском языке.\n"
            "Формат вывода: Русский. Ответ начинается сразу с факта.\n"
            "Конец инструкции."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        params = self.model_a_params.get_params_for_attempt(0, None)
        
        try:
            output = self.model_a.create_chat_completion(
                messages=messages,
                temperature=params['temperature'],
                max_tokens=min(params['max_tokens'], 512),
                top_p=params['top_p'],
                top_k=params['top_k'],
                repeat_penalty=params['repeat_penalty'],
                stop=["</s>"]
            )
            
            if output and output.get('choices'):
                raw_response = output['choices'][0]['message']['content'].strip()
                results.append(raw_response)
                logger.info(f"Model A чанк {i+1}: {raw_response[:100]}...")
        except Exception as e:
            logger.error(f"Model A: ошибка чанка {i+1}: {e}")
            continue
    
    if not results:
        return {
            'raw_response': '',
            'natural_response': '',
            'quality': {'is_gibberish': True, 'score': 0.0, 'reasons': ['Не удалось обработать чанки']},
            'model': 'Model A (Logic)',
            'tokens': 0,
            'chunks': len(chunks)
        }
    
    merged = ". ".join([r.strip().rstrip('.') for r in results if r.strip()])
    if merged:
        merged += "."
    
    return {
        'raw_response': merged,
        'natural_response': merged,
        'quality': {'is_gibberish': False, 'score': 0.8, 'reasons': ['OK']},
        'model': 'Model A (Logic)',
        'tokens': _chunker.estimate_tokens(merged),
        'chunks': len(chunks),
        'merged': True
    }


def generate_with_model_b(
    self,
    query: str,
    previous_response: str,
    max_retries: int = 9
) -> Dict[str, Any]:
    """Генерация ответа на Model B с адаптивными параметрами"""
    if not self.model_b:
        raise RuntimeError("Model B не загружена")
    
    _chunker = TextChunker()
    estimated_tokens = _chunker.estimate_tokens(query)
    
    if estimated_tokens > MAX_INPUT_TOKENS_MODEL_B:
        logger.warning(f"Model B: запрос слишком большой ({estimated_tokens} токенов), разбиение на чанки")
        return self._generate_with_chunking_b(query, previous_response)
    
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
            "Ты — Модуль Творческого Ядра eva_ai.\n"
            "Задача: Расширять запрос рассуждением о фактах.\n"
            "Спецификации:\n"
            "1. Расширяй концепциями.\n"
            "2. Максимум 30 предложений.\n"
            "3. Оценивай факт альтернативно.\n"
            "4. Рассуждай в ответе о факте\n"
            "5. Непонятно - переспроси.\n"
            "Ограничения:\n"
            "- Не добавляй заведомо ложной информации.\n"
            "- Не повторяй вопрос пользователя.\n"
            "- Отвечай строго на русском языке.\n"
            "Формат вывода: Русский. Ответ начинается с расширения факта\n"
            "Конец инструкции."
        )
        
        if failure_reasons:
            has_filler = any('фраз' in r.lower() or 'паразит' in r.lower() for r in failure_reasons)
            has_chinese = any('китайск' in r.lower() or 'chinese' in r.lower() for r in failure_reasons)
            
            if has_filler:
                system_prompt = (
                    "Ты — Модуль Развития Концепций eva_ai.\n"
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
                    "Ты — Модуль Развития Концепций eva_ai.\n"
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
            {"role": "user", "content": f"Контекст (факты от Model A):\n{previous_response}\n\nЗапрос пользователя:\n{query}\n\nРазверни ответ на основе фактов выше. Отвечай строго на русском языке."}
        ]
        
        # Logit bias против китайских токенов (диапазон 4E00-9FFF)
        logit_bias = {}
        if attempt > 0 and any('китайск' in r.lower() or 'chinese' in r.lower() for r in (failure_reasons or [])):
            # Агрессивно подавляем китайские токены
            logit_bias = {str(i): -100 for i in range(10000, 20000)}  # CJK диапазон в Qwen vocab
        
        start_time = time.time()
        try:
            output = self._generate_with_timeout(self.model_b, messages, params, timeout=None, logit_bias=logit_bias if logit_bias else None)
        except Exception as e:
            error_msg = str(e).lower()
            if "exceed" in error_msg or "context" in error_msg:
                logger.warning(f"Model B: ошибка контекста ({error_msg}), используем chunking")
                return self._generate_with_chunking_b(query, previous_response)
            raise
        
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
        russian_quality = check_russian_quality(raw_response)
        
        has_chinese = sum(1 for c in raw_response if '一' <= c <= '鿿') > 5
        if has_chinese:
            quality['reasons'].append('Содержит китайские символы')
            russian_quality['reasons'].append('Содержит китайские символы')
            self.model_b_params.record_failure(attempt + 1, quality['reasons'], params, raw_response)
            logger.warning(f"Model B: попытка {attempt+1} содержит китайские символы, retry")
            continue
        
        if not quality['is_gibberish'] and russian_quality['is_valid']:
            logger.info(f"Model B response (attempt {attempt+1}): '{raw_response[:200]}...'")
            logger.info(f"Model B tokens: {output['usage']['completion_tokens']}")
            self.model_b_params.record_success()
            
            combined_score = (quality['score'] + russian_quality['score']) / 2
            combined_reasons = quality['reasons'] + [r for r in russian_quality['reasons'] if r != 'OK']
            
            return {
                'raw_response': raw_response,
                'natural_response': raw_response,
                'quality': {
                    'is_gibberish': quality['is_gibberish'],
                    'score': combined_score,
                    'reasons': combined_reasons,
                    'russian_quality': russian_quality
                },
                'model': 'Model B (Concept)',
                'tokens': output['usage']['completion_tokens']
            }
        
        all_reasons = quality['reasons'] + russian_quality['reasons']
        self.model_b_params.record_failure(attempt + 1, all_reasons, params, raw_response)
        logger.warning(f"Model B: попытка {attempt+1} не прошла проверку качества: {all_reasons}")
    
    return {
        'raw_response': previous_response if previous_response else 'Не удалось сформировать ответ.',
        'natural_response': previous_response if previous_response else 'Не удалось сформировать ответ.',
        'quality': {'is_gibberish': True, 'score': 0.0, 'reasons': ['Все попытки не прошли проверку']},
        'model': 'Model B (Concept)',
        'tokens': 0
    }


def _generate_with_chunking_b(self, query: str, previous_response: str) -> Dict[str, Any]:
    """Генерация с разбиением больших запросов для Model B"""
    _chunker = TextChunker()
    
    chunks = _chunker.chunk_by_sentences(query, max_tokens=MAX_INPUT_TOKENS_MODEL_B - 300)
    logger.info(f"Model B: разбито на {len(chunks)} чанков")
    
    results = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Model B: обработка чанка {i+1}/{len(chunks)}")
        
        system_prompt = (
            "Ты EVA. Отвечай на русском языке. "
            "Расширь контекст, порассуждай на тему запроса. "
            "Без иностранных символов. "
            "Каждое предложение заканчивай точкой."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Разверни мысль:\n{chunk}\n\nКонтекст: {previous_response}"}
        ]
        
        params = self.model_b_params.get_params_for_attempt(0, None)
        
        try:
            output = self.model_b.create_chat_completion(
                messages=messages,
                temperature=params['temperature'],
                max_tokens=min(params['max_tokens'], 256),
                top_p=params['top_p'],
                top_k=params['top_k'],
                repeat_penalty=params['repeat_penalty'],
                stop=["</s>"]
            )
            
            if output and output.get('choices'):
                raw_response = output['choices'][0]['message']['content'].strip()
                results.append(raw_response)
        except Exception as e:
            logger.error(f"Model B: ошибка чанка {i+1}: {e}")
            continue
    
    if not results:
        return {
            'raw_response': previous_response,
            'natural_response': previous_response,
            'quality': {'is_gibberish': True, 'score': 0.0, 'reasons': ['Не удалось обработать чанки']},
            'model': 'Model B (Concept)',
            'tokens': 0,
            'chunks': len(chunks)
        }
    
    merged = "\n\n".join([r.strip() for r in results if r.strip()])
    
    return {
        'raw_response': merged,
        'natural_response': merged,
        'quality': {'is_gibberish': False, 'score': 0.8, 'reasons': ['OK']},
        'model': 'Model B (Concept)',
        'tokens': _chunker.estimate_tokens(merged),
        'chunks': len(chunks),
        'merged': True
    }


def _load_model_c(self) -> bool:
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


def _unload_model_c(self) -> None:
    """Выгрузка Model C из памяти после генерации."""
    if self.model_c is not None:
        try:
            del self.model_c
            self.model_c = None
            
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            
            logger.info("Model C выгружена из памяти")
        except Exception as e:
            logger.warning(f"Ошибка выгрузки Model C: {e}")


def generate_with_model_c(
    self,
    query: str,
    context: str,
    max_retries: int = 2
) -> Dict[str, Any]:
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
            'stop': self.STOP_TOKENS
        }
        
        # Model C - кодогенерация, тоже без hard timeout
        output = self._generate_with_timeout(self.model_c, messages, params, timeout=None)
        
        if output is None:
            # Генерация продолжается - NOT ошибка
            logger.warning(f"Model C: генерация продолжается (attempt {attempt+1})")
            return {
                'raw_response': '',
                'natural_response': '',
                'quality': {'is_gibberish': True, 'score': 0.0, 'reasons': ['Генерация в процессе']},
                'model': 'Model C (Coder)',
                'tokens': 0,
                'status': 'generating'
            }
        
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
