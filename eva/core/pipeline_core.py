"""
Pipeline Core - Main RecursiveModelPipeline class.
Orchestrates the 3-model pipeline for recursive answer generation.
"""

import os
import re
import logging
import json
import time
import threading
import atexit
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, Any, List, Optional
from llama_cpp import Llama

from .pipeline_adaptive import AdaptiveParameterController
from .pipeline_quality import (
    check_quality,
    _sanitize_response,
    _clean_filler_start,
    _remove_looping_blocks,
    _generate_with_timeout,
)
from .pipeline_models import (
    generate_with_model_a,
    generate_with_model_b,
    generate_with_model_c,
    _load_model_c,
)

logger = logging.getLogger(__name__)


class RecursiveModelPipeline:
    """
    Пайплайн для последовательной работы GGUF моделей:
    1. Model A (Qwen 2.5 3B) - даёт краткий логичный ответ
    2. Model B (Qwen 2.5 3B) - развивает мысль, добавляет детали
    3. Model C (Qwen 2.5 Coder 1.5B) - генерирует код, если нужен
    
    Использует create_chat_completion с автоматическим форматированием Qwen
    """
    
    MODEL_A_MAX_TOKENS = 1024
    MODEL_A_TEMPERATURE = 0.3
    MODEL_A_TOP_P = 0.9
    MODEL_A_TOP_K = 40
    MODEL_A_REPEAT_PENALTY = 1.5
    
    MODEL_B_MAX_TOKENS = 512
    MODEL_B_TEMPERATURE = 0.3
    MODEL_B_TOP_P = 0.9
    MODEL_B_TOP_K = 40
    MODEL_B_REPEAT_PENALTY = 2.0
    
    MODEL_C_MAX_TOKENS = 512
    MODEL_C_TEMPERATURE = 0.1
    MODEL_C_TOP_P = 0.9
    MODEL_C_TOP_K = 50
    MODEL_C_REPEAT_PENALTY = 1.3
    
    STOP_TOKENS = ["</s>"]
    
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
    
    def process_query(
        self,
        query: str,
        max_iterations: int = 1,
        gen_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Основной метод обработки запроса через 3-модельный пайплайн"""
        return self._process_query_with_timeout(query, max_iterations, gen_params)

    def _process_query_with_timeout(
        self,
        query: str,
        max_iterations: int = 1,
        gen_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Wrapper with 60s global timeout"""
        result_holder = {'done': False, 'result': None}

        def _run():
            result_holder['result'] = self._process_query_impl(query, max_iterations, gen_params)
            result_holder['done'] = True

        worker = threading.Thread(target=_run)
        worker.daemon = True
        worker.start()
        worker.join(timeout=60)

        if not result_holder['done']:
            logger.error("process_query: global timeout (60s)")
            return {
                'query': query,
                'response': 'Ответ не успел сгенерироваться за 60 секунд.',
                'final_response': 'Ответ не успел сгенерироваться за 60 секунд.',
                'status': 'timeout',
                'model_a_result': None,
                'model_b_result': None,
                'model_c_result': None,
                'reasoning_steps': [],
                'has_code': False,
                'fractal_context': None,
                'final_quality': {'is_gibberish': False, 'score': 0.0, 'reasons': ['Таймаут пайплайна']},
            }

        return result_holder['result']

    def _process_query_impl(
        self,
        query: str,
        max_iterations: int = 1,
        gen_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Implementation of query processing (called within timeout wrapper)"""
        if not self.model_a or not self.model_b:
            raise RuntimeError("Модели не загружены. Вызовите load_models()")
        
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
            logger.info(f"Применены динамические параметры: A(temp={a_temperature}), B(temp={b_temperature})")
        
        results = {
            'query': query,
            'model_a_result': None,
            'model_b_result': None,
            'model_c_result': None,
            'reasoning_steps': [],
            'has_code': False,
            'fractal_context': None
        }
        
        enriched_query = query
        if self.fractal_memory and hasattr(self.fractal_memory, 'get_context_for_query'):
            skip_context_keywords = ['привет', 'здравствуй', 'добрый', 'хай', 'hello', 'hi', 'hey', 'пока', 'до свидания', 'спасибо', 'благодар']
            query_lower = query.lower().strip()
            is_greeting = any(kw in query_lower for kw in skip_context_keywords) and len(query) < 60
            
            if not is_greeting:
                graph_context = self.fractal_memory.get_context_for_query(query)
                if graph_context:
                    enriched_query = f"{query}\n\nКонтекст из опыта:\n{graph_context}"
                    results['fractal_context'] = graph_context
                    logger.info(f"Контекст из графа обучения: {len(graph_context)} символов")
        
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
        
        results['final_response'] = model_b_result['natural_response']
        
        if not model_b_result.get('natural_response') or model_b_result['quality'].get('is_gibberish'):
            logger.warning("Model B failed, falling back to Model A response")
            results['final_response'] = model_a_result.get('natural_response', '')
        
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
            
            results['final_response'] = model_b_result['natural_response'] + "\n\n" + model_c_result['natural_response']
        else:
            results['final_response'] = model_b_result['natural_response']
        
        results['final_quality'] = model_b_result['quality']
        
        if self.fractal_memory and hasattr(self.fractal_memory, 'save_experience'):
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
        
        # Добавляем алиас 'response' для совместимости с brain_query.py
        results['response'] = results.get('final_response', '')
        
        return results
    
    def unload_models(self):
        """Выгружает модели и освобождает ресурсы."""
        self.model_a = None
        self.model_b = None
        self.model_c = None
        self.model_a_params.cleanup()
        self.model_b_params.cleanup()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Модели выгружены, ресурсы освобождены")
    
    def __del__(self):
        """Деструктор - освобождает ресурсы."""
        try:
            self.unload_models()
        except Exception:
            pass


def create_recursive_pipeline(
    model_a_path: str = None,
    model_b_path: str = None,
    model_c_path: str = None,
    n_ctx: int = 8192,
    n_threads: int = 8,
    fractal_memory = None
) -> 'RecursiveModelPipeline':
    """Фабричная функция для создания пайплайна"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if model_a_path is None:
        model_a_path = os.path.join(project_root, "eva", "memory", "fractal_torch_storage", "gguf_models", "qwen2.5-3b-instruct", "qwen2.5-3b-instruct-q4_k_m.gguf")
    
    if model_b_path is None:
        model_b_path = os.path.join(project_root, "eva", "memory", "fractal_torch_storage", "gguf_models", "qwen2.5-3b-instruct", "qwen2.5-3b-instruct-q4_k_m.gguf")
    
    if model_c_path is None:
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


RecursiveModelPipeline.check_quality = check_quality
RecursiveModelPipeline._sanitize_response = _sanitize_response
RecursiveModelPipeline._clean_filler_start = _clean_filler_start
RecursiveModelPipeline._remove_looping_blocks = _remove_looping_blocks
RecursiveModelPipeline._generate_with_timeout = _generate_with_timeout
RecursiveModelPipeline.generate_with_model_a = generate_with_model_a
RecursiveModelPipeline.generate_with_model_b = generate_with_model_b
RecursiveModelPipeline.generate_with_model_c = generate_with_model_c
RecursiveModelPipeline._load_model_c = _load_model_c
