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
    _unload_model_c,
    _generate_response,
)
from .resource_manager import ResourceManager
from .contradiction_resolver import ContradictionResolver
from .knowledge_rollback import KnowledgeRollback
from .event_bus import Event

logger = logging.getLogger(__name__)


class RecursiveModelPipeline:
    """
    Пайплайн для последовательной работы GGUF моделей:
    1. Model A (Qwen 2.5 3B) - даёт краткий логичный ответ
    2. Model B (Qwen 2.5 3B) - развивает мысль, добавляет детали
    3. Model C (Qwen 2.5 Coder 1.5B) - генерирует код, если нужен
    
    Использует create_chat_completion с автоматическим форматированием Qwen
    """
    
    MODEL_A_MAX_TOKENS = 4096
    MODEL_A_TEMPERATURE = 0.30
    
    MODEL_B_MAX_TOKENS = 4096
    MODEL_B_TEMPERATURE = 0.45
    
    MODEL_C_MAX_TOKENS = 4096
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
        n_ctx: int = 16384,
        n_threads: int = None,  # None = испольовать все ядра (12 для i5-12450H)
        fractal_memory = None,
        event_bus = None,
        resource_manager = None,
        attention_system = None
    ):
        self.model_a_path = model_a_path
        self.model_b_path = model_b_path
        self.model_c_path = model_c_path
        self.n_ctx = n_ctx
        self.context_size = n_ctx  # Alias for compatibility
        self.n_threads = n_threads
        self.model_a = None
        self.model_b = None
        self.model_c = None
        self.fractal_memory = fractal_memory
        self.quality_checker = None
        self.event_bus = event_bus
        self.resource_manager = resource_manager
        self.attention_system = attention_system
        
        self.contradiction_resolver = None
        self.knowledge_rollback = None
        if self.attention_system:
            self.contradiction_resolver = ContradictionResolver(self.attention_system)
            self.knowledge_rollback = KnowledgeRollback(
                event_bus=self.event_bus
            )
        
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
        
        try:
            self.resource_manager = ResourceManager()
            self.resource_manager.start_monitoring(interval=10.0)
        except Exception:
            self.resource_manager = None
        
        logger.info(f"RecursiveModelPipeline инициализирован (3-модельный, адаптивные параметры)")
        
        self.ethics_framework = None
        self._init_ethics_framework()

    def _init_ethics_framework(self):
        try:
            from eva_ai.ethics.ethics_core import EthicsFramework
            self.ethics_framework = EthicsFramework(brain=None, event_bus=self.event_bus)
            logger.info("EthicsFramework инициализирован")
        except ImportError as e:
            logger.warning(f"EthicsFramework не найден: {e}")
            self.ethics_framework = None

    def _check_ethics(self, query: str) -> Dict[str, Any]:
        if not self.ethics_framework:
            return {'allowed': True, 'reason': 'Ethics not available'}
        
        try:
            result = self.ethics_framework.assess_ethics({'query': query})
            decision = result.decision if hasattr(result, 'decision') else 'allow'
            is_high_risk = hasattr(result, 'requires_human_review') and result.requires_human_review
            return {
                'allowed': decision != 'block',
                'reason': result.justification if hasattr(result, 'justification') else '',
                'risk_level': 'high' if is_high_risk else 'low'
            }
        except Exception as e:
            logger.error(f"Ошибка ethical check: {e}")
            return {'allowed': True, 'reason': f'Check error: {e}'}
    
    def _publish_event(self, event_type: str, data: Dict):
        if self.event_bus:
            try:
                self.event_bus.publish(event_type, data)
            except Exception:
                pass
    
    def _get_adaptive_generation_params(self):
        """Получить адаптированные параметры генерации"""
        if not self.resource_manager:
            return {
                'max_tokens': self.MODEL_A_MAX_TOKENS,
                'temperature': self.MODEL_A_TEMPERATURE
            }
        
        recommended_ctx = self.resource_manager.get_recommended_context_size()
        
        if recommended_ctx <= 4096:
            return {'max_tokens': 4096, 'temperature': 0.3}
        elif recommended_ctx <= 8192:
            return {'max_tokens': 4096, 'temperature': 0.4}
        else:
            return {'max_tokens': 4096, 'temperature': 0.5}
    
    def load_models(self):
        """Загрузка GGUF моделей - Model A и B как отдельные экземпляры"""
        if self.resource_manager:
            recommended_ctx = self.resource_manager.get_recommended_context_size()
        else:
            recommended_ctx = 16384
        
        a_ctx = min(recommended_ctx, 16384)
        
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
    
    def _review_with_model_a(
        self,
        original_query: str,
        model_b_response: str,
        model_a_facts: str
    ) -> Dict[str, Any]:
        """Model A проверяет и исправляет ответ Model B"""
        if not self.model_a or not model_b_response:
            return {'improved': False, 'changes': [], 'corrected_response': model_b_response}
        
        changes = []
        
        # Проверка на иностранные символы
        foreign_chars = re.findall(r'[a-zA-Z\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', model_b_response)
        if foreign_chars:
            unique_foreign = set(foreign_chars)
            changes.append(f"Найдены иностранные символы: {len(unique_foreign)} уникальных")
            
            # Исправление - заменяем на русские аналоги или удаляем
            corrected = model_b_response
            # Удаляем китайские и японские символы
            corrected = re.sub(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', '', corrected)
            # Оставляем латиницу в коде, но удаляем в тексте
            corrected = re.sub(r'(?<![a-zA-Z_])([a-zA-Z]+)(?![a-zA-Z_(])', '', corrected)
            
            return {
                'improved': True,
                'changes': changes,
                'corrected_response': corrected
            }
        
        # Проверка на грамматические ошибки (базовая)
        # Проверка повторяющихся слов
        words = model_b_response.split()
        word_counts = {}
        for word in words:
            word_lower = word.lower().strip('.,!?;:"()[]')
            if len(word_lower) > 3:
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        repeated = [(w, c) for w, c in word_counts.items() if c > 3]
        if repeated:
            changes.append(f"Повторяющиеся слова: {repeated[:3]}")
        
        # Проверка на "мусорные" паттерны
        garbage_patterns = [
            r'\{[^}]{100,}\}',  # слишком длинные JSON-подобные конструкции
            r'\.{10,}',          # слишком много точек подряд
            r'-{5,}',            # слишком много дефисов
        ]
        for pattern in garbage_patterns:
            if re.search(pattern, model_b_response):
                changes.append(f"Найден мусорный паттерн: {pattern}")
                model_b_response = re.sub(pattern, '', model_b_response)
        
        # Если есть изменения, перегенерируем с Model A
        if changes and self.model_a:
            logger.info(f"Model A исправляет {len(changes)} проблем в ответе B")
            
            review_prompt = (
                f"Проверь и исправь следующий ответ. Исправь ошибки, грамматику, форматирование.\n"
                f"Оригинальный запрос: {original_query}\n"
                f"Известные факты: {model_a_facts[:500]}\n"
                f"Ответ для проверки: {model_b_response}\n\n"
                f"Инструкции:\n"
                f"1. Сохрани смысл ответа\n"
                f"2. Исправь грамматические ошибки\n"
                f"3. Убери лишние повторения\n"
                f"4. Используй норм русский язык\n"
                f"5. Отвечай строго на русском\n\n"
                f"Верни только исправленный ответ без комментариев."
            )
            
            try:
                from .pipeline_models import _generate_response
                corrected = _generate_response(
                    self.model_a,
                    review_prompt,
                    max_tokens=len(model_b_response) * 2,
                    temperature=0.3,
                    max_context=self.context_size
                )
                if corrected and len(corrected) > 50:
                    return {
                        'improved': True,
                        'changes': changes,
                        'corrected_response': corrected
                    }
            except Exception as e:
                logger.debug(f"Error in Model A review: {e}")
        
        return {
            'improved': len(changes) > 0,
            'changes': changes,
            'corrected_response': model_b_response
        }
    
    def process_query(
        self,
        query: str,
        max_iterations: int = 1,
        gen_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Основной метод обработки запроса через 3-модельный пайплайн"""
        return self._process_query_impl(query, max_iterations, gen_params)

    def _process_query_impl(
        self,
        query: str,
        max_iterations: int = 1,
        gen_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Implementation of query processing"""
        pipeline_start = time.time()
        self._publish_event('pipeline.start', {'query': query[:100]})
        try:
            return self._process_query_impl_inner(query, max_iterations, gen_params, pipeline_start)
        except Exception as e:
            self._publish_event('pipeline.failed', {'error': str(e)})
            raise

    def _process_query_impl_inner(
        self,
        query: str,
        max_iterations: int = 1,
        gen_params: Dict[str, Any] = None,
        pipeline_start: float = None
    ) -> Dict[str, Any]:
        """Inner implementation of query processing."""
        if not self.model_a or not self.model_b:
            raise RuntimeError("Модели не загружены. Вызовите load_models()")
        
        ethics_result = self._check_ethics(query)
        if not ethics_result.get('allowed'):
            logger.warning(f"Запрос отклонён по этическим причинам: {ethics_result.get('reason')}")
            if self.event_bus:
                self.event_bus.publish(Event(
                    event_type='pipeline.ethics.blocked',
                    source='pipeline',
                    data={'query': query[:100], 'reason': ethics_result.get('reason')}
                ))
            return {
                'response': 'Извините, я не могу ответить на этот запрос.',
                'final_response': 'Извините, я не могу ответить на этот запрос.',
                'ethics_blocked': True,
                'block_reason': ethics_result.get('reason'),
                'query': query,
                'model_a_result': None,
                'model_b_result': None,
                'model_c_result': None,
                'reasoning_steps': [],
                'has_code': False,
                'fractal_context': None,
                'final_quality': {'is_gibberish': False, 'score': 0.0, 'reasons': [' ethics_blocked']}
            }
        
        if self.event_bus:
            self.event_bus.publish(Event(
                event_type='pipeline.ethics.check_complete',
                source='pipeline',
                data={'query': query[:100], 'allowed': ethics_result.get('allowed')}
            ))
        
        a_max_tokens = self.MODEL_A_MAX_TOKENS
        a_temperature = self.MODEL_A_TEMPERATURE
        b_max_tokens = self.MODEL_B_MAX_TOKENS
        b_temperature = self.MODEL_B_TEMPERATURE
        
        resource_usage = {'cpu': 0.0, 'ram': 0.0}
        if self.resource_manager:
            try:
                resource_usage = {
                    'cpu': self.resource_manager.get_cpu_usage(),
                    'ram': self.resource_manager.get_memory_usage()
                }
                logger.debug(f"Resource usage before generation: CPU={resource_usage['cpu']:.2f}, RAM={resource_usage['ram']:.2f}")
            except Exception as e:
                logger.debug(f"Error getting resource usage: {e}")
        
        adapted_params_a = self.model_a_params.adapt_to_resources(resource_usage)
        adapted_params_b = self.model_b_params.adapt_to_resources(resource_usage)
        
        if adapted_params_a.get('max_tokens'):
            a_max_tokens = adapted_params_a['max_tokens']
        if adapted_params_a.get('temperature'):
            a_temperature = adapted_params_a['temperature']
        if adapted_params_b.get('max_tokens'):
            b_max_tokens = adapted_params_b['max_tokens']
        if adapted_params_b.get('temperature'):
            b_temperature = adapted_params_b['temperature']
        
        skip_model_c = self.model_a_params.should_skip_model_c(resource_usage)
        if skip_model_c:
            logger.info(f"Skipping Model C due to high resource usage: CPU={resource_usage['cpu']:.2f}, RAM={resource_usage['ram']:.2f}")
        
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
        self._publish_event('pipeline.model_a.start', {'query_length': len(enriched_query)})
        model_a_start = time.time()
        model_a_result = self.generate_with_model_a(enriched_query)
        model_a_elapsed = time.time() - model_a_start
        
        # Проверяем: если генерация ещё идёт (status='generating'), используем Model B
        if model_a_result.get('status') == 'generating':
            logger.warning(f"Model A: генерация продолжается ({model_a_elapsed:.1f}с), используем Model B")
            # Переходим сразу к Model B с оригинальным запросом
            model_a_result = {
                'natural_response': '',
                'quality': {'is_gibberish': True, 'score': 0.0, 'reasons': ['Model A ещё генерирует, переход к Model B']},
                'status': 'fallback_to_b'
            }
        
        logger.info(f"Model A ответ: {model_a_result['natural_response'][:150]}...")
        results['model_a_result'] = model_a_result
        self._publish_event('pipeline.model_a.complete', {
            'response_length': len(model_a_result.get('natural_response', '')),
            'quality': model_a_result['quality'].get('score', 0.0),
            'time': round(model_a_elapsed, 2)
        })
        
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
        self._publish_event('pipeline.model_b.start', {'query_length': len(query), 'model_a_response_length': len(model_a_result.get('natural_response', ''))})
        model_b_start = time.time()
        model_b_result = self.generate_with_model_b(query, model_a_result.get('natural_response', ''))
        model_b_elapsed = time.time() - model_b_start
        logger.info(f"Model B ответ: {model_b_result['natural_response'][:150]}...")
        results['model_b_result'] = model_b_result
        self._publish_event('pipeline.model_b.complete', {
            'response_length': len(model_b_result.get('natural_response', '')),
            'quality': model_b_result['quality'].get('score', 0.0),
            'time': round(model_b_elapsed, 2)
        })

        # === Проверка и перегенерация Model B при низком качестве ===
        model_b_quality = model_b_result.get('quality', {}).get('score', 0.0)
        model_b_needs_regen = model_b_quality < 0.5 or model_b_result.get('quality', {}).get('is_gibberish', False)
        
        if model_b_needs_regen:
            logger.warning(f"Model B quality low ({model_b_quality:.2f}), attempting regeneration...")
            self._publish_event('pipeline.model_b.regen.start', {'attempt': 2, 'previous_quality': model_b_quality})
            
            regen_result = self.generate_with_model_b(
                query, 
                model_a_result.get('natural_response', ''),
                max_retries=1
            )
            
            regen_quality = regen_result.get('quality', {}).get('score', 0.0)
            logger.info(f"Model B регенерация: quality={regen_quality:.2f}")
            
            if regen_quality > model_b_quality:
                model_b_result = regen_result
                model_b_elapsed = time.time() - model_b_start
                logger.info(f"Model B улучшен до {regen_quality:.2f}")
                
                self._publish_event('pipeline.model_b.regen.complete', {'new_quality': regen_quality})
        
        # === Model A проверяет и исправляет ответ Model B ===
        logger.info("=== Шаг 2.5: Model A проверяет ответ Model B ===")
        self._publish_event('pipeline.model_b.review.start', {'model_b_length': len(model_b_result.get('natural_response', ''))})
        
        review_result = self._review_with_model_a(
            original_query=query,
            model_b_response=model_b_result.get('natural_response', ''),
            model_a_facts=model_a_result.get('natural_response', '')
        )
        
        if review_result.get('improved'):
            logger.info(f"Model A исправила ответ B: {review_result.get('changes', [])}")
            model_b_result['natural_response'] = review_result.get('corrected_response', model_b_result['natural_response'])
            model_b_result['review_applied'] = True
            model_b_result['review_changes'] = review_result.get('changes', [])
        
        self._publish_event('pipeline.model_b.review.complete', {
            'improved': review_result.get('improved', False),
            'changes_count': len(review_result.get('changes', []))
        })
        
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
        
        if self.model_c and self._is_code_request(query) and not skip_model_c:
            logger.info("=== Шаг 3: Генерация кода на Model C (Coder) ===")
            results['has_code'] = True
            self._publish_event('pipeline.model_c.start', {'query_length': len(query), 'context_length': len(model_b_result.get('natural_response', ''))})
            model_c_start = time.time()
            model_c_result = self.generate_with_model_c(query, model_b_result['natural_response'])
            model_c_elapsed = time.time() - model_c_start
            logger.info(f"Model C ответ: {model_c_result['natural_response'][:150]}...")
            results['model_c_result'] = model_c_result
            self._publish_event('pipeline.model_c.complete', {
                'response_length': len(model_c_result.get('natural_response', '')),
                'quality': model_c_result['quality'].get('score', 0.0),
                'time': round(model_c_elapsed, 2)
            })
            
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
        
        # Выгружаем Model C после генерации кода (экономим ~1GB RAM)
        if results.get('has_code'):
            self._unload_model_c()
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
            
            if self.contradiction_resolver:
                contradictions = self.contradiction_resolver.check_response(query, model_b_result['natural_response'])
                if contradictions and self.knowledge_rollback:
                    try:
                        if hasattr(self.fractal_memory, 'learning_loop') and self.fractal_memory.learning_loop:
                            self.knowledge_rollback.set_graph_learning(self.fractal_memory.learning_loop)
                            for contr in contradictions:
                                if 'confidence' in contr and contr['confidence'] > 0.5:
                                    exp_id = f"exp_{hash(query) % 1000000}"
                                    self.knowledge_rollback.rollback_knowledge(
                                        exp_id,
                                        f"Противоречие в ответе: {contr.get('indicator', 'unknown')}"
                                    )
                    except Exception as e:
                        logger.debug(f"Ошибка проверки противоречий: {e}")
        
        logger.info("Three-GGUF пайплайн завершён")
        
        total_elapsed = time.time() - pipeline_start
        self._publish_event('pipeline.complete', {
            'total_time': round(total_elapsed, 2),
            'final_quality': results.get('final_quality', {}).get('score', 0.0)
        })
        
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
    n_ctx: int = 16384,
    n_threads: int = None,  # None = испольовать все ядра (12 для i5-12450H)
    fractal_memory = None,
    event_bus = None,
    resource_manager = None,
    attention_system = None
) -> 'RecursiveModelPipeline':
    """Фабричная функция для создания пайплайна"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if model_a_path is None:
        model_a_path = os.path.join(project_root, "eva_ai", "memory", "fractal_torch_storage", "gguf_models", "qwen2.5-3b-instruct", "qwen2.5-3b-instruct-q4_k_m.gguf")
    
    if model_b_path is None:
        model_b_path = os.path.join(project_root, "eva_ai", "memory", "fractal_torch_storage", "gguf_models", "qwen2.5-3b-instruct", "qwen2.5-3b-instruct-q4_k_m.gguf")
    
    if model_c_path is None:
        model_c_path = os.path.join(project_root, "eva_ai", "memory", "fractal_torch_storage", "gguf_models", "qwen2.5-coder-1.5b-instruct", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
    
    pipeline = RecursiveModelPipeline(
        model_a_path=model_a_path,
        model_b_path=model_b_path,
        model_c_path=model_c_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        fractal_memory=fractal_memory,
        event_bus=event_bus,
        resource_manager=resource_manager,
        attention_system=attention_system
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
RecursiveModelPipeline._unload_model_c = _unload_model_c
