"""
Enhanced Reasoning Engine - главный движок рассуждения с регенерацией
Query -> Qwen -> Analytics -> 3 Modules (parallel) -> Composer -> Qwen -> Stability + Metric check
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from .analytics_module import AnalyticsModule, AnalyticsResult
from .prompt_composer import PromptComposer, ComposedPrompt
from .semantic_stability import SemanticStabilityChecker, StabilityResult
from .combined_metric import CombinedMetricCalculator, ImprovementResult
from .entity_extractor import EntityExtractor, ExtractedEntity
from .correlation_calculator import CorrelationCalculator, CorrelationResult


@dataclass
class ReasoningIteration:
    """Одна итерация рассуждения"""
    iteration: int
    query: str
    response: str
    analytics_result: Optional[AnalyticsResult] = None
    contradiction_result: Optional[Dict] = None
    ethics_result: Optional[Dict] = None
    websearch_result: Optional[Dict] = None
    stability_result: Optional[StabilityResult] = None
    improvement_result: Optional[ImprovementResult] = None
    confidence: float = 0.0
    is_final: bool = False
    module_prompts: Dict[str, str] = field(default_factory=dict)


class EnhancedReasoningEngine:
    """
    Улучшенный движок рассуждения с полным циклом регенерации
    
    Workflow:
    1. Query -> Qwen (initial response)
    2. Analytics module extracts logical components
    3. Three parallel modules process components:
       - Contradiction module: check against query, knowledge, history
       - Ethics module: check query and response
       - Web search module: enrich context
    4. PromptComposer combines all module prompts
    5. Combined prompt -> Qwen (regeneration)
    6. Stopping criteria:
       - Semantic stability (similarity > 0.95)
       - Combined improvement metric
    7. Self-learning: entities from contradictions saved to knowledge graph
    """
    
    def __init__(self, brain=None, config: Optional[Dict[str, Any]] = None):
        self.brain = brain
        self.config = config or {}
        
        # Параметры
        self.max_iterations = self.config.get('max_iterations', 5)
        self.stability_threshold = self.config.get('stability_threshold', 0.95)
        self.improvement_threshold = self.config.get('improvement_threshold', 0.05)
        self.min_confidence = self.config.get('min_confidence', 0.7)
        
        # Инициализируем модули
        self.analytics = AnalyticsModule(brain)
        self.composer = PromptComposer(brain)
        self.stability_checker = SemanticStabilityChecker(brain)
        self.combined_metric = CombinedMetricCalculator(brain)
        self.entity_extractor = EntityExtractor(brain)
        self.correlation_checker = CorrelationCalculator(brain)
        
        # Модули из brain (если доступны)
        self.contradiction_manager = None
        self.ethics_framework = None
        self.web_search = None
        self._init_modules_from_brain()
        
        # История итераций
        self.iterations: List[ReasoningIteration] = []
        
        # Qwen для генерации ответов (GPU)
        self._qwen = None
        
        # Fractal Qwen DISABLED - using UnifiedGenerator
        self._fractal_qwen = None
        self._use_fractal_for_prompts = False
        
        logger.info(f"EnhancedReasoningEngine initialized: max_iterations={self.max_iterations}, fractal_qwen={self._use_fractal_for_prompts}")
    
    def _init_modules_from_brain(self):
        """Инициализирует модули из brain"""
        if self.brain is None:
            return
        
        self.contradiction_manager = getattr(self.brain, 'contradiction_manager', None)
        self.ethics_framework = getattr(self.brain, 'ethics_framework', None)
        self.web_search = getattr(self.brain, 'web_search_engine', None)
        
        if self.contradiction_manager:
            logger.info("ContradictionManager connected")
        if self.ethics_framework:
            logger.info("EthicsFramework connected")
        if self.web_search:
            logger.info("WebSearch connected")
    
    def _get_qwen(self):
        """Получает экземпляр генератора - используем UnifiedGenerator (two_model_pipeline)"""
        if self._qwen is not None:
            return self._qwen
        
        # Используем UnifiedGenerator через two_model_pipeline
        try:
            if self.brain is not None:
                # Проверяем two_model_pipeline (UnifiedGenerator через PipelineAdapter)
                pipeline = getattr(self.brain, 'two_model_pipeline', None)
                if pipeline and getattr(self.brain, 'two_model_pipeline_ready', False):
                    logger.info("Using UnifiedGenerator (two_model_pipeline) for generation")
                    self._qwen = pipeline
                    return self._qwen
        except Exception as e:
            logger.debug(f"Cannot get UnifiedGenerator from brain: {e}")
        
        # Fallback: пробуем старые методы
        try:
            if self.brain is not None:
                self._qwen = getattr(self.brain, 'qwen_model_manager', None)
                if self._qwen and getattr(self._qwen, 'initialized', False):
                    return self._qwen
        except Exception as e:
            logger.debug(f"Cannot get Qwen from brain: {e}")
        
        logger.warning("No generation model available (UnifiedGenerator not ready)")
        return None
    
    def _get_fractal_qwen(self):
        """Получает экземпляр Fractal Qwen для генерации промтов"""
        if self._fractal_qwen is not None:
            return self._fractal_qwen
        
        if not self._use_fractal_for_prompts:
            return None
        
        try:
            from eva_ai.mlearning.fractal_qwen_manager import get_fractal_qwen
            self._fractal_qwen = get_fractal_qwen(device="cpu")
            if self._fractal_qwen and self._fractal_qwen.initialized:
                logger.info("Fractal Qwen initialized for prompt generation")
                return self._fractal_qwen
            else:
                logger.warning("Fractal Qwen not initialized")
                return None
        except Exception as e:
            logger.debug(f"Cannot get Fractal Qwen: {e}")
            return None
    
    async def process_query_async(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        knowledge_context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Обрабатывает запрос через расширенный цикл рассуждения
        
        Args:
            query: Запрос пользователя
            conversation_history: История диалогов
            knowledge_context: Контекст из knowledge graph
            
        Returns:
            Dict с response, iterations, confidence
        """
        start_time = time.time()
        
        logger.info(f"Starting enhanced reasoning for: {query[:50]}...")
        
        # Очищаем историю итераций
        self.iterations = []
        
        # Шаг 1: Генерируем начальный ответ через Qwen (GPU)
        original_response = await self._generate_response(query, conversation_history)
        
        if not original_response:
            return {
                'response': 'Не удалось сгенерировать ответ',
                'status': 'error',
                'iterations': 0,
                'confidence': 0.0
            }
        
        # Сохраняем оригинальный ответ для корреляции
        self._original_response = original_response
        
        # Основной цикл регенерации с проверкой корреляции
        iteration = 0
        current_response = original_response
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"=== Iteration {iteration}/{self.max_iterations} ===")
            
            # Шаг 2: Анализируем текущий ответ через Analytics
            analytics_result = self.analytics.analyze_text(current_response)
            
            # Шаг 3: Запускаем модули параллельно для генерации промтов
            contr_result, ethics_result, web_result = await self._run_modules_parallel(
                query=query,
                response=current_response,
                conversation_history=conversation_history,
                knowledge_context=knowledge_context,
                iteration=iteration
            )
            
            # Шаг 4: Каждый модуль генерирует промт через Fractal Qwen (CPU)
            module_prompts = await self._generate_module_prompts(
                query=query,
                response=current_response,
                conversation_history=conversation_history,
                contr_result=contr_result,
                ethics_result=ethics_result,
                web_result=web_result
            )
            
            # Шаг 5: Объединяем все промты в один контекст
            combined_context = self._combine_prompts_context(
                query=query,
                original_response=self._original_response,
                current_response=current_response,
                module_prompts=module_prompts,
                conversation_history=conversation_history
            )
            
            # Шаг 6: Генерируем улучшенный ответ через Qwen (GPU)
            refined_response = await self._generate_with_context(
                combined_context, 
                query
            )
            
            if not refined_response:
                logger.warning("Не удалось сгенерировать улучшенный ответ, используем текущий")
                refined_response = current_response
            
            # Шаг 7: Проверяем корреляцию и релевантность
            correlation_result = self.correlation_checker.check_correlation(
                original_response=self._original_response,
                refined_response=refined_response,
                knowledge_context=knowledge_context,
                web_context=web_result.get('summary') if web_result else None,
                contradiction_score=1.0 - contr_result.get('contradiction_level', 0.0) if contr_result else 0.5,
                ethics_score=ethics_result.get('is_ethical', 1.0) if ethics_result else 1.0
            )
            
            logger.info(
                f"Correlation: score={correlation_result.correlation_score:.3f}, "
                f"acceptable={correlation_result.is_acceptable}"
            )
            
            # Шаг 8: Проверяем стабильность
            stability_result = None
            if current_response != refined_response:
                stability_result = self.stability_checker.analyze_changes(
                    current_response, refined_response
                )
                logger.info(f"Stability: similarity={stability_result.similarity:.3f}")
            
            # Рассчитываем confidence с учётом корреляции
            confidence = self._calculate_confidence(
                ethics_result, contr_result, analytics_result, iteration
            )
            confidence = (confidence + correlation_result.correlation_score) / 2
            
            # Сохраняем итерацию
            iter_data = ReasoningIteration(
                iteration=iteration,
                query=query,
                response=refined_response,
                analytics_result=analytics_result,
                contradiction_result=contr_result,
                ethics_result=ethics_result,
                websearch_result=web_result,
                stability_result=stability_result,
                improvement_result=None,
                confidence=confidence,
                module_prompts=module_prompts
            )
            self.iterations.append(iter_data)
            
            # Шаг 9: Проверяем критерии остановки
            should_stop = self._should_stop(
                stability_result=stability_result,
                improvement_result=None,
                iteration=iteration,
                confidence=confidence
            )
            
            # Если корреляция приемлемая - завершаем
            if correlation_result.is_acceptable and stability_result and stability_result.is_stable:
                logger.info(f"Stopping: correlation acceptable and stable")
                iter_data.is_final = True
                current_response = refined_response
                break
            
            if should_stop:
                logger.info(f"Stopping at iteration {iteration}: criteria met")
                iter_data.is_final = True
                break
            
            # Продолжаем цикл с улучшенным ответом
            current_response = refined_response
        
        # Шаг 8: Самообучение - извлекаем и сохраняем сущности
        self._perform_self_learning(query, current_response)
        
        processing_time = time.time() - start_time
        
        final_result = {
            'response': current_response,
            'status': 'ok',
            'iterations': iteration,
            'confidence': confidence,
            'processing_time': processing_time,
            'reasoning_chain': [self._iteration_to_dict(it) for it in self.iterations]
        }
        
        logger.info(f"Enhanced reasoning completed: {iteration} iterations, "
                   f"confidence={confidence:.2f}, time={processing_time:.2f}s")
        
        return final_result
    
    def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        knowledge_context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Синхронная версия process_query_async"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.process_query_async(query, conversation_history, knowledge_context)
        )
    
    async def _run_modules_parallel(
        self,
        query: str,
        response: str,
        conversation_history: Optional[List[Dict]],
        knowledge_context: Optional[List[str]],
        iteration: int
    ) -> tuple:
        """Запускает три модуля параллельно"""
        
        async def run_contradiction():
            if self.contradiction_manager and hasattr(self.contradiction_manager, 'check_with_context'):
                return await asyncio.to_thread(
                    self.contradiction_manager.check_with_context,
                    text=response,
                    query=query,
                    conversation_history=conversation_history,
                    knowledge_context=knowledge_context
                )
            return {'contradictions': [], 'has_conflicts': False}
        
        async def run_ethics():
            if self.ethics_framework and hasattr(self.ethics_framework, 'check_with_context'):
                return await asyncio.to_thread(
                    self.ethics_framework.check_with_context,
                    text=response,
                    query=query
                )
            return {'violations': [], 'has_violations': False, 'overall_score': 1.0}
        
        async def run_websearch():
            if self.web_search and hasattr(self.web_search, 'enrich_with_context'):
                return await asyncio.to_thread(
                    self.web_search.enrich_with_context,
                    query=query,
                    response=response
                )
            return {'success': False, 'context': ''}
        
        # Запускаем все три параллельно
        contr_task = asyncio.create_task(run_contradiction())
        ethics_task = asyncio.create_task(run_ethics())
        web_task = asyncio.create_task(run_websearch())
        
        # Ждём все задачи
        contr_result, ethics_result, web_result = await asyncio.gather(
            contr_task, ethics_task, web_task
        )
        
        return contr_result, ethics_result, web_result
    
    async def _generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Генерирует начальный ответ через UnifiedGenerator.
        Унифицированный API: всегда передаем строку prompt, получаем строку response.
        """
        generator = self._get_qwen()
        
        if generator is None:
            logger.warning("Generator not available, using fallback response")
            return self._fallback_response(query)
        
        # Формируем промпт с историей
        prompt = self._build_prompt(query, conversation_history)
        
        try:
            # Унифицированный вызов: всегда строка
            if hasattr(generator, 'generate'):
                result = generator.generate(
                    prompt,  # Строка!
                    max_tokens=2048,
                    temperature=0.7
                )
                # Обрабатываем результат (может быть строкой или объектом)
                if isinstance(result, str):
                    return result if result.strip() else self._fallback_response(query)
                elif hasattr(result, 'text'):
                    return result.text if result.text and result.text.strip() else self._fallback_response(query)
                else:
                    response_str = str(result)
                    return response_str if response_str.strip() else self._fallback_response(query)
            else:
                logger.warning("Generator has no generate method")
                return self._fallback_response(query)
        except Exception as e:
            logger.error(f"Generation error in _generate_response: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_response(query)
    
    async def _regenerate_response(
        self,
        query: str,
        previous_response: str,
        contradiction_result: Dict,
        ethics_result: Dict,
        websearch_result: Dict,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Генерирует улучшенный ответ через PromptComposer с учётом контекста"""
        
        # Формируем контекст из истории диалога
        context_text = ""
        if conversation_history and len(conversation_history) > 0:
            recent = conversation_history[-5:]  # Последние 5 сообщений
            context_parts = []
            for msg in recent:
                role = msg.get('role', 'user')
                content = msg.get('content', '')[:150]  # Ограничиваем длину
                label = 'Пользователь' if role == 'user' else 'ЕВА'
                context_parts.append(f"{label}: {content}")
            context_text = "История разговора:\n" + "\n".join(context_parts) + "\n\n"
        
        # Генерируем промпты от модулей
        contr_prompt = ""
        ethics_prompt = ""
        web_context = ""
        
        if self.contradiction_manager and hasattr(self.contradiction_manager, 'generate_refinement_prompt'):
            contr_prompt = self.contradiction_manager.generate_refinement_prompt(
                contradiction_result, query, previous_response
            )
        
        if self.ethics_framework and hasattr(self.ethics_framework, 'generate_regeneration_prompt'):
            ethics_prompt = self.ethics_framework.generate_regeneration_prompt(
                ethics_result, query, previous_response
            )
        
        if self.web_search and hasattr(self.web_search, 'format_enrichment_prompt'):
            web_context = self.web_search.format_enrichment_prompt(websearch_result)
        
        # Если нет промптов от модулей - возвращаем предыдущий ответ
        if not contr_prompt and not ethics_prompt and not web_context:
            logger.info("No refinement needed, returning previous response")
            return previous_response
        
        # Собираем feedback от модулей для Fractal Qwen
        module_feedback = {
            "contradiction": contr_prompt,
            "ethics": ethics_prompt,
            "websearch": web_context
        }
        
        # Пробуем использовать Fractal Qwen для генерации улучшенного промпта
        fractal_qwen = self._get_fractal_qwen()
        
        if fractal_qwen and fractal_qwen.initialized:
            logger.info("Using Fractal Qwen for prompt generation")
            
            try:
                improved_prompt = fractal_qwen.generate_prompt(
                    query=query,
                    previous_response=previous_response,
                    module_feedback=module_feedback,
                    max_tokens=512
                )
                
                logger.info(f"Fractal Qwen generated prompt: {improved_prompt[:100]}...")
                
                # Формируем финальный промпт с контекстом
                final_prompt = f"""{context_text}Ты - ЕВА, система искусственного интеллекта.

Текущий запрос: {query}

Предыдущий ответ: {previous_response}

Рекомендации по улучшению: {improved_prompt}

Дай улучшенный ответ с учётом контекста разговора и рекомендаций.

Улучшенный ответ:"""
                
            except Exception as e:
                logger.warning(f"Fractal Qwen prompt generation failed: {e}, using standard")
                final_prompt = None
        else:
            logger.info("Using standard PromptComposer")
            final_prompt = None
        
        # Если Fractal Qwen не использовался - компонуем стандартно с контекстом
        if final_prompt is None:
            composed = self.composer.compose(
                query=query,
                previous_response=previous_response,
                contradiction_prompt=contr_prompt,
                ethics_prompt=ethics_prompt,
                websearch_context=web_context
            )
            # Добавляем контекст к промпту
            final_prompt = context_text + composed.full_prompt if context_text else composed.full_prompt
        
        # Генерируем через UnifiedGenerator
        generator = self._get_qwen()
        if generator is None:
            logger.warning("Generator not available for regeneration")
            return previous_response
        
        try:
            result = generator.generate(
                final_prompt,  # Строка!
                max_tokens=1024,
                temperature=0.7
            )
            # Унифицированная обработка результата
            if isinstance(result, str):
                return result if result.strip() else previous_response
            elif hasattr(result, 'text'):
                return result.text if result.text and result.text.strip() else previous_response
            else:
                response_str = str(result)
                return response_str if response_str.strip() else previous_response
        except Exception as e:
            logger.error(f"Regeneration error: {e}")
            import traceback
            traceback.print_exc()
            return previous_response
    
    def _build_prompt(self, query: str, history: Optional[List[Dict]] = None) -> str:
        """Строит промпт с контекстом истории"""
        if not history:
            return query
        
        recent = history[-5:]
        context_parts = []
        for msg in recent:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            label = 'User' if role == 'user' else 'EVA'
            context_parts.append(f"{label}: {content[:200]}")
        
        return f"""Context:
{chr(10).join(context_parts)}

Question: {query}

Answer:"""
    
    async def _generate_module_prompts(
        self,
        query: str,
        response: str,
        conversation_history: Optional[List[Dict]],
        contr_result: Dict,
        ethics_result: Dict,
        web_result: Dict
    ) -> Dict[str, str]:
        """
        Генерирует промты от каждого модуля через Fractal Qwen (CPU).
        
        Returns:
            Dict с ключами: contradiction, ethics, websearch
        """
        prompts = {}
        
        fractal_qwen = self._get_fractal_qwen()
        
        if not fractal_qwen or not fractal_qwen.initialized:
            logger.warning("Fractal Qwen not available for module prompts")
            return prompts
        
        # Промт для модуля противоречий
        if contr_result:
            try:
                contr_feedback = f"""Проанализируй ответ на противоречия:
Ответ: {response[:500]}
Запрос: {query}

Определи противоречия и сформируй промт для их устранения."""
                
                prompts['contradiction'] = fractal_qwen.generate_prompt(
                    query=query,
                    previous_response=response,
                    module_feedback={'contradiction': contr_feedback},
                    max_tokens=256
                )
            except Exception as e:
                logger.warning(f"Contradiction prompt generation failed: {e}")
        
        # Промт для этического модуля
        if ethics_result:
            try:
                ethics_feedback = f"""Проанализируй ответ на этичность:
Ответ: {response[:500]}
Запрос: {query}

Определи этические проблемы и сформируй промт для их исправления."""
                
                prompts['ethics'] = fractal_qwen.generate_prompt(
                    query=query,
                    previous_response=response,
                    module_feedback={'ethics': ethics_feedback},
                    max_tokens=256
                )
            except Exception as e:
                logger.warning(f"Ethics prompt generation failed: {e}")
        
        # Промт для веб-поиска
        if web_result and web_result.get('summary'):
            try:
                web_feedback = f"""Обогати ответ веб-данными:
Ответ: {response[:500]}
Веб-контекст: {web_result.get('summary', '')[:300]}

Сформируй промт для добавления релевантной информации из веб-источников."""
                
                prompts['websearch'] = fractal_qwen.generate_prompt(
                    query=query,
                    previous_response=response,
                    module_feedback={'websearch': web_feedback},
                    max_tokens=256
                )
            except Exception as e:
                logger.warning(f"Web search prompt generation failed: {e}")
        
        return prompts
    
    def _combine_prompts_context(
        self,
        query: str,
        original_response: str,
        current_response: str,
        module_prompts: Dict[str, str],
        conversation_history: Optional[List[Dict]]
    ) -> str:
        """Объединяет все промты в один контекст для генерации."""
        
        # История диалога
        history_text = ""
        if conversation_history:
            recent = conversation_history[-5:]
            parts = []
            for msg in recent:
                role = 'Пользователь' if msg.get('role') == 'user' else 'ЕВА'
                content = msg.get('content', '')[:200]
                parts.append(f"{role}: {content}")
            history_text = "История разговора:\n" + "\n".join(parts) + "\n\n"
        
        # Объединяем промты модулей
        prompts_parts = []
        for module, prompt in module_prompts.items():
            if prompt:
                prompts_parts.append(f"[{module.upper()}]: {prompt}")
        
        prompts_text = "\n".join(prompts_parts) if prompts_parts else "Нет дополнительных рекомендаций"
        
        # Формируем системный промт
        combined = f"""{history_text}Ты - ЕВА, система искусственного интеллекта с внутренним рассуждением.

=== КОНТЕКСТ ===
Оригинальный запрос: {query}

Первый ответ (до рассуждения): {original_response[:300]}

Текущий ответ: {current_response[:500]}

=== РЕКОМЕНДАЦИИ ОТ МОДУЛЕЙ ===
{prompts_text}

=== ЗАДАНИЕ ===
На основе рекомендаций от модулей противоречий, этики и веб-поиска,
создай финальный улучшенный ответ.

Улучшенный ответ должен:
1. Устранить противоречия
2. Соответствовать этическим нормам
3. Включить релевантную информацию из веб-источников
4. Учитывать историю разговора

Финальный ответ:"""
        
        return combined
    
    async def _generate_with_context(self, context: str, query: str) -> str:
        """
        Генерирует ответ с объединённым контекстом через UnifiedGenerator.
        Унифицированный API: всегда передаем строку context.
        """
        generator = self._get_qwen()
        
        if generator is None:
            logger.warning("Generator not available for context generation")
            return ""
        
        try:
            result = generator.generate(
                context,  # Строка!
                max_tokens=1024,
                temperature=0.7
            )
            # Унифицированная обработка результата
            if isinstance(result, str):
                return result if result.strip() else ""
            elif hasattr(result, 'text'):
                return result.text if result.text and result.text.strip() else ""
            else:
                response_str = str(result)
                return response_str if response_str.strip() else ""
        except Exception as e:
            logger.error(f"Context generation error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _fallback_response(self, query: str) -> str:
        """Fallback ответ если Qwen недоступен"""
        return f"Я получил ваш запрос: '{query}'. Для полного ответа требуется доступ к модели."
    
    def _should_stop(
        self,
        stability_result: Optional[StabilityResult],
        improvement_result: Optional[ImprovementResult],
        iteration: int,
        confidence: float = 0.0
    ) -> bool:
        """Проверяет критерии остановки"""
        
        # Достигли максимума итераций
        if iteration >= self.max_iterations:
            logger.info(f"Stopping: max iterations ({self.max_iterations}) reached")
            return True
        
        # Confidence высокий + минимум 2 итерации ИЛИ стабильность
        if confidence >= self.min_confidence:
            if iteration >= 2:
                logger.info(f"Stopping: confidence {confidence:.2f} >= min {self.min_confidence} after {iteration} iterations")
                return True
            elif stability_result and stability_result.is_stable:
                logger.info(f"Stopping: confidence {confidence:.2f} high AND stable after {iteration} iterations")
                return True
        
        # Проверяем стабильность
        if stability_result and stability_result.is_stable:
            logger.info("Stopping: response is semantically stable")
            return True
        
        # Проверяем улучшение
        if improvement_result and not improvement_result.is_improved:
            logger.info("Stopping: no further improvement")
            return True
        
        return False
    
    def _calculate_confidence(
        self,
        ethics_result: Dict,
        contradiction_result: Dict,
        analytics_result: AnalyticsResult,
        iteration: int
    ) -> float:
        """Рассчитывает общую уверенность"""
        # Ethics score
        ethics_score = ethics_result.get('overall_score', 0.5) if ethics_result else 0.5
        
        # Contradiction score (fewer = better)
        contr_count = len(contradiction_result.get('significant_contradictions', [])) if contradiction_result else 0
        contr_score = 1.0 - (contr_count * 0.3)
        
        # Analytics score
        analytics_score = analytics_result.coherence_score if analytics_result else 0.5
        
        # Weighted average (итеративно снижаем этику)
        if iteration <= 1:
            weights = {'ethics': 0.4, 'contradiction': 0.3, 'analytics': 0.3}
        elif iteration <= 3:
            weights = {'ethics': 0.3, 'contradiction': 0.4, 'analytics': 0.3}
        else:
            weights = {'ethics': 0.2, 'contradiction': 0.4, 'analytics': 0.4}
        
        confidence = (
            ethics_score * weights['ethics'] +
            contr_score * weights['contradiction'] +
            analytics_score * weights['analytics']
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _perform_self_learning(self, query: str, response: str):
        """Выполняет самообучение - извлекает и сохраняет сущности"""
        try:
            # Собираем противоречия из всех итераций
            all_contradictions = []
            for it in self.iterations:
                if it.contradiction_result:
                    contrs = it.contradiction_result.get('significant_contradictions', [])
                    contrs += it.contradiction_result.get('minor_contradictions', [])
                    all_contradictions.extend(contrs)
            
            # Извлекаем сущности
            entities = self.entity_extractor.extract_all(
                query=query,
                response=response,
                contradictions=all_contradictions
            )
            
            if entities:
                # Пытаемся сохранить в knowledge graph
                saved = self.entity_extractor.save_to_knowledge_graph(
                    entities=entities,
                    knowledge_graph=getattr(self.brain, 'knowledge_graph', None) if self.brain else None
                )
                logger.info(f"Self-learning: saved {saved} entities")
            
        except Exception as e:
            logger.warning(f"Self-learning error: {e}")
    
    def _iteration_to_dict(self, iteration: ReasoningIteration) -> Dict:
        """Конвертирует итерацию в словарь для GUI"""
        # Получаем промты от модулей если они были сохранены
        module_prompts = getattr(iteration, 'module_prompts', {})
        
        return {
            'iteration': iteration.iteration,
            'response': iteration.response[:200] + '...' if len(iteration.response) > 200 else iteration.response,
            'confidence': iteration.confidence,
            'has_contradictions': bool(iteration.contradiction_result and 
                                      iteration.contradiction_result.get('significant_count', 0) > 0),
            'has_ethics_issues': bool(iteration.ethics_result and 
                                     iteration.ethics_result.get('has_violations', False)),
            'is_final': iteration.is_final,
            'module_prompts': module_prompts,
            'analytics': iteration.analytics_result.to_dict() if iteration.analytics_result else {}
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику"""
        return {
            'total_iterations': len(self.iterations),
            'max_iterations': self.max_iterations,
            'final_confidence': self.iterations[-1].confidence if self.iterations else 0.0,
            'stopped_early': len(self.iterations) < self.max_iterations
        }


__all__ = ['EnhancedReasoningEngine', 'ReasoningIteration']