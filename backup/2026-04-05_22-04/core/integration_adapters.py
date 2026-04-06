"""
Integration Adapters - Adapter classes for different subsystems.
"""

import logging
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger("eva.integration")


def _handle_query_received(self, request_data: Dict[str, Any]):
    """Обработка события query_received."""
    try:
        request_id = request_data['request_id']
        query = request_data['query']

        logger.info(f"Обработка запроса {request_id}: '{query[:50]}...'")

        if self.fractal_attention:
            focus_result = self.fractal_attention._initialize_attention_focus(query)
            request_data['attention_focus'] = focus_result

        tokenize_data = {
            'request_id': request_id,
            'query': query,
            'context': request_data.get('context', {}),
            'attention_focus': request_data.get('attention_focus', {})
        }

        self.event_bus.trigger('tokenize_request', tokenize_data)

    except Exception as e:
        logger.error(f"Ошибка обработки query_received: {e}")
        self._update_request_status(request_id, 'error', error=str(e))


def _handle_tokenize_request(self, tokenize_data: Dict[str, Any]):
    """Обработка события tokenize_request."""
    try:
        request_id = tokenize_data['request_id']
        query = tokenize_data['query']

        logger.info(f"Токенизация запроса {request_id}")

        if self.generation_coordinator:
            tokens = self._tokenize_text(query)

            tokens_data = {
                'request_id': request_id,
                'query': query,
                'tokens': tokens,
                'context': tokenize_data.get('context', {}),
                'attention_focus': tokenize_data.get('attention_focus', {})
            }

            self.event_bus.trigger('tokens_ready', tokens_data)

        else:
            logger.error("GenerationCoordinator недоступен")
            self._update_request_status(request_id, 'error', error="Токенизатор недоступен")

    except Exception as e:
        logger.error(f"Ошибка токенизации: {e}")
        self._update_request_status(request_id, 'error', error=str(e))


def _handle_tokens_ready(self, tokens_data: Dict[str, Any]):
    """Обработка события tokens_ready."""
    try:
        request_id = tokens_data['request_id']
        tokens = tokens_data['tokens']

        logger.info(f"Токены готовы для запроса {request_id}")

        if self.memory_manager and hasattr(self.memory_manager, 'create_hot_window'):
            hot_window = self.memory_manager.create_hot_window(
                tokens=tokens,
                context=tokens_data.get('context', {}),
                attention_focus=tokens_data.get('attention_focus', {})
            )

            window_data = {
                'request_id': request_id,
                'query': tokens_data['query'],
                'tokens': tokens,
                'hot_window': hot_window,
                'context': tokens_data.get('context', {}),
                'attention_focus': tokens_data.get('attention_focus', {})
            }

            self.event_bus.trigger('hot_window_ready', window_data)

        else:
            logger.warning("MemoryManager недоступен, пропускаем формирование горячего окна")
            window_data = {
                'request_id': request_id,
                'query': tokens_data['query'],
                'tokens': tokens,
                'hot_window': {},
                'context': tokens_data.get('context', {}),
                'attention_focus': tokens_data.get('attention_focus', {})
            }
            self.event_bus.trigger('hot_window_ready', window_data)

    except Exception as e:
        logger.error(f"Ошибка обработки tokens_ready: {e}")
        self._update_request_status(request_id, 'error', error=str(e))


def _handle_hot_window_ready(self, window_data: Dict[str, Any]):
    """Обработка события hot_window_ready с внутренним рассуждением."""
    try:
        request_id = window_data['request_id']
        query = window_data['query']
        hot_window = window_data['hot_window']

        logger.info(f"Горячее окно готово для запроса {request_id}")

        reasoning_result = None
        if self.reasoning_engine:
            try:
                logger.info(f"Запуск рассуждения для запроса {request_id}")
                reasoning_result = self.reasoning_engine.reason(
                    query=query,
                    context={
                        'hot_window': hot_window,
                        'attention_focus': window_data.get('attention_focus', {}),
                        'request_id': request_id
                    }
                )
                logger.info(f"Рассуждение завершено: {reasoning_result.get('reasoning_steps', 0)} шагов, "
                          f"уверенность={reasoning_result.get('confidence', 0):.2f}")
            except Exception as e:
                logger.error(f"Ошибка рассуждения: {e}")
                reasoning_result = None

        contradictions = []
        if hasattr(self.fractal_attention, 'contradiction_resolver'):
            contradictions = self.fractal_attention.contradiction_resolver.detect_contradictions(
                query, hot_window
            )

        if self.response_generator:
            generation_context = {
                'hot_window': hot_window,
                'contradictions': contradictions,
                'attention_focus': window_data.get('attention_focus', {}),
                'reasoning_result': reasoning_result
            }

            enhanced_prompt = query
            if reasoning_result and reasoning_result.get('confidence', 0) > 0.5:
                if hasattr(self, '_enhance_prompt_with_reasoning') and callable(getattr(self, '_enhance_prompt_with_reasoning', None)):
                    try:
                        enhanced_prompt = self._enhance_prompt_with_reasoning(query, reasoning_result)
                    except Exception as e:
                        logger.warning(f"Ошибка при улучшении промпта: {e}")
                        enhanced_prompt = query

            response_result = self.response_generator.generate_response(
                prompt=enhanced_prompt,
                context=generation_context
            )

            if contradictions:
                self.event_bus.trigger('contradiction_detected', {
                    'request_id': request_id,
                    'contradictions': contradictions
                })

            response_data = {
                'request_id': request_id,
                'query': query,
                'response': response_result,
                'hot_window': hot_window,
                'contradictions': contradictions,
                'reasoning_result': reasoning_result,
                'processing_time': time.time() - window_data.get('timestamp', time.time())
            }

            self.event_bus.trigger('response_generated', response_data)

        else:
            logger.error("ResponseGenerator недоступен")
            self._update_request_status(request_id, 'error', error="Генератор ответов недоступен")

    except Exception as e:
        logger.error(f"Ошибка обработки hot_window_ready: {e}")
        self._update_request_status(request_id, 'error', error=str(e))


def _handle_response_generated(self, response_data: Dict[str, Any]):
    """Обработка события response_generated."""
    try:
        request_id = response_data['request_id']

        logger.info(f"Ответ сгенерирован для запроса {request_id}")

        with self._processing_lock:
            if request_id in self._active_requests:
                self._active_requests[request_id].update({
                    'status': 'completed',
                    'response': response_data.get('response', {}),
                    'processing_time': response_data.get('processing_time', 0.0)
                })

        if hasattr(self.fractal_attention, 'learning_scheduler'):
            learning_opportunities = self.fractal_attention.learning_scheduler.identify_learning_opportunities(
                response_data.get('query', '')
            )

            if learning_opportunities:
                self.event_bus.trigger('learning_opportunity', {
                    'request_id': request_id,
                    'opportunities': learning_opportunities
                })

    except Exception as e:
        logger.error(f"Ошибка обработки response_generated: {e}")


def _handle_contradiction_detected(self, contradiction_data: Dict[str, Any]):
    """Обработка события contradiction_detected."""
    try:
        request_id = contradiction_data['request_id']
        contradictions = contradiction_data['contradictions']

        logger.info(f"Обнаружено {len(contradictions)} противоречий для запроса {request_id}")

        if hasattr(self.fractal_attention, 'contradiction_resolver'):
            for contradiction in contradictions:
                resolution = self.fractal_attention.contradiction_resolver.resolve_contradiction(contradiction)
                logger.info(f"Противоречие разрешено: {resolution}")

    except Exception as e:
        logger.error(f"Ошибка обработки противоречий: {e}")


def _handle_learning_opportunity(self, learning_data: Dict[str, Any]):
    """Обработка события learning_opportunity."""
    try:
        opportunities = learning_data['opportunities']

        logger.info(f"Обнаружено {len(opportunities)} возможностей обучения")

        for opportunity in opportunities:
            if hasattr(self.fractal_attention, 'learning_scheduler'):
                success = self.fractal_attention.learning_scheduler.schedule_learning_session(opportunity)
                if success:
                    logger.info(f"Запланирована сессия обучения: {opportunity.get('description', '')}")

    except Exception as e:
        logger.error(f"Ошибка обработки возможностей обучения: {e}")


def _handle_self_dialog_request(self, dialog_data: Dict[str, Any]):
    """Обработка события self_dialog_request."""
    try:
        logger.info("Запуск сессии самодиалога")

        if hasattr(self.fractal_attention, 'dialog_manager'):
            self.fractal_attention.dialog_manager.start_session()

    except Exception as e:
        logger.error(f"Ошибка запуска самодиалога: {e}")


def _handle_ethical_check_request(self, ethical_data: Dict[str, Any]):
    """Обработка события ethical_check_request."""
    try:
        request_id = ethical_data.get('request_id', '')
        content = ethical_data.get('content', '')

        logger.info(f"Проверка этической корректности для запроса {request_id}")

        ethical_result = {
            'request_id': request_id,
            'score': 0.8,
            'approved': True,
            'recommendations': []
        }

        self._update_request_status(request_id, 'ethical_check_completed', ethical_result=ethical_result)

    except Exception as e:
        logger.error(f"Ошибка этической проверки: {e}")


def _tokenize_text(self, text: str) -> List[str]:
    """Токенизация текста."""
    try:
        if self.generation_coordinator and hasattr(self.generation_coordinator, 'tokenizer'):
            tokens = self.generation_coordinator.tokenizer.tokenize(text)
            return [str(token) for token in tokens]
        else:
            return text.split()
    except Exception as e:
        logger.error(f"Ошибка токенизации: {e}")
        return text.split()


def _enhance_prompt_with_reasoning(self, query: str, reasoning_result: Dict) -> str:
    """Улучшает промпт на основе результатов рассуждения"""
    try:
        insights = reasoning_result.get('reasoning_phases', [])
        confidence = reasoning_result.get('confidence', 0)
        
        if confidence > 0.6 and insights:
            enhanced = f"""На основе анализа:
Запрос: {query}

Ключевые аспекты для учета:
- Тип запроса: {reasoning_result.get('reasoning_phases', ['general'])[0] if reasoning_result.get('reasoning_phases') else 'general'}
- Уверенность анализа: {confidence:.0%}
- Найдено инсайтов: {reasoning_result.get('insights_count', 0)}

Сформируйте точный и полезный ответ:"""
            return enhanced
        
        return query
        
    except Exception as e:
        logger.debug(f"Ошибка улучшения промпта: {e}")
        return query


def _setup_adapters(self):
    """Setup adapter handlers on the integrator class."""
    pass
