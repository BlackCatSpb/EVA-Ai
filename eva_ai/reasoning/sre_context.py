"""
SRE Context Module — contextual query building, Wikipedia lookup, query type detection, generation params, and Qwen generation fallback chain.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Явно экспортируем все функции для import *
__all__ = [
    '_build_contextual_query',
    '_get_wikipedia_context',
    '_determine_query_type',
    '_get_generation_params',
    '_generate_with_qwen',
    '_generate_simple_response',
    '_get_knowledge_response',
]


def _build_contextual_query(self, query: str, conversation_history: List[Dict]) -> str:
    """
    Формирует расширенный промпт с историей диалогов и Wikipedia контекстом.
    """
    parts = []

    wiki_context = _get_wikipedia_context(self, query)
    if wiki_context:
        parts.append(f"Справочная информация из Википедии:\n{wiki_context}")

    if conversation_history:
        recent_history = conversation_history[-10:]
        context_parts = []
        for msg in recent_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if content:
                role_label = 'Пользователь' if role == 'user' else 'Ассистент'
                context_parts.append(f"{role_label}: {content[:300]}")
        if context_parts:
            parts.append(f"Предыдущий контекст разговора:\n" + "\n".join(context_parts))

    if parts:
        enhanced = f"{'\n\n'.join(parts)}\n\nТекущий вопрос: {query}\n\nДай ответ с учётом контекста."
        logger.info(f"Сформирован расширенный промпт: {len(parts)} источников контекста")
        return enhanced

    return query


def _get_wikipedia_context(self, query: str) -> Optional[str]:
    """Получает контекст из Wikipedia Knowledge Base с таймаутом 5 секунд."""
    result = [None]

    def _search():
        try:
            brain = getattr(self, 'brain', None)
            if brain is None:
                return
            wiki_kb = getattr(brain, 'wikipedia_kb', None)
            if wiki_kb is None:
                return

            results = wiki_kb.search(query, limit=3, min_similarity=0.25)
            if not results:
                return

            context_parts = []
            for r in results[:3]:
                context_parts.append(f"- {r['title']}: {r['text'][:200]}")

            result[0] = "\n".join(context_parts)
        except Exception as e:
            logger.debug("Ошибка получения Wikipedia контекста: %s", e)

    thread = threading.Thread(target=_search, daemon=True)
    thread.start()
    thread.join(timeout=5)

    if thread.is_alive():
        logger.debug("Wikipedia search timed out after 5s")
        return None

    return result[0]


def _determine_query_type(self, query: str) -> str:
    """Определяет тип запроса: кратко или подробно"""
    query_lower = query.lower()
    short_keywords = ['кратко', 'вкратце', 'суть', 'что такое', 'кто такой', 'чем является', 'дай определение', 'назови', 'перечисли']
    long_keywords = ['подробно', 'детально', 'развернуто', 'расскажи', 'объясни', 'опиши', 'проанализируй', 'рассмотри', 'подробнее', 'как работает', 'почему', 'зачем']

    for kw in short_keywords:
        if kw in query_lower:
            return 'кратко'
    for kw in long_keywords:
        if kw in query_lower:
            return 'подробно'
    return 'подробно'


def _get_generation_params(self, query_type: str) -> Dict[str, Any]:
    """Возвращает динамические параметры генерации в зависимости от типа запроса"""
    base_params = {
        'model_a': {'temperature': 0.2, 'max_tokens': 256, 'top_p': 0.85, 'top_k': 50, 'repeat_penalty': 1.5},
        'model_b': {'temperature': 0.6, 'max_tokens': 512, 'top_p': 0.85, 'top_k': 50, 'repeat_penalty': 1.5},
        'translate': {'temperature': 0.3, 'max_tokens': 512, 'top_p': 0.9, 'repeat_penalty': 1.1}
    }

    if query_type == 'кратко':
        return {
            'model_a': {'temperature': 0.1, 'max_tokens': 128, 'top_p': 0.8, 'top_k': 40, 'repeat_penalty': 1.5},
            'model_b': {'temperature': 0.5, 'max_tokens': 256, 'top_p': 0.8, 'top_k': 40, 'repeat_penalty': 1.5},
            'translate': {'temperature': 0.2, 'max_tokens': 256, 'top_p': 0.85, 'repeat_penalty': 1.2}
        }
    return base_params


def _generate_with_qwen(self, prompt: str) -> str:
    """
    Генерация ответа с fallback на доступные модели.
    Упрощенная цепочка: GenerationCoordinator → ResponseGenerator → Simple Response
    (QwenModelManager и FractalModelManager отключены - используется UnifiedGenerator)
    """
    # 1. Пробуем GenerationCoordinator (основной, использует UnifiedGenerator)
    try:
        if self.brain is not None:
            gen_coord = getattr(self.brain, 'generation_coordinator', None)
            if gen_coord and hasattr(gen_coord, 'generate'):
                result = gen_coord.generate(text=prompt, source="reasoning_engine")
                if result:
                    text = result.text if hasattr(result, 'text') else str(result)
                    if text and len(text.strip()) > 10:
                        logger.debug("Generated via GenerationCoordinator")
                        return text
    except Exception as e:
        logger.debug(f"GenerationCoordinator failed: {e}")

    # 2. Пробуем ResponseGenerator
    try:
        if self.brain is not None:
            resp_gen = getattr(self.brain, 'response_generator', None)
            if resp_gen and hasattr(resp_gen, 'generate'):
                result = resp_gen.generate(prompt)
                if result:
                    if isinstance(result, dict):
                        text = result.get('text', result.get('response', ''))
                    else:
                        text = str(result)
                    if text and len(text.strip()) > 10:
                        logger.debug("Generated via ResponseGenerator")
                        return text
    except Exception as e:
        logger.debug(f"ResponseGenerator generation failed: {e}")

    # 3. Пробуем two_model_pipeline (UnifiedGenerator через PipelineAdapter)
    try:
        if self.brain is not None:
            pipeline = getattr(self.brain, 'two_model_pipeline', None)
            if pipeline and hasattr(pipeline, 'generate'):
                result = pipeline.generate(prompt, max_tokens=512, temperature=0.7)
                if result and len(result.strip()) > 10:
                    logger.debug("Generated via two_model_pipeline")
                    return result
    except Exception as e:
        logger.debug(f"TwoModelPipeline generation failed: {e}")

    # Примечание: QwenModelManager и FractalModelManager отключены
    # Они возвращают None через get_qwen_model_manager() и get_fractal_qwen()

    logger.warning("Все генераторы недоступны, используем простой ответ")
    return self._generate_simple_response(prompt)


def _generate_simple_response(self, prompt: str) -> str:
    """Простой fallback ответ без модели."""
    prompt_lower = prompt.lower().strip()

    greetings = ['привет', 'здравствуй', 'hello', 'hi', 'хай', 'прив', 'здорово']
    if any(g in prompt_lower for g in greetings):
        return "Здравствуйте! Я ЕВА. Чем могу помочь?"

    kg_response = self._get_knowledge_response(prompt_lower)
    if kg_response:
        return kg_response

    keyword_responses = {
        'погода': 'Для информации о погоде я могу выполнить поиск в интернете.',
        'новост': 'Могу найти последние новости по вашему запросу.',
        'помощ': 'Я могу помочь с ответами на вопросы, анализом информации и поиском данных.',
        'что такое': 'Для объяснения понятий мне нужно больше контекста.',
        'как работает': 'Могу объяснить принципы работы, но для точного ответа уточните область.',
        'кто такой': 'Для идентификации личности нужны дополнительные детали.',
        'спасиб': 'Пожалуйста! Рада была помочь.',
        'благодар': 'Спасибо! Обращайтесь ещё.',
        'пока': 'До свидания! Возвращайтесь с новыми вопросами.',
        'помоги': 'Опишите подробнее, что именно вам нужно, и я постараюсь помочь.',
    }

    for keyword, response in keyword_responses.items():
        if keyword in prompt_lower:
            return response

    if '?' in prompt:
        return f"Интересный вопрос: '{prompt}'. Для полного ответа требуется больше контекста."

    return f"Получил ваш запрос: '{prompt}'. Чтобы дать точный ответ, уточните детали."


def _get_knowledge_response(self, prompt: str) -> Optional[str]:
    """Попытка получить ответ из knowledge graph"""
    try:
        if self.brain is not None and hasattr(self.brain, 'knowledge_graph'):
            kg = self.brain.knowledge_graph
            if kg is None:
                return None
            search_method = getattr(kg, 'search_nodes', getattr(kg, 'search', getattr(kg, 'search_by_concept', None)))
            if search_method:
                results = search_method(prompt, limit=3)
                if results and isinstance(results, list) and len(results) > 0:
                    best = results[0]
                    if isinstance(best, dict):
                        content = best.get('content', best.get('text', ''))
                        if content:
                            return f"Известно: {content[:200]}..."
                    else:
                        content = getattr(best, 'content', getattr(best, 'description', ''))
                        if content:
                            return f"Известно: {content[:200]}..."
    except Exception as e:
        logger.debug(f"Knowledge search error: {e}")
    return None
