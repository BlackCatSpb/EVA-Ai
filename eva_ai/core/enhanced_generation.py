"""
EnhancedGenerationMixin - Улучшенные методы генерации текста.

Включает:
1. Динамический расчет длины ответа
2. Chain of Thought (CoT) генерация
3. Рекурсивное расширение контекста
"""

import re
import logging
from typing import Dict, List, Optional, AsyncIterator, Any

logger = logging.getLogger("eva_ai.core.enhanced_generation")


class EnhancedGenerationMixin:
    """Миксин для улучшения процессов генерации текста."""

    def calculate_dynamic_max_tokens(self, query: str, context_length: int = 0) -> int:
        """
        Динамически рассчитывает лимит токенов на основе сложности запроса.
        
        Args:
            query: Текст запроса пользователя.
            context_length: Длина текущего контекста.
            
        Returns:
            Рекомендуемый лимит токенов (512 - 4096).
        """
        base_limit = 512
        complexity_score = 0
        
        word_count = len(query.split())
        if word_count > 15:
            complexity_score += 1
        if word_count > 30:
            complexity_score += 1
        if word_count > 50:
            complexity_score += 1
        
        complex_keywords = [
            'сравни', 'анализ', 'почему', 'как работает', 'объясни механизм',
            'различия', 'преимущества', 'недостатки', 'в чем причина',
            'докажи', 'обоснуй', 'исследуй', 'рассмотри все'
        ]
        query_lower = query.lower()
        for keyword in complex_keywords:
            if keyword in query_lower:
                complexity_score += 1
                break
        
        if context_length > 2000:
            complexity_score += 1
        if context_length > 4000:
            complexity_score += 1
        
        limits = {
            0: 512,
            1: 768,
            2: 1024,
            3: 1536,
            4: 2048,
            5: 3072,
            6: 4096
        }
        return limits.get(complexity_score, 2048)

    async def generate_with_cot(
        self, 
        prompt: str, 
        model_instance: Any,
        max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        """
        Генерация с использованием Chain of Thought (Мысли вслух).
        Скрывает промежуточные рассуждения от пользователя.
        
        Args:
            prompt: Исходный промпт.
            model_instance: Экземпляр модели с методом generate().
            max_tokens: Максимум токенов.
            
        Yields:
            Токены финального ответа.
        """
        cot_instruction = (
            "\n\nИНСТРУКЦИЯ: Сначала кратко проанализируй вопрос и составь план ответа "
            "внутри тегов <thinking>. Затем дай финальный ответ внутри тегов <answer>. "
            "Не показывай внутренние рассуждения пользователю - только ответ в <answer>."
        )
        full_prompt = f"{prompt}{cot_instruction}"
        
        full_response = ""
        in_thought = False
        in_answer = False
        thought_buffer = ""
        answer_buffer = ""
        
        try:
            if hasattr(model_instance, 'generate'):
                response = model_instance.generate(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=0.35,
                    repeat_penalty=1.8,
                    echo=False
                )
            elif callable(model_instance):
                response = model_instance(full_prompt, max_tokens=max_tokens)
            else:
                logger.warning("Model instance doesn't have generate method")
                return
            
            if isinstance(response, dict):
                text = response.get('choices', [{}])[0].get('text', '')
            else:
                text = str(response)
            
            full_response = text
            
            thought_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
            if thought_match:
                thought_buffer = thought_match.group(1).strip()
                logger.debug(f"CoT thought captured: {len(thought_buffer)} chars")
            
            answer_match = re.search(r'<answer>(.*?)</answer>', full_response, re.DOTALL)
            if answer_match:
                answer_buffer = answer_match.group(1).strip()
                yield answer_buffer
            else:
                yield full_response
                
        except Exception as e:
            logger.error(f"CoT generation error: {e}")
            yield prompt

    async def generate_with_recursive_context(
        self, 
        query: str, 
        initial_context: List[str],
        threshold: float = 0.6,
        max_iterations: int = 2
    ) -> List[str]:
        """
        Рекурсивно расширяет контекст, если текущий недостаточно релевантен.
        
        Args:
            query: Запрос пользователя.
            initial_context: Начальный список узлов контекста.
            threshold: Порог релевантности (0.0 - 1.0).
            max_iterations: Максимум итераций.
            
        Returns:
            Расширенный список контекста.
        """
        try:
            from eva_ai.memory.fractal_graph_v2 import FractalGraphV2
            fg = FractalGraphV2.get_instance()
        except Exception:
            logger.warning("FractalGraphV2 not available, using initial context")
            return initial_context
        
        current_context = list(initial_context)
        iteration = 0
        
        while iteration < max_iterations:
            relevance_score = self._estimate_context_relevance(query, current_context)
            
            if relevance_score >= threshold or iteration == max_iterations - 1:
                break
            
            missing_keywords = self._extract_missing_keywords(query, current_context)
            
            if missing_keywords:
                search_query = " ".join(missing_keywords[:5])
                try:
                    extra_nodes = fg.search_semantic(search_query, top_k=3)
                    if extra_nodes:
                        new_contexts = []
                        for n in extra_nodes:
                            if hasattr(n, 'content'):
                                new_contexts.append(n.content)
                            elif isinstance(n, dict):
                                new_contexts.append(n.get('content', ''))
                        current_context.extend(new_contexts)
                        iteration += 1
                    else:
                        break
                except Exception as e:
                    logger.debug(f"Recursive context search error: {e}")
                    break
            else:
                break
                
        return current_context

    def _estimate_context_relevance(self, query: str, context: List[str]) -> float:
        """Оценивает перекрытие токенов между запросом и контекстом."""
        if not context:
            return 0.0
            
        q_tokens = set(query.lower().split())
        if not q_tokens:
            return 0.0
            
        c_text = " ".join(context)
        c_tokens = set(c_text.lower().split())
        
        if not c_tokens:
            return 0.0
            
        overlap = len(q_tokens & c_tokens) / len(q_tokens)
        return min(1.0, overlap * 1.5)

    def _extract_missing_keywords(self, query: str, context: List[str]) -> List[str]:
        """Извлекает слова из запроса, отсутствующие в контексте."""
        c_text = " ".join(context).lower()
        return [w for w in query.lower().split() if len(w) > 3 and w not in c_text]