"""
ReflectiveThinkingMixin - Рефлексивное мышление для Model B.

Позволяет модели:
1. Генерировать первичное рассуждение
2. Извлекать мета-данные (посылки, неопределённости, пропуски)
3. Рефлексировать через "зеркало" 
4. Углублять рассуждение итеративно

Цикл: рассуждение → мета-извлечение → зеркало → углубление → повтор
"""

import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("eva_ai.learning.reflective_thinking")


@dataclass
class MetaThoughts:
    """Мета-данные из рассуждения модели."""
    premises: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    logical_steps: List[str] = field(default_factory=list)
    raw_text: str = ""


class ReflectiveThinkingMixin:
    """
    Миксин для рефлексивного мышления.
    
    Использование с Model B:
    ```python
    class ReflectiveModelB(ReflectiveThinkingMixin):
        def __init__(self, model_instance):
            self.model = model_instance
            
        async def generate(self, prompt):
            return await self.generate_with_reflection(prompt)
    ```
    """
    
    MAX_ITERATIONS = 3
    MIN_IMPROVEMENT_THRESHOLD = 0.3
    
    PREMISE_PATTERNS = [
        r'(?:потому что|так как|значит|из этого следует|=>)',
        r'(?:исходя из|на основе|учитывая)',
        r'([А-Яа-яё]+\s+это\s+[А-Яа-яё]+)',
        r'(?:допустим|предположим|примем\s+за)',
    ]
    
    UNCERTAINTY_PATTERNS = [
        r'(возможно|вероятно|скорее всего)',
        r'(?:но\s+это\s+не\s+точно|не\s+уверен)',
        r'(?:may\s+be|might|perhaps)',
        r'(?:однако\s+есть\s+сомнения)',
    ]
    
    TRANSITION_PATTERNS = [
        r'(?:но\s+|однако|тем\s+не\s+менее)',
        r'(?:с\s+другой\s+стороны|тем\s+не\s+менее)',
    ]

    async def generate_with_reflection(
        self,
        prompt: str,
        model_instance: Any,
        max_tokens: int = 2048,
        iterations: int = None
    ) -> str:
        """
        Генерирует рассуждение с рефлексией.
        
        Args:
            prompt: Базовый промт.
            model_instance: Модель для генерации.
            max_tokens: Максимум токенов.
            iterations: Количество итераций (по умолчанию MAX_ITERATIONS).
            
        Returns:
            Финальное рассуждение после рефлексии.
        """
        iterations = iterations or self.MAX_ITERATIONS
        
        current_thought = await self._generate_thought(prompt, model_instance, max_tokens)
        prev_quality = 0
        
        for i in range(iterations):
            meta = self._extract_meta_thoughts(current_thought)
            
            if i == 0:
                logger.debug(f"Iteration {i+1}: первичное рассуждение {len(current_thought)} символов")
                current_thought = await self._apply_mirror(
                    prompt, meta, model_instance, max_tokens, first_reflect=True
                )
            else:
                quality = self._assess_thought_quality(meta)
                logger.debug(f"Iteration {i+1}: quality={quality:.2f}, prev={prev_quality:.2f}")
                
                if quality - prev_quality < self.MIN_IMPROVEMENT_THRESHOLD:
                    logger.debug(f"Рефлексия стабилизировалась на итерации {i+1}")
                    break
                
                current_thought = await self._apply_mirror(
                    prompt, meta, model_instance, max_tokens, first_reflect=False
                )
                prev_quality = quality
        
        return current_thought

    async def _generate_thought(
        self,
        prompt: str,
        model_instance: Any,
        max_tokens: int
    ) -> str:
        """Генерирует первичное рассуждение."""
        full_prompt = f"{prompt}\n\nРассуждай подробно, шаг за шагом."
        
        try:
            if hasattr(model_instance, 'generate'):
                response = model_instance.generate(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=0.35,
                    repeat_penalty=1.8,
                    echo=False
                )
            else:
                response = model_instance(full_prompt, max_tokens=max_tokens)
            
            if isinstance(response, dict):
                return response.get('choices', [{}])[0].get('text', '')
            return str(response)
        except Exception as e:
            logger.error(f"Thought generation error: {e}")
            return ""

    def _extract_meta_thoughts(self, thought_text: str) -> MetaThoughts:
        """
        Извлекает мета-данные из текста рассуждения.
        
        Args:
            thought_text: Текст рассуждения модели.
            
        Returns:
            MetaThoughts с извлечёнными данными.
        """
        meta = MetaThoughts(raw_text=thought_text)
        
        thought_lower = thought_text.lower()
        
        for pattern in self.PREMISE_PATTERNS:
            matches = re.findall(pattern, thought_text, re.IGNORECASE)
            for m in matches:
                if len(m) > 10:
                    meta.premises.append(m.strip())
        
        for pattern in self.UNCERTAINTY_PATTERNS:
            matches = re.findall(pattern, thought_text, re.IGNORECASE)
            for m in matches:
                if len(m) > 5:
                    meta.uncertainties.append(m.strip())
        
        lines = thought_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(t in line.lower() for t in ['значит', 'следовательно', 'вывод', '=>', '→']):
                meta.conclusions.append(line)
            elif re.search(r'\d+\.', line):
                meta.logical_steps.append(line)
        
        if len(meta.premises) > 3:
            meta.gaps.append("Много посылок - возможно есть неучтённые факторы")
        if len(meta.uncertainties) > 2:
            meta.gaps.append("Высокая неопределённость - нужно уточнить")
        
        if meta.premises and meta.conclusions:
            if len(meta.conclusions) > len(meta.premises):
                meta.contradictions.append("Выводов больше чем посылок")
        
        return meta

    async def _apply_mirror(
        self,
        original_prompt: str,
        meta: MetaThoughts,
        model_instance: Any,
        max_tokens: int,
        first_reflect: bool = False
    ) -> str:
        """
        Применяет "зеркало" - возвращает модели её же рассуждения для рефлексии.
        
        Args:
            original_prompt: Оригинальный промт.
            meta: Извлечённые мета-данные.
            model_instance: Модель для генерации.
            max_tokens: Максимум токенов.
            first_reflect: Первая итерация рефлексии.
            
        Returns:
            Углублённое рассуждение.
        """
        mirror_prompt = self._build_mirror_prompt(meta, first_reflect)
        
        full_prompt = f"""{original_prompt}

Предыдущее рассуждение:
{meta.raw_text[:1000]}

{mirror_prompt}"""

        try:
            response = model_instance.generate(
                full_prompt,
                max_tokens=max_tokens,
                temperature=0.3,
                repeat_penalty=1.9,
                echo=False
            )
            
            if isinstance(response, dict):
                return response.get('choices', [{}])[0].get('text', '')
            return str(response)
        except Exception as e:
            logger.error(f"Mirror reflection error: {e}")
            return meta.raw_text

    def _build_mirror_prompt(self, meta: MetaThoughts, first_reflect: bool) -> str:
        """
        Формирует промт для зеркальной рефлексии.
        
        Args:
            meta: Мета-данные.
            first_reflect: Это первая рефлексия.
            
        Returns:
            Промт для модели.
        """
        prompt_parts = []
        
        if first_reflect:
            prompt_parts.append("Проанализируй своё рассуждение:")
        else:
            prompt_parts.append("Проверь и углубь своё рассуждение:")
        
        if meta.premises:
            prompt_parts.append(f"\nПосылки, которые ты принял: {', '.join(meta.premises[:3])}")
        
        if meta.uncertainties:
            prompt_parts.append(f"\nНеопределённости: {', '.join(meta.uncertainties[:3])}")
        
        if meta.gaps:
            prompt_parts.append(f"\nВозможные пропуски: {', '.join(meta.gaps)}")
        
        if not first_reflect:
            prompt_parts.append("\nИсправь ошибки и добавь детали.")
        
        prompt_parts.append(
            "\n\nОтвечай в формате:"
            "\n- Что ты проверил/исправил:"
            "\n- Новые выводы:"
            "\n- Финальный ответ:"
        )
        
        return "\n".join(prompt_parts)

    def _assess_thought_quality(self, meta: MetaThoughts) -> float:
        """
        Оценивает качество рассуждения.
        
        Args:
            meta: Мета-данные.
            
        Returns:
            Оценка качества 0.0 - 1.0.
        """
        quality = 0.5
        
        if meta.logical_steps:
            quality += min(0.2, len(meta.logical_steps) * 0.05)
        
        if meta.conclusions:
            quality += min(0.15, len(meta.conclusions) * 0.03)
        
        if not meta.uncertainties:
            quality += 0.1
        
        if not meta.gaps:
            quality += 0.1
        
        if meta.contradictions:
            quality -= min(0.15, len(meta.contradictions) * 0.05)
        
        return min(1.0, max(0.0, quality))

    def format_thoughts_for_context(
        self,
        thought_text: str,
        max_length: int = 500
    ) -> str:
        """
        Форматирует рассуждение для использования как контекст для Model A.
        
        Args:
            thought_text: Текст рассуждения.
            max_length: Максимальная длина.
            
        Returns:
            Отформатированная строка для контекста.
        """
        meta = self._extract_meta_thoughts(thought_text)
        
        parts = []
        
        if meta.premises:
            parts.append(f"Посылки: {'; '.join(meta.premises[:2])}")
        
        if meta.logical_steps:
            parts.append(f"Шаги: {'; '.join(meta.logical_steps[:3])}")
        
        if meta.uncertainties:
            parts.append(f"Неопределённости: {'; '.join(meta.uncertainties[:2])}")
        
        if meta.conclusions:
            parts.append(f"Выводы: {'; '.join(meta.conclusions[:2])}")
        
        result = " | ".join(parts)
        
        if len(result) > max_length:
            result = result[:max_length] + "..."
        
        return result