"""
Prompt Composer - объединение промптов от модулей противоречий, этики и веб-поиска
Создаёт единый промпт для Qwen на основе результатов анализа всех модулей
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModulePrompt:
    """Промпт от отдельного модуля"""
    module_name: str
    prompt_text: str
    priority: int = 1  # 1-5, выше = важнее
    weight: float = 1.0


@dataclass
class ComposedPrompt:
    """Результат объединения промптов"""
    full_prompt: str
    module_prompts: List[ModulePrompt] = field(default_factory=list)
    token_count: int = 0


class PromptComposer:
    """
    Компоновщик промптов для Qwen
    
    Объединяет результаты трёх модулей:
    - Contradiction module (проверка противоречий)
    - Ethics module (проверка этики)
    - Web search module (обогащение контекста)
    
    Создаёт единый промпт для регенерации ответа
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        
        # Максимальная длина промпта (в токенах, примерно 4 символа на токен)
        self.max_tokens = 8000
        self.max_chars = self.max_tokens * 4
        
        # Шаблон для финального промпта
        self.prompt_template = """Ты - ЕВА, система искусственного интеллекта. 
Твоя задача - улучшить свой предыдущий ответ на основе обратной связи от аналитических модулей.

Исходный запрос: {query}

Предыдущий ответ: {previous_response}

=== АНАЛИЗ МОДУЛЕЙ ===

{module_feedback}

=== ИНСТРУКЦИЯ ===

Улучши предыдущий ответ с учётом обратной связи от модулей.
Сохрани основную информацию, но исправь указанные проблемы.
Дай более точный, этичный и непротиворечивый ответ.

Улучшенный ответ:"""

        logger.info("PromptComposer инициализирован")
    
    def compose(
        self,
        query: str,
        previous_response: str,
        contradiction_prompt: Optional[str] = None,
        ethics_prompt: Optional[str] = None,
        websearch_context: Optional[str] = None
    ) -> ComposedPrompt:
        """
        Компонует единый промпт из всех модулей
        
        Args:
            query: Оригинальный запрос пользователя
            previous_response: Предыдущий ответ Qwen
            contradiction_prompt: Промпт от модуля противоречий
            ethics_prompt: Промпт от модуля этики
            websearch_context: Контекст от веб-поиска
            
        Returns:
            ComposedPrompt: Скомпонованный промпт
        """
        module_prompts = []
        
        # Добавляем промпт от Contradiction модуля
        if contradiction_prompt:
            module_prompts.append(ModulePrompt(
                module_name="contradiction",
                prompt_text=contradiction_prompt,
                priority=3,
                weight=1.0
            ))
        
        # Добавляем промпт от Ethics модуля
        if ethics_prompt:
            module_prompts.append(ModulePrompt(
                module_name="ethics",
                prompt_text=ethics_prompt,
                priority=4,  # Этика имеет высокий приоритет
                weight=1.0
            ))
        
        # Добавляем контекст от Web Search
        if websearch_context:
            module_prompts.append(ModulePrompt(
                module_name="websearch",
                prompt_text=websearch_context,
                priority=2,
                weight=0.8
            ))
        
        # Сортируем по приоритету
        module_prompts.sort(key=lambda x: x.priority, reverse=True)
        
        # Формируем объединённый текст обратной связи
        module_feedback = self._combine_module_feedback(module_prompts)
        
        # Создаём финальный промпт
        full_prompt = self.prompt_template.format(
            query=query[:500],  # Ограничиваем длину запроса
            previous_response=previous_response[:2000],  # Ограничиваем длину ответа
            module_feedback=module_feedback
        )
        
        # Проверяем длину и обрезаем если нужно
        if len(full_prompt) > self.max_chars:
            full_prompt = self._truncate_prompt(full_prompt, module_prompts)
        
        return ComposedPrompt(
            full_prompt=full_prompt,
            module_prompts=module_prompts,
            token_count=len(full_prompt) // 4
        )
    
    def _combine_module_feedback(self, module_prompts: List[ModulePrompt]) -> str:
        """Объединяет тексты обратной связи от модулей"""
        parts = []
        
        for mp in module_prompts:
            parts.append(f"\n### {mp.module_name.upper()} модуль ###\n{mp.prompt_text}")
        
        return "\n".join(parts)
    
    def _truncate_prompt(self, prompt: str, module_prompts: List[ModulePrompt]) -> str:
        """Обрезает промпт если он слишком длинный"""
        # Удаляем модули с низким приоритетом
        truncated_modules = [mp for mp in module_prompts if mp.priority >= 3]
        
        if not truncated_modules:
            truncated_modules = module_prompts[:2]
        
        module_feedback = self._combine_module_feedback(truncated_modules)
        
        # Пересоздаём промпт
        prompt = self.prompt_template.format(
            query="{query}",
            previous_response="{previous_response}",
            module_feedback=module_feedback
        )
        
        # Если всё ещё слишком длинно, обрезаем
        if len(prompt) > self.max_chars:
            return prompt[:self.max_chars] + "..."
        
        return prompt
    
    def compose_from_results(
        self,
        query: str,
        previous_response: str,
        contradiction_result: Optional[Dict] = None,
        ethics_result: Optional[Dict] = None,
        websearch_result: Optional[Dict] = None
    ) -> ComposedPrompt:
        """
        Компонует промпт из результатов модулей (более высокий уровень)
        
        Args:
            query: Запрос пользователя
            previous_response: Предыдущий ответ
            contradiction_result: Результат проверки противоречий
            ethics_result: Результат проверки этики
            websearch_result: Результат веб-поиска
            
        Returns:
            ComposedPrompt: Скомпонованный промпт
        """
        # Генерируем промпты из результатов
        contradiction_prompt = self._generate_contradiction_prompt(contradiction_result)
        ethics_prompt = self._generate_ethics_prompt(ethics_result)
        websearch_context = self._generate_websearch_context(websearch_result)
        
        return self.compose(
            query=query,
            previous_response=previous_response,
            contradiction_prompt=contradiction_prompt,
            ethics_prompt=ethics_prompt,
            websearch_context=websearch_context
        )
    
    def _generate_contradiction_prompt(self, result: Optional[Dict]) -> Optional[str]:
        """Генерирует промпт для противоречий из результата"""
        if not result:
            return None
        
        contradictions = result.get('contradictions', [])
        if not contradictions:
            return None
        
        parts = []
        for i, contr in enumerate(contradictions[:3], 1):
            if isinstance(contr, dict):
                concept = contr.get('concept', 'unknown')
                conflicts = contr.get('conflicting_facts', [])
                parts.append(f"{i}. Противоречие в '{concept}': {conflicts[:2]}")
        
        if parts:
            return "Обнаружены противоречия:\n" + "\n".join(parts)
        
        return None
    
    def _generate_ethics_prompt(self, result: Optional[Dict]) -> Optional[str]:
        """Генерирует промпт для этики из результата"""
        if not result:
            return None
        
        violations = result.get('violations', [])
        if not violations:
            return None
        
        parts = []
        for violation in violations[:3]:
            if isinstance(violation, dict):
                desc = violation.get('description', violation.get('message', ''))
                parts.append(f"- {desc}")
        
        if parts:
            return "Этические нарушения:\n" + "\n".join(parts)
        
        return None
    
    def _generate_websearch_context(self, result: Optional[Dict]) -> Optional[str]:
        """Генерирует контекст из веб-поиска"""
        if not result:
            return None
        
        results = result.get('results', [])
        if not results:
            return None
        
        parts = []
        for r in results[:3]:
            if isinstance(r, dict):
                title = r.get('title', '')
                snippet = r.get('snippet', r.get('text', ''))[:200]
                if title or snippet:
                    parts.append(f"- {title}: {snippet}")
        
        if parts:
            return "Контекст из веб-поиска:\n" + "\n".join(parts)
        
        return None
    
    def get_module_weights(self, iteration: int) -> Dict[str, float]:
        """
        Возвращает веса модулей для текущей итерации
        
        Args:
            iteration: Номер итерации (1-5)
            
        Returns:
            Dict с весами модулей
        """
        # Адаптивные веса - на ранних итерациях этика важнее
        if iteration <= 1:
            return {'ethics': 0.40, 'contradiction': 0.30, 'websearch': 0.30}
        elif iteration <= 3:
            return {'ethics': 0.30, 'contradiction': 0.40, 'websearch': 0.30}
        else:
            return {'ethics': 0.20, 'contradiction': 0.40, 'websearch': 0.40}


__all__ = ['PromptComposer', 'ModulePrompt', 'ComposedPrompt']