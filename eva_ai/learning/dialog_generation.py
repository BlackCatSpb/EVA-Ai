"""Dialog generation, response creation, and quality checks for self-dialog learning."""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional

from eva_ai.learning.dialog_types import DialogRole, DialogTurn, LearningType, SelfDialog

logger = logging.getLogger("eva_ai.self_dialog_learning")


class DialogGenerationMixin:
    """Mixin for dialog generation, response simulation, and quality assessment."""

    def _run_dialog(self, dialog: SelfDialog, context: Optional[Dict[str, Any]] = None):
        """Выполняет самодиалог."""
        assistant_prompt = self._generate_assistant_prompt(dialog.topic, context)
        conversation_context = self._get_conversation_context()

        turn_count = 0
        while turn_count < self.max_dialog_turns and len(dialog.turns) < self.max_dialog_turns:
            turn_count += 1

            if turn_count == 1:
                role = DialogRole.ASSISTANT
                content = self._simulate_assistant_response(assistant_prompt, conversation_context)
            elif turn_count == 2:
                role = DialogRole.CRITIC
                content = self._simulate_critic_response(dialog.turns, dialog.topic)
            elif turn_count == 3:
                role = DialogRole.LEARNER
                content = self._simulate_learner_response(dialog.turns, dialog.topic)
            elif turn_count == 4:
                role = DialogRole.TEACHER
                content = self._simulate_teacher_response(dialog.turns, dialog.topic)
            else:
                role = DialogRole.OBSERVER
                content = self._simulate_observer_response(dialog.turns, dialog.topic)

            turn = DialogTurn(
                role=role,
                content=content,
                timestamp=time.time()
            )
            turn.quality_score = self._assess_turn_quality(turn, dialog)
            dialog.turns.append(turn)

            if self._should_end_dialog(dialog):
                break

        self._analyze_dialog_outcome(dialog)

    def _generate_assistant_prompt(self, topic: str, context: Optional[Dict[str, Any]]) -> str:
        """Генерирует промпт для первого хода ассистента."""
        prompt = f"Тема: {topic}\n"
        if context:
            if "user_query" in context:
                prompt += f"Вопрос пользователя: {context['user_query']}\n"
            if "system_response" in context:
                prompt += f"Ответ системы: {context['system_response']}\n"
            if "knowledge_gaps" in context:
                prompt += f"Известные пробелы: {', '.join(context['knowledge_gaps'])}\n"
        prompt += "\nСистема анализирует эту тему..."
        return prompt

    def _get_conversation_context(self) -> Dict[str, Any]:
        """Получает контекст из истории диалогов для использования в самодиалоге."""
        context = {"conversation_history": []}
        if not self.brain:
            return context
        if not hasattr(self.brain, 'memory_manager') or not self.brain.memory_manager:
            return context
        try:
            if hasattr(self.brain.memory_manager, 'get_conversation_history'):
                history = self.brain.memory_manager.get_conversation_history(
                    user_id="default_user",
                    limit=10
                )
                if history and isinstance(history, list):
                    context["conversation_history"] = [
                        {"role": "user" if i % 2 == 0 else "assistant", "content": h.get('query', h.get('response', ''))}
                        for i, h in enumerate(history)
                    ]
                    logger.info(f"Получено {len(context['conversation_history'])} сообщений из истории")
        except Exception as e:
            logger.debug(f"Ошибка получения истории диалогов: {e}")
        return context

    def _simulate_assistant_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Симулирует ответ ассистента с использованием SelfReasoningEngine."""
        topic_match = prompt.split("Тема:")[1].split("\n")[0].strip() if "Тема:" in prompt else "тему"

        relevant_context = ""
        if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
            try:
                kg = self.brain.knowledge_graph
                if hasattr(kg, 'search_nodes'):
                    results = kg.search_nodes(topic_match, limit=3)
                    if results:
                        relevant_context = "Известная информация: " + "; ".join([getattr(r, 'description', '')[:100] for r in results[:2]])
                        logger.info(f"Найден контекст из knowledge graph: {len(results)} результатов")
                elif hasattr(kg, 'search'):
                    results = kg.search(topic_match, limit=3)
                    if results:
                        relevant_context = "Известная информация: " + "; ".join([getattr(r, 'description', '')[:100] for r in results[:2]])
                        logger.info(f"Найден контекст из knowledge graph: {len(results)} результатов")
            except Exception as e:
                logger.debug(f"Ошибка поиска в knowledge graph: {e}")

        full_query = f"Объясни: {topic_match}"
        if relevant_context:
            full_query = relevant_context + "\n\n" + full_query

        if self._in_self_dialog:
            logger.debug("Рекурсия в _simulate_assistant_response - использую fallback")
            return f"Анализ темы '{topic_match}': система исследует базовые аспекты проблемы."

        if self.brain and hasattr(self.brain, 'self_reasoning_engine') and self.brain.self_reasoning_engine:
            if hasattr(self.brain.self_reasoning_engine, 'process_query'):
                try:
                    result = self.brain.self_reasoning_engine.process_query(
                        full_query,
                        user_context=context if context else {}
                    )
                    if result and result.get('response'):
                        return result['response'][:500]
                except Exception as e:
                    logger.debug(f"Не удалось использовать SelfReasoningEngine: {e}")

        if self.brain and hasattr(self.brain, 'process_query'):
            try:
                self._in_self_dialog = True
                try:
                    response = self.brain.process_query(full_query, user_context=context if context else {})
                    if response and isinstance(response, dict):
                        response = response.get('response', response.get('text', ''))
                    return response[:500] if response and len(response) > 500 else (response or '')
                finally:
                    self._in_self_dialog = False
            except Exception as e:
                logger.debug(f"Не удалось получить ответ от brain: {e}")

        return f"Анализ темы '{topic_match}': система исследует базовые аспекты проблемы."

    def _simulate_critic_response(self, turns: List[DialogTurn], topic: str) -> str:
        """Симулирует ответ критика с проверкой противоречий."""
        if not turns:
            return "Нет данных для критики."

        last_content = turns[-1].content if turns else ""

        contradictions_found = []
        if (self.brain and hasattr(self.brain, 'contradiction_manager') and
            self.brain.contradiction_manager and
            hasattr(self.brain.contradiction_manager, 'detect_contradictions')):
            try:
                contr_result = self.brain.contradiction_manager.detect_contradictions()
                if contr_result and isinstance(contr_result, dict):
                    contradictions = contr_result.get('contradictions', [])
                    if contradictions:
                        contradictions_found = [str(c)[:50] for c in contradictions[:3]]
            except Exception as e:
                logger.debug(f"Ошибка проверки противоречий: {e}")

        gaps = self._identify_knowledge_gaps(last_content, topic)

        result_parts = []
        if contradictions_found:
            result_parts.append(f"Противоречия: {', '.join(contradictions_found)}")

        gaps_text = "; ".join(gaps) if gaps else "не выявлены"
        result_parts.append(f"Пробелы в знаниях: {gaps_text}")

        return "Критический анализ: " + ". ".join(result_parts) + "."

    def _simulate_learner_response(self, turns: List[DialogTurn], topic: str) -> str:
        """Симулирует ответ обучающегося."""
        gaps = self._identify_knowledge_gaps("", topic)
        if gaps:
            actions = [f"Изучить: {gap}" for gap in gaps[:3]]
            return f"План обучения: {'; '.join(actions)}."
        return "Обучение: текущие знания достаточны для данной темы."

    def _simulate_teacher_response(self, turns: List[DialogTurn], topic: str) -> str:
        """Симулирует ответ учителя."""
        for turn in turns:
            if turn.role == DialogRole.CRITIC:
                _ = turn.content
            elif turn.role == DialogRole.LEARNER:
                _ = turn.content

        gaps = self._identify_knowledge_gaps("", topic)
        if gaps:
            recommendations = [f"Рекомендация: углубить знания в области {gap}" for gap in gaps[:2]]
            return f"Наставление: {'; '.join(recommendations)}."
        return "Текущий уровень знаний соответствует требованиям."

    def _simulate_observer_response(self, turns: List[DialogTurn], topic: str) -> str:
        """Симулирует ответ наблюдателя."""
        avg_quality = sum(t.quality_score for t in turns) / len(turns) if turns else 0

        if avg_quality >= self.min_quality_threshold:
            outcome = "положительный"
        elif avg_quality >= 0.4:
            outcome = "нейтральный"
        else:
            outcome = "требует улучшения"

        gaps = self._identify_knowledge_gaps("", topic)
        return f"Наблюдение: общий исход диалога {outcome}. Требуется работа над {len(gaps)} аспектами."

    def _identify_knowledge_gaps(self, content: str, topic: str) -> List[str]:
        """Выявляет пробелы в знаниях на основе анализа контента."""
        gaps = []
        topic_lower = topic.lower()
        content_lower = content.lower() if content else ""

        uncertainty_indicators = [
            "не знаю", "не могу", "не уверен", "возможно", "вероятно",
            "недостаточно", "сложно", "требует", "необходимо изучить"
        ]
        has_uncertainty = any(indicator in content_lower for indicator in uncertainty_indicators)

        topic_keywords = {
            "анализ": "методы анализа данных",
            "analysis": "методы анализа данных",
            "обучение": "алгоритмы машинного обучения",
            "learning": "алгоритмы машинного обучения",
            "знание": "управление знаниями",
            "knowledge": "управление знаниями",
            "когнитив": "когнитивные функции",
            "cognitive": "когнитивные функции",
            "мышление": "когнитивные функции",
            "thinking": "когнитивные функции",
            "нейро": "нейронные сети",
            "neuro": "нейронные сети"
        }

        for keyword, gap in topic_keywords.items():
            if keyword in topic_lower:
                gaps.append(gap)

        if not gaps and has_uncertainty:
            gaps.append("общие концепции предметной области")
        elif not gaps:
            return []

        self.stats["knowledge_gaps_identified"] += len(gaps)
        return gaps

    def _assess_turn_quality(self, turn: DialogTurn, dialog: SelfDialog) -> float:
        """Оценивает качество хода диалога."""
        score = 0.5
        if len(turn.content) > 50:
            score += 0.1
        if any(word in turn.content.lower() for word in ["анализ", "проблем", "решени", "вывод"]):
            score += 0.2
        if turn.role == DialogRole.CRITIC and "пробел" in turn.content.lower():
            score += 0.1
        if turn.role == DialogRole.LEARNER and ("изучить" in turn.content.lower() or "план" in turn.content.lower()):
            score += 0.1
        return min(1.0, max(0.0, score))

    def _should_end_dialog(self, dialog: SelfDialog) -> bool:
        """Определяет, нужно ли завершить диалог."""
        if len(dialog.turns) >= self.max_dialog_turns:
            return True
        if len(dialog.turns) >= 4:
            observer_turns = [t for t in dialog.turns if t.role == DialogRole.OBSERVER]
            if observer_turns and observer_turns[-1].quality_score >= self.min_quality_threshold:
                return True
        return False

    def _analyze_dialog_outcome(self, dialog: SelfDialog) -> None:
        """Анализирует исход диалога."""
        if not dialog.turns:
            dialog.outcome = "no_content"
            dialog.learning_type = None
            return

        avg_quality = sum(t.quality_score for t in dialog.turns) / len(dialog.turns)

        if avg_quality >= self.min_quality_threshold:
            dialog.outcome = "successful"
            if avg_quality >= 0.8:
                dialog.learning_type = LearningType.REFINEMENT
            else:
                dialog.learning_type = LearningType.EXPANSION
        else:
            dialog.outcome = "needs_improvement"
            dialog.learning_type = LearningType.UPDATING

        gaps = []
        for turn in dialog.turns:
            if turn.role == DialogRole.CRITIC:
                gap_text = turn.content
                if "пробелы" in gap_text.lower():
                    gap_start = gap_text.lower().find(":") + 1
                    gap_end = gap_text.find(".")
                    if gap_start > 0 and gap_end > gap_start:
                        gap_content = gap_text[gap_start:gap_end].strip()
                        gaps = [g.strip() for g in gap_content.split(";") if g.strip()]

        dialog.knowledge_gaps = gaps

        actions = []
        for turn in dialog.turns:
            if turn.role == DialogRole.LEARNER:
                if "план" in turn.content.lower():
                    actions.append("Создан план обучения")
                if "изучить" in turn.content.lower():
                    actions.append("Инициировано изучение нового материала")
            elif turn.role == DialogRole.TEACHER:
                if "рекомендац" in turn.content.lower():
                    actions.append("Даны рекомендации по улучшению")

        dialog.actions_taken = actions

        if dialog.outcome == "successful":
            self.stats["successful_learning"] += 1
        self.stats["actions_taken"] += len(actions)
