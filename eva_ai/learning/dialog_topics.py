"""Topic generation, management, selection, and prioritization for self-dialog learning."""
from __future__ import annotations

import logging
import time
import re
from typing import Dict, List, Any, Optional

from eva_ai.learning.dialog_types import DialogRole, DialogTurn, LearningType, SelfDialog

logger = logging.getLogger("eva_ai.self_dialog_learning")


class DialogTopicsMixin:
    """Mixin for topic generation, management, selection, and prioritization."""

    def _generate_dialog_from_conversations(self) -> None:
        """Generates self-dialog from recent conversation history."""
        if not self.brain:
            return

        if not hasattr(self.brain, 'memory_manager') or not self.brain.memory_manager:
            return

        current_time = time.time()

        self._recently_processed_topics = {
            k: v for k, v in self._recently_processed_topics.items()
            if current_time - v < self._topic_ttl
        }

        if current_time - self.last_dialog_check < self.auto_dialog_interval:
            return

        self.last_dialog_check = current_time

        try:
            if (self.brain and hasattr(self.brain, 'memory_manager') and
                self.brain.memory_manager and
                hasattr(self.brain.memory_manager, 'get_recent_interactions')):
                conversation_history = self.brain.memory_manager.get_recent_interactions(limit=10)
            else:
                conversation_history = []

            if not conversation_history or not isinstance(conversation_history, list):
                logger.debug("No conversation history available for self-dialog generation")
                return

            if not all(isinstance(conv, dict) and 'query' in conv for conv in conversation_history):
                logger.debug("Invalid conversation history structure")
                return

            topics = []
            for conv in conversation_history:
                query = conv.get('query', '')
                if query and len(query) > 10:
                    topic = query[:100]
                    if topic not in self._recently_processed_topics:
                        topics.append(topic)

            if topics:
                topic = topics[0]
                self._recently_processed_topics[topic] = current_time
                logger.info(f"Creating self-dialog from conversation: {topic[:50]}...")
                self.create_dialog(topic=topic, context={"source": "conversation_history"})
                self.stats["total_dialogs"] += 1
            else:
                logger.debug("All recent topics already processed, skipping dialog generation")

        except Exception as e:
            logger.debug(f"Error generating dialog from conversations: {e}")

    def extract_key_concepts(self, query: str) -> List[str]:
        """
        Извлекает ключевые понятия из запроса для анализа.

        Args:
            query: Запрос пользователя

        Returns:
            Список ключевых понятий
        """
        query = query.lower().strip()

        simple_patterns = ['привет', 'здравствуй', 'как дела', 'спасибо', 'пока', 'да', 'нет']
        for pattern in simple_patterns:
            if query == pattern or query.startswith(pattern + ' '):
                return []

        words = re.findall(r'\b[а-яёa-z]{3,}\b', query)

        stop_words = {'это', 'что', 'как', 'где', 'когда', 'почему', 'потому', 'для', 'от', 'до', 'при', 'над', 'под', 'между', 'среди', 'который', 'которая', 'которое', 'которые', 'этот', 'эта', 'эти', 'тот', 'та', 'те', 'свой', 'своя', 'своё', 'свои', 'весь', 'всё', 'все', 'один', 'одна', 'одно', 'одни', 'два', 'три', 'четыре', 'пять', 'либо', 'нибудь', 'только', 'уже', 'ещё', 'еще', 'быть', 'был', 'была', 'было', 'были', 'иметь', 'есть', 'will', 'are', 'was', 'were', 'have', 'has', 'had'}

        concepts = [w for w in words if w not in stop_words and len(w) > 2]
        concepts = list(set(concepts))

        return concepts[:10]

    def analyze_unknown_concepts(self, query: str, response: str) -> List[Dict[str, Any]]:
        """
        Анализирует какие понятия из запроса модель не знает.

        Args:
            query: Запрос пользователя
            response: Ответ модели

        Returns:
            Список неизвестных понятий с метаданными
        """
        unknown_patterns = [
            'я не знаю', 'не знаю', 'не могу ответить', 'не имею информации',
            'неизвестно', 'затрудняюсь', 'недостаточно информации', 'мне неизвестно',
            "i don't know", 'i cannot', 'i do not know', 'unable to'
        ]

        response_lower = response.lower()

        if not any(p in response_lower for p in unknown_patterns):
            return []

        concepts = self.extract_key_concepts(query)

        unknown_concepts = []
        for concept in concepts:
            if concept not in response_lower or len(response_lower) < len(concept) * 3:
                unknown_concepts.append({
                    'concept': concept,
                    'source': query,
                    'type': 'semantic_gap',
                    'priority': 0.7
                })

        return unknown_concepts

    def search_and_learn_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Выполняет поиск и обучение по неизвестным понятиям.

        Args:
            concepts: Список неизвестных понятий

        Returns:
            Результаты обучения
        """
        if not self.brain:
            return []

        results = []

        for concept_info in concepts[:5]:
            concept = concept_info.get('concept', '')
            if not concept or len(concept) < 3:
                continue

            try:
                web_search = getattr(self.brain, 'web_search_engine', None)
                if web_search and hasattr(web_search, 'search'):
                    search_result = web_search.search(concept, max_results=3)

                    results.append({
                        'concept': concept,
                        'search_results': search_result.get('results', []) if search_result else [],
                        'status': 'learned'
                    })

                    logger.info(f"Самодиалог: изучено понятие '{concept}' через веб-поиск")

                kg = getattr(self.brain, 'knowledge_graph', None)
                if kg and hasattr(kg, 'add_entity'):
                    try:
                        kg.add_entity(
                            name=concept,
                            entity_type='learned_concept',
                            properties={'source': 'self_dialog_learning', 'learned_from': 'web_search'}
                        )
                    except Exception as e:
                        logger.debug(f"Не удалось сохранить в граф: {e}")

            except Exception as e:
                logger.error(f"Ошибка обучения понятия {concept}: {e}")

        return results
