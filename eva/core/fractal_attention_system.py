# fractal_attention_system.py
"""Система фрактального внимания для ЕВА."""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger("eva.fractal_attention")

# Импорты из других модулей системы
try:
    from .self_dialog_manager import SelfDialogManager
except ImportError:
    # Заглушка для случаев, когда модуль ещё не создан
    class SelfDialogManager:
        def __init__(self, core_brain):
            self.core_brain = core_brain

        def start_dialog(self):
            pass

        def stop(self):
            pass

try:
    from .contradiction_resolver import ContradictionResolver
except ImportError:
    # Заглушка для случаев, когда модуль ещё не создан
    class ContradictionResolver:
        def __init__(self, core_brain):
            self.core_brain = core_brain
            self.active_contradictions = []

        def has_active_contradictions(self):
            return len(self.active_contradictions) > 0

        def check_response(self, question, response):
            pass

try:
    from .learning_scheduler import LearningScheduler
except ImportError:
    # Заглушка для случаев, когда модуль ещё не создан
    class LearningScheduler:
        def __init__(self, core_brain):
            self.core_brain = core_brain
            self.pending_opportunities = []

        def schedule_learning(self, question, response):
            pass

try:
    from .system_optimizer import SystemOptimizer
except ImportError:
    # Заглушка для случаев, когда модуль ещё не создан
    class SystemOptimizer:
        def __init__(self, core_brain):
            self.core_brain = core_brain

        def enter_power_saving_mode(self):
            pass


class FractalAttentionSystem:
    """Ядро системы, объединяющее фрактальную память и динамический фокус внимания."""

    def __init__(self, core_brain):
        self.core_brain = core_brain
        self.current_focus = None
        self.dialog_manager = SelfDialogManager(self)
        self.contradiction_resolver = ContradictionResolver(self)
        self.learning_scheduler = LearningScheduler(self)
        self.system_optimizer = SystemOptimizer(self)
        self.logger = logging.getLogger("eva.fractal_attention")

        # Инициализация начального фокуса
        self._reset_focus()

    def _reset_focus(self):
        """Сбрасывает текущий фокус внимания."""
        self.current_focus = {
            "primary_concepts": [],
            "context": "",
            "timestamp": time.time(),
            "priority": 1.0,
            "dialog_type": "exploration",
            "attention_span": 0.0
        }

    def update_focus(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Обновляет фокус внимания на основе нового запроса и контекста."""
        try:
            # Определяем домен знаний для запроса
            domain = self._identify_domain(query)

            # Определяем тип диалога
            dialog_type = self._determine_dialog_type(query)

            # Извлекаем первичные концепты
            primary_concepts = self._extract_primary_concepts(query)

            # Обновляем фокус
            self.current_focus = {
                "primary_concepts": primary_concepts,
                "domain": domain,
                "context": query,
                "timestamp": time.time(),
                "priority": self._calculate_priority(query, context),
                "dialog_type": dialog_type,
                "attention_span": self._calculate_attention_span(query),
                "raw_query": query
            }

            # Строим горячее окно
            self._build_hot_window()

            # Запускаем самодиалог при необходимости
            if self._should_start_dialog():
                self.dialog_manager.start_dialog()

            return self.current_focus
        except Exception as e:
            self.logger.error(f"Ошибка обновления фокуса внимания: {e}")
            return self.current_focus

    def _identify_domain(self, query: str) -> str:
        """Определяет домен знаний для запроса."""
        # Простая классификация по ключевым словам
        domains = {
            "технологии": ["программирование", "алгоритм", "данные", "система", "компьютер"],
            "наука": ["физика", "математика", "химия", "биология", "исследование"],
            "искусство": ["музыка", "живопись", "литература", "творчество"],
            "философия": ["смысл", "сущность", "бытие", "сознание", "этика"],
            "бизнес": ["менеджмент", "финансы", "маркетинг", "стратегия", "экономика"],
            "здоровье": ["медицина", "питание", "упражнения", "психология", "здоровый образ жизни"]
        }

        query_lower = query.lower()
        max_matches = 0
        best_domain = "general"

        for domain, keywords in domains.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches > max_matches:
                max_matches = matches
                best_domain = domain

        return best_domain

    def _determine_dialog_type(self, query: str) -> str:
        """Определяет тип диалога на основе запроса."""
        # Проверяем на наличие противоречий
        if self._contains_contradiction(query):
            return "contradiction_resolution"

        # Проверяем на временные индикаторы
        if self._is_time_sensitive(query):
            return "time_sensitive"

        # Проверяем, является ли запрос уточнением
        if self._is_refinement(query):
            return "refinement"

        # По умолчанию - исследовательский диалог
        return "exploration"

    def _contains_contradiction(self, query: str) -> bool:
        """Проверяет, содержит ли запрос противоречие."""
        contradiction_indicators = ["но", "однако", "противоречит", "несоответствие", "несогласованность"]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in contradiction_indicators)

    def _is_time_sensitive(self, query: str) -> bool:
        """Проверяет, является ли запрос временно-зависимым."""
        time_indicators = ["сейчас", "сегодня", "вчера", "завтра", "последний", "новый", "актуальный"]
        return any(indicator in query.lower() for indicator in time_indicators)

    def _is_refinement(self, query: str) -> bool:
        """Проверяет, является ли запрос уточнением предыдущего."""
        refinement_indicators = ["более подробно", "подробнее", "уточнить", "развить", "расширить"]
        return any(indicator in query.lower() for indicator in refinement_indicators)

    def _extract_primary_concepts(self, query: str) -> List[str]:
        """Извлекает первичные концепты из запроса."""
        # Здесь должна быть интеграция с NLP-моделями для извлечения сущностей
        # Для упрощения используем простой метод
        common_words = ["и", "в", "на", "с", "к", "от", "до", "по", "для", "о", "об", "перед", "при"]
        words = [word.lower().strip(".,?!") for word in query.split()
                if word.lower() not in common_words and len(word) > 3]

        # Уникальные слова как первичные концепты
        return list(set(words))[:5]  # Ограничиваем до 5 концептов

    def _calculate_priority(self, query: str, context: Optional[Dict[str, Any]]) -> float:
        """Рассчитывает приоритет фокуса внимания."""
        # Базовый приоритет
        priority = 0.5

        # Увеличиваем приоритет для коротких, конкретных запросов
        if len(query.split()) < 8:
            priority += 0.2

        # Увеличиваем приоритет для запросов с вопросительными словами
        question_words = ["как", "почему", "зачем", "когда", "где", "кто"]
        if any(word in query.lower() for word in question_words):
            priority += 0.15

        # Увеличиваем приоритет для временно-зависимых запросов
        if self._is_time_sensitive(query):
            priority += 0.1

        # Нормализуем в диапазон [0, 1]
        return min(1.0, priority)

    def _calculate_attention_span(self, query: str) -> float:
        """Рассчитывает продолжительность внимания для запроса."""
        # Базовая продолжительность
        base_span = 0.3  # от 0 до 1

        # Увеличиваем для сложных запросов
        if len(query.split()) > 15:
            base_span += 0.2

        # Увеличиваем для исследовательских запросов
        if self._determine_dialog_type(query) == "exploration":
            base_span += 0.15

        # Нормализуем в диапазон [0, 1]
        return min(1.0, base_span)

    def _build_hot_window(self):
        """Строит горячее окно на основе текущего фокуса."""
        try:
            if not hasattr(self.core_brain, 'memory_manager') or not self.core_brain.memory_manager:
                self.logger.debug("MemoryManager недоступен для построения горячего окна")
                return

            # Получаем данные из горячего окна
            hot_window_data = self._get_hot_window_data()

            # Обрабатываем восходящее направление
            upward_result = self._process_upward(self.current_focus["context"])

            # Выполняем горизонтальную обработку
            self._process_horizontal(upward_result)

            # Обновляем горячее окно
            self._update_hot_window(upward_result)
        except Exception as e:
            self.logger.error(f"Ошибка построения горячего окна: {e}")

    def _get_hot_window_data(self) -> Dict:
        """Получает данные из горячего окна."""
        try:
            if hasattr(self.core_brain, 'memory_manager') and self.core_brain.memory_manager:
                return self.core_brain.memory_manager.get_hot_window_data()
            return {}
        except Exception as e:
            self.logger.debug(f"Ошибка получения данных горячего окна: {e}")
            return {}

    def _process_upward(self, query: str) -> Dict:
        """Выполняет восходящую обработку от деталей к обобщениям."""
        try:
            # Получаем данные из горячего окна
            hot_window = self._get_hot_window_data()

            # Здесь должна быть сложная логика анализа и обобщения
            # Для упрощения возвращаем заглушку
            return {
                "aggregated_info": {},
                "high_level_summary": query,
                "primary_concept": "general"
            }
        except Exception as e:
            self.logger.error(f"Ошибка восходящей обработки: {e}")
            return {
                "aggregated_info": {},
                "high_level_summary": query,
                "primary_concept": "general"
            }

    def _process_horizontal(self, upward_result: Dict) -> Dict:
        """Выполняет горизонтальную обработку для установления новых связей."""
        try:
            primary_concept = upward_result["primary_concept"]

            # Здесь должна быть логика поиска связей между концептами
            # Для упрощения возвращаем заглушку
            return {
                "related_concepts": [primary_concept],
                "new_connections": []
            }
        except Exception as e:
            self.logger.error(f"Ошибка горизонтальной обработки: {e}")
            return {
                "related_concepts": [],
                "new_connections": []
            }

    def _update_hot_window(self, processing_result: Dict):
        """Обновляет горячее окно на основе результатов обработки."""
        try:
            if not hasattr(self.core_brain, 'memory_manager') or not self.core_brain.memory_manager:
                return

            # Добавляем основной концепт
            if "primary_concept" in processing_result:
                self.core_brain.memory_manager.add_to_hot_window(
                    processing_result["primary_concept"], 0.9, "core_concept"
                )

            # Добавляем связанные концепты
            for concept in processing_result.get("related_concepts", []):
                self.core_brain.memory_manager.add_to_hot_window(
                    concept, 0.7, "related_concept"
                )
        except Exception as e:
            self.logger.error(f"Ошибка обновления горячего окна: {e}")

    def _should_start_dialog(self) -> bool:
        """Определяет, следует ли начать самодиалог."""
        try:
            # Проверяем тип диалога
            if self.current_focus["dialog_type"] == "contradiction_resolution":
                return True

            # Проверяем активность системы
            if hasattr(self.core_brain, 'system_monitor') and self.core_brain.system_monitor:
                health = self.core_brain.system_monitor.get_system_status()
                if health.get("status") == "degraded":
                    return True

            # Проверяем, есть ли противоречия
            if hasattr(self.contradiction_resolver, 'has_active_contradictions') and \
               self.contradiction_resolver.has_active_contradictions():
                return True

            return False
        except Exception as e:
            self.logger.debug(f"Ошибка определения необходимости диалога: {e}")
            return False

    def signal_user_activity(self):
        """Сигнализирует о пользовательской активности."""
        try:
            # Сброс таймера бездействия
            if hasattr(self, '_inactivity_timer'):
                self._inactivity_timer.cancel()

            # Перезапуск таймера
            self._inactivity_timer = threading.Timer(300.0, self._handle_inactivity)  # 5 минут
            self._inactivity_timer.start()
        except Exception as e:
            self.logger.debug(f"Ошибка сигнализации активности: {e}")

    def _handle_inactivity(self):
        """Обрабатывает длительное бездействие пользователя."""
        try:
            self.logger.info("Обнаружено длительное бездействие пользователя")

            # Переход в режим энергосбережения
            if hasattr(self.core_brain, 'system_optimizer'):
                self.core_brain.system_optimizer.enter_power_saving_mode()

            # Сброс фокуса внимания
            self._reset_focus()
        except Exception as e:
            self.logger.error(f"Ошибка обработки бездействия: {e}")

    def process_query(self, query: str) -> str:
        """
        Обрабатывает запрос через динамический фокус внимания.

        Args:
            query: Текст запроса

        Returns:
            str: Ответ системы
        """
        try:
            # 1. Инициализация фокуса внимания
            self._initialize_attention_focus(query)

            # 2. Восходящая обработка
            high_level_concepts = self._process_upward(query)

            # 3. Горизонтальная связность
            related_concepts = self._process_horizontal(high_level_concepts)

            # 4. Нисходящая обработка
            detailed_analysis = self._process_downward(related_concepts)

            # 5. Формирование ответа
            response = self._generate_response(query, detailed_analysis)

            # 6. Анализ этической корректности
            if hasattr(self.core_brain, 'ethics_framework'):
                ethical_decision = self.core_brain.ethics_framework.analyze_response(query, response)
                if not ethical_decision.get('overall_score', 1.0) > 0.7:
                    response = self._apply_ethical_corrections(response, ethical_decision)

            # 7. Адаптация горячего окна для будущих запросов
            self._adapt_hot_window_for_follow_up(query, response)

            return response

        except Exception as e:
            self.logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
            return "Извините, произошла ошибка при обработке запроса."

    def _initialize_attention_focus(self, query: str):
        """Инициализирует фокус внимания на основе запроса."""
        try:
            # Определяем основную область знаний
            domain = self._identify_domain(query)

            # Определяем тип запроса
            query_type = self._determine_query_type(query)

            # Формируем начальный фокус
            self.current_focus = {
                "domain": domain,
                "query_type": query_type,
                "primary_concepts": self._extract_primary_concepts(query),
                "secondary_concepts": self._extract_secondary_concepts(query),
                "time_sensitive": self._is_time_sensitive(query),
                "focus_strength": 1.0
            }

            # Создаем начальное горячее окно
            self._build_initial_hot_window()

        except Exception as e:
            self.logger.error(f"Ошибка инициализации фокуса внимания: {e}")

    def _extract_secondary_concepts(self, query: str) -> List[str]:
        """Извлекает вторичные концепты."""
        words = query.split()
        primary = self.current_focus.get("primary_concepts", []) if self.current_focus else []
        return [word for word in words if len(word) > 2 and word not in primary][:5]

    def _build_initial_hot_window(self):
        """Строит начальное горячее окно на основе текущего фокуса."""
        try:
            if not hasattr(self.core_brain, 'memory_manager') or not self.core_brain.memory_manager:
                self.logger.debug("MemoryManager недоступен для построения горячего окна")
                return

            # Получаем релевантные концепты из базы знаний
            primary_concepts = self.current_focus["primary_concepts"]
            candidates = []

            for concept in primary_concepts:
                try:
                    # Ищем связанные узлы в графе знаний
                    if hasattr(self, 'knowledge_graph') and self.knowledge_graph:
                        related_nodes = self.knowledge_graph.find_related(concept, limit=3)
                        for node in related_nodes:
                            priority = 0.8 if node.get('type') == 'core_concept' else 0.6
                            candidates.append((node.get('id', concept), priority, "contextual_attention"))
                except Exception as e:
                    self.logger.debug(f"Ошибка поиска связанных концептов: {e}")

            # Добавляем первичные концепты с высоким приоритетом
            for concept in primary_concepts:
                candidates.append((concept, 0.9, "core_attention"))

            # Добавляем в горячее окно через memory_manager
            if hasattr(self.core_brain.memory_manager, 'add_to_hot_window'):
                for item_id, priority, attention_type in candidates[:10]:  # Ограничиваем до 10
                    try:
                        self.core_brain.memory_manager.add_to_hot_window(item_id, priority, attention_type)
                    except Exception as e:
                        self.logger.debug(f"Ошибка добавления в горячее окно: {e}")

        except Exception as e:
            self.logger.error(f"Ошибка построения горячего окна: {e}")

    def _process_downward(self, horizontal_result: Dict) -> Dict:
        """Выполняет нисходящую обработку для применения обобщений к конкретике."""
        try:
            # Определяем, какие связи наиболее значимы
            significant_relations = []
            for concept in horizontal_result["related_concepts"][:3]:  # Ограничиваем до 3
                significant_relations.append({
                    "concept": concept,
                    "relation": "связано с",
                    "description": f"Связь между {horizontal_result['primary_concept']} и {concept}"
                })

            return {
                "primary_concept": horizontal_result["primary_concept"],
                "detailed_relations": significant_relations,
                "significant_relations": significant_relations
            }

        except Exception as e:
            self.logger.error(f"Ошибка нисходящей обработки: {e}")
            return horizontal_result

    def _generate_response(self, query: str, downward_result: Dict) -> str:
        """Генерирует ответ на основе анализа."""
        try:
            primary_concept = downward_result["primary_concept"]
            significant_relations = downward_result["significant_relations"]

            # Формируем структурированный ответ
            response_parts = []

            if significant_relations:
                response_parts.append(f"Анализируя запрос о '{primary_concept}', я нашел следующие связи:")
                for i, relation in enumerate(significant_relations, 1):
                    response_parts.append(f"{i}. {relation['description']}")
            else:
                response_parts.append(f"Запрос касается концепта '{primary_concept}'.")

            # Добавляем дополнительную информацию
            response_parts.append("\nЭто базовый анализ. Для более детального ответа уточните запрос.")

            return "\n".join(response_parts)

        except Exception as e:
            self.logger.error(f"Ошибка генерации ответа: {e}")
            return f"На основе анализа, ваш запрос касается: {query}"

    def _apply_ethical_corrections(self, response: str, ethical_decision: Dict) -> str:
        """Применяет этические коррекции к ответу."""
        try:
            if ethical_decision.get('corrections_needed', False):
                corrections = ethical_decision.get('corrections', [])
                corrected_response = response

                for correction in corrections:
                    if correction.get('type') == 'add_disclaimer':
                        corrected_response += "\n\nПримечание: " + correction.get('text', '')

                return corrected_response
            return response

        except Exception as e:
            self.logger.error(f"Ошибка применения этических коррекций: {e}")
            return response

    def _adapt_hot_window_for_follow_up(self, query: str, response: str):
        """Адаптирует горячее окно для будущих запросов."""
        try:
            # Извлекаем концепты из ответа
            response_concepts = self._extract_concepts_from_response(response)

            # Добавляем новые концепты в горячее окно
            if hasattr(self.core_brain, 'memory_manager') and self.core_brain.memory_manager:
                for concept in response_concepts[:3]:  # Ограничиваем до 3
                    try:
                        self.core_brain.memory_manager.add_to_hot_window(concept, 0.7, "response_driven")
                    except Exception as e:
                        self.logger.debug(f"Ошибка добавления концепта в горячее окно: {e}")

        except Exception as e:
            self.logger.error(f"Ошибка адаптации горячего окна: {e}")

    def _extract_concepts_from_response(self, response: str) -> List[str]:
        """Извлекает концепты из ответа."""
        # Простое извлечение ключевых слов
        words = response.split()
        concepts = [word.strip('.,!?') for word in words if len(word) > 4]
        return list(set(concepts))  # Убираем дубликаты

    def run_self_dialog(self):
        """Запускает фоновую сессию самодиалога."""
        try:
            self.dialog_manager.start_dialog()
        except Exception as e:
            self.logger.error(f"Ошибка запуска самодиалога: {e}")

    def get_system_health(self) -> Dict[str, Any]:
        """Возвращает комплексный отчет о состоянии системы."""
        try:
            health = {
                "attention_focus": self.current_focus is not None,
                "dialog_active": self.dialog_manager.running if hasattr(self.dialog_manager, 'running') else False,
                "contradictions_count": len(self.contradiction_resolver.active_contradictions) if hasattr(self.contradiction_resolver, 'active_contradictions') else 0,
                "learning_opportunities": len(self.learning_scheduler.pending_opportunities) if hasattr(self.learning_scheduler, 'pending_opportunities') else 0
            }

            # НЕ добавляем данные из core_brain - это вызывает рекурсию
            # if hasattr(self.core_brain, 'get_system_health'):
            #     core_health = self.core_brain.get_system_health()
            #     health.update(core_health)

            return health

        except Exception as e:
            self.logger.error(f"Ошибка получения здоровья системы: {e}")
            return {"status": "error", "error": str(e)}

    def _determine_query_type(self, query: str) -> str:
        """
        Определяет тип запроса.

        Args:
            query: Текст запроса

        Returns:
            str: Тип запроса
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ["что", "кто", "где", "когда", "почему", "как"]):
            return "factual"
        elif any(word in query_lower for word in ["анализ", "исследовать", "изучить"]):
            return "analytical"
        elif any(word in query_lower for word in ["создать", "написать", "разработать"]):
            return "creative"
        else:
            return "conversational"
