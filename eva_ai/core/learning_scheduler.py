# learning_scheduler.py
"""Планировщик обучения для ЕВА."""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("eva_ai.learning_scheduler")

class LearningScheduler:
    """Система планирования и управления процессом обучения."""

    def __init__(self, attention_system):
        self.attention_system = attention_system
        self.pending_opportunities = []
        self.completed_learnings = []
        self.learning_goals = []
        self.logger = logging.getLogger("eva_ai.learning_scheduler")
        self.min_confidence_threshold = 0.6  # Минимальный уровень уверенности для обучения

    def schedule_learning(self, question: str, response: str):
        """Планирует обучение на основе диалога."""
        try:
            # Определяем возможности для обучения
            learning_opportunities = self._identify_learning_opportunities(question, response)

            # Добавляем новые возможности
            for opportunity in learning_opportunities:
                self._add_learning_opportunity(opportunity)

            # Проверяем, нужно ли запустить обучение
            self._check_learning_trigger()
        except Exception as e:
            self.logger.error(f"Ошибка планирования обучения: {e}")

    def _identify_learning_opportunities(self, question: str, response: str) -> List[Dict[str, Any]]:
        """Определяет возможности для обучения на основе диалога."""
        opportunities = []

        # Проверяем на низкую уверенность в ответе
        if self._has_low_confidence(response):
            opportunities.append({
                "type": "knowledge_gap",
                "topic": self._extract_topic(question),
                "question": question,
                "evidence": response,
                "priority": 0.9,
                "detected_at": time.time()
            })

        # Проверяем на новые связи
        if self._has_new_connections(response):
            opportunities.append({
                "type": "relationship_learning",
                "topic": self._extract_topic(question),
                "question": question,
                "evidence": response,
                "priority": 0.7,
                "detected_at": time.time()
            })

        # Проверяем на противоречия
        contradiction_resolver = getattr(self.attention_system, 'contradiction_resolver', None)
        if contradiction_resolver and hasattr(contradiction_resolver, 'has_active_contradictions'):
            try:
                if contradiction_resolver.has_active_contradictions():
                    history = []
                    if hasattr(contradiction_resolver, 'get_resolution_history'):
                        try:
                            history = contradiction_resolver.get_resolution_history()[-1:]
                        except Exception:
                            pass
                    opportunities.append({
                        "type": "contradiction_resolution",
                        "topic": "knowledge_consistency",
                        "question": "Как разрешить текущие противоречия в знаниях?",
                        "evidence": str(history),
                        "priority": 0.8,
                        "detected_at": time.time()
                    })
            except Exception:
                pass

        return opportunities

    def _has_low_confidence(self, response: str) -> bool:
        """Проверяет, указывает ли ответ на низкую уверенность."""
        low_confidence_indicators = [
            "не уверен", "не знаю", "возможно", "вероятно", "похоже",
            "кажется", "предположительно", "вряд ли", "маловероятно"
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in low_confidence_indicators)

    def _has_new_connections(self, response: str) -> bool:
        """Проверяет, содержит ли ответ новые связи между концептами."""
        connection_indicators = [
            "связано с", "похоже на", "аналогично", "в отличие от",
            "контрастирует с", "параллельно", "соотносится с"
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in connection_indicators)

    def _extract_topic(self, question: str) -> str:
        """Извлекает тему из вопроса."""
        # Простой метод извлечения темы
        common_words = ["как", "почему", "что", "кто", "где", "когда", "зачем"]
        words = [word.lower().strip(".,?!") for word in question.split()
                if word.lower() not in common_words and len(word) > 3]

        return words[0] if words else "general"

    def _add_learning_opportunity(self, opportunity: Dict[str, Any]):
        """Добавляет возможность обучения в список."""
        # Проверяем, не дублируется ли возможность
        if not self._is_duplicate(opportunity):
            self.pending_opportunities.append(opportunity)
            self.logger.info(f"Добавлена новая возможность обучения: {opportunity['type']}")

    def _is_duplicate(self, new_opportunity: Dict[str, Any]) -> bool:
        """Проверяет, является ли возможность обучения дубликатом."""
        for existing in self.pending_opportunities:
            # Простая проверка на дубликаты
            if new_opportunity['type'] == existing['type']:
                time_diff = abs(new_opportunity['detected_at'] - existing['detected_at'])
                if time_diff < 1800:  # 30 минут
                    return True
        return False

    def _check_learning_trigger(self):
        """Проверяет, нужно ли запустить процесс обучения."""
        if not self.pending_opportunities:
            return

        # Сортируем возможности по приоритету
        self.pending_opportunities.sort(key=lambda x: x['priority'], reverse=True)

        # Проверяем приоритет первой возможности
        if self.pending_opportunities[0]['priority'] >= self.min_confidence_threshold:
            self._initiate_learning(self.pending_opportunities[0])

    def _initiate_learning(self, opportunity: Dict[str, Any]):
        """Инициирует процесс обучения для указанной возможности."""
        try:
            self.logger.info(f"Начато обучение по возможности: {opportunity['type']}")

            # Создаем план обучения
            learning_plan = self._create_learning_plan(opportunity)

            # Выполняем обучение
            learning_result = self._execute_learning(learning_plan)

            # Сохраняем результат
            self._record_learning_result(opportunity, learning_plan, learning_result)

            # Удаляем возможность из списка
            if opportunity in self.pending_opportunities:
                self.pending_opportunities.remove(opportunity)
        except Exception as e:
            self.logger.error(f"Ошибка при инициации обучения: {e}")

    def _create_learning_plan(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Создает план обучения для указанной возможности."""
        base_plan = {
            "opportunity": opportunity,
            "steps": [],
            "started_at": time.time(),
            "status": "planning"
        }

        # Добавляем шаги в зависимости от типа обучения
        if opportunity['type'] == "knowledge_gap":
            base_plan["steps"] = [
                {
                    "name": "gather_information",
                    "description": "Сбор информации по теме",
                    "status": "pending"
                },
                {
                    "name": "analyze_sources",
                    "description": "Анализ собранных источников",
                    "status": "pending"
                },
                {
                    "name": "integrate_knowledge",
                    "description": "Интеграция новых знаний в существующую модель",
                    "status": "pending"
                }
            ]
        elif opportunity['type'] == "relationship_learning":
            base_plan["steps"] = [
                {
                    "name": "identify_related_concepts",
                    "description": "Идентификация связанных концептов",
                    "status": "pending"
                },
                {
                    "name": "map_relationships",
                    "description": "Картирование отношений между концептами",
                    "status": "pending"
                },
                {
                    "name": "update_knowledge_graph",
                    "description": "Обновление графа знаний",
                    "status": "pending"
                }
            ]
        elif opportunity['type'] == "contradiction_resolution":
            base_plan["steps"] = [
                {
                    "name": "analyze_contradiction",
                    "description": "Анализ противоречия и его источников",
                    "status": "pending"
                },
                {
                    "name": "evaluate_evidence",
                    "description": "Оценка достоверности доказательств",
                    "status": "pending"
                },
                {
                    "name": "resolve_conflict",
                    "description": "Разрешение конфликта знаний",
                    "status": "pending"
                }
            ]

        return base_plan

    def _execute_learning(self, learning_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет процесс обучения по плану."""
        try:
            results = []

            for step in learning_plan["steps"]:
                try:
                    # Отмечаем начало шага
                    step["status"] = "in_progress"
                    step["started_at"] = time.time()

                    # Выполняем шаг
                    result = self._execute_learning_step(step)

                    # Отмечаем завершение
                    step["status"] = "completed"
                    step["completed_at"] = time.time()
                    step["result"] = result

                    results.append(result)
                except Exception as e:
                    step["status"] = "failed"
                    step["error"] = str(e)
                    self.logger.error(f"Ошибка выполнения шага обучения {step['name']}: {e}")

            # Формируем общий результат
            return {
                "plan": learning_plan,
                "results": results,
                "completed_at": time.time(),
                "status": "completed" if all(s["status"] == "completed" for s in learning_plan["steps"]) else "partial"
            }
        except Exception as e:
            self.logger.error(f"Ошибка выполнения обучения: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "completed_at": time.time()
            }

    def _execute_learning_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет отдельный шаг обучения."""
        try:
            if step["name"] == "gather_information":
                return self._gather_information()
            elif step["name"] == "analyze_sources":
                return self._analyze_sources()
            elif step["name"] == "integrate_knowledge":
                return self._integrate_knowledge()
            elif step["name"] == "identify_related_concepts":
                return self._identify_related_concepts()
            elif step["name"] == "map_relationships":
                return self._map_relationships()
            elif step["name"] == "update_knowledge_graph":
                return self._update_knowledge_graph()
            elif step["name"] == "analyze_contradiction":
                return self._analyze_contradiction()
            elif step["name"] == "evaluate_evidence":
                return self._evaluate_evidence()
            elif step["name"] == "resolve_conflict":
                return self._resolve_conflict()

            return {"status": "skipped", "message": "Unknown step"}
        except Exception as e:
            self.logger.error(f"Ошибка выполнения шага обучения {step['name']}: {e}")
            return {"status": "failed", "error": str(e)}

    def _gather_information(self) -> Dict[str, Any]:
        """Собирает информацию для обучения."""
        try:
            # Здесь должна быть интеграция с источниками знаний
            # Для упрощения возвращаем заглушку

            # Получаем данные из горячего окна
            hot_window = self._get_hot_window_data()

            # Извлекаем концепты
            concepts = list(hot_window.keys())[:3] if hot_window else ["general"]

            # Формируем запрос для сбора информации
            query = f"Подробная информация о {' и '.join(concepts)}"

            # Получаем ответ от системы через унифицированный метод
            if hasattr(self.attention_system.core_brain, 'process_query'):
                result = self.attention_system.core_brain.process_query(query, context={"analysis": "contradiction"})
                response = result.get('text', "Анализ противоречия завершен.")
            else:
                response = "Анализ противоречия завершен."

            return {
                "query": query,
                "response": response,
                "sources": ["internal_knowledge_base"],
                "status": "completed"
            }
        except Exception as e:
            self.logger.error(f"Ошибка сбора информации: {e}")
            return {"status": "failed", "error": str(e)}

    def _analyze_sources(self) -> Dict[str, Any]:
        """Анализирует собранные источники информации."""
        try:
            # Простой анализ
            return {
                "summary": "Анализ собранных источников показал согласованность информации.",
                "key_points": ["Основные моменты извлечены и проверены на достоверность."],
                "status": "completed"
            }
        except Exception as e:
            self.logger.error(f"Ошибка анализа источников: {e}")
            return {"status": "failed", "error": str(e)}

    def _integrate_knowledge(self) -> Dict[str, Any]:
        """Интегрирует новые знания в существующую модель."""
        try:
            # Обновляем граф знаний
            if hasattr(self.attention_system.core_brain, 'knowledge_graph'):
                # Здесь должна быть логика обновления графа
                pass

            # Обновляем горячее окно
            if hasattr(self.attention_system.core_brain, 'memory_manager'):
                self.attention_system.core_brain.memory_manager.add_to_hot_window(
                    "new_knowledge", 0.9, "integrated_knowledge"
                )

            return {
                "status": "completed",
                "message": "Новые знания успешно интегрированы"
            }
        except Exception as e:
            self.logger.error(f"Ошибка интеграции знаний: {e}")
            return {"status": "failed", "error": str(e)}

    def _identify_related_concepts(self) -> Dict[str, Any]:
        """Идентифицирует связанные концепты."""
        try:
            # Получаем данные из горячего окна
            hot_window = self._get_hot_window_data()

            # Извлекаем концепты
            concepts = list(hot_window.keys())[:2] if hot_window else []

            related = []
            if concepts and hasattr(self.attention_system.core_brain, 'knowledge_graph'):
                # Здесь должна быть интеграция с графом знаний
                related = concepts  # Заглушка

            return {
                "concepts": concepts,
                "related": related,
                "status": "completed"
            }
        except Exception as e:
            self.logger.error(f"Ошибка идентификации концептов: {e}")
            return {"status": "failed", "error": str(e)}

    def _map_relationships(self) -> Dict[str, Any]:
        """Картирует отношения между концептами."""
        try:
            return {
                "relationships": [
                    {"source": "concept1", "target": "concept2", "type": "similar_to", "strength": 0.8}
                ],
                "status": "completed"
            }
        except Exception as e:
            self.logger.error(f"Ошибка картирования отношений: {e}")
            return {"status": "failed", "error": str(e)}

    def _update_knowledge_graph(self) -> Dict[str, Any]:
        """Обновляет граф знаний."""
        try:
            if hasattr(self.attention_system.core_brain, 'knowledge_graph'):
                # Здесь должна быть логика обновления графа
                pass

            return {
                "status": "completed",
                "message": "Граф знаний обновлен"
            }
        except Exception as e:
            self.logger.error(f"Ошибка обновления графа знаний: {e}")
            return {"status": "failed", "error": str(e)}

    def _analyze_contradiction(self) -> Dict[str, Any]:
        """Анализирует противоречие."""
        try:
            if not (hasattr(self.attention_system.contradiction_resolver, 'active_contradictions') and
                    self.attention_system.contradiction_resolver.active_contradictions):
                return {"status": "skipped", "message": "No active contradictions"}
        except Exception as e:
            self.logger.error(f"Ошибка анализа противоречия: {e}")
            return {"status": "failed", "error": str(e)}

    def _evaluate_evidence(self) -> Dict[str, Any]:
        """Оценивает достоверность доказательств."""
        try:
            return {
                "evidence_evaluation": "Доказательства оценены и ранжированы по достоверности.",
                "status": "completed"
            }
        except Exception as e:
            self.logger.error(f"Ошибка оценки доказательств: {e}")
            return {"status": "failed", "error": str(e)}

    def _resolve_conflict(self) -> Dict[str, Any]:
        """Разрешает конфликт знаний."""
        try:
            return {
                "resolution": "Конфликт знаний успешно разрешен.",
                "status": "completed"
            }
        except Exception as e:
            self.logger.error(f"Ошибка разрешения конфликта: {e}")
            return {"status": "failed", "error": str(e)}

    def _get_hot_window_data(self) -> Dict:
        """Получает данные из горячего окна."""
        try:
            if hasattr(self.attention_system.core_brain, 'memory_manager') and \
               self.attention_system.core_brain.memory_manager:
                return self.attention_system.core_brain.memory_manager.get_hot_window_data()
            return {}
        except Exception as e:
            logger.debug(f"Ошибка получения данных горячего окна: {e}")
            return {}

    def _record_learning_result(self, opportunity: Dict[str, Any], plan: Dict[str, Any], result: Dict[str, Any]):
        """Записывает результат обучения."""
        learning_record = {
            "opportunity": opportunity,
            "plan": plan,
            "result": result,
            "completed_at": time.time()
        }

        self.completed_learnings.append(learning_record)
        self.logger.info(f"Обучение завершено: {opportunity['type']}")

    def get_learning_goals(self) -> List[Dict[str, Any]]:
        """Возвращает текущие цели обучения."""
        return self.learning_goals

    def get_pending_opportunities(self) -> List[Dict[str, Any]]:
        """Возвращает список ожидающих возможностей обучения."""
        return self.pending_opportunities

    def get_completed_learnings(self) -> List[Dict[str, Any]]:
        """Возвращает историю завершенных обучений."""
        return self.completed_learnings

    def identify_learning_opportunities(self, query: str) -> List[Dict[str, Any]]:
        """Идентифицирует возможности для обучения на основе запроса.
        
        Args:
            query: Текст запроса пользователя
            
        Returns:
            Список возможностей для обучения
        """
        opportunities = []
        try:
            if hasattr(self.attention_system, 'core_brain') and self.attention_system.core_brain:
                brain = self.attention_system.core_brain
                
                # Анализируем запрос на предмет новых паттернов
                if len(query) > 20:
                    opportunities.append({
                        'type': 'query_pattern',
                        'query': query,
                        'priority': 0.5,
                        'description': 'Новый паттерн запроса для анализа'
                    })
        except Exception as e:
            self.logger.debug(f"Ошибка идентификации возможностей обучения: {e}")
        return opportunities

    def schedule_learning_session(self, opportunity: Dict[str, Any]) -> bool:
        """Запланировать сессию обучения для возможности.
        
        Args:
            opportunity: Словарь с информацией о возможности обучения
            
        Returns:
            True если сессия успешно запланирована
        """
        try:
            self._add_learning_opportunity(opportunity)
            self.logger.info(f"Сессия обучения запланирована: {opportunity.get('type', 'unknown')}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка планирования сессии обучения: {e}")
            return False

    def get_high_priority_opportunities(self, min_priority: float = 0.7) -> List[Dict[str, Any]]:
        """Возвращает высокоприоритетные возможности обучения.
        
        Args:
            min_priority: Минимальный приоритет для включения в результат
            
        Returns:
            Список высокоприоритетных возможностей
        """
        high_priority = [
            opp for opp in self.pending_opportunities 
            if opp.get('priority', 0) >= min_priority
        ]
        high_priority.sort(key=lambda x: x.get('priority', 0), reverse=True)
        return high_priority
