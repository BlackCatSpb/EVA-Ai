# self_dialog_manager.py
"""Менеджер самодиалога для внутреннего мышления системы."""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("eva_ai.self_dialog")

class SelfDialogManager:
    """Менеджер самодиалога для внутреннего мышления системы."""

    def __init__(self, attention_system):
        self.attention_system = attention_system
        self.running = False
        self.stop_event = threading.Event()
        self.dialog_thread = None
        self.current_dialog = None
        self.logger = logging.getLogger("eva_ai.self_dialog.manager")

    def start_dialog(self):
        """Запускает процесс самодиалога."""
        if self.running:
            return  # Уже запущен

        self.stop_event.clear()
        self.running = True

        def dialog_loop():
            try:
                while self.running and not self.stop_event.is_set():
                    # Определяем тип диалога
                    dialog_type = self._determine_dialog_type()

                    # Генерируем вопросы
                    questions = self._generate_dialog_questions()

                    if not questions:
                        time.sleep(1)
                        continue

                    # Обрабатываем каждый вопрос
                    for question in questions:
                        if self.stop_event.is_set():
                            break

                        # Получаем ответ от системы
                        response = self._get_system_response(question)

                        # Обрабатываем ответ
                        self._process_response(question, response)

                        # Небольшая пауза между вопросами
                        time.sleep(0.5)

                    # Проверяем, нужно ли продолжать диалог
                    if not self._should_continue_dialog():
                        break

                    # Задержка перед следующим циклом
                    time.sleep(2)
            finally:
                self.running = False
                self.logger.info("Процесс самодиалога завершен")

        self.dialog_thread = threading.Thread(
            target=dialog_loop,
            name="SelfDialogThread",
            daemon=True
        )
        self.dialog_thread.start()
        self.logger.info("Запущен процесс самодиалога")

    def stop(self):
        """Останавливает процесс самодиалога."""
        self.stop_event.set()
        if self.dialog_thread and self.dialog_thread.is_alive():
            self.dialog_thread.join(timeout=2.0)
        self.running = False
        self.logger.info("Процесс самодиалога остановлен")

    def _determine_dialog_type(self) -> str:
        """Определяет тип текущего диалога."""
        try:
            # FIX: Check current_focus exists and is a dict before accessing
            current_focus = getattr(self.attention_system, 'current_focus', None)
            if isinstance(current_focus, dict) and current_focus.get("dialog_type") == "contradiction_resolution":
                return "contradiction_resolution"
            return "exploration"
        except Exception as e:
            self.logger.debug(f"Ошибка определения типа диалога: {e}")
            return "exploration"

    def _generate_dialog_questions(self) -> List[str]:
        """Генерирует вопросы для самодиалога."""
        questions = []
        try:
            concept = None
            try:
                focus = getattr(self.attention_system, 'current_focus', None) or {}
                concepts = focus.get('primary_concepts') if isinstance(focus, dict) else None
                if isinstance(concepts, list) and concepts:
                    concept = concepts[0]
                elif isinstance(concepts, str) and concepts:
                    concept = concepts
            except Exception:
                concept = None

            if self.current_dialog is None:
                # Начинаем новый диалог
                if self._determine_dialog_type() == "contradiction_resolution":
                    questions.extend([
                        f"Как разрешить противоречия связанные с концептом '{concept if concept else 'текущим контекстом'}'?",
                        "Какие альтернативные интерпретации существуют для этого вопроса?",
                        "Какие факты подтверждают или опровергают текущую позицию?"
                    ])
                else:
                    questions.extend([
                        f"Какие аспекты концепта '{concept if concept else 'текущим контекстом'}' я упустил?",
                        "Как эта информация связана с предыдущими знаниями?",
                        "Какие практические применения можно найти для этой информации?"
                    ])
            else:
                # Продолжаем текущий диалог
                questions.append(self._generate_next_question())

            return questions[:3]  # Ограничиваем количество вопросов
        except Exception as e:
            self.logger.error(f"Ошибка генерации вопросов для самодиалога: {e}")
            return []

    def _generate_next_question(self) -> str:
        """Генерирует следующий вопрос в текущем диалоге."""
        try:
            # Анализируем последний ответ для формирования следующего вопроса
            last_response = self.current_dialog[-1]["response"] if self.current_dialog else ""

            # Простая логика для генерации следующего вопроса
            if "не уверен" in last_response.lower() or "не знаю" in last_response.lower():
                return "Можешь уточнить или привести пример?"
            elif "следовательно" in last_response.lower() or "поэтому" in last_response.lower():
                return "Каковы возможные исключения из этого вывода?"
            else:
                return "Как это связано с другими концептами в горячем окне?"
        except Exception as e:
            self.logger.debug(f"Ошибка генерации следующего вопроса: {e}")
            return "Можешь развить эту мысль дальше?"

    def _get_system_response(self, question: str) -> str:
        """Получает ответ системы на вопрос."""
        try:
            # Здесь должна быть интеграция с основной системой для генерации ответов
            # Для упрощения возвращаем заглушку

            # Используем координатор генерации, если доступен
            if hasattr(self.attention_system.core_brain, 'generation_coordinator'):
                return self.attention_system.core_brain.generation_coordinator.generate_response(question)

            # Простой ответ на основе горячего окна
            hot_window = self._get_hot_window_data()
            if hot_window:
                return f"На основе текущего контекста: {question} - это связано с {', '.join(hot_window.keys()[:2])}."

            return f"Я думаю, что {question} требует дополнительного анализа."
        except Exception as e:
            self.logger.error(f"Ошибка получения системного ответа: {e}")
            return "Я временно не могу дать подробный ответ на этот вопрос."

    def _get_hot_window_data(self) -> Dict:
        """Получает данные из горячего окна."""
        try:
            if hasattr(self.attention_system.core_brain, 'memory_manager') and \
               self.attention_system.core_brain.memory_manager:
                return self.attention_system.core_brain.memory_manager.get_hot_window_data()
            return {}
        except Exception as e:
            self.logger.debug(f"Ошибка получения данных горячего окна: {e}")
            return {}

    def _process_response(self, question: str, response: str):
        """Обрабатывает ответ системы."""
        try:
            # Сохраняем диалог
            if self.current_dialog is None:
                self.current_dialog = []

            self.current_dialog.append({
                "question": question,
                "response": response,
                "timestamp": time.time()
            })

            # Обновляем горячее окно на основе ответа
            self._update_hot_window(question, response)

            # Проверяем на наличие противоречий
            self._check_for_contradictions(question, response)

            # Планируем обучение на основе ответа
            self._schedule_learning(question, response)
        except Exception as e:
            self.logger.error(f"Ошибка обработки ответа: {e}")

    def _update_hot_window(self, question: str, response: str):
        """Обновляет горячее окно на основе диалога."""
        try:
            if not hasattr(self.attention_system.core_brain, 'memory_manager') or \
               not self.attention_system.core_brain.memory_manager:
                return

            # Извлекаем ключевые концепты из ответа
            concepts = self._extract_concepts(response)

            # Добавляем концепты в горячее окно
            for concept in concepts:
                self.attention_system.core_brain.memory_manager.add_to_hot_window(
                    concept, 0.8, "dialog_concept"
                )
        except Exception as e:
            self.logger.error(f"Ошибка обновления горячего окна: {e}")

    def _extract_concepts(self, text: str) -> List[str]:
        """Извлекает ключевые концепты из текста."""
        # Простой метод извлечения сущностей
        common_words = ["и", "в", "на", "с", "к", "от", "до", "по", "для", "о", "об", "перед", "при"]
        words = [word.lower().strip(".,?!") for word in text.split()
                if word.lower() not in common_words and len(word) > 3]

        return list(set(words))[:5]

    def _check_for_contradictions(self, question: str, response: str):
        """Проверяет ответ на наличие противоречий."""
        try:
            # FIX: Check contradiction_resolver exists on attention_system first
            contradiction_resolver = getattr(self.attention_system, 'contradiction_resolver', None)
            if contradiction_resolver and hasattr(contradiction_resolver, 'check_response'):
                contradiction_resolver.check_response(question, response)
        except Exception as e:
            self.logger.debug(f"Ошибка проверки на противоречия: {e}")

    def _schedule_learning(self, question: str, response: str):
        """Планирует обучение на основе диалога."""
        try:
            # FIX: Check learning_scheduler exists on attention_system first
            learning_scheduler = getattr(self.attention_system, 'learning_scheduler', None)
            if learning_scheduler and hasattr(learning_scheduler, 'schedule_learning'):
                learning_scheduler.schedule_learning(question, response)
        except Exception as e:
            self.logger.debug(f"Ошибка планирования обучения: {e}")

    def _should_continue_dialog(self) -> bool:
        """Определяет, следует ли продолжать диалог."""
        try:
            # Проверяем длину диалога
            if self.current_dialog and len(self.current_dialog) >= 5:
                return False

            # Проверяем тип диалога
            contradiction_resolver = getattr(self.attention_system, 'contradiction_resolver', None)
            if self._determine_dialog_type() == "contradiction_resolution" and \
               contradiction_resolver and \
               hasattr(contradiction_resolver, 'has_active_contradictions'):
                try:
                    if not contradiction_resolver.has_active_contradictions():
                        return False
                except Exception:
                    pass

            # Проверяем активность системы
            if hasattr(self.attention_system, 'core_brain') and self.attention_system.core_brain and \
               hasattr(self.attention_system.core_brain, 'system_monitor') and \
               self.attention_system.core_brain.system_monitor:
                try:
                    health = self.attention_system.core_brain.system_monitor.get_system_status()
                    if health.get("status") == "degraded":
                        return False
                except Exception:
                    pass

            return True
        except Exception as e:
            self.logger.debug(f"Ошибка определения продолжения диалога: {e}")
            return False
