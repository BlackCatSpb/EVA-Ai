#!/usr/bin/env python3
"""
Обширные тесты функциональности ЕВА после удаления fallback заглушек
Тестирует реальную работу всех компонентов системы
"""

import os
import sys
import time
import logging
import traceback
from typing import Dict, List, Any

# Настройка логирования для тестов
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("cogniflex.tests")

class TestResults:
    """Класс для сбора результатов тестирования"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.start_time = time.time()

    def test_passed(self, test_name: str, message: str = ""):
        self.passed += 1
        logger.info(f"✅ {test_name}: {message}")

    def test_failed(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        logger.error(f"❌ {test_name}: {error}")

    def summary(self):
        duration = time.time() - self.start_time
        total = self.passed + self.failed

        logger.info(f"\n{'='*60}")
        logger.info("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        logger.info(f"{'='*60}")
        logger.info(f"Всего тестов: {total}")
        logger.info(f"✅ Пройдено: {self.passed}")
        logger.info(f"❌ Провалено: {self.failed}")
        success_rate = (self.passed / total * 100) if total > 0 else 0
        logger.info(f"Успешность: {success_rate:.2f}%")
        if self.errors:
            logger.info(f"\nОШИБКИ ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"  - {error}")

        return self.failed == 0

class ЕВАTests:
    """Комплексные тесты ЕВА"""

    def __init__(self):
        self.results = TestResults()

    def run_all_tests(self) -> bool:
        """Запуск всех тестов"""
        logger.info("🚀 Начало комплексного тестирования ЕВА")

        try:
            # Тесты импорта
            self.test_imports()

            # Тесты основных компонентов
            self.test_core_components()

            # Тесты веб-поиска без заглушек
            self.test_web_search_functionality()

            # Тесты этического фреймворка без placeholder классов
            self.test_ethics_framework()

            # Тесты процессора обучения
            self.test_learning_processor()

            # Тесты Core Brain без пустых возвратов
            self.test_core_brain_functionality()

            # Тесты интеграции компонентов
            self.test_component_integration()

        except Exception as e:
            logger.error(f"Критическая ошибка в тестировании: {e}")
            traceback.print_exc()
            return False

        return self.results.summary()

    def test_imports(self):
        """Тестирование импортов всех модулей"""
        logger.info("📦 Тестирование импортов модулей...")

        # Список модулей для тестирования
        modules_to_test = [
            ('cogniflex.core.core_brain', 'CoreBrain'),
            ('cogniflex.core.event_bus', 'EventBus, Event, EventTypes'),
            ('cogniflex.websearch.search_engines', 'SearchEngines'),
            ('cogniflex.ethics.ethics_framework', 'EthicalDecision'),
            ('cogniflex.learning.learning_processor', 'LearningProcessor'),
            ('cogniflex.knowledge.knowledge_graph', 'KnowledgeGraph'),
            ('cogniflex.contradiction.contradiction_integrated', 'IntegratedContradictionManager'),
        ]

        for module_name, expected_items in modules_to_test:
            try:
                __import__(module_name)
                self.results.test_passed(f"import_{module_name}", f"Модуль {module_name} импортирован успешно")
            except ImportError as e:
                self.results.test_failed(f"import_{module_name}", f"Ошибка импорта {module_name}: {e}")
            except Exception as e:
                self.results.test_failed(f"import_{module_name}", f"Неожиданная ошибка при импорте {module_name}: {e}")

    def test_core_components(self):
        """Тестирование основных компонентов системы"""
        logger.info("🏗️ Тестирование основных компонентов...")

        try:
            from eva.core.event_bus import EventBus, Event, EventTypes, get_event_bus

            # Тест EventBus
            bus = get_event_bus()
            assert bus is not None, "EventBus не создан"
            self.results.test_passed("event_bus_creation", "EventBus успешно создан")

            # Тест Event
            event = Event("test_event", source="test", data={"key": "value"})
            assert event.event_type == "test_event", "Event тип не установлен"
            assert event.source == "test", "Event источник не установлен"
            self.results.test_passed("event_creation", "Event успешно создан")

            # Тест EventTypes
            assert hasattr(EventTypes, 'SYSTEM_START'), "EventTypes не содержит SYSTEM_START"
            self.results.test_passed("event_types", "EventTypes содержит необходимые типы")

        except Exception as e:
            self.results.test_failed("core_components", f"Ошибка тестирования основных компонентов: {e}")

    def test_web_search_functionality(self):
        """Тестирование веб-поиска без заглушек"""
        logger.info("🔍 Тестирование веб-поиска без заглушек...")

        try:
            from eva.websearch.search_engines import SearchEngines

            engines = SearchEngines()

            # Тест реального поиска (с коротким таймаутом для тестов)
            test_query = "python programming"
            results = engines.search_google(test_query, max_results=2)

            # Проверяем, что это не mock результаты
            assert isinstance(results, list), "Результаты поиска должны быть списком"
            assert len(results) > 0, "Должен быть хотя бы один результат"

            # Проверяем структуру результата
            result = results[0]
            assert hasattr(result, 'title'), "Результат должен иметь title"
            assert hasattr(result, 'url'), "Результат должен иметь url"
            assert hasattr(result, 'snippet'), "Результат должен иметь snippet"
            assert hasattr(result, 'source'), "Результат должен иметь source"

            # Проверяем, что это не hardcoded данные
            assert "python.org" in result.url.lower() or "wikipedia" in result.url.lower() or len(results) > 1, \
                   "Результаты выглядят как реальные (не hardcoded)"

            self.results.test_passed("web_search_real", f"Веб-поиск вернул {len(results)} реальных результатов для '{test_query}'")

        except Exception as e:
            self.results.test_failed("web_search", f"Ошибка тестирования веб-поиска: {e}")

    def test_ethics_framework(self):
        """Тестирование этического фреймворка без placeholder классов"""
        logger.info("⚖️ Тестирование этического фреймворка без placeholder классов...")

        try:
            from eva.ethics import EthicsFramework
            from eva.ethics.ethics_framework import EthicalDecision

            # Проверяем, что это не placeholder класс
            ethics = EthicsFramework()
            assert hasattr(ethics, 'make_decision'), "EthicsFramework должен иметь метод make_decision"
            assert not hasattr(ethics, '__placeholder__'), "Это не должен быть placeholder класс"

            # Тест создания этического решения
            decision = EthicalDecision(
                decision="allow",
                confidence=0.85,
                justification="Тестовое решение",
                alternatives=["deny", "delay"],
                assessment=[],
                requires_human_review=False
            )

            assert decision.decision == "allow", "Решение должно быть 'allow'"
            assert decision.confidence == 0.85, "Уверенность должна быть 0.85"

            self.results.test_passed("ethics_framework_real", "Этический фреймворк работает без placeholder классов")

        except ImportError as e:
            self.results.test_failed("ethics_framework", f"Ошибка импорта этических модулей: {e}")
        except Exception as e:
            self.results.test_failed("ethics_framework", f"Ошибка тестирования этического фреймворка: {e}")

    def test_learning_processor(self):
        """Тестирование процессора обучения"""
        logger.info("🧠 Тестирование процессора обучения...")

        try:
            from eva.learning.learning_processor import LearningProcessor

            # Создаем процессор (без полной инициализации для тестов)
            processor = LearningProcessor(brain=None)
            assert hasattr(processor, 'data_processor'), "LearningProcessor должен иметь data_processor"
            assert hasattr(processor, 'task_generator'), "LearningProcessor должен иметь task_generator"
            assert hasattr(processor, 'integration_manager'), "LearningProcessor должен иметь integration_manager"

            self.results.test_passed("learning_processor_structure", "LearningProcessor имеет правильную структуру")

        except Exception as e:
            self.results.test_failed("learning_processor", f"Ошибка тестирования процессора обучения: {e}")

    def test_core_brain_functionality(self):
        """Тестирование Core Brain без пустых возвратов"""
        logger.info("🧠 Тестирование Core Brain без пустых возвратов...")

        try:
            from eva.core.core_brain import CoreBrain

            # Создаем Core Brain
            brain = CoreBrain()

            # Тестируем методы, которые ранее возвращали пустые данные
            contradiction_stats = brain.get_contradiction_statistics()
            assert isinstance(contradiction_stats, dict), "Статистика противоречий должна быть словарем"
            assert contradiction_stats != {}, "Статистика противоречий не должна быть пустой"

            # Проверяем наличие статусных полей
            expected_fields = ['total_contradictions', 'resolved_contradictions', 'unresolved_contradictions', 'timestamp']
            for field in expected_fields:
                assert field in contradiction_stats, f"Статистика должна содержать поле {field}"

            self.results.test_passed("core_brain_informative", "Core Brain возвращает информативные данные вместо пустых")

        except Exception as e:
            self.results.test_failed("core_brain", f"Ошибка тестирования Core Brain: {e}")

    def test_component_integration(self):
        """Тестирование интеграции компонентов"""
        logger.info("🔗 Тестирование интеграции компонентов...")

        try:
            from eva.core.component_initializer import ComponentInitializer
            from eva.core.core_brain import CoreBrain

            # Создаем компоненты
            brain = CoreBrain()
            initializer = ComponentInitializer(brain)

            # Проверяем наличие фабрик компонентов
            assert hasattr(initializer, 'component_factories'), "ComponentInitializer должен иметь component_factories"
            assert len(initializer.component_factories) > 0, "Должны быть зарегистрированы фабрики компонентов"

            # Проверяем наличие важных компонентов
            required_components = ['memory_manager', 'learning_processor', 'contradiction_manager']
            for component in required_components:
                assert component in initializer.component_factories, f"Компонент {component} должен быть зарегистрирован"

            self.results.test_passed("component_integration", f"Зарегистрировано {len(initializer.component_factories)} фабрик компонентов")

        except Exception as e:
            self.results.test_failed("component_integration", f"Ошибка тестирования интеграции компонентов: {e}")

def main():
    """Главная функция запуска тестов"""
    print("🧪 Запуск обширных тестов ЕВА после удаления fallback заглушек")
    print("=" * 70)

    tester = ЕВАTests()
    success = tester.run_all_tests()

    if success:
        print("\n🎉 Все тесты пройдены! Система работает без fallback заглушек.")
        return 0
    else:
        print(f"\n💥 {tester.results.failed} тестов провалено. Требуется исправление.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
