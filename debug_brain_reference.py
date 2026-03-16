#!/usr/bin/env python3
"""
Отладка проблемы инициализации компонентов - почему brain.fractal_model_manager = None
"""

import sys
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_brain_reference")

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def debug_brain_reference_setting():
    """Отладка проблемы с установкой ссылок на brain."""
    try:
        logger.info("=== ОТЛАДКА ПРОБЛЕМЫ BRAIN.FRACTAL_MODEL_MANAGER ===")

        # Создаем MockBrain с поддержкой динамических атрибутов
        class MockBrain:
            def __init__(self):
                self.cache_dir = './test_cache'
                self.components = {}
                self.memory_manager = None
                self.knowledge_graph = None
                self.model_manager = None
                self.fractal_model_manager = None
                self.ml_unit = None
                self._dynamic_attrs = {}

            def __setattr__(self, name, value):
                print(f"DEBUG: Setting brain.{name} = {type(value).__name__ if value else None}")
                self.__dict__[name] = value

            def __getattr__(self, name):
                print(f"DEBUG: Getting brain.{name}")
                if name in self.__dict__:
                    return self.__dict__[name]
                return None

        brain = MockBrain()

        # Импортируем ComponentInitializer
        from cogniflex.core.component_initializer import ComponentInitializer
        initializer = ComponentInitializer(brain)

        # Проверяем начальное состояние
        logger.info("Начальное состояние brain:")
        logger.info(f"  brain.fractal_model_manager: {brain.fractal_model_manager}")
        logger.info(f"  hasattr(brain, 'fractal_model_manager'): {hasattr(brain, 'fractal_model_manager')}")

        # Проверяем фабрику model_manager
        logger.info("Проверка фабрики model_manager...")
        factory = initializer.component_factories.get('model_manager')
        if factory:
            logger.info("Фабрика найдена, создаем компонент...")

            # Создаем компонент напрямую через фабрику
            component = factory()
            logger.info(f"Компонент создан: {type(component).__name__}")

            # Проверяем состояние brain после создания компонента
            logger.info("Состояние brain после создания компонента:")
            logger.info(f"  brain.fractal_model_manager: {brain.fractal_model_manager}")
            logger.info(f"  brain.model_manager: {brain.model_manager}")

            # Проверяем методы компонента
            if hasattr(component, 'get_model_info'):
                info = component.get_model_info()
                logger.info(f"Model info: {info}")

        else:
            logger.error("Фабрика model_manager не найдена!")

        # Теперь попробуем инициализацию через ComponentInitializer
        logger.info("Попытка инициализации через ComponentInitializer...")

        # Сбрасываем brain
        brain = MockBrain()
        initializer = ComponentInitializer(brain)

        # Инициализируем только model_manager
        logger.info("Инициализация только model_manager...")
        success = initializer._initialize_component('model_manager')

        logger.info(f"Результат инициализации: {success}")
        logger.info("Состояние brain после инициализации через ComponentInitializer:")
        logger.info(f"  brain.fractal_model_manager: {brain.fractal_model_manager}")
        logger.info(f"  brain.model_manager: {brain.model_manager}")
        logger.info(f"  type(brain.fractal_model_manager): {type(brain.fractal_model_manager) if brain.fractal_model_manager else None}")

        # Проверяем полную инициализацию компонентов
        logger.info("Полная инициализация компонентов...")

        # Сбрасываем brain снова
        brain = MockBrain()
        initializer = ComponentInitializer(brain)

        success = initializer.initialize_components()

        logger.info(f"Результат полной инициализации: {success}")
        logger.info("Финальное состояние brain:")
        logger.info(f"  brain.fractal_model_manager: {brain.fractal_model_manager}")
        logger.info(f"  brain.model_manager: {brain.model_manager}")

        if brain.fractal_model_manager:
            logger.info("✅ brain.fractal_model_manager успешно установлен!")
        else:
            logger.error("❌ brain.fractal_model_manager все еще None")

        return success

    except Exception as e:
        logger.error(f"Критическая ошибка при отладке: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = debug_brain_reference_setting()
    if success:
        print("\n✅ Отладка завершена")
    else:
        print("\n❌ Ошибка при отладке")
        sys.exit(1)
