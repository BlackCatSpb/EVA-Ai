"""
Тест горячего развертывания EVA
"""
import sys
import os

# Добавляем путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eva_ai.mlearning.hot_deployment import (
    get_hot_deployment_manager,
    initialize_hot_deployment,
    FractalGraph,
    GraphNode,
    NodeState
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_fractal_graph():
    """Тестирует создание графа"""
    logger.info("=== Тест FractalGraph ===")
    
    graph = FractalGraph(max_depth=3, max_children=5)
    
    # Создаем дочерние узлы
    node1 = graph.create_child_node("0", "Основной", "root")
    node2 = graph.create_child_node("0", "Альтернативный", "alt")
    
    logger.info(f"Создано узлов: {len(graph._nodes)}")
    logger.info(f"Статистика: {graph.get_stats()}")
    
    # Создаем вложенные узлы
    if node1:
        child1 = graph.create_child_node(node1.address, "Уточнение 1", "refinement")
        if child1:
            child2 = graph.create_child_node(child1.address, "Уточнение 2", "refinement")
    
    logger.info(f"Всего узлов после вложения: {len(graph._nodes)}")
    logger.info(f"Статистика: {graph.get_stats()}")
    
    # Получаем горячие узлы (пока нет)
    hot = graph.get_hot_nodes()
    logger.info(f"Горячие узлы: {len(hot)}")
    
    return True


def test_hot_deployment():
    """Тестирует горячее развертывание с моделью"""
    logger.info("=== Тест HotDeploymentManager ===")
    
    # Определяем путь к модели
    # __file__ = eva/mlearning/hot_deployment/test_hot_deployment.py
    # project_root = eva
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(project_root, "mlearning", "eva_models", "qwen3.5-0.8b")
    
    logger.info(f"Путь к модели: {model_path}")
    logger.info(f"Существует: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        logger.error(f"Модель не найдена: {model_path}")
        return False
    
    # Инициализируем
    manager = get_hot_deployment_manager(model_path=model_path)
    
    logger.info("Загрузка модели...")
    success = manager.initialize(preload_root=True)
    
    if not success:
        logger.error("Ошибка инициализации")
        return False
    
    logger.info("Модель загружена успешно!")
    
    # Проверяем статус
    status = manager.get_status()
    logger.info(f"Статус: {status}")
    
    # Тест генерации (меньше токенов для скорости)
    logger.info("Тест генерации (10 токенов)...")
    response = manager.generate(
        prompt="Привет!",
        max_new_tokens=10
    )
    
    logger.info(f"Ответ: {response[:200] if response else 'None'}")
    
    return True


if __name__ == "__main__":
    logger.info("Запуск тестов горячего развертывания...")
    
    # Тест графа (без модели)
    test_fractal_graph()
    
    # Тест с моделью
    try:
        test_hot_deployment()
    except Exception as e:
        logger.error(f"Ошибка теста горячего развертывания: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("Тесты завершены")