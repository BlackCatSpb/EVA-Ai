"""
Скрипт для тестирования FG - загрузка архитектуры модели и базовые операции.
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("test_fg")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    logger.info("=== Тестирование FractalGraphV2 ===")
    
    try:
        # Создаём FG напрямую
        from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph
        
        logger.info("Создание FractalMemoryGraph...")
        fg = FractalMemoryGraph()
        
        # Проверяем stats
        stats = fg.get_stats() if hasattr(fg, 'get_stats') else {}
        logger.info(f"FG Stats: {stats}")
        
        # Добавляем тестовый узел
        logger.info("Добавление тестового узла...")
        node = fg.add_node(
            content="Тестовый концепт из скрипта",
            node_type="test_concept",
            level=2,
            confidence=0.8,
            metadata={'source': 'test_script'}
        )
        logger.info(f"Узел создан: {node.id if node else 'FAILED'}")
        
        # Тестируем поиск
        logger.info("Тестируем семантический поиск...")
        results = fg.semantic_search("тестовый концепт", top_k=3, min_similarity=0.3)
        logger.info(f"Найдено результатов: {len(results)}")
        
        # Добавляем знание
        logger.info("Добавление знания (S-P-O)...")
        fg.add_knowledge(
            subject="Тест",
            relation="связан с",
            object_="Пример"
        )
        
        logger.info("✅ FG работает!")
        
        # Проверяем статистику после добавления
        stats = fg.get_stats() if hasattr(fg, 'get_stats') else {}
        logger.info(f"FG Stats после теста: {stats}")
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()