"""
Скрипт миграции данных из KnowledgeGraph в FractalGraphV2.
Запустить: python -m eva_ai.scripts.migrate_kg_to_fg
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("migrate")

def main():
    # Добавить путь
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    logger.info("=== Миграция KnowledgeGraph → FractalGraphV2 ===")
    
    try:
        # Импорт модулей
        from eva_ai.core.core_brain import CoreBrain
        
        logger.info("Создание CoreBrain...")
        brain = CoreBrain()
        
        logger.info("Brain создан, ищем компоненты...")
        
        # Выводим все атрибуты brain
        all_attrs = [a for a in dir(brain) if not a.startswith('_') and 'graph' in a.lower()]
        logger.info(f"Graph-related attributes: {all_attrs}")
        
        # Пробуем напрямую создать FG
        try:
            from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph
            logger.info("Создаём FractalMemoryGraph напрямую...")
            fg = FractalMemoryGraph()
            logger.info(f"FractalGraphV2 создан напрямую: {'да'}")
            
            # Ищем KnowledgeGraph
            kg = brain.components.get('knowledge_graph')
            logger.info(f"KnowledgeGraph из components: {'да' if kg else 'нет'}")
            
            if not kg:
                # Пробуем через knowledge_manager
                km = getattr(brain, 'knowledge_manager', None)
                if km and hasattr(km, 'knowledge_graph'):
                    kg = km.knowledge_graph
                    logger.info(f"KnowledgeGraph из km: {'да' if kg else 'нет'}")
            
            if kg and fg:
                logger.info("Запуск миграции...")
                from eva_ai.knowledge.kg_to_fg_migration import migrate_knowledge_graph
                result = migrate_knowledge_graph(brain)
                logger.info(f"Результат: {result}")
            elif not kg:
                logger.warning("⚠️ KnowledgeGraph не найден - проверяем что есть в components")
                logger.info(f"Все components: {list(brain.components.keys())}")
                
        except Exception as e:
            logger.error(f"Ошибка создания FG: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()