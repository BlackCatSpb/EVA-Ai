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
        from eva_ai.core.component_initializer import ComponentInitializer
        
        logger.info("Инициализация системы...")
        initializer = ComponentInitializer()
        
        # Инициализировать brain
        if not initializer.initialize():
            logger.error("Ошибка инициализации")
            return
        
        brain = initializer.core_brain
        if not brain:
            logger.error("Brain не инициализирован")
            return
        
        logger.info("Brain инициализирован")
        
        # Проверить наличие компонентов
        kg = getattr(brain, 'knowledge_graph', None)
        fg = getattr(brain, 'fractal_graph_v2', None)
        
        logger.info(f"KnowledgeGraph: {'да' if kg else 'нет'}")
        logger.info(f"FractalGraphV2: {'да' if fg else 'нет'}")
        
        if not kg:
            logger.warning("KnowledgeGraph не найден")
            return
            
        if not fg:
            logger.warning("FractalGraphV2 не найден")
            return
        
        # Запустить миграцию
        from eva_ai.knowledge.kg_to_fg_migration import migrate_knowledge_graph
        
        logger.info("Запуск миграции...")
        result = migrate_knowledge_graph(brain)
        
        logger.info(f"Результат: {result}")
        
        if result.get('status') == 'complete':
            logger.info("✅ Миграция завершена успешно!")
        else:
            logger.warning(f"⚠️ Миграция завершена с проблемами: {result}")
            
    except Exception as e:
        logger.error(f"Ошибка миграции: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()