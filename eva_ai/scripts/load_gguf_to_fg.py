"""
Скрипт загрузки GGUF архитектуры модели в FG.
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("gguf_to_fg")

def main():
    # Определяем правильный корень
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Поднимаемся: scripts -> eva_ai -> корень
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    logger.info(f"Project root: {project_root}")
    
    # Добавляем в path
    sys.path.insert(0, project_root)
    
    logger.info("=== Загрузка GGUF архитектуры в FG ===")
    
    try:
        from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph
        from eva_ai.memory.fg_gguf_architecture_mapper import create_architecture_mapper
        
        # Создаём FG
        logger.info("Создание FractalMemoryGraph...")
        fg = FractalMemoryGraph()
        
        # Проверяем текущее состояние
        stats = fg.get_stats()
        logger.info(f"FG до загрузки: {stats.get('total_nodes', 0)} узлов")
        
        # Путь к GGUF модели (3B) - правильный
        gguf_path = r"C:\Users\black\OneDrive\Desktop\CogniFlex\eva_ai\memory\fractal_torch_storage\gguf_models\qwen2.5-3b-instruct\qwen2.5-3b-instruct-q4_k_m.gguf"
        
        logger.info(f"GGUF модель: {gguf_path}")
        logger.info(f"GGUF существует: {os.path.exists(gguf_path)}")
        
        if not os.path.exists(gguf_path):
            logger.error(f"GGUF файл не найден: {gguf_path}")
            return
        
        # Создаём маппер и загружаем архитектуру
        logger.info("Создание маппера...")
        mapper = create_architecture_mapper(fg)
        
        logger.info("Загрузка архитектуры модели...")
        success = mapper.load_model_architecture(gguf_path)
        
        if success:
            logger.info("✅ Архитектура загружена!")
        else:
            logger.warning("⚠️ Архитектура не загружена (возможно GGUF парсер не работает)")
        
        # Проверяем результат
        stats = fg.get_stats()
        logger.info(f"FG после загрузки: {stats}")
        
        # Тестируем ACI концепт
        logger.info("Тестирование ACI create_aci_concept_from_context...")
        concept_id = mapper.create_aci_concept_from_context(
            context="Тестовый контекст для создания концепта",
            query="Что такое тест"
        )
        logger.info(f"Создан концепт: {concept_id}")
        
        logger.info("=== Завершено ===")
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()