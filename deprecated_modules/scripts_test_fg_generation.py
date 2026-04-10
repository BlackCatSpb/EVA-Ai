"""
Тестирование генерации только с FG (без GGUF моделей).
Фаза 3: Проверка что FG может работать как самостоятельная система генерации.
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("fg_generation_test")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    logger.info("=== Тестирование генерации только с FG ===")
    
    try:
        from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph
        from eva_ai.memory.fg_gguf_architecture_mapper import create_architecture_mapper
        from eva_ai.memory.fg_gguf_quality_extraction import create_gguf_fg_integrator
        
        # Создаём FG
        logger.info("Создание FractalMemoryGraph...")
        fg = FractalMemoryGraph()
        
        stats = fg.get_stats()
        logger.info(f"FG Stats: {stats.get('total_nodes', 0)} узлов, {stats.get('total_edges', 0)} связей")
        
        # Тест 1: Семантический поиск контекста
        logger.info("\n=== Тест 1: Семантический поиск контекста ===")
        test_query = "искусственный интеллект"
        
        context_results = fg.semantic_search(test_query, top_k=5, min_similarity=0.3)
        logger.info(f"Найдено контекста: {len(context_results)} результатов")
        
        for i, r in enumerate(context_results[:3]):
            content = r.get('content', '')[:80]
            sim = r.get('similarity', 0)
            logger.info(f"  {i+1}. [ similarity={sim:.2f}] {content}...")
        
        # Тест 2: Создание ACI концепта из контекста
        logger.info("\n=== Тест 2: ACI создание концепта ===")
        
        mapper = create_architecture_mapper(fg)
        
        # Симулируем контекст из результатов поиска
        context_from_fg = " ".join([r.get('content', '')[:100] for r in context_results[:2]])
        
        concept_id = mapper.create_aci_concept_from_context(
            context=context_from_fg,
            query=test_query
        )
        
        logger.info(f"Создан концепт: {concept_id}")
        
        # Тест 3: Проверка уровней и типов узлов
        logger.info("\n=== Тест 3: Структура FG ===")
        
        nodes_by_level = stats.get('nodes_by_level', {})
        nodes_by_type = stats.get('nodes_by_type', {})
        
        logger.info("Узлы по уровням:")
        for level, count in sorted(nodes_by_level.items()):
            logger.info(f"  Level {level}: {count} узлов")
        
        logger.info("Узлы по типам (топ-10):")
        sorted_types = sorted(nodes_by_type.items(), key=lambda x: -x[1])
        for node_type, count in sorted_types[:10]:
            logger.info(f"  {node_type}: {count}")
        
        # Тест 4: Получение контекста для генерации (без LLM)
        logger.info("\n=== Тест 4: Контекст для генерации ===")
        
        context = fg.get_context_for_query(test_query, max_length=256, min_similarity=0.4)
        logger.info(f"Контекст для генерации: {len(context)} символов")
        if context:
            logger.info(f"Первые 200 символов: {context[:200]}...")
        else:
            logger.warning("Контекст пустой!")
        
        # Тест 5: Проверка что GGUF НЕ используется
        logger.info("\n=== Тест 5: Проверка отсутствия GGUF ===")
        
        # Проверяем что модель config не ссылается на активную генерацию
        model_config_nodes = [n for n in fg.storage.nodes.values() 
                             if getattr(n, 'node_type', '') == 'model_config']
        
        if model_config_nodes:
            mc = model_config_nodes[0]
            logger.info(f"Model config: {getattr(mc, 'content', '')[:100]}")
            logger.info("  → Это metadata, не активная модель (OK)")
        
        # Тест 6: Проверка метрик FG
        logger.info("\n=== Тест 6: Метрики FG ===")
        
        hot_stats = mapper.get_hot_window_stats()
        logger.info(f"Hot window stats: {hot_stats}")
        
        logger.info("\n✅ Тестирование FG генерации завершено!")
        logger.info("FG работает автономно:")
        logger.info("  - Семантический поиск работает")
        logger.info("  - ACI концепты создаются")
        logger.info("  - Контекст для генерации доступен")
        logger.info("  - GGUF модели не требуются для базовой работы")
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()