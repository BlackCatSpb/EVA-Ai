"""
Анализ методов для интеграции fractal_graph_v2 в систему

Сравнение UnifiedFractalMemory vs FractalMemoryGraph
"""

# ==============================================================================
# Методы UnifiedFractalMemory (СТАРАЯ СИСТЕМА)
# ==============================================================================

OLD_METHODS = {
    # Основные методы
    "save_experience": "Сохранить опыт (query/response) в граф",
    "get_context_for_query": "Получить контекст для запроса", 
    "add_knowledge": "Добавить знание в граф",
    "retrieve_knowledge": "Извлечь знания по запросу",
    
    # Управление моделями
    "register_model_instance": "Зарегистрировать Llama инстанс модели",
    "export_model_to_graph": "Экспортировать GGUF в граф",
    "get_model_instance": "Получить Llama инстанс",
    "add_model_node": "Добавить узел модели",
    "get_model_context": "Получить контекст модели",
    "get_static_models": "Получить статичные модели",
    
    # Управление
    "get_stats": "Получить статистику",
    "flush": "Сохранить на диск",
    "close": "Закрыть/очистить",
}

# ==============================================================================
# Методы FractalMemoryGraph (НОВАЯ СИСТЕМА)
# ==============================================================================

NEW_METHODS = {
    # Основные методы
    "add_node": "Добавить узел в граф",
    "add_knowledge": "Добавить S-P-O знание",
    "add_edge": "Добавить связь",
    
    # Поиск
    "semantic_search": "Семантический поиск (векторный)",
    "keyword_search": "Поиск по ключевым словам",
    "get_context": "Получить контекст узла",
    
    # Группы
    "create_group": "Создать семантическую группу",
    "auto_cluster": "Автоматическая кластеризация",
    "get_groups": "Получить группы",
    
    # Модели GGUF
    "load_gguf_knowledge": "Загрузить знания из GGUF модели",
    "get_model_info": "Получить информацию о GGUF модели",
    
    # Векторизация
    "vectorize_all": "Векторизовать все узлы",
    "vectorize_groups": "Векторизовать группы",
    
    # Противоречия
    "check_contradiction": "Проверить на противоречие",
    "resolve_contradiction": "Разрешить противоречие",
    "self_dialogue": "Самодиалог/верификация",
    
    # Управление
    "get_stats": "Получить статистику",
    "get_node": "Получить узел",
    "get_all_nodes": "Получить все узлы",
}

# ==============================================================================
# ЧТО НУЖНО ДЛЯ ИНТЕГРАЦИИ
# ==============================================================================

REQUIRED_FOR_INTEGRATION = """
Для полной замены UnifiedFractalMemory на FractalMemoryGraph нужно:

1. АДАПТЕР (Wrapper/Proxy):
   - Создать класс-адаптер, который реализует интерфейс UnifiedFractalMemory
   - Но внутри использует FractalMemoryGraph
   - Методы: save_experience, get_context_for_query, add_knowledge и т.д.

2. НОВЫЕ МЕТОДЫ (необходимо добавить в FractalMemoryGraph):
   - register_model_instance(model_type, llama) - для хранения Llama инстансов
   - get_model_context(model_type) - контекст для конкретной модели
   - get_static_models() - информация о моделях A, B, C
   - export_model_to_graph(model_type, gguf_path) - экспорт GGUF

3. ОБНОВЛЕНИЕ БД:
   - Поля для хранения Llama инстансов (не в JSON)
   - Индексы для быстрого поиска

4. СОВМЕСТИМОСТЬ:
   - Сохранение формата JSON (для совместимости)
   - Или постепенный переход

5. ТЕСТИРОВАНИЕ:
   - Интеграционные тесты
   - Проверка всех сценариев
"""

print("=" * 60)
print("АНАЛИЗ ИНТЕГРАЦИИ")
print("=" * 60)

print("\n### Методы которые нужно реализовать в FractalMemoryGraph:")
new_features_needed = [
    "register_model_instance(model_type, llama_instance)",
    "get_model_instance(model_type)", 
    "get_model_context(model_type)",
    "get_static_models()",
    "export_model_to_graph(model_type, gguf_path)",
    "save_experience(query, response, model, quality)",
    "get_context_for_query(query)",
    "retrieve_knowledge(query, top_k)",
]
for f in new_features_needed:
    print(f"  - {f}")

print("\n### Что уже реализовано:")
existing = [
    "add_node / add_knowledge / add_edge",
    "semantic_search / keyword_search",
    "create_group / auto_cluster",
    "check_contradiction / self_dialogue",
    "load_gguf_knowledge (GGUF extractor)",
    "vectorize_all / vectorize_groups",
]
for f in existing:
    print(f"  ✓ {f}")

print("\n### Что НЕ перенесено из старой системы:")
not_migrated = [
    "Llama инстансы моделей (model_instances)",
    "Tier management (hot/warm/cold nodes)",
    "Migration events (publish/subscribe)",
    "LLM инференс в графе (run_model_inference)",
]
for f in not_migrated:
    print(f"  ✗ {f}")

print("\n" + "=" * 60)
print("ВЫВОД: Для полной интеграции нужно создать адаптер")
print("=" * 60)