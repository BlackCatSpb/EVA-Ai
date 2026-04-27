# Анализ Memory System EVA

## 1. FractalGraphV2 - Основной граф знаний

### Расположение
`eva_ai/memory/fractal_graph_v2/`

### Ключевые компоненты
- **FractalMemoryGraph** - главный класс графа
- **HNSWIndex** - индекс для similarity search
- **Storage layer** - персистентное хранилище

### Методы
- `add_node()`, `add_edge()` - создание элементов
- `semantic_search()` - поиск по эмбеддингам
- `get_clusters()` - получение кластеров

### Статус: АКТИВЕН
Используется в ConceptMiner, ContradictionMiner, ContradictionGenerator

---

## 2. HybridTokenCache

### Расположение
`eva_ai/core/hybrid_token_cache.py`

### Назначение
Гибридное кэширование токенов с LRU + semantic eviction

### Статус: ЧАСТИЧНО ИСПОЛЬЗУЕТСЯ

---

## Проблемы

1. **memory_system.md** - stub файл (37 bytes)
2. Некоторые методы FGv2 не документированы
3. Нет unit-тестов

---

## Вывод

Memory система - основа для всех mining компонентов. Работает стабильно.