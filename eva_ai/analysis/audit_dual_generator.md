# Аудит DualGenerator систем в EVA AI

**Дата:** 2026-04-14  
**Аудитор:** System Audit  
**Версия EVA:** Conceptual Architecture v2  

---

## 1. Найденные реализации

### 1.1 Основная реализация: dual_generator.py

| Параметр | Значение |
|----------|----------|
| **Путь** | eva_ai/memory/fractal_graph_v2/dual_generator.py |
| **Классы** | DualGenerator, CondensedGenerator, ExtendedGenerator |
| **Размер** | 1031 строка |
| **Статус** | АКТИВНО ИСПОЛЬЗУЕТСЯ |

#### Структура:

`
DualGenerator (класс-обёртка)
├── CondensedGenerator (быстрый, 512-1024 токенов, temperature=0.1)
│   └── generate() - краткие ответы
├── ExtendedGenerator (развёрнутый, 4096 токенов, temperature=0.35)
│   ├── generate() - базовая генерация
│   └── generate_chunked() - блочная генерация для больших ответов
└── DocumentManager (опционально)
`

#### Ключевые методы:
- generate_condensed(query) - быстрый режим
- generate_extended(query) - развёрнутый режим
- generate_large(query) - chunked generation (до 4096+ токенов)
- generate_streaming(query) - потоковая генерация
- load_document(), query_document(), generate_with_document()

---

### 1.2 Альтернативная реализация: dual_generator_pie.py

| Параметр | Значение |
|----------|----------|
| **Путь** | eva_ai/memory/fractal_graph_v2/dual_generator_pie.py |
| **Классы** | DualGeneratorPie, PieEnabledDualGenerator |
| **Размер** | 346 строк |
| **Статус** | НЕ ИСПОЛЬЗУЕТСЯ (мёртвый код) |

#### Структура:
`
DualGeneratorPie (обёртка с Pie)
├── dual_generator (оригинальный DualGenerator)
├── pie (PieIntegration)
├── fallback (PieFallbackPipeline)
└── generate() - маршрутизация между источниками
`

---

## 2. Какая версия ИСПОЛЬЗУЕТСЯ

### Используется: dual_generator.py

**Доказательства:**

1. **HybridPipelineAdapter** (строка 179):
   from eva_ai.memory.fractal_graph_v2.dual_generator import DualGenerator

2. **brain_components.py** (строка 230):
   # dual - DualGenerator с 2 физическими моделями (БЫСТРО)

3. **server_routes.py** (строки 510-516):
   dg = web_gui_instance.brain.two_model_pipeline.dual_generator

4. **async_pipeline.py** (строки 103-109):
   self.dual_generator = dual_generator

### НЕ используется: dual_generator_pie.py

- Нет упоминаний в HybridPipelineAdapter
- Нет импортов в brain_components.py
- Нет обращений в server_routes.py
- DualGeneratorPie импортируется только в самом файле

---

## 3. Дублирование функциональности

| Функция | dual_generator.py | dual_generator_pie.py | Дублирование |
|---------|-------------------|----------------------|--------------|
| generate() | Да | Да (делегирует) | Частичное |
| generate_condensed() | Да | Нет | Нет |
| generate_extended() | Да | Нет | Нет |
| generate_large() | Да | Нет | Нет |
| generate_streaming() | Да | Нет | Нет |
| DocumentManager | Да | Нет | Нет |
| Pie Integration | Нет | Да | Нет |
| Fallback Pipeline | Нет | Да | Нет |

---

## 4. Интеграция с памятью

### dual_generator.py - ИНТЕГРАЦИИ:

- HybridTokenizer - токенизация
- SemanticContextCache - кэширование контекста
- GGUFShadowProfiler - профилирование
- self_dialog_learning.compact_context() - компактификация
- DocumentVirtualMemory - управление документами
- FractalGraphV2 - semantic_search(), nodes

### dual_generator_pie.py - ИНТЕГРАЦИИ:

- PieIntegration - профилирование и маршрутизация
- PieFallbackPipeline - fallback механизм

---

## 5. Оценка по 10-балльной шкале

### dual_generator.py

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Функциональность | 9/10 | Полный набор: condensed, extended, large, streaming, documents |
| Интеграция с памятью | 8/10 | Хорошая интеграция с FGv2, semantic search, компактификация |
| Код | 7/10 | Есть дублирование методов очистки, 1031 строка |
| Использование | 10/10 | Активно используется через HybridPipelineAdapter |
| Поддержка | 6/10 | Сложная архитектура, нет документации API |
| **ИТОГО** | **8/10** | Хорошая основная реализация |

### dual_generator_pie.py

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| Функциональность | 6/10 | Обёртка с дополнительными фичами (Pie, Fallback) |
| Интеграция с памятью | 5/10 | Зависит от оригинального DualGenerator |
| Код | 6/10 | 346 строк, в основном маршрутизация |
| Использование | 1/10 | НЕ используется в системе |
| Поддержка | 3/10 | Мёртвый код, не поддерживается |
| **ИТОГО** | **4/10** | Не используется, удалить |

---

## 6. Конкретные рекомендации

### Рекомендация 1: УДАЛИТЬ dual_generator_pie.py (Приоритет: КРИТИЧЕСКИЙ)

**Обоснование:**
- Файл не используется в системе
- Является мёртвым кодом
- Создаёт путаницу при аудите
- Увеличивает время анализа

### Рекомендация 2: Рефакторинг dual_generator.py (Приоритет: ВЫСОКИЙ)

**Проблемы:**
1. Дублирование _clean_response() в CondensedGenerator и ExtendedGenerator
2. Метод _get_context() дублируется в ExtendedGenerator
3. Нет абстрактного базового класса

### Рекомендация 3: Добавить документацию API (Приоритет: СРЕДНИЙ)

### Рекомендация 4: Вынести константы в конфиг (Приоритет: НИЗКИЙ)

---

## 7. Заключение

### Вердикт

| Файл | Статус | Действие |
|------|--------|----------|
| dual_generator.py | ИСПОЛЬЗУЕТСЯ | Оставить, провести рефакторинг |
| dual_generator_pie.py | НЕ ИСПОЛЬЗУЕТСЯ | УДАЛИТЬ |

### Итоговая оценка системы DualGenerator

**Текущее состояние:** 8/10

**После выполнения рекомендаций:** 9.5/10

---

## Приложение: Chain of Use

 HybridPipelineAdapter (core/hybrid_pipeline_adapter.py)
   └── _init_dual_generator() [line 176]
         └── from eva_ai.memory.fractal_graph_v2.dual_generator import DualGenerator [line 179]
               └── DualGenerator(...) [line 190]
