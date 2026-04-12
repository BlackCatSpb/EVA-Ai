# IMPORT_SCHEME.md - Схема импортов EVA AI (Pie Architecture)

**Дата:** 2026-04-12  
**Версия:** UnifiedGenerator (Pie-based)  
**Статус:** ✅ SelfReasoningEngine работает, тест проходит

---

## 1. АРХИТЕКТУРА UNIFIEDGENERATOR (Текущая)

### 1.1 Основная схема

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CORE BRAIN                                   │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  UnifiedGenerator (eva_ai/core/unified_generator.py)         │ │
│  │                                                               │ │
│  │  ├─ ModelType.LOGIC    → ruadapt_qwen3_4b_q4_k_m.gguf      │ │
│  │  │   n_ctx=4096, threads=4, temperature=0.3                  │ │
│  │  ├─ ModelType.CONTEXT  → ruadapt_qwen3_4b_q4_k_m.gguf      │ │
│  │  │   n_ctx=32768, threads=4, temperature=0.3                  │ │
│  │  └─ ModelType.CODER    → qwen2.5-coder-1.5b.gguf           │ │
│  │      n_ctx=4096, threads=4, temperature=0.2                   │ │
│  │                                                               │ │
│  │  SimpleRouter (L2 routing по ключевым словам)                  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                           ↓                                         │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  PipelineAdapter (eva_ai/core/pipeline_adapter.py)             │ │
│  │  ├─ Атрибуты: model_a, model_b, model_c (обратная совм.)   │ │
│  │  ├─ generate() → UnifiedGenerator                             │ │
│  │  └─ process_query() → структура для SelfReasoningEngine     │ │
│  │      (model_a_result, model_b_result, reasoning_steps)        │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                           ↓                                         │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  GenerationCoordinator (eva_ai/core/generation/)              │ │
│  │  └─ UnifiedGeneratorProvider (priority=0)                     │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    REASONING ENGINES                                │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  SelfReasoningEngine (eva_ai/reasoning/)                   │ │
│  │  ├─ process_query() → PipelineAdapter.process_query()        │ │
│  │  ├─ _build_contextual_query() → sre_context.py             │ │
│  │  ├─ _get_wikipedia_context() → sre_context.py             │ │
│  │  ├─ _generate_with_qwen() → sre_context.py                 │ │
│  │  ├─ _check_relevance() → sre_quality.py                   │ │
│  │  ├─ _check_response_quality() → sre_quality.py            │ │
│  │  └─ _determine_query_type() → sre_context.py               │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  EnhancedReasoningEngine (eva_ai/reasoning/)                │ │
│  │  ├─ _get_qwen() → brain.two_model_pipeline                │ │
│  │  ├─ _generate_response() → UnifiedGenerator.generate()       │ │
│  │  ├─ _regenerate_response() → UnifiedGenerator.generate()    │ │
│  │  └─ _generate_with_context() → UnifiedGenerator.generate() │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. КАРТА ИМПОРТОВ

### 2.1 Core Imports

```
eva_ai/
├── core/
│   ├── core_brain.py
│   │   └── from eva.core.brain_init import initialize_generation_coordinator
│   │
│   ├── unified_generator.py          # UNIFIED GENERATOR
│   │   ├── import llama_cpp
│   │   ├── from eva_ai.core.generation.generation_coordinator import GenerationCoordinator
│   │   └── from eva_ai.core.pipeline_adapter import PipelineAdapter
│   │
│   ├── pipeline_adapter.py           # PIPELINE ADAPTER
│   │   ├── from eva_ai.core.unified_generator import UnifiedGenerator
│   │   └── from eva_ai.core.generation.generation_result import GenerationResult
│   │
│   ├── brain_components.py
│   │   ├── from eva_ai.core.unified_generator import _init_unified_generator
│   │   ├── from eva_ai.core.pipeline_adapter import PipelineAdapter
│   │   └── from eva_ai.core.generation.generation_coordinator import GenerationCoordinator
│   │
│   └── generation/
│       ├── generation_coordinator.py
│       │   └── (UnifiedGeneratorProvider)
│       └── generation_result.py
│
├── reasoning/
│   ├── self_reasoning_engine.py    # SELF REASONING ENGINE
│   │   ├── from .sre_context import (...)
│   │   ├── from .sre_quality import (...)
│   │   ├── from .sre_feedback import *
│   │   └── from .sre_recursive import (...)
│   │
│   ├── enhanced_reasoning_engine.py  # ENHANCED REASONING ENGINE
│   │   ├── from eva_ai.reasoning.sre_context import ...
│   │   └── from eva_ai.contradiction.contradiction_manager import ...
│   │
│   ├── sre_context.py              # CONTEXT + FALLBACK
│   │   ├── from eva.knowledge.wikipedia_kb import get_wikipedia_kb
│   │   └── __all__ = [_build_contextual_query, _get_wikipedia_context, ...]
│   │
│   ├── sre_quality.py             # QUALITY CHECKING
│   │   └── __all__ = [check_quality, _sanitize_response, ...]
│   │
│   ├── sre_feedback.py            # FEEDBACK
│   │   └── __all__ = [...]
│   │
│   ├── sre_recursive.py          # RECURSIVE REASONING
│   │   └── __all__ = [...]
│   │
│   └── integration.py             # INTEGRATION
│
├── mlearning/
│   ├── qwen_model_manager.py      # DISABLED (returns None)
│   │   └── get_qwen_model_manager() → None
│   │
│   └── fractal_qwen_manager.py   # DISABLED (returns None)
│       └── get_fractal_qwen() → None
│
└── memory/
    └── fractal_graph_v2/
        ├── __init__.py
        ├── storage.py
        └── embeddings.py
```

### 2.2 Flow диаграмма генерации

```
User Query
    ↓
CoreBrain.process_query() [brain_query.py]
    ↓
QueryMixin._execute_with_web_search()
    ↓
GenerationCoordinator.generate() [generation_coordinator.py:319]
    ↓
UnifiedGeneratorProvider.generate()
    ↓
PipelineAdapter.generate() ИЛИ PipelineAdapter.process_query()
    ↓
UnifiedGenerator.generate() [unified_generator.py]
    ↓
SimpleRouter.route() → ModelType (LOGIC/CONTEXT/CODER)
    ↓
Llama(model_path)(prompt, max_tokens, temperature)
```

---

## 3. КРИТИЧЕСКИЕ МЕТОДЫ ГЕНЕРАЦИИ

| Метод | Файл | Строка | Возвращает | Статус |
|-------|------|--------|------------|--------|
| `generate()` | `unified_generator.py` | ~200 | `GenerationResult` | ✅ Работает |
| `generate()` | `pipeline_adapter.py` | 96 | `GenerationResult` | ✅ Работает |
| `process_query()` | `pipeline_adapter.py` | 47 | `Dict` с model_a/b_result | ✅ Работает |
| `_generate_response()` | `enhanced_reasoning_engine.py` | 402 | `str` | ✅ Работает |
| `_regenerate_response()` | `enhanced_reasoning_engine.py` | 433 | `str` | ✅ Работает |
| `process_query()` | `self_reasoning_engine.py` | 167 | `Dict` | ✅ Работает |

---

## 4. ОТКЛЮЧЁННЫЕ КОМПОНЕНТЫ

| Компонент | Файл | Возвращает | Причина |
|-----------|------|-----------|---------|
| `QwenModelManager` | `qwen_model_manager.py:142` | `None` | Заменён UnifiedGenerator |
| `FractalQwenManager` | `fractal_qwen_manager.py:273` | `None` | Заменён UnifiedGenerator |
| `FractalModelManager` | `brain_components.py` | Не инициализируется | Заменён UnifiedGenerator |
| `MLUnit` | `brain_components.py` | Не используется | Заменён UnifiedGenerator |
| `_use_fractal_for_prompts` | `enhanced_reasoning_engine.py` | `False` | Не используется |

---

## 5. КОНФИГУРАЦИЯ (brain_config.json)

```json
{
  "model": {
    "use_unified_generator": true,
    "llama_cpp_n_ctx": 4096,
    "llama_cpp_threads": 4,
    "models": {
      "logic": "eva_pie_architecture/models/gguf_models/ruadapt_qwen3_4b_q4_k_m.gguf",
      "context": "eva_pie_architecture/models/gguf_models/ruadapt_qwen3_4b_q4_k_m.gguf",
      "coder": "eva_pie_architecture/models/gguf_models/qwen2.5-coder-1.5b-instruct/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    },
    "two_model_pipeline": {
      "logic": {
        "n_ctx": 4096,
        "temperature": 0.3,
        "max_tokens": 2048
      },
      "context": {
        "n_ctx": 32768,
        "temperature": 0.3,
        "max_tokens": 4096
      },
      "coder": {
        "n_ctx": 4096,
        "temperature": 0.2,
        "max_tokens": 1024
      }
    }
  }
}
```

---

## 6. ИЗВЕСТНЫЕ ОШИБКИ И ПРОБЛЕМЫ

### 6.1 Исправленные ошибки

| # | Ошибка | Файл | Исправление |
|---|--------|------|-------------|
| 1 | `import *` не импортирует `_func` | `self_reasoning_engine.py` | Явные импорты вместо `import *` |
| 2 | Методы после класса | `self_reasoning_engine.py:881-889` | Перемещены внутрь класса |
| 3 | `_get_wikipedia_context` не найден | `sre_context.py:30` | Вызов через `_get_wikipedia_context(self, query)` |
| 4 | Пустая структура ответа | `pipeline_adapter.py` | Добавлены `model_a_result`, `model_b_result` |
| 5 | Отключённые fallback | `sre_context.py` | Упрощена цепочка без QwenModelManager |

### 6.2 Текущие предупреждения

| Компонент | Проблема | Серьёзность |
|-----------|----------|-------------|
| `memory_percent: 96.5%` | Высокое потребление памяти | ⚠️ Средняя |
| `GraphCurator.consolidation` | `name 'level' is not defined` | ⚠️ Средняя |
| `Tokenizer initialization` | `False` result | ⚠️ Низкая |
| `llama_context: n_ctx_seq (256) < n_ctx_train (262144)` | Ограничен контекст | ℹ️ Информация |

### 6.3 Мёртвый код

| Файл | Код | Причина |
|------|-----|---------|
| `enhanced_reasoning_engine.py:88-90` | `self._fractal_qwen = None` | Fractal Qwen отключён |
| `enhanced_reasoning_engine.py:487-498` | `fractal_qwen.generate_prompt()` | Никогда не выполнится |
| `sre_context.py:200+` | QwenModelManager fallback | Отключён |

---

## 7. ТОЧКИ ВХОДА

### 7.1 Запуск системы

```bash
# Полный запуск Eva AI
python -m eva_ai

# Через Web GUI
python start_webgui.py

# Инициализация только мозга
python -c "from eva_ai.core.core_brain import CoreBrain; b = CoreBrain(); b.initialize()"
```

### 7.2 Тестирование

```bash
# Тест SelfReasoningEngine
python -c "
from eva_ai.core.core_brain import CoreBrain
brain = CoreBrain()
brain.initialize()
result = brain.self_reasoning_engine.process_query('Что такое ИИ?', {})
print(result.get('status'))
"
```

---

## 8. ЛОГИ ДЛЯ ДЕБАГА

| Лог | Что проверить |
|-----|--------------|
| `test_sre.log` | Тест SelfReasoningEngine (статус PASSED/FAILED) |
| `initialization.log` | Инициализация UnifiedGenerator |
| `sre_test2.log` | Последний успешный тест |
| `full_test.log` | Полный цикл обработки запроса |

---

## 9. СЛЕДУЮЩИЕ ШАГИ

### Приоритет 1: Стабильность
- [ ] Исправить `GraphCurator.consolidation` ошибку
- [ ] Проверить высокое потребление памяти (96.5%)

### Приоритет 2: Оптимизация
- [ ] Добавить параметры generation в brain_config.json
- [ ] Увеличить `llama_cpp_threads` (сейчас 4, можно 8-12)
- [ ] Добавить `batch_size` параметр

### Приоритет 3: Очистка
- [ ] Удалить мёртвый код Fractal Qwen
- [ ] Удалить отключённые fallback в sre_context.py
- [ ] Добавить проверки на None перед вызовами

---

*Документ создан: 2026-04-12*  
*Автор: AI Architect Agent*
