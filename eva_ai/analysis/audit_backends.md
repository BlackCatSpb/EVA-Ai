# Отчёт: Backends
**Дата:** 2026-04-14  
**Объект аудита:** eva_ai/backends/ и eva_ai/core/

---

## 1. Структура

### 1.1 Директория `eva_ai/backends/pie/`

```
backends/
└── pie/
    ├── __init__.py           # Экспорт LayerWiseEmbedder
    ├── base.py               # BaseBackend, GenerationResult, GenerationConfig
    ├── gguf_backend.py       # GGUFBackend (llama.cpp)
    ├── transformers_backend.py # TransformersBackend (ЗАГЛУШКА)
    ├── onnx_backend.py       # ONNXBackend (ЗАГЛУШКА)
    └── layer_wise.py         # LayerWiseEmbedder система
```

### 1.2 Директория `eva_ai/core/` (ключевые файлы)

```
core/
├── unified_generator.py          # UnifiedGenerator - ОСНОВНОЙ генератор
├── hybrid_pipeline_adapter.py    # HybridPipelineAdapter - адаптер с dual/fractal режимами
├── openvino_generator.py       # OpenVINOGenerator + Registry (шаринг GPU)
├── model_access_manager.py      # ModelAccessManager - координация доступа
├── pipeline_adapter.py          # PipelineAdapter - совместимость с TwoModelPipeline
├── fractal_pipeline.py          # FractalPipeline
├── recursive_model_pipeline.py  # RecursiveModelPipeline
└── generation/
    └── generation_coordinator.py # UnifiedGenerationCoordinator
```

### 1.3 DualGenerator (FractalGraph v2)

```
memory/fractal_graph_v2/
├── dual_generator.py          # DualGenerator (2 физических инстанса)
└── dual_generator_pie.py      # DualGeneratorPie (Pie-специфичный)
```

---

## 2. Импорты

### 2.1 Бэкенды (`backends/pie/`)

| Файл | Импорты | Статус |
|------|---------|--------|
| `base.py` | `abc, typing, dataclasses, pathlib` | ✅ Валидны |
| `gguf_backend.py` | `llama_cpp`, `logging`, `pathlib`, `time` | ✅ Валидны |
| `transformers_backend.py` | `...base` | ✅ Валидны (заглушка) |
| `onnx_backend.py` | `...base` | ✅ Валидны (заглушка) |
| `layer_wise.py` | Нет внешних | ✅ Валидны |

### 2.2 Core генерация (`core/`)

| Файл | Импорты | Статус |
|------|---------|--------|
| `unified_generator.py` | `llama_cpp`, `context_chunking`, `model_access_manager`, `openvino_generator` | ✅ Валидны |
| `hybrid_pipeline_adapter.py` | `llama_cpp`, `fractal_pipeline`, `dual_generator`, `recursive_model_pipeline` | ✅ Валидны |
| `openvino_generator.py` | `openvino_genai`, `asyncio`, `threading` | ⚠️ Опционально (try/except) |
| `model_access_manager.py` | `queue, threading, concurrent.futures` | ✅ Валидны |
| `generation_coordinator.py` | Внутренние компоненты | ✅ Валидны |

### 2.3 Потенциальные проблемы импортов

```python
# gguf_backend.py
from llama_cpp import Llama  # Требует pip install llama-cpp-python

# openvino_generator.py  
import openvino_genai as ov_genai  # Требует pip install openvino-genai
```

---

## 3. Реализация

### 3.1 Базовая архитектура (`BaseBackend`)

```python
# eva_ai/backends/pie/base.py
class BaseBackend(ABC):
    @abstractmethod
    def load_model(self, path: str, **kwargs)
    @abstractmethod  
    def generate(self, prompt, config) -> GenerationResult
    @abstractmethod
    def generate_stream(self, prompt, config) -> Iterator[str]
    @abstractmethod
    def tokenize(self, text: str) -> List[int]
    @abstractmethod
    def detokenize(self, tokens: List[int]) -> str
```

**Реализации:**
- ✅ `GGUFBackend` - полная реализация через llama.cpp
- ⚠️ `TransformersBackend` - заглушка (NotImplementedError)
- ⚠️ `ONNXBackend` - заглушка (NotImplementedError)

### 3.2 UnifiedGenerator (ОСНОВНОЙ)

```python
# eva_ai/core/unified_generator.py
class UnifiedGenerator:
    # Три модели: LOGIC, CONTEXT, CODER
    # CPU: Logic/Context модель
    # GPU.0: Coder модель
    
    def generate(query, context, max_tokens, temperature, task_type)
    def generate_dual(query, ...)        # 2-этапная: LOGIC -> CONTEXT
    def generate_unified(query, ...)     # 1-этапная с XML-выводом
    def generate_iterative(query, ...)    # iterative с концептами/противоречиями
    def generate_streaming(query, ...)    # стриминг
```

**Роутер SimpleRouter:**
- CODER: по ключевым словам кода (python, function, api...)
- CONTEXT: длинные запросы, вопросительные слова
- LOGIC: по умолчанию

### 3.3 HybridPipelineAdapter

```python
# eva_ai/core/hybrid_pipeline_adapter.py
class HybridPipelineAdapter:
    MODE_FRACTAL = 'fractal'      # Только FractalPipeline
    MODE_DUAL = 'dual'            # DualGenerator (БЫСТРЫЙ)
    MODE_RECURSIVE = 'recursive'  # RecursiveModelPipeline
    MODE_HYBRID = 'hybrid'        # Fractal + fallback
```

### 3.4 DualGenerator (FractalGraph v2)

```python
# eva_ai/memory/fractal_graph_v2/dual_generator.py
class DualGenerator:
    # 2 физических инстанса:
    # - CondensedGenerator: краткие ответы (max_tokens=1024, temp=0.1)
    # - ExtendedGenerator: развёрнутые (max_tokens=4096, temp=0.35)
    
    def generate_condensed(query)
    def generate_extended(query)  
    def generate_large(query)        # chunked generation
    def generate_streaming(query)
```

### 3.5 OpenVINOGenerator

```python
# eva_ai/core/openvino_generator.py
class OpenVINOGenerator:
    # GPU Model Sharing через Registry (синглтон)
    # Поддержка CPU/GPU
    
    def generate(prompt, max_tokens, temperature)
    def generate_streaming(prompt, ...)
```

**OpenVINOGeneratorRegistry:**
- Синглтон для шаринга модели между генераторами
- Ключ: (model_path, device)
- Reference counting

### 3.6 ModelAccessManager

```python
# eva_ai/core/model_access_manager.py
class ModelAccessManager:
    # Singleton
    # Приоритетная очередь: CRITICAL > HIGH > NORMAL > LOW
    # EventBus интеграция
```

---

## 4. Интеграция

### 4.1 Инициализация в CoreBrain

```python
# eva_ai/core/brain_components.py

# 1. HybridPipelineAdapter
brain.two_model_pipeline = HybridPipelineAdapter(
    mode=pipeline_mode,  # dual/fractal/recursive/hybrid
    model_a=model_a,
    model_b=model_b,
    fractal_graph=graph
)

# 2. UnifiedGenerator (отдельный, не через HybridPipelineAdapter)
brain.unified_generator = create_unified_generator(config, graph, brain)
```

### 4.2 GenerationCoordinator

```python
# eva_ai/generation/generation_coordinator.py

# Провайдеры по приоритету:
# 0. UnifiedGeneratorProvider (brain.two_model_pipeline)
# 1. HybridModelProvider (HybridModelManager)
# 2. FractalModelProvider (fractal_model_manager)  
# 3. ResponseGeneratorProvider
# 4. MLUnitProvider
```

### 4.3 ModelAccessManager - координация

```
UnifiedGenerator.generate()
    │
    └── ModelAccessManager.request_access(priority, task_type, callback)
            │
            ├── CRITICAL: query (пользовательские запросы)
            ├── HIGH: self_dialog, concept_mining, contradiction_mining, coder
            ├── NORMAL: default
            └── LOW: долгосрочное обучение
```

### 4.4 OpenVINO + llama.cpp сосуществование

```python
# UnifiedGenerator._init_openvino_devices()
# CPU: Logic/Context модель (openvino_cpu)
# GPU: Coder модель (openvino_coder)

# llama.cpp модели:
# - загружаются через GGUFBackend
# - используются в DualGenerator (fractal_graph_v2)
# - используются в RecursiveModelPipeline
```

---

## 5. Оценка

### 5.1 Плюсы

| Компонент | Оценка | Комментарий |
|-----------|--------|-------------|
| BaseBackend абстракция | ✅ Хорошо | Чёткий интерфейс для扩展 |
| UnifiedGenerator | ✅ Хорошо | 3 модели, роутинг, итеративная генерация |
| HybridPipelineAdapter | ✅ Хорошо | 4 режима, прозрачный fallback |
| DualGenerator | ✅ Хорошо | 2 физических инстанса, chunked generation |
| OpenVINO + Registry | ✅ Хорошо | GPU шаринг, CPU/GPU архитектура |
| ModelAccessManager | ✅ Хорошо | Приоритизация, EventBus |
| GenerationCoordinator | ✅ Хорошо | Провайдеры с fallback |

### 5.2 Проблемы и риски

| Проблема | Серьёзность | Описание |
|----------|-------------|----------|
| TransformersBackend заглушка | ⚠️ Средняя | Не реализована, но есть абстракция |
| ONNXBackend заглушка | ⚠️ Средняя | Не реализована, но есть абстракция |
| Сложность архитектуры | 🔴 Высокая | 3 системы генерации (Unified, Hybrid, Dual) |
| Импорты llama.cpp | ⚠️ Средняя | Требует системной установки |
| OpenVINO опциональность | ⚠️ Средняя | try/except, fallback на llama.cpp |

### 5.3 Рекомендации

1. **Документация:** Архитектура сложная - добавить диаграмму последовательности
2. **Тестирование:** Критичные компоненты (UnifiedGenerator, ModelAccessManager) требуют unit-тестов
3. **Оптимизация:** ModelAccessManager использует polling - рассмотреть async/await
4. **Резервирование:** OpenVINO failure обрабатывается, но llama.cpp fallback не автоматический

### 5.4 Итоговая оценка: 7/10

**Сильные стороны:**
- Гибкая модульная архитектура
- Несколько уровней абстракции
- Хорошее разделение concerns

**Слабые стороны:**
- Высокая сложность
- Не все бэкенды реализованы
- Требуется координация между 3+ системами

---

## Файлы для проверки

```
eva_ai/backends/pie/
├── __init__.py
├── base.py
├── gguf_backend.py
├── transformers_backend.py
├── onnx_backend.py
└── layer_wise.py

eva_ai/core/
├── unified_generator.py
├── hybrid_pipeline_adapter.py
├── openvino_generator.py
├── model_access_manager.py
├── pipeline_adapter.py
└── generation/
    └── generation_coordinator.py

eva_ai/memory/fractal_graph_v2/
├── dual_generator.py
└── dual_generator_pie.py
```
